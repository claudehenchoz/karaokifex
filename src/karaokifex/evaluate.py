"""karaokifex-eval: how close the karaoke word timings are to a hand-made reference.

Put a reference into a song folder and run `karaokifex-eval <song folder>...`:

- `reference.ass`: an ASS file with `\\k`/`\\kf`/`\\ko` syllable timing, e.g. made in
  Aegisub against the *video* (not the album recording);
- `reference.lrc`: enhanced LRC with `<mm:ss.xx>` word tags, timed against the video.

Reported per song (and per timing source): median onset error, and the share of
words starting within 100 ms and 300 ms of the reference. `--recompute` re-runs the
alignment from the cached intermediate files (transcript, vocal activity, forced
alignment), so thresholds can be tuned without the GPU; `--baseline` does so with
whisper matching only, for comparison. Keep the temporary files of golden-set songs
(run karaokifex with --keep-temp).
"""

from __future__ import annotations

import difflib
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import click
from rich.table import Table

from karaokifex.models import TimedWord

REFERENCE_ASS = "reference.ass"
REFERENCE_LRC = "reference.lrc"

_TAG_BLOCK = re.compile(r"(\{[^}]*\})")
_K_TAG = re.compile(r"\\(?:kf|ko|k|K)(\d+)")
_ASS_TIME = re.compile(r"(\d+):(\d+):(\d+(?:\.\d+)?)")


@dataclass
class Report:
    words: int = 0  # reference words paired with one of ours
    reference: int = 0
    errors: list[float] = field(default_factory=list)  # seconds, absolute
    by_source: dict[str, list[float]] = field(default_factory=dict)

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.errors) * 1000 if self.errors else None

    def within(self, seconds: float, errors: list[float] | None = None) -> float:
        errors = self.errors if errors is None else errors
        return sum(e <= seconds for e in errors) / len(errors) if errors else 0.0


def parse_reference_ass(text: str) -> list[TimedWord]:
    """Word start times from the `\\k` tags of every Dialogue line."""
    from karaokifex.timing import normalize

    words: list[TimedWord] = []
    for row in text.splitlines():
        if not row.startswith("Dialogue:"):
            continue
        fields = row.split(",", 9)
        if len(fields) < 10 or not (start := _ass_time(fields[1])):
            continue
        cursor, duration = start, 0.0
        open_word: list[str] | None = None
        begun = start
        for piece in _TAG_BLOCK.split(fields[9]):
            if piece.startswith("{"):
                cursor += duration
                duration = sum(int(v) for v in _K_TAG.findall(piece)) / 100
                continue
            for index, chunk in enumerate(re.split(r"(\s+)", piece)):
                if not chunk:
                    continue
                if chunk.isspace():
                    if open_word is not None:
                        words.append(TimedWord("".join(open_word), begun, begun))
                        open_word = None
                elif open_word is None:
                    open_word, begun = [chunk], cursor
                else:
                    open_word.append(chunk)
        if open_word is not None:
            words.append(TimedWord("".join(open_word), begun, begun))
    return [word for word in words if normalize(word.text)]


def parse_reference_lrc(text: str) -> list[TimedWord]:
    from karaokifex.steps.lyrics import parse_lrc
    from karaokifex.timing import split_words

    words = []
    for line in parse_lrc(text):
        if line.word_starts:
            words += [TimedWord(word, start, start) for word, start in zip(split_words(line.text), line.word_starts)
                      if start is not None]
    return words


def compare(reference: list[TimedWord], ours: list[TimedWord]) -> Report:
    """Pair identical words in order and measure how far our starts are from the reference's."""
    from karaokifex.timing import normalize

    matcher = difflib.SequenceMatcher(None, [normalize(w.text) for w in reference],
                                      [normalize(w.text) for w in ours], autojunk=False)
    report = Report(reference=len(reference))
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            ref, our = reference[block.a + offset], ours[block.b + offset]
            error = abs(our.start - ref.start)
            report.errors.append(error)
            report.by_source.setdefault(our.source or "?", []).append(error)
    report.words = len(report.errors)
    return report


def load_reference(folder: Path) -> list[TimedWord] | None:
    if (path := folder / REFERENCE_ASS).exists():
        return parse_reference_ass(path.read_text(encoding="utf-8-sig"))
    if (path := folder / REFERENCE_LRC).exists():
        return parse_reference_lrc(path.read_text(encoding="utf-8-sig"))
    return None


def load_ours(folder: Path) -> list[TimedWord]:
    data = json.loads((folder / "timings.json").read_text(encoding="utf-8"))
    return [TimedWord(**word) for line in data["lines"] for word in line]


def recompute(folder: Path, *, baseline: bool = False) -> list[TimedWord]:
    """Re-run the (pure) alignment from the cached intermediate files of a song folder."""
    from karaokifex.activity import Activity
    from karaokifex.pipeline import align_candidates, load_forced
    from karaokifex.steps import lyrics, transcription
    from karaokifex.timing import filter_heard, lines_from_words
    from karaokifex.workspace import Workspace

    ws = Workspace(folder, folder.name)
    candidates = lyrics.load_lyrics(ws.lyrics_json)
    words, language = transcription.load_transcript(ws.transcript_json)
    activity = None if baseline or not ws.lead_activity.exists() else Activity.load(ws.lead_activity)
    mix = transcription.load_transcript(ws.transcript_mix_json)[0] if ws.transcript_mix_json.exists() else None
    if not candidates:
        return [w for line in lines_from_words(filter_heard(words, activity)) for w in line.words]
    forced = [] if baseline else load_forced(ws.forced_json)
    # An lrc reference is usually the lrclib entry itself: its word tags would make the comparison circular.
    ranked = align_candidates(candidates, words, language, activity, forced, mix=None if baseline else mix,
                              use_word_tags=not (folder / REFERENCE_LRC).exists())
    return [word for line in ranked[0][1].lines for word in line.words]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("folders", nargs=-1, required=True, type=click.Path(file_okay=False, exists=True, path_type=Path))
@click.option("--recompute", "recompute_timings", is_flag=True,
              help="Re-run the alignment from the cached files instead of reading timings.json.")
@click.option("--baseline", is_flag=True, help="With --recompute: whisper matching only (no forced alignment or VAD).")
def main(folders: tuple[Path, ...], recompute_timings: bool, baseline: bool) -> None:
    """Compare the karaoke timings of song FOLDERS with their reference.ass / reference.lrc."""
    from karaokifex.console import console

    table = Table(title="Word onset accuracy", title_justify="left")
    for column in ("song", "source", "words", "median", "≤100 ms", "≤300 ms"):
        table.add_column(column, justify="left" if column in ("song", "source") else "right")
    everything = Report()
    for folder in folders:
        reference = load_reference(folder)
        if reference is None:
            console.print(f"[yellow]{folder.name}: no {REFERENCE_ASS} or {REFERENCE_LRC}, skipped")
            continue
        ours = recompute(folder, baseline=baseline) if recompute_timings else load_ours(folder)
        report = compare(reference, ours)
        everything.errors += report.errors
        everything.reference += report.reference
        table.add_row(folder.name, "all", f"{report.words}/{report.reference}", _ms(report.median_ms),
                      f"{report.within(0.1):.0%}", f"{report.within(0.3):.0%}")
        for source, errors in sorted(report.by_source.items()):
            table.add_row("", source, str(len(errors)), _ms(statistics.median(errors) * 1000),
                          f"{report.within(0.1, errors):.0%}", f"{report.within(0.3, errors):.0%}")
    if len(folders) > 1 and everything.errors:
        table.add_row("[bold]all songs", "", f"{len(everything.errors)}/{everything.reference}",
                      _ms(everything.median_ms), f"{everything.within(0.1):.0%}", f"{everything.within(0.3):.0%}")
    console.print(table)


def _ass_time(text: str) -> float | None:
    if not (match := _ASS_TIME.fullmatch(text.strip())):
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _ms(value: float | None) -> str:
    return "–" if value is None else f"{value:.0f} ms"

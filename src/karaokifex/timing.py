"""Time lyric lines (lrclib) word by word — pure functions, no I/O.

The lyrics decide *what* is displayed and how it is split into lines; the audio
decides *when* each word is sung. Several sources of word times are combined,
best first:

- `lrc-tag`: enhanced-LRC word tags, mapped onto the video's timeline;
- `forced`: wav2vec2 forced alignment of the known lyric line inside its window;
- `whisper`: words whisperx heard, paired with lyric words by a DP sequence
  alignment (only inside the line's window, so a repeated chorus is matched to
  the right occurrence);
- `lrc-line` / `interpolated`: everything else, packed into the voiced parts
  of the gap it falls in.

`plan_alignment` first maps lrclib's timeline onto the video's (mapping.py),
derives a search window per line and finds lines that were cut from the video.
The lead-vocal activity (activity.py), when given, vetoes timings: heard words in
silence are dropped, words don't start in silence, and held notes last until
the singing stops.
"""

from __future__ import annotations

import difflib
import functools
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING, Any, Sequence

import jellyfish

from karaokifex.mapping import MIN_OFFSET_SAMPLES, TimeMap, fit_time_map, xcorr_offset
from karaokifex.models import LyricLine, TimedLine, TimedWord

if TYPE_CHECKING:
    from karaokifex.activity import Activity

MATCH_WINDOW = 5.0  # seconds a matched word may lie outside its line window
ANCHOR_SLACK = 15.0  # the same while collecting anchors for the time map (the map isn't known yet)
FUZZY_RATIO = 0.6  # minimum spelling similarity to pair two non-identical words
PHONETIC = 0.85  # similarity of words that sound alike ("there"/"their"); English only
MERGE_MIN = 0.8  # similarity needed to pair one word with two ("alright" / "all right")
LAST_LINE_WINDOW = 15.0  # assumed length of the final lrclib line
WINDOW_PAD = 1.0  # seconds added around a line's window for forced alignment
SECONDS_PER_CHAR = 0.08  # natural word duration estimate, clamped to the range below
MIN_WORD, MAX_WORD = 0.2, 0.8
MIN_DURATION = 0.05
FORCED_LINE_MIN = 0.25  # mean forced-alignment score above which a line trusts forced alignment
FORCED_WORD_MIN = 0.1  # forced words below this lose to a whisper match or get interpolated
MIN_HEARD_VOICED = 0.2  # heard words with less of their span voiced are hallucinations
CUT_FIT = 0.4  # unheard lines need this share of their natural duration as voiced time, or they were cut
MAX_CUT_SHARE = 0.35  # more cut lines than this means the time map is wrong: cut nothing
ONSET_SNAP = 0.3  # seconds a line's first word may move to a phrase onset
MAX_HOLD = 4.0  # seconds a line's last word may be stretched to the end of a held note
VOICED_PACK_MIN = 0.5  # interpolate into voiced time only if there is at least this share of the natural duration
VOTE_AGREE = 0.3  # seconds within which the lead-stem and full-mix transcriptions agree

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
# Spellings that differ between lyric sites and whisper; compared with apostrophes and spaces removed.
_EQUIVALENTS = {
    "wanna": "wantto", "gonna": "goingto", "gotta": "gotto", "cause": "because", "cuz": "because",
    "coz": "because", "alright": "allright", "kinda": "kindof", "outta": "outof", "lemme": "letme",
    "gimme": "giveme", "ok": "okay",
}

Window = tuple[float, float]
Times = tuple[float, float, float | None]  # start, end, score


@dataclass(frozen=True)
class Alignment:
    lines: list[TimedLine]
    matched: int  # lyric words whisperx heard
    total: int
    forced: int = 0  # lyric words timed by forced alignment
    forced_score: float | None = None  # mean forced score over the displayed words
    cut: int = 0  # lyric lines left out because the video doesn't contain them

    @property
    def match_ratio(self) -> float:
        return self.matched / self.total if self.total else 0.0

    @property
    def quality(self) -> float:
        """How well these lyrics fit the audio (0–1), for choosing between lyrics versions."""
        if self.forced_score is None:
            return self.match_ratio
        return 0.5 * self.match_ratio + 0.5 * self.forced_score


@dataclass(frozen=True)
class AlignmentPlan:
    """Where each lyric line is searched for in the video (indices: lines with words, see `kept_lines`)."""

    time_map: TimeMap = field(default_factory=TimeMap)
    starts: tuple[float | None, ...] = ()  # lrclib line starts mapped onto the video
    windows: tuple[Window | None, ...] = ()
    cut: frozenset[int] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {"time_map": self.time_map.to_dict(), "starts": list(self.starts),
                "windows": [None if w is None else list(w) for w in self.windows], "cut": sorted(self.cut)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentPlan:
        return cls(TimeMap.from_dict(data["time_map"]), tuple(data["starts"]),
                   tuple(None if w is None else (w[0], w[1]) for w in data["windows"]), frozenset(data["cut"]))


@dataclass(frozen=True)
class _Token:
    line: int
    position: int  # word index within its line
    text: str
    norm: str

    @property
    def natural(self) -> float:
        """Rough time it takes to sing this word."""
        return min(max(len(self.norm) * SECONDS_PER_CHAR, MIN_WORD), MAX_WORD)


@dataclass(frozen=True)
class _Match:
    """Lyric tokens [i, i + ni) sung as heard words [j, j + nj); ni, nj ∈ {1, 2}, not both 2."""

    i: int
    ni: int
    j: int
    nj: int


def normalize(word: str) -> str:
    """Spelling used for comparing words: "Don’t!" -> "don't", "Café" -> "cafe"."""
    decomposed = unicodedata.normalize("NFKD", word.translate(_APOSTROPHES))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^\w']|_", "", stripped.casefold()).strip("'")


def split_words(text: str) -> list[str]:
    """Split a lyric line into words, gluing punctuation-only tokens ("-", "...") onto a neighbour."""
    words: list[str] = []
    leading = ""
    for token in text.split():
        if normalize(token):
            words.append(leading + token)
            leading = ""
        elif words:
            words[-1] += " " + token
        else:
            leading += token + " "
    return words


def kept_lines(lines: Sequence[LyricLine]) -> list[tuple[LyricLine, list[str]]]:
    """The lines that have words, with their words — what every per-line index here refers to."""
    return [(line, words) for line in lines if (words := split_words(line.text))]


def filter_heard(heard: Sequence[TimedWord], activity: Activity | None = None) -> list[TimedWord]:
    """Heard words worth matching: real words, and (with activity) not in silence — whisper hallucinates there."""
    words = [word for word in heard if normalize(word.text)]
    if activity is None:
        return words
    return [w for w in words if activity.voiced_fraction(w.start, max(w.end, w.start + 0.02)) >= MIN_HEARD_VOICED]


def _tokens(kept: Sequence[tuple[LyricLine, list[str]]]) -> list[_Token]:
    return [_Token(index, position, word, normalize(word))
            for index, (_, words) in enumerate(kept) for position, word in enumerate(words)]


# --- planning: time map, windows, cut lines -------------------------------------------


def plan_alignment(lines: Sequence[LyricLine], heard: Sequence[TimedWord], activity: Activity | None = None, *,
                   language: str | None = None, window: float = MATCH_WINDOW) -> AlignmentPlan:
    kept = kept_lines(lines)
    tokens = _tokens(kept)
    heard = filter_heard(heard, activity)
    starts = [line.start for line, _ in kept]
    synced = any(start is not None for start in starts)
    time_map, supported = TimeMap(), False
    if synced:
        # Candidate maps: from anchors found without any timing constraint, and from anchors found near the
        # onset cross-correlation's offset. The correlation can lock onto a wrong peak (repetitive songs), so
        # the map under which more lyric words match what whisper heard wins; ties go to the first.
        prior = xcorr_offset(starts, activity.onset_envelope(), activity.hop) if activity is not None else None
        searches = [([None] * len(kept), window)]
        if prior is not None:
            searches.append((_line_windows([None if s is None else s + prior for s in starts]), ANCHOR_SLACK))
        candidates = []
        for search, slack in searches:
            anchors = _anchors(kept, tokens, heard, _match_pairs(tokens, heard, search, slack, language))
            candidate = fit_time_map(anchors, prior)
            mapped = [None if s is None else candidate.map(s) for s in starts]
            matched = len(_match_pairs(tokens, heard, _line_windows(mapped), window, language))
            candidates.append((matched, candidate, anchors))
        _, time_map, anchors = max(candidates, key=lambda c: c[0])
        supported = time_map.support(anchors) >= MIN_OFFSET_SAMPLES
    mapped = tuple(None if s is None else time_map.map(s) for s in starts)
    times = _matched_times(tokens, heard, _match_pairs(tokens, heard, _line_windows(mapped), window, language))
    _drop_non_monotonic(times)

    by_line = [[k for k in range(len(tokens)) if tokens[k].line == index] for index in range(len(kept))]
    natural = [sum(tokens[k].natural for k in indices) for indices in by_line]
    heard_lines = [any(times[k] is not None for k in indices) for indices in by_line]
    ends_before, starts_after = _neighbour_times(by_line, times)

    windows: list[Window | None] = []
    for index in range(len(kept)):
        windows.append(_line_search_window(index, mapped, by_line[index], times, natural[index],
                                           ends_before[index], starts_after[index], activity))
    cut = _cut_lines(mapped, heard_lines, natural, ends_before, starts_after, activity)
    # Only trust "the video doesn't contain this line" when heard words back the time map, and never for
    # a large part of the song: that means the map is wrong, not that the video cut half of it.
    if (synced and not supported) or len(cut) > MAX_CUT_SHARE * len(kept):
        cut = set()
    return AlignmentPlan(time_map, mapped, tuple(windows), frozenset(cut))


def forced_requests(lines: Sequence[LyricLine], plan: AlignmentPlan, *,
                    use_word_tags: bool = True) -> list[tuple[int, list[str], Window]]:
    """(line index, normalised words, window) for every line forced alignment should time."""
    requests = []
    for index, (line, words) in enumerate(kept_lines(lines)):
        if index in plan.cut or index >= len(plan.windows) or (window := plan.windows[index]) is None:
            continue
        if use_word_tags and line.word_starts:
            continue
        requests.append((index, [normalize(word) for word in words], window))
    return requests


def _anchors(kept: Sequence[tuple[LyricLine, list[str]]], tokens: Sequence[_Token], heard: Sequence[TimedWord],
             matches: Sequence[_Match]) -> list[tuple[float, float]]:
    """(lrclib time, video time) pairs: heard first words of synced lines, and heard tagged words."""
    anchors = []
    for match in matches:
        token = tokens[match.i]
        line = kept[token.line][0]
        if token.position == 0 and line.start is not None:
            anchors.append((line.start, heard[match.j].start))
        elif line.word_starts and (tag := line.word_starts[token.position]) is not None:
            anchors.append((tag, heard[match.j].start))
    return anchors


def _line_windows(starts: Sequence[float | None]) -> list[Window | None]:
    """(start, end) of each lyric line according to lrclib, or None when unsynced."""
    windows: list[Window | None] = []
    for index, start in enumerate(starts):
        if start is None:
            windows.append(None)
            continue
        following = next((s for s in starts[index + 1:] if s is not None), None)
        windows.append((start, following if following is not None else start + LAST_LINE_WINDOW))
    return windows


def _neighbour_times(by_line: Sequence[list[int]], times: Sequence[Times | None]
                     ) -> tuple[list[float | None], list[float | None]]:
    """Per line: end of the last heard word before it, start of the first heard word after it."""
    ends_before: list[float | None] = []
    last: float | None = None
    for indices in by_line:
        ends_before.append(last)
        for k in indices:
            if (t := times[k]) is not None:
                last = t[1]
    starts_after: list[float | None] = [None] * len(by_line)
    first: float | None = None
    for index in range(len(by_line) - 1, -1, -1):
        starts_after[index] = first
        for k in reversed(by_line[index]):
            if (t := times[k]) is not None:
                first = t[0]
    return ends_before, starts_after


def _line_search_window(index: int, mapped: Sequence[float | None], indices: Sequence[int],
                        times: Sequence[Times | None], natural: float, end_before: float | None,
                        start_after: float | None, activity: Activity | None) -> Window | None:
    heard_times = [t for k in indices if (t := times[k]) is not None]
    if (start := mapped[index]) is not None:
        following = next((s for s in mapped[index + 1:] if s is not None), start + LAST_LINE_WINDOW)
        low, high = start - WINDOW_PAD, following + WINDOW_PAD
    elif heard_times:
        low, high = heard_times[0][0] - WINDOW_PAD, heard_times[-1][1] + WINDOW_PAD
    elif end_before is not None or start_after is not None:
        low = end_before if end_before is not None else max(0.0, start_after - 2 * natural)  # type: ignore[operator]
        high = start_after if start_after is not None else low + 2 * natural
    else:
        return None
    # Don't reach into the neighbours' heard words (unless that leaves too little room).
    clamped_low = max(low, end_before - 0.2) if end_before is not None else low
    clamped_high = min(high, start_after + 0.2) if start_after is not None else high
    if heard_times:
        clamped_low, clamped_high = min(clamped_low, heard_times[0][0]), max(clamped_high, heard_times[-1][1])
    if clamped_high - clamped_low >= natural:
        low, high = clamped_low, clamped_high
    if activity is not None and (voiced := activity.voiced_intervals(low, high)):
        low, high = max(low, voiced[0][0] - 0.3), min(high, voiced[-1][1] + 0.3)
    low = max(0.0, low)
    return (low, max(high, low + natural))


def _cut_lines(mapped: Sequence[float | None], heard_lines: Sequence[bool], natural: Sequence[float],
               ends_before: Sequence[float | None], starts_after: Sequence[float | None],
               activity: Activity | None) -> set[int]:
    """Unheard lines the video can't contain: silence where they belong, or no room between heard neighbours."""
    if activity is None:
        return set()

    def voiced_seconds(start: float, end: float) -> float:
        return sum(b - a for a, b in activity.voiced_intervals(start, end))

    cut: set[int] = set()
    for index, start in enumerate(mapped):
        if start is None or heard_lines[index]:
            continue
        following = next((s for s in mapped[index + 1:] if s is not None), start + LAST_LINE_WINDOW)
        if voiced_seconds(start, following) < CUT_FIT * natural[index]:
            cut.add(index)
    index = 0
    while index < len(heard_lines):  # runs of unheard lines between two heard ones
        if heard_lines[index]:
            index += 1
            continue
        end = index
        while end < len(heard_lines) and not heard_lines[end]:
            end += 1
        left, right = ends_before[index], starts_after[end - 1]
        if left is not None and right is not None:
            if voiced_seconds(left, right) < CUT_FIT * sum(natural[index:end]):
                cut.update(range(index, end))
        index = end
    return cut


# --- matching lyric words with heard words -----------------------------------------------


@functools.lru_cache(maxsize=16384)
def _phonetic_code(word: str) -> str:
    return jellyfish.metaphone(word.replace("'", "")) if word else ""


@functools.lru_cache(maxsize=65536)
def _similarity(lyric: str, heard: str, phonetic: bool = False) -> float:
    """1 for identical words, a bit less for near-misses ("colour"/"color", "there"/"their"), 0 otherwise."""
    lyric, heard = lyric.replace("'", ""), heard.replace("'", "")
    lyric, heard = _EQUIVALENTS.get(lyric, lyric), _EQUIVALENTS.get(heard, heard)
    if lyric == heard:
        return 1.0
    ratio = difflib.SequenceMatcher(None, lyric, heard).ratio()
    similarity = ratio * 0.9 if ratio >= FUZZY_RATIO else 0.0
    if phonetic and similarity < PHONETIC and _phonetic_code(lyric) and _phonetic_code(lyric) == _phonetic_code(heard):
        return PHONETIC
    return similarity


def _weight(word: TimedWord) -> float:
    """Trust in a heard word, from its alignment score."""
    return 1.0 if word.score is None else 0.5 + 0.5 * min(max(word.score, 0.0), 1.0)


def _match_pairs(tokens: Sequence[_Token], heard: Sequence[TimedWord], windows: Sequence[Window | None],
                 slack: float, language: str | None = None) -> list[_Match]:
    """Order-preserving pairing of lyric and heard words maximising total (score-weighted) similarity.

    An LCS-style dynamic programme that also allows one lyric word sung as two heard
    words ("alright" / "all right") and vice versa. A pair is only allowed when the
    heard word lies within its lyric line's window (± slack).
    """
    phonetic = language == "en"
    heard_norm = [normalize(word.text) for word in heard]
    weights = [_weight(word) for word in heard]
    rows, columns = len(tokens) + 1, len(heard) + 1
    score = [[0.0] * columns for _ in range(rows)]
    move = [bytearray(columns) for _ in range(rows)]  # 0 skip lyric, 1 skip heard, 2 = 1:1, 3 = 1:2, 4 = 2:1
    bounds = []
    for token in tokens:
        window = windows[token.line]
        bounds.append((window[0] - slack, window[1] + slack) if window else (-math.inf, math.inf))

    for i in range(1, rows):
        token = tokens[i - 1]
        low, high = bounds[i - 1]
        previous, row, moves = score[i - 1], score[i], move[i]
        before = score[i - 2] if i >= 2 else None
        pair_ok = before is not None and tokens[i - 2].line == token.line
        for j in range(1, columns):
            if previous[j] >= row[j - 1]:
                best, how = previous[j], 0
            else:
                best, how = row[j - 1], 1
            start = heard[j - 1].start
            if low <= start <= high:
                similarity = _similarity(token.norm, heard_norm[j - 1], phonetic)
                if similarity and previous[j - 1] + similarity * weights[j - 1] > best:
                    best, how = previous[j - 1] + similarity * weights[j - 1], 2
                if j >= 2 and token.norm[:1] == heard_norm[j - 2][:1] and low <= heard[j - 2].start:
                    similarity = _similarity(token.norm, heard_norm[j - 2] + heard_norm[j - 1], phonetic)
                    gain = similarity * (weights[j - 2] + weights[j - 1]) / 2
                    if similarity >= MERGE_MIN and previous[j - 2] + gain > best:
                        best, how = previous[j - 2] + gain, 3
                if pair_ok and tokens[i - 2].norm[:1] == heard_norm[j - 1][:1]:
                    similarity = _similarity(tokens[i - 2].norm + token.norm, heard_norm[j - 1], phonetic)
                    gain = similarity * weights[j - 1] * 1.5  # two lyric words explained by one heard word
                    if similarity >= MERGE_MIN and before[j - 1] + gain > best:  # type: ignore[index]
                        best, how = before[j - 1] + gain, 4  # type: ignore[index]
            row[j], moves[j] = best, how

    matches: list[_Match] = []
    i, j = rows - 1, columns - 1
    while i > 0 and j > 0:
        how = move[i][j]
        if how == 0:
            i -= 1
        elif how == 1:
            j -= 1
        elif how == 2:
            matches.append(_Match(i - 1, 1, j - 1, 1))
            i, j = i - 1, j - 1
        elif how == 3:
            matches.append(_Match(i - 1, 1, j - 2, 2))
            i, j = i - 1, j - 2
        else:
            matches.append(_Match(i - 2, 2, j - 1, 1))
            i, j = i - 2, j - 1
    return matches[::-1]


def _matched_times(tokens: Sequence[_Token], heard: Sequence[TimedWord], matches: Sequence[_Match]
                   ) -> list[Times | None]:
    times: list[Times | None] = [None] * len(tokens)
    for m in matches:
        first, last = heard[m.j], heard[m.j + m.nj - 1]
        scores = [w.score for w in heard[m.j: m.j + m.nj] if w.score is not None]
        score = statistics.fmean(scores) if scores else None
        if m.ni == 1:
            times[m.i] = (first.start, last.end, score)
        else:  # one heard word covers two lyric words: split it by their natural lengths
            a, b = tokens[m.i].natural, tokens[m.i + 1].natural
            middle = first.start + (first.end - first.start) * a / (a + b)
            times[m.i], times[m.i + 1] = (first.start, middle, score), (middle, first.end, score)
    return times


def _drop_non_monotonic(times: list[Times | None]) -> None:
    """Safety net in case whisperx reports words out of order."""
    last = -math.inf
    for index, value in enumerate(times):
        if value is None:
            continue
        if value[0] < last:
            times[index] = None
        else:
            last = value[0]


# --- merging -------------------------------------------------------------------------------


def align_lyrics(lines: Sequence[LyricLine], heard: Sequence[TimedWord], *, plan: AlignmentPlan | None = None,
                 forced: Sequence[Sequence[TimedWord] | None] | None = None, activity: Activity | None = None,
                 language: str | None = None, use_word_tags: bool = True,
                 mix_heard: Sequence[TimedWord] | None = None, window: float = MATCH_WINDOW) -> Alignment:
    """Time every word of `lines`.

    `forced` holds forced-alignment results per kept line (None where it wasn't run),
    `mix_heard` a second transcription (of the full mix) that votes with `heard`.
    """
    kept = kept_lines(lines)
    tokens = _tokens(kept)
    if not tokens:
        return Alignment([], 0, 0)
    if plan is None:
        plan = plan_alignment(lines, heard, activity, language=language, window=window)
    heard = filter_heard(heard, activity)
    windows = _line_windows(plan.starts)
    times = _matched_times(tokens, heard, _match_pairs(tokens, heard, windows, window, language))
    if mix_heard is not None:
        mix = filter_heard(mix_heard, activity)
        times = _vote(times, _matched_times(tokens, mix, _match_pairs(tokens, mix, windows, window, language)))
    _drop_non_monotonic(times)
    matched = sum(t is not None for t in times)
    sources = ["whisper" if t is not None else "" for t in times]
    forced_scores = _apply_better_sources(kept, tokens, times, sources, plan, forced, use_word_tags)
    _drop_non_monotonic(times)
    sources = [source if t is not None else "" for source, t in zip(sources, times)]

    all_tokens = tokens
    active = [k for k in range(len(tokens)) if tokens[k].line not in plan.cut]
    tokens = [tokens[k] for k in active]
    times = [times[k] for k in active]
    sources = [sources[k] for k in active]
    _label(sources, times, lambda: _time_unheard_synced_lines(tokens, times, plan.starts, activity), "lrc-line")
    _label(sources, times, lambda: _fill_gaps(tokens, times, activity), "interpolated")
    if activity is not None:
        _snap_to_voice(tokens, times, activity)  # type: ignore[arg-type]  # every slot is filled by now
    final = _make_monotonic(times)  # type: ignore[arg-type]

    timed_lines = [
        TimedLine(tuple(TimedWord(tokens[k].text, *final[k], sources[k]) for k, _ in group))
        for _, group in groupby(enumerate(tokens), key=lambda item: item[1].line)
    ]
    shown = [score for k, score in forced_scores.items() if all_tokens[k].line not in plan.cut]
    return Alignment(timed_lines, matched, len(all_tokens), forced=sum(source == "forced" for source in sources),
                     forced_score=statistics.fmean(shown) if shown else None, cut=len(plan.cut))


def lines_from_words(words: Sequence[TimedWord], *, max_words: int = 8, max_pause: float = 0.8) -> list[TimedLine]:
    """Fallback when there are no lyrics: build display lines straight from the transcription."""
    lines: list[TimedLine] = []
    current: list[TimedWord] = []

    def flush() -> None:
        if current:
            lines.append(TimedLine(tuple(current)))
            current.clear()

    for word in words:
        if not normalize(word.text):
            continue
        if current and (word.start - current[-1].end > max_pause or len(current) >= max_words):
            flush()
        current.append(TimedWord(word.text.strip(), word.start, word.end, word.score, "whisper"))
        if len(current) >= 3 and word.text.rstrip().endswith((".", "?", "!")):
            flush()
    flush()
    return lines


def _apply_better_sources(kept: Sequence[tuple[LyricLine, list[str]]], tokens: Sequence[_Token],
                          times: list[Times | None], sources: list[str], plan: AlignmentPlan,
                          forced: Sequence[Sequence[TimedWord] | None] | None, use_word_tags: bool) -> dict[int, float]:
    """Overwrite whisper times with enhanced-LRC tags and good forced alignments; returns forced scores per token."""
    forced_scores: dict[int, float] = {}
    for index, (line, _) in enumerate(kept):
        indices = [k for k in range(len(tokens)) if tokens[k].line == index]
        if use_word_tags and line.word_starts:
            tags = line.word_starts
            for position, k in enumerate(indices):
                if tags[position] is None:
                    continue
                start = plan.time_map.map(tags[position])  # type: ignore[arg-type]
                following = tags[position + 1] if position + 1 < len(tags) else line.end
                end = plan.time_map.map(following) if following is not None else start + tokens[k].natural
                times[k], sources[k] = (start, max(end, start + MIN_DURATION), None), "lrc-tag"
            continue
        line_forced = forced[index] if forced is not None and index < len(forced) else None
        if not line_forced or len(line_forced) != len(indices):
            continue
        scores = [word.score or 0.0 for word in line_forced]
        forced_scores.update(zip(indices, scores))
        trusted_line = statistics.fmean(scores) >= FORCED_LINE_MIN
        for k, word, score in zip(indices, line_forced, scores):
            if trusted_line:
                use = score >= FORCED_WORD_MIN or times[k] is None
            else:
                use = times[k] is None and score >= FORCED_WORD_MIN
            if use:
                times[k], sources[k] = (word.start, max(word.end, word.start + MIN_DURATION), word.score), "forced"
    return forced_scores


def _vote(primary: Sequence[Times | None], secondary: Sequence[Times | None]) -> list[Times | None]:
    """Combine two transcriptions' matches: agreeing ones are averaged, otherwise the more confident wins."""
    result: list[Times | None] = []
    for a, b in zip(primary, secondary):
        if a is None or b is None:
            result.append(a or b)
            continue
        score_a, score_b = (0.5 if a[2] is None else max(a[2], 1e-3)), (0.5 if b[2] is None else max(b[2], 1e-3))
        if abs(a[0] - b[0]) <= VOTE_AGREE:
            total = score_a + score_b
            result.append(((a[0] * score_a + b[0] * score_b) / total, (a[1] * score_a + b[1] * score_b) / total,
                           max(score_a, score_b)))
        else:
            result.append(a if score_a >= score_b else b)
    return result


def _label(sources: list[str], times: list[Times | None], fill: Any, label: str) -> None:
    """Run `fill` and label the words it timed."""
    untimed = [t is None for t in times]
    fill()
    for k, was_untimed in enumerate(untimed):
        if was_untimed and times[k] is not None:
            sources[k] = label


def _time_unheard_synced_lines(tokens: list[_Token], times: list[Times | None], line_starts: Sequence[float | None],
                               activity: Activity | None) -> None:
    """Lines no source timed but lrclib has a timestamp for start at that (mapped) timestamp."""
    for line, group in groupby(range(len(tokens)), key=lambda k: tokens[k].line):
        indices = list(group)
        start = line_starts[line] if line < len(line_starts) else None
        if start is None or any(times[k] is not None for k in indices):
            continue
        first, last = indices[0], indices[-1]
        previous_end = next((t[1] for k in range(first - 1, -1, -1) if (t := times[k]) is not None), 0.0)
        next_start = next((t[0] for k in range(last + 1, len(times)) if (t := times[k]) is not None), math.inf)
        next_line = next((s for s in line_starts[line + 1:] if s is not None), math.inf)
        natural = sum(tokens[k].natural for k in indices)
        begin = max(start, previous_end)
        if activity is not None and (voiced := activity.next_voiced(begin, min(next_line, next_start))) is not None:
            begin = voiced
        finish = min(next_line - 0.1, next_start, begin + natural * 2.5)
        _pack_voiced(times, indices, [tokens[k].natural for k in indices], begin, max(finish, begin + natural),
                     activity)


def _fill_gaps(tokens: list[_Token], times: list[Times | None], activity: Activity | None) -> None:
    """Interpolate every remaining run of untimed words between its timed neighbours."""
    count, i = len(times), 0
    while i < count:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < count and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else None  # type: ignore[index]
        right = times[j][0] if j < count else None  # type: ignore[index]
        _fill_run(tokens, times, i, j, left, right, activity)
        i = j


def _fill_run(tokens: list[_Token], times: list[Times | None], i: int, j: int,
              left: float | None, right: float | None, activity: Activity | None) -> None:
    run = list(range(i, j))
    natural = sum(tokens[k].natural for k in run)
    if left is None and right is None:
        left, right = 0.0, natural
    elif left is None:
        left = max(0.0, right - natural)  # type: ignore[operator]
    elif right is None:
        right = left + natural
    assert left is not None and right is not None
    right = max(right, left)

    def weights(indices: list[int]) -> list[float]:
        return [tokens[k].natural for k in indices]

    previous_line = tokens[i - 1].line if i > 0 else None
    next_line = tokens[j].line if j < len(tokens) else None
    if right - left <= natural * 1.5 or previous_line == next_line:
        _pack_voiced(times, run, weights(run), left, right, activity)
        return

    # A long gap (instrumental break, unheard lines): keep words close to the line they belong to.
    head = [k for k in run if tokens[k].line == previous_line]
    tail = [k for k in run if tokens[k].line == next_line]
    middle = [k for k in run if k not in head and k not in tail]
    if head:
        need = sum(weights(head))
        left = _pack_voiced(times, head, weights(head), left, right, activity, need=need)
    if tail:
        need = sum(weights(tail))
        right = _pack_voiced(times, tail, weights(tail), left, right, activity, need=need, from_end=True)
    groups = [list(g) for _, g in groupby(middle, key=lambda k: tokens[k].line)]
    if groups:
        slot = (right - left) / len(groups)
        for number, group in enumerate(groups):
            start = left + number * slot
            _pack_voiced(times, group, weights(group), start, start + slot, activity,
                         need=min(slot, sum(weights(group))))


def _pack(times: list[Times | None], indices: list[int], weights: list[float], start: float, end: float) -> None:
    """Spread consecutive words over [start, end], proportionally to their weights."""
    total = sum(weights) or 1.0
    cursor = start
    for index, weight in zip(indices, weights):
        length = (end - start) * weight / total
        times[index] = (cursor, cursor + length, None)
        cursor += length


def _pack_voiced(times: list[Times | None], indices: list[int], weights: list[float], start: float, end: float,
                 activity: Activity | None, *, need: float | None = None, from_end: bool = False) -> float:
    """Like _pack, but only into the voiced parts of [start, end].

    With `need`, only the first (or, `from_end`, the last) `need` seconds of voiced
    time are used. Returns the far edge of the time used, so callers can pack more
    words next to it. Falls back to plain packing when there is too little voice.
    """
    natural = sum(weights)
    intervals = activity.voiced_intervals(start, end) if activity is not None and end > start else []
    voiced = sum(b - a for a, b in intervals)
    if voiced < VOICED_PACK_MIN * natural:
        if need is None:
            _pack(times, indices, weights, start, end)
            return end
        if from_end:
            _pack(times, indices, weights, end - need, end)
            return end - need
        _pack(times, indices, weights, start, start + need)
        return start + need
    use = voiced if need is None else min(need, voiced)
    offset = voiced - use if from_end else 0.0
    total = natural or 1.0
    for index, weight in zip(indices, weights):
        length = use * weight / total
        times[index] = (_voiced_time(intervals, offset, False), _voiced_time(intervals, offset + length, True), None)
        offset += length
    return _voiced_time(intervals, voiced - use, False) if from_end else _voiced_time(intervals, offset, True)


def _voiced_time(intervals: Sequence[tuple[float, float]], offset: float, is_end: bool) -> float:
    """The moment `offset` seconds of voiced time into the intervals."""
    for a, b in intervals:
        span = b - a
        if offset < span or (is_end and offset <= span + 1e-9):
            return a + offset
        offset -= span
    return intervals[-1][1]


def _snap_to_voice(tokens: Sequence[_Token], times: list[Times], activity: Activity) -> None:
    """Line starts move to a phrase onset, words don't start in silence, last words last as long as the note."""
    count = len(times)
    for k in range(count):
        start, end, score = times[k]
        first_of_line = k == 0 or tokens[k - 1].line != tokens[k].line
        last_of_line = k == count - 1 or tokens[k + 1].line != tokens[k].line
        following = times[k + 1][0] if k + 1 < count else math.inf
        if first_of_line and (onset := activity.onset_near(start, ONSET_SNAP)) is not None and onset < end:
            start = onset
        elif not activity.is_voiced(start):
            if (voiced := activity.next_voiced(start, min(end, following))) is not None:
                start = voiced
        if last_of_line:
            limit = min(following - 0.1, end + MAX_HOLD)
            if limit > end:
                end = max(end, activity.phrase_end(max(start, end - 0.05), limit))
        times[k] = (start, max(end, start + MIN_DURATION), score)


def _make_monotonic(times: list[Times]) -> list[Times]:
    result: list[Times] = []
    previous_start = 0.0
    for start, end, score in times:
        start = max(start, previous_start)
        result.append((start, max(end, start + MIN_DURATION), score))
        previous_start = start
    for index in range(len(result) - 1):
        start, end, score = result[index]
        next_start = result[index + 1][0]
        if end > next_start:
            result[index] = (start, max(next_start, start + MIN_DURATION), score)
    return result

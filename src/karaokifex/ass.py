"""Render timed lyric lines as a karaoke-style ASS subtitle file — pure functions, no I/O.

Each line is a Dialogue event whose words carry `\\kf` tags, so libass sweeps the
highlight colour across every word exactly while it is sung. Lines alternate
between an upper and a lower slot and appear a little before they are sung, so
the singer can always read ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from karaokifex.models import TimedLine, TimedWord

LEAD_IN = 1.5  # seconds a line is shown before its first word
LINGER = 0.5  # seconds a line stays after its last word
TITLE_MAX = 5.0  # longest the opening title card is shown


@dataclass(frozen=True)
class KaraokeStyle:
    font: str = "Arial"
    # ASS colours are &HAABBGGRR (alpha 00 = opaque).
    sung_colour: str = "&H0000D7FF"  # gold
    unsung_colour: str = "&H00FFFFFF"  # white
    outline_colour: str = "&H00000000"
    shadow_colour: str = "&H96000000"
    font_scale: float = 0.065  # font size relative to video height


def format_timestamp(seconds: float) -> str:
    """ASS time format H:MM:SS.cc."""
    centis = max(0, round(seconds * 100))
    hours, rest = divmod(centis, 360_000)
    minutes, rest = divmod(rest, 6_000)
    secs, centis = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def escape_text(text: str) -> str:
    """Neutralise characters that ASS would interpret as override tags."""
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")")


# Debug mode: unsung words take the colour of what timed them (&HBBGGRR); a low score underlines them.
SOURCE_COLOURS = {
    "lrc-tag": "&HFF60C0&",  # violet
    "forced": "&H40E040&",  # green
    "whisper": "&HE0E000&",  # cyan
    "lrc-line": "&H0090FF&",  # orange
    "interpolated": "&H3030FF&",  # red
}
LOW_SCORE = 0.3


def _debug_tags(word: TimedWord) -> str:
    colour = SOURCE_COLOURS.get(word.source, "&HFFFFFF&")
    low = word.score is not None and word.score < LOW_SCORE
    return f"\\2c{colour}\\1c&HFFFFFF&\\u{int(low)}"


def karaoke_text(line: TimedLine, shown_at: float, *, debug: bool = False) -> str:
    """The `{\\k..}`-tagged text for one line, relative to when the line appears.

    Durations are differences of rounded absolute times, so rounding errors never
    accumulate and the tags always add up to the line's length.
    """

    def centis(t: float) -> int:
        return round((t - shown_at) * 100)

    parts: list[str] = []
    cursor = 0
    lead = max(0, centis(line.start))
    if lead:
        parts.append(f"{{\\k{lead}}}")
        cursor = lead
    for index, word in enumerate(line.words):
        end = max(centis(word.end), cursor)
        tags = _debug_tags(word) if debug else ""
        parts.append(f"{{\\kf{end - cursor}{tags}}}{escape_text(word.text)}")
        cursor = end
        if index + 1 < len(line.words):
            next_start = max(centis(line.words[index + 1].start), cursor)
            parts.append(f"{{\\k{next_start - cursor}}} ")
            cursor = next_start
    return "".join(parts)


def display_windows(lines: Sequence[TimedLine]) -> list[tuple[float, float]]:
    """When each line appears and disappears.

    Line i shares its screen slot with line i+2, so it has to be gone before
    that one appears — but it never appears after its own first word.
    """
    windows: list[tuple[float, float]] = []
    for index, line in enumerate(lines):
        shown = max(0.0, line.start - LEAD_IN)
        if index >= 2:
            shown = max(shown, windows[index - 2][1] + 0.05)
        shown = min(shown, line.start)
        hidden = line.end + LINGER
        if index + 2 < len(lines):
            hidden = max(line.end, min(hidden, lines[index + 2].start - 0.2))
        windows.append((shown, hidden))
    return windows


def build_ass(lines: Sequence[TimedLine], *, width: int, height: int, title: str | None = None,
              style: KaraokeStyle = KaraokeStyle(), debug: bool = False) -> str:
    """The karaoke subtitle file; `debug` colours every word by what timed it and adds a legend."""
    size = round(height * style.font_scale)
    outline = max(1, round(height / 360))
    shadow = max(1, round(height / 540))
    margin_lower = round(height * 0.08)
    margin_upper = margin_lower + round(size * 1.45)
    margin_side = round(width * 0.04)

    def style_line(name: str, font_size: int, alignment: int, margin_v: int) -> str:
        return (
            f"Style: {name},{style.font},{font_size},{style.sung_colour},{style.unsung_colour},"
            f"{style.outline_colour},{style.shadow_colour},-1,0,0,0,100,100,0,0,1,{outline},{shadow},"
            f"{alignment},{margin_side},{margin_side},{margin_v},1"
        )

    def dialogue(start: float, end: float, style_name: str, text: str) -> str:
        return f"Dialogue: 0,{format_timestamp(start)},{format_timestamp(end)},{style_name},,0,0,0,,{text}"

    header = [
        "[Script Info]",
        "; Generated by karaokifex",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        style_line("Upper", size, 2, margin_upper),
        style_line("Lower", size, 2, margin_lower),
        style_line("Title", round(size * 1.2), 5, 0),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []
    windows = display_windows(lines)
    if title and lines and windows[0][0] > 1.5:
        events.append(dialogue(0.0, min(TITLE_MAX, windows[0][0] - 0.3), "Title",
                               "{\\fad(300,300)}" + escape_text(title)))
    for index, (line, (shown, hidden)) in enumerate(zip(lines, windows)):
        slot = "Upper" if index % 2 == 0 else "Lower"
        events.append(dialogue(shown, hidden, slot, "{\\fad(150,200)}" + karaoke_text(line, shown, debug=debug)))
    if debug and lines:
        header.insert(header.index("[Events]") - 1, style_line("Legend", round(size * 0.45), 7, margin_side))
        legend = " ".join(f"{{\\c{colour}}}{name}" for name, colour in SOURCE_COLOURS.items())
        events.append(dialogue(0.0, lines[-1].end + LINGER, "Legend",
                               legend + "{\\c&HFFFFFF&} · underlined: low score"))
    return "\n".join(header + events) + "\n"

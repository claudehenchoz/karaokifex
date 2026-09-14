import re

from karaokifex.ass import build_ass, display_windows, escape_text, format_timestamp, karaoke_text
from karaokifex.models import TimedLine, TimedWord

K_TAG = re.compile(r"\\kf?(\d+)")


def line(*items: tuple[str, float, float]) -> TimedLine:
    return TimedLine(tuple(TimedWord(text, start, end) for text, start, end in items))


def test_format_timestamp():
    assert format_timestamp(0) == "0:00:00.00"
    assert format_timestamp(61.5) == "0:01:01.50"
    assert format_timestamp(3661.234) == "1:01:01.23"
    assert format_timestamp(59.999) == "0:01:00.00"
    assert format_timestamp(-1) == "0:00:00.00"


def test_escape_text_removes_override_characters():
    escaped = escape_text("{a}\\b")
    assert not set(escaped) & set("{}\\")


def test_karaoke_durations_add_up_to_the_line():
    timed = line(("Hello", 10.004, 10.5), ("big", 10.73, 11.0), ("world", 11.0, 12.337))
    text = karaoke_text(timed, shown_at=8.5)
    assert text.startswith("{\\k150}{\\kf")
    assert sum(int(value) for value in K_TAG.findall(text)) == round((12.337 - 8.5) * 100)
    assert "Hello" in text and "world" in text


def test_display_windows_share_slots_without_overlap():
    lines = [line((f"w{i}", i * 2.0 + 2, i * 2.0 + 3.5)) for i in range(6)]
    windows = display_windows(lines)
    for index, (shown, hidden) in enumerate(windows):
        assert shown <= lines[index].start
        assert hidden >= lines[index].end
        if index >= 2:
            assert shown >= windows[index - 2][1]


def test_debug_mode_colours_words_by_source():
    timed = TimedLine((TimedWord("One", 5, 6, 0.9, "forced"), TimedWord("Two", 6, 7, 0.1, "interpolated")))
    ass = build_ass([timed], width=1280, height=720, debug=True)
    assert "\\2c&H40E040&" in ass and "\\2c&H3030FF&" in ass
    assert "\\u1}Two" in ass  # low score is underlined
    assert "Legend" in ass
    plain = build_ass([timed], width=1280, height=720)
    assert "\\2c" not in plain and "Legend" not in plain
    assert sum(int(v) for v in K_TAG.findall(karaoke_text(timed, 4.0, debug=True))) == 300


def test_build_ass_structure():
    lines = [line(("One", 5, 6)), line(("Two", 7, 8)), line(("Three", 9, 10))]
    ass = build_ass(lines, width=1280, height=720, title="Artist – Song")
    assert "PlayResX: 1280" in ass and "PlayResY: 720" in ass
    dialogues = [row for row in ass.splitlines() if row.startswith("Dialogue:")]
    assert [row.split(",")[3] for row in dialogues] == ["Title", "Upper", "Lower", "Upper"]

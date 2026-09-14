import json

import pytest
from click.testing import CliRunner

from karaokifex.evaluate import compare, load_reference, main, parse_reference_ass, parse_reference_lrc
from karaokifex.models import TimedWord


def test_parse_reference_ass_syllables_and_spaces():
    text = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:10.00,0:00:13.00,Default,,0,0,0,,{\\k50}one {\\kf30}twen{\\k20}ty {\\ko100}three\n"
        "Comment: 0,0:00:20.00,0:00:21.00,Default,,0,0,0,,{\\k50}ignored\n"
    )
    words = parse_reference_ass(text)
    assert [(w.text, w.start) for w in words] == [("one", 10.0), ("twenty", 10.5), ("three", 11.0)]


def test_parse_reference_lrc_uses_word_tags():
    words = parse_reference_lrc("[00:01.00]<00:01.00>one <00:01.50>two\n[00:03.00]untagged line")
    assert [(w.text, w.start) for w in words] == [("one", 1.0), ("two", 1.5)]


def test_compare_pairs_words_in_order():
    reference = [TimedWord("one", 1.0, 1.0), TimedWord("two", 2.0, 2.0), TimedWord("three", 3.0, 3.0)]
    ours = [TimedWord("One", 1.05, 1.3, source="forced"), TimedWord("extra", 1.5, 1.6),
            TimedWord("three", 3.4, 3.6, source="interpolated")]
    report = compare(reference, ours)
    assert (report.words, report.reference) == (2, 3)
    assert report.median_ms == pytest.approx(225)
    assert report.within(0.1) == 0.5 and report.within(0.3) == 0.5
    assert set(report.by_source) == {"forced", "interpolated"}


def test_cli_reports_a_folder(tmp_path):
    (tmp_path / "reference.lrc").write_text("[00:01.00]<00:01.00>one <00:01.50>two", encoding="utf-8")
    words = [{"text": "one", "start": 1.02, "end": 1.4, "score": None, "source": "forced"},
             {"text": "two", "start": 1.9, "end": 2.2, "score": None, "source": "whisper"}]
    (tmp_path / "timings.json").write_text(json.dumps({"lines": [words]}), encoding="utf-8")
    assert len(load_reference(tmp_path)) == 2
    result = CliRunner().invoke(main, [str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "2/2" in result.output and "forced" in result.output

import numpy as np
import pytest

from karaokifex.activity import HOP, Activity
from karaokifex.models import LyricLine, TimedWord
from karaokifex.timing import (
    PHONETIC,
    AlignmentPlan,
    _similarity,
    align_lyrics,
    filter_heard,
    forced_requests,
    lines_from_words,
    normalize,
    plan_alignment,
    split_words,
)


def heard(*items: tuple[str, float, float]) -> list[TimedWord]:
    return [TimedWord(text, start, end) for text, start, end in items]


def all_words(alignment):
    return [word for line in alignment.lines for word in line.words]


def test_normalize():
    assert normalize("Don’t!") == "don't"
    assert normalize("Café") == "cafe"
    assert normalize("'Cause") == "cause"
    assert normalize("...") == ""


def test_split_words_glues_punctuation_to_neighbours():
    assert split_words("Hello - world ...") == ["Hello -", "world ..."]
    assert split_words("... and then") == ["... and", "then"]
    assert split_words("♪ ♪") == []


def test_exact_match_uses_whisper_times():
    lines = [LyricLine(1.0, "Hello world"), LyricLine(3.0, "Goodbye moon")]
    words = heard(("hello", 1.1, 1.5), ("world", 1.6, 2.0), ("goodbye", 3.1, 3.6), ("moon", 3.7, 4.2))
    result = align_lyrics(lines, words)
    assert result.matched == result.total == 4
    assert [line.text for line in result.lines] == ["Hello world", "Goodbye moon"]
    assert result.lines[0].words[0] == TimedWord("Hello", 1.1, 1.5, source="whisper")
    assert result.lines[1].words[1].start == 3.7


def test_missing_word_is_interpolated_between_neighbours():
    lines = [LyricLine(1.0, "Hello world"), LyricLine(3.0, "Goodbye moon")]
    words = heard(("hello", 1.1, 1.5), ("goodbye", 3.1, 3.6), ("moon", 3.7, 4.2))
    result = align_lyrics(lines, words)
    world = result.lines[0].words[1]
    assert result.matched == 3
    assert 1.5 <= world.start < world.end <= 3.1


def test_extra_and_misspelled_words():
    lines = [LyricLine(None, "The colour of love")]
    words = heard(("uh", 0.2, 0.4), ("the", 1.0, 1.2), ("color", 1.3, 1.7), ("of", 1.8, 1.9),
                  ("love", 2.0, 2.6), ("yeah", 3.0, 3.5))
    result = align_lyrics(lines, words)
    assert result.matched == 4
    assert [w.start for w in all_words(result)] == [1.0, 1.3, 1.8, 2.0]


def test_repeated_chorus_matches_the_right_occurrence():
    lines = [LyricLine(10.0, "Na na hey"), LyricLine(20.0, "Something else"), LyricLine(50.0, "Na na hey")]
    words = heard(("something", 20.1, 20.5), ("else", 20.6, 21.0),
                  ("na", 50.1, 50.3), ("na", 50.4, 50.6), ("hey", 50.7, 51.0))
    result = align_lyrics(lines, words)
    assert result.matched == 5
    assert result.lines[2].words[0].start == 50.1
    # The unheard first chorus falls back to lrclib's timestamp.
    assert 10.0 <= result.lines[0].words[0].start < 20.0


def test_constant_offset_between_lyrics_and_video_is_compensated():
    # The video has a 12 s longer intro than the lrclib version.
    lines = [LyricLine(float(t), text) for t, text in [(1, "one two"), (3, "three four"), (5, "five six"), (7, "seven eight")]]
    words = heard(*[(word, 12 + t + i * 0.4, 12 + t + i * 0.4 + 0.3)
                    for t, text in [(1, "one two"), (3, "three four"), (5, "five six"), (7, "seven eight")]
                    for i, word in enumerate(text.split())])
    result = align_lyrics(lines, words)
    assert result.matched == 8


def test_plain_lyrics_without_timestamps():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three four")]
    words = heard(("one", 1, 1.3), ("two", 1.4, 1.8), ("three", 2, 2.4), ("four", 2.5, 3))
    assert align_lyrics(lines, words).matched == 4


def test_unheard_plain_line_lands_in_the_gap():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three four"), LyricLine(None, "five six")]
    words = heard(("one", 1, 1.4), ("two", 1.5, 2), ("five", 10, 10.4), ("six", 10.5, 11))
    middle = align_lyrics(lines, words).lines[1]
    assert 2.0 <= middle.start and middle.end <= 10.0


def test_times_are_monotonic_and_positive():
    lines = [LyricLine(None, "a b c d e f g")]
    words = heard(("a", 1, 1.2), ("c", 0.5, 0.7), ("d", 2, 2.2), ("x", 2.3, 2.4), ("g", 3, 3.2))
    result = all_words(align_lyrics(lines, words))
    assert len(result) == 7
    assert all(w.end > w.start for w in result)
    assert all(a.start <= b.start for a, b in zip(result, result[1:]))


def test_edge_cases():
    assert align_lyrics([], []).total == 0
    lonely = align_lyrics([LyricLine(None, "hi")], [])
    assert lonely.matched == 0 and lonely.lines[0].words[0].start == 0.0


def voice(duration: float, *voiced: tuple[float, float]) -> Activity:
    """A fake lead-vocal activity: singing in the given intervals, silence elsewhere."""
    frames = round(duration / HOP)
    mask = np.zeros(frames, dtype=bool)
    for start, end in voiced:
        mask[round(start / HOP): round(end / HOP)] = True
    return Activity(np.where(mask, -20.0, -80.0), mask, HOP)


def test_good_forced_alignment_beats_interpolation():
    lines = [LyricLine(1.0, "one two"), LyricLine(3.0, "three four")]
    words = heard(("one", 1.1, 1.4), ("two", 1.5, 1.9))
    forced = [None, [TimedWord("three", 3.2, 3.5, 0.8), TimedWord("four", 3.6, 4.0, 0.7)]]
    result = align_lyrics(lines, words, forced=forced)
    three = result.lines[1].words[0]
    assert (three.start, three.source) == (3.2, "forced")
    assert result.forced == 2 and result.matched == 2
    assert result.forced_score == pytest.approx(0.75)


def test_poor_forced_line_keeps_whisper_times():
    lines = [LyricLine(1.0, "one two")]
    words = heard(("one", 1.1, 1.4), ("two", 1.5, 1.9))
    forced = [[TimedWord("one", 2.0, 2.2, 0.05), TimedWord("two", 2.3, 2.6, 0.05)]]
    result = align_lyrics(lines, words, forced=forced)
    assert [w.source for w in all_words(result)] == ["whisper", "whisper"]
    assert all_words(result)[0].start == 1.1


def test_heard_words_in_silence_are_dropped():
    words = heard(("one", 1.0, 1.4), ("ghost", 5.0, 5.3))
    assert [w.text for w in filter_heard(words, voice(8, (0.9, 2.0)))] == ["one"]
    assert len(filter_heard(words)) == 2


def test_lines_on_silence_were_cut_from_the_video():
    texts = ["one two", "three four", "five six", "seven eight", "nine ten", "eleven twelve", "red blue",
             "green gold"]
    lines = [LyricLine(10.0 * (i + 1), text) for i, text in enumerate(texts)]
    sung = [i for i in range(len(texts)) if i not in (2, 3)]  # the video has no "five six" / "seven eight"
    words = heard(*[(word, 10.0 * (i + 1) + 0.1 + k * 0.5, 10.0 * (i + 1) + 0.5 + k * 0.5)
                    for i in sung for k, word in enumerate(texts[i].split())])
    activity = voice(100, *[(10.0 * (i + 1), 10.0 * (i + 1) + 1.2) for i in sung])
    result = align_lyrics(lines, words, activity=activity)
    assert [line.text for line in result.lines] == [texts[i] for i in sung]
    assert result.cut == 2


SIX = ["one two", "three four", "five six", "seven eight", "nine ten", "eleven twelve"]


def six_lines():
    return [LyricLine(10.0 * (i + 1), text) for i, text in enumerate(SIX)]


def six_voiced():
    return voice(80, *[(10.0 * (i + 1), 10.0 * (i + 1) + 1.0) for i in range(len(SIX))])


def test_misleading_onset_correlation_loses_to_what_whisper_heard(monkeypatch):
    monkeypatch.setattr("karaokifex.timing.xcorr_offset", lambda *args, **kwargs: 59.6)
    words = heard(*[(word, 10.0 * (i + 1) + 0.1 + k * 0.4, 10.0 * (i + 1) + 0.4 + k * 0.4)
                    for i, text in enumerate(SIX) for k, word in enumerate(text.split())])
    plan = plan_alignment(six_lines(), words, six_voiced())
    assert plan.time_map.offset == pytest.approx(0.0, abs=0.3)
    assert not plan.cut
    assert align_lyrics(six_lines(), words, activity=six_voiced()).matched == 12


def test_nothing_is_cut_when_no_heard_words_back_the_time_map(monkeypatch):
    monkeypatch.setattr("karaokifex.timing.xcorr_offset", lambda *args, **kwargs: 59.6)
    plan = plan_alignment(six_lines(), [], six_voiced())  # the prior maps every line onto silence
    assert not plan.cut


def test_held_last_note_extends_the_line():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three")]
    words = heard(("one", 1.0, 1.3), ("two", 1.4, 1.7), ("three", 10.0, 10.5))
    result = align_lyrics(lines, words, activity=voice(12, (1.0, 3.0), (10.0, 10.6)))
    assert result.lines[0].end == pytest.approx(3.0, abs=HOP)


def test_interpolated_line_lands_where_someone_sings():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three four"), LyricLine(None, "five six")]
    words = heard(("one", 1, 1.4), ("two", 1.5, 2), ("five", 10, 10.4), ("six", 10.5, 11))
    result = align_lyrics(lines, words, activity=voice(12, (1, 2), (6, 7), (10, 11)))
    middle = result.lines[1]
    assert 6.0 - HOP <= middle.start and middle.end <= 7.0 + HOP
    assert {w.source for w in middle.words} == {"interpolated"}


def test_one_lyric_word_sung_as_two_heard_words_and_back():
    result = align_lyrics([LyricLine(None, "alright now")], heard(("all", 1.0, 1.2), ("right", 1.3, 1.6),
                                                                  ("now", 1.7, 2.0)))
    assert result.matched == 2
    assert (all_words(result)[0].start, all_words(result)[0].end) == (1.0, 1.6)
    result = align_lyrics([LyricLine(None, "want to go")], heard(("wanna", 1.0, 1.4), ("go", 1.5, 1.8)))
    assert result.matched == 3
    assert all_words(result)[1].end == pytest.approx(1.4)


def test_phonetic_similarity_only_for_english():
    assert _similarity("cue", "queue") == 0.0
    assert _similarity("cue", "queue", True) == PHONETIC


def test_low_score_heard_words_count_less():
    lines = [LyricLine(None, "la")]
    words = [TimedWord("la", 1.0, 1.2, 0.01), TimedWord("la", 2.0, 2.2, 0.9)]
    assert all_words(align_lyrics(lines, words))[0].start == 2.0


def test_enhanced_lrc_tags_time_words():
    lines = [LyricLine(1.0, "one two", (1.0, 1.5), 2.0)]
    words = all_words(align_lyrics(lines, []))
    assert [(w.start, w.source) for w in words] == [(1.0, "lrc-tag"), (1.5, "lrc-tag")]
    assert all_words(align_lyrics(lines, [], use_word_tags=False))[0].source == "lrc-line"


def test_full_mix_transcription_votes():
    lines = [LyricLine(None, "one two three")]
    lead = heard(("one", 1.0, 1.3), ("three", 2.0, 2.3))
    mix = [TimedWord("one", 1.1, 1.4, 0.5), TimedWord("two", 1.5, 1.8, 0.5)]
    result = align_lyrics(lines, lead, mix_heard=mix)
    assert result.matched == 3
    assert all_words(result)[1].start == 1.5


def test_plan_roundtrip_and_forced_requests():
    lines = [LyricLine(10.0, "one two"), LyricLine(20.0, "three four", (20.0, 20.5)), LyricLine(30.0, "five six")]
    words = heard(("one", 10.1, 10.5), ("two", 10.6, 11.0), ("five", 30.1, 30.5), ("six", 30.6, 31.0))
    plan = plan_alignment(lines, words)
    assert AlignmentPlan.from_dict(plan.to_dict()) == plan
    requests = forced_requests(lines, plan)
    assert [index for index, _, _ in requests] == [0, 2]  # line 1 has word tags
    assert requests[0][1] == ["one", "two"]
    low, high = requests[0][2]
    assert low <= 10.1 and high >= 11.0


def test_lines_from_words_splits_on_pauses_sentences_and_length():
    words = heard(("Hello", 0, 0.4), ("there", 0.5, 0.9), ("friend.", 1.0, 1.4), ("How", 1.5, 1.7),
                  ("are", 1.8, 2.0), ("you", 5.0, 5.4))
    assert [line.text for line in lines_from_words(words)] == ["Hello there friend.", "How are", "you"]
    many = heard(*[(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(10)])
    assert [len(line.words) for line in lines_from_words(many, max_words=8)] == [8, 2]

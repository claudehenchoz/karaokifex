from karaokifex.models import LyricLine, TimedWord
from karaokifex.timing import align_lyrics, lines_from_words, normalize, split_words


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
    assert result.lines[0].words[0] == TimedWord("Hello", 1.1, 1.5)
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


def test_lines_from_words_splits_on_pauses_sentences_and_length():
    words = heard(("Hello", 0, 0.4), ("there", 0.5, 0.9), ("friend.", 1.0, 1.4), ("How", 1.5, 1.7),
                  ("are", 1.8, 2.0), ("you", 5.0, 5.4))
    assert [line.text for line in lines_from_words(words)] == ["Hello there friend.", "How are", "you"]
    many = heard(*[(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(10)])
    assert [len(line.words) for line in lines_from_words(many, max_words=8)] == [8, 2]

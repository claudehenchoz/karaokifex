import json

import requests

from karaokifex.models import LyricLine, Lyrics
from karaokifex.steps.lyrics import (
    SEARCH_URL,
    choose_best,
    fetch_lyrics,
    guess_language,
    load_lyrics,
    parse_lrc,
    parse_plain,
    prompt_text,
    rank_candidates,
    save_lyrics,
)


def candidate(duration, *, synced=True, instrumental=False, id=1, text="Hi there"):
    return {
        "id": id,
        "artistName": "Artist",
        "trackName": "Song",
        "albumName": None,
        "duration": duration,
        "syncedLyrics": f"[00:01.00]{text}" if synced else None,
        "plainLyrics": text,
        "instrumental": instrumental,
    }


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._data


def test_parse_lrc():
    text = (
        "[ar:Someone]\n"
        "[00:01.50]First line\n"
        "[00:03.00][01:03.00]Chorus\n"
        "[00:05.00]\n"
        "[00:04.20]<00:04.20>Word <00:04.80>tags<00:05.10>\n"
        "no timestamp"
    )
    assert parse_lrc(text) == [
        LyricLine(1.5, "First line"),
        LyricLine(3.0, "Chorus"),
        LyricLine(4.2, "Word tags", (4.2, 4.8), 5.1),
        LyricLine(63.0, "Chorus"),
    ]


def test_word_tags_cover_untagged_words_and_punctuation():
    [line] = parse_lrc("[00:10.00]<00:10.00>one two - <00:11.00>three")
    assert line.text == "one two - three"
    assert line.word_starts == (10.0, None, 11.0)
    assert line.end is None


def test_parse_plain_drops_blank_lines():
    assert parse_plain("a\n\n  b  \n") == [LyricLine(None, "a"), LyricLine(None, "b")]


def test_choose_best_prefers_synced_within_tolerance():
    assert choose_best([candidate(200, synced=False, id=1), candidate(205, id=2)], 200)["id"] == 2


def test_choose_best_takes_closest_duration_when_nothing_is_close():
    assert choose_best([candidate(300, id=1), candidate(250, synced=False, id=2)], 200)["id"] == 2


def test_choose_best_ignores_instrumentals_and_empty_results():
    assert choose_best([candidate(200, instrumental=True)], 200) is None
    assert choose_best([], 200) is None


def test_rank_candidates_drops_duplicate_texts():
    ranked = rank_candidates([candidate(200, id=1), candidate(201, id=2), candidate(202, id=3, text="Other words")],
                             200)
    assert [c["id"] for c in ranked] == [1, 3]


def test_fetch_lyrics_falls_back_to_free_text_search():
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(params)
        assert url == SEARCH_URL and "User-Agent" in headers
        return FakeResponse([] if "track_name" in params else [candidate(200, id=7)])

    [lyrics] = fetch_lyrics("Artist", "Song", 201, get=fake_get)
    assert len(calls) == 2
    assert lyrics.synced and lyrics.lrclib_id == 7
    assert lyrics.lines == (LyricLine(1.0, "Hi there"),)


def test_fetch_lyrics_keeps_several_candidates():
    results = [candidate(200 + i, id=i, text=f"Version {i}") for i in range(5)]
    found = fetch_lyrics("Artist", "Song", 200, get=lambda *a, **k: FakeResponse(results))
    assert [lyrics.lrclib_id for lyrics in found] == [0, 1, 2]


def test_fetch_lyrics_miss():
    assert fetch_lyrics("Artist", "Song", 200, get=lambda *a, **k: FakeResponse([])) == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "lyrics.json"
    lyrics = Lyrics(1, "Artist", "Song", None, 200.0, True,
                    (LyricLine(1.0, "Hi", (1.0,), 1.4), LyricLine(2.0, "there")))
    save_lyrics([lyrics, lyrics], path)
    assert load_lyrics(path) == [lyrics, lyrics]
    save_lyrics([], path)
    assert load_lyrics(path) == []


def test_load_lyrics_reads_the_old_single_match_format(tmp_path):
    path = tmp_path / "lyrics.json"
    old = {"found": True, "lrclib_id": 1, "artist": "A", "track": "S", "album": None, "duration": 200.0,
           "synced": True, "lines": [{"start": 1.0, "text": "Hi"}]}
    path.write_text(json.dumps(old), encoding="utf-8")
    assert load_lyrics(path) == [Lyrics(1, "A", "S", None, 200.0, True, (LyricLine(1.0, "Hi"),))]


def test_prompt_text_dedupes_and_cuts_at_a_word():
    lines = [LyricLine(None, "la la la"), LyricLine(None, "la la la"), LyricLine(None, "something new here")]
    assert prompt_text(lines) == "la la la something new here"
    assert prompt_text(lines, limit=14) == "la la la"


def test_guess_language():
    english = [LyricLine(None, "I walked along the river and the evening light was fading slowly")]
    french = [LyricLine(None, "Je marchais le long de la rivière et la lumière du soir tombait doucement")]
    assert guess_language(english) == "en"
    assert guess_language(french) == "fr"
    assert guess_language([LyricLine(None, "hey")]) is None


def test_a_busy_lrclib_is_asked_again(monkeypatch):
    monkeypatch.setattr("karaokifex.steps.lyrics.RETRY_WAIT", 0.0)
    answers = [FakeResponse([], status_code=503), FakeResponse([], status_code=502), FakeResponse([candidate(200, id=9)])]
    found = fetch_lyrics("Artist", "Song", 200, get=lambda *a, **k: answers.pop(0))
    assert [lyrics.lrclib_id for lyrics in found] == [9] and not answers

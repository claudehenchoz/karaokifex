from karaokifex.models import LyricLine, Lyrics
from karaokifex.steps.lyrics import SEARCH_URL, choose_best, fetch_lyrics, load_lyrics, parse_lrc, parse_plain, save_lyrics


def candidate(duration, *, synced=True, instrumental=False, id=1):
    return {
        "id": id,
        "artistName": "Artist",
        "trackName": "Song",
        "albumName": None,
        "duration": duration,
        "syncedLyrics": "[00:01.00]Hi there" if synced else None,
        "plainLyrics": "Hi there",
        "instrumental": instrumental,
    }


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_parse_lrc():
    text = (
        "[ar:Someone]\n"
        "[00:01.50]First line\n"
        "[00:03.00][01:03.00]Chorus\n"
        "[00:05.00]\n"
        "[00:04.20]<00:04.20>Word <00:04.80>tags\n"
        "no timestamp"
    )
    assert parse_lrc(text) == [
        LyricLine(1.5, "First line"),
        LyricLine(3.0, "Chorus"),
        LyricLine(4.2, "Word tags"),
        LyricLine(63.0, "Chorus"),
    ]


def test_parse_plain_drops_blank_lines():
    assert parse_plain("a\n\n  b  \n") == [LyricLine(None, "a"), LyricLine(None, "b")]


def test_choose_best_prefers_synced_within_tolerance():
    assert choose_best([candidate(200, synced=False, id=1), candidate(205, id=2)], 200)["id"] == 2


def test_choose_best_takes_closest_duration_when_nothing_is_close():
    assert choose_best([candidate(300, id=1), candidate(250, synced=False, id=2)], 200)["id"] == 2


def test_choose_best_ignores_instrumentals_and_empty_results():
    assert choose_best([candidate(200, instrumental=True)], 200) is None
    assert choose_best([], 200) is None


def test_fetch_lyrics_falls_back_to_free_text_search():
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(params)
        assert url == SEARCH_URL and "User-Agent" in headers
        return FakeResponse([] if "track_name" in params else [candidate(200, id=7)])

    lyrics = fetch_lyrics("Artist", "Song", 201, get=fake_get)
    assert len(calls) == 2
    assert lyrics is not None and lyrics.synced and lyrics.lrclib_id == 7
    assert lyrics.lines == (LyricLine(1.0, "Hi there"),)


def test_fetch_lyrics_miss():
    assert fetch_lyrics("Artist", "Song", 200, get=lambda *a, **k: FakeResponse([])) is None


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "lyrics.json"
    lyrics = Lyrics(1, "Artist", "Song", None, 200.0, True, (LyricLine(1.0, "Hi"), LyricLine(2.0, "there")))
    save_lyrics(lyrics, path)
    assert load_lyrics(path) == lyrics
    save_lyrics(None, path)
    assert load_lyrics(path) is None

import pytest
import requests

from karaokifex.steps import musicbrainz
from karaokifex.steps.musicbrainz import Match, best_match, build_query


def rec(artist, title, credited=None, joined=None):
    """A search result as MusicBrainz sends it; `joined` makes a credit of several artists."""
    credits = joined or [{"name": credited or artist, "joinphrase": "", "artist": {"name": artist}}]
    return {"title": title, "artist-credit": credits, "score": 100}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(musicbrainz, "MIN_INTERVAL", 0.0)


def test_query_asks_for_every_guess_at_once():
    query = build_query([("Culture Beat", "Mr. Vain"), ("Mr. Vain", "Culture Beat")], "Sinéad O’Connor")
    assert query == ('(recording:"mr vain" AND artist:"culture beat") OR '
                     '(recording:"culture beat" AND artist:"mr vain") OR '
                     "(recording:(sinéad o'connor) AND artist:(sinéad o'connor))")


def test_takes_the_spelling_most_recordings_use():
    found = [rec("Culture Beat", "Mr. Vain"), rec("Culture Beat", "Mr Vain"), rec("Culture Beat", "Mr. Vain"),
             rec("Culture Beat", "Culture Beat / Mr Vain")]
    assert best_match(found, [("Culture Beat", "Mr Vain")]) == Match("Culture Beat", "Mr. Vain", 3,
                                                                       ("Culture Beat", "Mr Vain"))


def test_artist_as_credited_but_in_its_own_spelling():
    wings = rec("Wings", "Live and Let Die", credited="Paul McCartney & Wings")
    assert best_match([wings], [("Paul McCartney and Wings", "Live and Let Die")]).artist == "Paul McCartney & Wings"
    pumpkins = rec("The Smashing Pumpkins", "Mayonaise", credited="Smashing Pumpkins")
    assert best_match([pumpkins], [("Smashing Pumpkins", "Mayonaise")]).artist == "The Smashing Pumpkins"
    duet = rec("", "Under Pressure", joined=[
        {"name": "Queen", "joinphrase": " & ", "artist": {"name": "Queen"}},
        {"name": "David Bowie", "joinphrase": "", "artist": {"name": "David Bowie"}}])
    assert best_match([duet], [("Queen & David Bowie", "Under Pressure")]).artist == "Queen & David Bowie"


def test_finds_the_names_among_the_parts_of_a_title():
    found = [rec("Say Anything", "Say Anything"), rec("Peter Gabriel", "In Your Eyes (version from ‘Say Anything’)"),
             rec("Peter Gabriel", "In Your Eyes"), rec("Peter Gabriel", "In your eyes")]
    guesses = [("Say Anything", "In Your Eyes"), ("Say Anything", "Peter Gabriel"), ("In Your Eyes", "Say Anything"),
               ("Peter Gabriel", "Say Anything"), ("Peter Gabriel", "In Your Eyes")]
    match = best_match(found, guesses)
    assert (match.artist, match.song, match.recordings) == ("Peter Gabriel", "In Your Eyes", 3)


def test_title_additions_are_dropped_when_no_recording_goes_without():
    found = [rec("The Doors", "The End (from Apocalypse Now)")]
    assert best_match(found, [("The Doors", "The End")]).song == "The End"


def test_a_title_that_holds_both_names():
    found = [rec("The Smashing Pumpkins", "Smashing Pumpkins Interview Disc"),
             rec("The Smashing Pumpkins", "Mayonaise", credited="Smashing Pumpkins")]
    match = best_match(found, [("ag4321", "Smashing Pumpkins Mayonaise")], "Smashing Pumpkins Mayonaise")
    assert (match.artist, match.song) == ("The Smashing Pumpkins", "Mayonaise")


def test_no_confident_match():
    covers = [rec("Some Tribute Band", "Mr. Vain"), rec("Culture Beat", "Mr. Vain Recall")]
    assert best_match(covers, [("Culture Beat", "Mr. Vain")]) is None
    assert best_match([], [("Culture Beat", "Mr. Vain")]) is None


def test_typographic_punctuation_becomes_plain():
    found = [rec("Sinéad O’Connor", "Nothing Compares 2 U"), rec("Sinéad O’Connor", "Nothing Compares 2 U")]
    match = best_match(found, [("Sinéad O'Connor", "Nothing Compares 2 U")])
    assert (match.artist, match.song) == ("Sinéad O'Connor", "Nothing Compares 2 U")


class FakeResponse:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        return self._data


def test_search_identifies_itself_and_tries_again_when_busy():
    answers = [FakeResponse(503), FakeResponse(200, {"recordings": [rec("A", "B")]})]
    calls = []

    def get(url, **options):
        calls.append(options)
        return answers.pop(0)

    assert musicbrainz.search("q", get=get) == [rec("A", "B")]
    assert len(calls) == 2
    assert "github.com/samluescher/karaokifex" in calls[0]["headers"]["User-Agent"]
    assert calls[0]["params"] == {"query": "q", "fmt": "json", "limit": 100}


def test_search_gives_up_eventually():
    with pytest.raises(requests.HTTPError):
        musicbrainz.search("q", get=lambda url, **options: FakeResponse(503))

    def offline(url, **options):
        raise requests.ConnectionError("no network")

    with pytest.raises(requests.ConnectionError):
        musicbrainz.search("q", get=offline)


# --- describe -------------------------------------------------------------------------

BAND = {"id": "band", "name": "Band"}
VARIOUS = {"id": musicbrainz.VARIOUS_ARTISTS, "name": "Various Artists"}


def release(title, date, primary="Album", secondary=None, by=BAND, status="Official"):
    return {"status": status, "date": date, "artist-credit": [{"name": by["name"], "artist": by}],
            "release-group": {"id": f"rg-{title}", "title": title, "primary-type": primary,
                              "secondary-types": secondary}}


def song(rid, first, *releases, title="Song"):
    return {"id": rid, "title": title, "first-release-date": first, "releases": list(releases),
            "artist-credit": [{"name": "Band", "joinphrase": "", "artist": BAND}]}


def test_the_album_is_the_earliest_of_the_artists_own():
    found = [song("hits", "1999", release("Party Hits", "1999", secondary=["Compilation"], by=VARIOUS)),
             song("live", "2001", release("Live in Oslo", "2001", secondary=["Live"])),
             song("single", "1984-10-19", release("Song", "1984-10-19", primary="Single")),
             song("studio", "1985", release("Second Album", "1987-01-01"), release("First Album", "1985-06-01")),
             song("mixed", "1986", release("Throwback 80s", "1986", by=VARIOUS)),
             song("boot", "1980", release("Taped", "1980", status="Bootleg"))]
    recording, group = musicbrainz.first_album(found)
    assert (recording["id"], group["title"]) == ("studio", "First Album")
    assert musicbrainz.first_year(found) == 1980
    assert musicbrainz.first_album([found[2]]) is None


def test_genres_the_recording_votes_for_first():
    genres = musicbrainz.top_genres([{"name": "pop", "count": 19}, {"name": "synth-pop", "count": 7},
                                     {"name": "trance", "count": 1}],
                                    [{"name": "new wave", "count": 5}, {"name": "pop", "count": 12}],
                                    [{"name": "new wave", "count": 15}])
    assert genres == ("pop", "new wave", "synth-pop")
    assert musicbrainz.top_genres(None, [], None) == ()


def test_writers_once_each_with_every_role_and_the_language():
    work = {"language": "deu", "languages": ["deu"], "relations": [
        {"type": "composer", "artist": {"name": "Anna"}}, {"type": "lyricist", "artist": {"name": "Ben"}},
        {"type": "lyricist", "artist": {"name": "Anna"}}, {"type": "arranger", "artist": {"name": "Carl"}}]}
    assert musicbrainz.writers_of(work) == (musicbrainz.Writer("Anna", ("composer", "lyricist")),
                                            musicbrainz.Writer("Ben", ("lyricist",)))
    assert musicbrainz.language_of(work) == "de"
    assert musicbrainz.language_of({"languages": ["zxx"]}) is None
    assert musicbrainz.language_of({"languages": ["eng", "fra"]}) == "mul"
    assert musicbrainz.language_of({"language": "tlh"}) == "tlh"


class Answer:
    def __init__(self, data):
        self.status_code, self.data = 200, data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_describe_asks_for_the_recording_its_work_album_and_artist():
    other = {"id": "o", "title": "Song", "artist-credit": [{"name": "Other", "artist": {"id": "x", "name": "Other"}}]}
    answers = {
        "recording": {"recordings": [other, song("studio", "1985", release("First Album", "1985-06-01"))]},
        "recording/studio": {"genres": [{"name": "pop", "count": 4}],
                             "relations": [{"type": "performance", "work": {"id": "w"}}],
                             "artist-credit": [{"name": "Band", "artist": BAND}]},
        "work/w": {"id": "w", "languages": ["ita"], "relations": [{"type": "writer", "artist": {"name": "Anna"}}]},
        "release-group/rg-First Album": {"genres": [{"name": "italo disco", "count": 3}]},
        "artist/band": {"country": "IT", "genres": []},
    }
    asked = []

    def get(url, params, **_):
        asked.append((url.removeprefix(musicbrainz.API + "/"), params.get("inc")))
        return Answer(answers[asked[-1][0]])

    details = musicbrainz.describe("Band", "Song", get=get)
    assert details.to_dict() == {"album": "First Album", "year": 1985, "genres": ["pop", "italo disco"],
                                 "writers": [{"name": "Anna", "roles": ["writer"]}], "language": "it",
                                 "country": "IT", "musicbrainz": {"recording": "studio",
                                                                  "release-group": "rg-First Album",
                                                                  "work": "w", "artist": "band"}}
    assert [a for a, _ in asked] == ["recording", "recording/studio", "work/w", "release-group/rg-First Album",
                                     "artist/band"]
    queries = []

    def nothing(url, params, **_):
        queries.append(params["query"])
        return Answer({"recordings": []})

    assert musicbrainz.describe("Paul McCartney & Wings", "Live and Let Die", get=nothing) is None
    assert queries == ['(recording:"live and let die" AND artist:"paul mccartney wings")',
                       'recording:"live and let die" AND artist:(paul mccartney wings)']

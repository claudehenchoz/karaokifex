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

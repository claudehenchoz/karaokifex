import pytest

from karaokifex.metadata import (
    channel_name,
    guess_artist_song,
    name_guesses,
    normalize_name,
    split_title,
    title_segments,
    without_brackets,
)
from karaokifex.models import VideoInfo


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Queen - Bohemian Rhapsody (Official Video Remastered)", ("Queen", "Bohemian Rhapsody")),
        ("Rick Astley – Never Gonna Give You Up [4K]", ("Rick Astley", "Never Gonna Give You Up")),
        ('Daft Punk - "Get Lucky" (feat. Pharrell)', ("Daft Punk", "Get Lucky")),
        ("Just A Song Title (Lyrics)", (None, "Just A Song Title")),
        ('Smashing Pumpkins "Mayonaise"', ("Smashing Pumpkins", "Mayonaise")),
    ],
)
def test_split_title(title, expected):
    assert split_title(title) == expected


def test_channel_name():
    assert channel_name("AdeleVEVO") == "Adele"
    assert channel_name("Adele - Topic") == "Adele"
    assert channel_name(None) is None


def test_guess_prefers_explicit_values_then_metadata_then_title():
    info = VideoInfo(id="x", title="Foo - Bar (Official Video)", artist="Meta Artist", uploader="FooVEVO")
    assert guess_artist_song(info) == ("Meta Artist", "Bar")
    assert guess_artist_song(info, artist="A", song="S") == ("A", "S")
    assert guess_artist_song(VideoInfo(id="x", title="Untitled", uploader="Adele - Topic")) == ("Adele", "Untitled")


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Say Anything • In Your Eyes • Peter Gabriel", ["Say Anything", "In Your Eyes", "Peter Gabriel"]),
        ('Smashing Pumpkins "Mayonaise"', ["Smashing Pumpkins", "Mayonaise"]),
        ("Culture Beat - Mr. Vain (Official Video)", ["Culture Beat", "Mr. Vain"]),
        ("Sinéad O'Connor - Nothing Compares 2 U", ["Sinéad O'Connor", "Nothing Compares 2 U"]),
        ("Hello", ["Hello"]),
    ],
)
def test_title_segments(title, expected):
    assert title_segments(title) == expected


def test_name_guesses_try_every_order_of_the_title_parts():
    info = VideoInfo(id="x", title="Apocalypse Now • The End • The Doors", uploader="HD Film Tributes")
    guesses = name_guesses(info)
    assert guesses[0] == ("Apocalypse Now", "The End")
    assert ("The Doors", "The End") in guesses and ("HD Film Tributes", "The End") in guesses
    assert ("The End", "The End") not in guesses


def test_name_guesses_start_with_the_music_metadata_and_keep_given_names():
    info = VideoInfo(id="x", title="Foo - Bar", artist="Meta Artist", track="Meta Track")
    assert name_guesses(info)[0] == ("Meta Artist", "Meta Track")
    assert {artist for artist, _ in name_guesses(info, artist="Given")} == {"Given"}


def test_normalize_name():
    assert normalize_name("The Smashing Pumpkins") == normalize_name("smashing pumpkins")
    assert normalize_name("Mr. Vain") == normalize_name("Mr Vain")
    assert normalize_name("Sinéad O’Connor") == normalize_name("Sinead O'Connor") == "sinead oconnor"
    assert normalize_name("Paul McCartney & Wings") == normalize_name("Paul McCartney and Wings")
    assert normalize_name("The The") == "the"


def test_without_brackets():
    assert without_brackets("In Your Eyes (version from ‘Say Anything’)") == "In Your Eyes"
    assert without_brackets("(Untitled)") == "(Untitled)"

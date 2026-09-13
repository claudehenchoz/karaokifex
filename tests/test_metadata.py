import pytest

from karaokifex.metadata import channel_name, guess_artist_song, split_title
from karaokifex.models import VideoInfo


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Queen - Bohemian Rhapsody (Official Video Remastered)", ("Queen", "Bohemian Rhapsody")),
        ("Rick Astley – Never Gonna Give You Up [4K]", ("Rick Astley", "Never Gonna Give You Up")),
        ('Daft Punk - "Get Lucky" (feat. Pharrell)', ("Daft Punk", "Get Lucky")),
        ("Just A Song Title (Lyrics)", (None, "Just A Song Title")),
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

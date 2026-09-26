"""Working out artist and song name for a video (pure functions)."""

from __future__ import annotations

import itertools
import re
import unicodedata

from karaokifex.models import VideoInfo

# Bracketed title decorations that are not part of the song name, e.g. "(Official Video)", "[4K Remaster]".
_NOISE = re.compile(
    r"\s*[\(\[【][^\)\]】]*"
    r"\b(official|video|audio|lyrics?|visuali[sz]er|hd|hq|4k|remaster(ed)?|mv|m/v|live|clip|feat|ft)\b"
    r"[^\)\]】]*[\)\]】]",
    re.IGNORECASE,
)
_SEPARATORS = (" - ", " – ", " — ", " ~ ", " | ")
_QUOTES = "\"'“”‘’"
_CHANNEL_SUFFIX = re.compile(r"(\s*-\s*Topic|VEVO)$", re.IGNORECASE)
# Where titles like "Film • Song • Artist" or 'Artist "Song"' break into their parts.
_SEGMENT_BREAKS = re.compile(r"\s+[-–—~|•·]\s+|[\"“”]")
_QUOTED = re.compile(r'^(?P<artist>[^"“”]+?)\s*["“](?P<song>[^"“”]+)["”]\s*$')
MAX_SEGMENTS = 4
_BRACKETS = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


def clean_title(title: str) -> str:
    """Strip decorations like "(Official Video)" from a video title."""
    cleaned = _NOISE.sub("", title)
    return re.sub(r"\s+", " ", cleaned).strip(" -–—|")


def split_title(title: str) -> tuple[str | None, str]:
    """Split "Artist - Song (Official Video)" into ("Artist", "Song"); artist is None if there is no separator."""
    cleaned = clean_title(title)
    for separator in _SEPARATORS:
        if separator in cleaned:
            artist, song = cleaned.split(separator, 1)
            return artist.strip(), song.strip().strip(_QUOTES)
    if quoted := _QUOTED.match(cleaned):  # Artist "Song"
        return quoted["artist"].strip(), quoted["song"].strip()
    return None, cleaned.strip(_QUOTES)


def channel_name(uploader: str | None) -> str | None:
    """"AdeleVEVO" -> "Adele", "Adele - Topic" -> "Adele"."""
    if not uploader:
        return None
    return _CHANNEL_SUFFIX.sub("", uploader).strip() or None


def guess_artist_song(info: VideoInfo, artist: str | None = None, song: str | None = None) -> tuple[str, str]:
    """Explicit values win, then yt-dlp's music metadata, then whatever the video title says."""
    title_artist, title_song = split_title(info.title)
    artist = artist or info.artist or title_artist or channel_name(info.uploader) or "Unknown Artist"
    song = song or info.track or title_song or info.title or info.id
    return artist.strip(), song.strip()


def title_segments(title: str) -> list[str]:
    """The parts of a video title: "Say Anything • In Your Eyes • Peter Gabriel" -> its three names."""
    parts = (part.strip(" -–—|") for part in _SEGMENT_BREAKS.split(clean_title(title)))
    return [part for part in parts if part][:MAX_SEGMENTS]


def name_guesses(info: VideoInfo, artist: str | None = None, song: str | None = None) -> list[tuple[str, str]]:
    """(artist, song) pairs worth checking against MusicBrainz, the likeliest first.

    Explicit values stay fixed. Otherwise yt-dlp's music metadata comes first, then every ordered
    pair of title parts (titles put the artist first, last, or around a quoted song), then the channel.
    """
    segments = title_segments(info.title)
    artists = [artist] if artist else [info.artist, *segments, channel_name(info.uploader)]
    songs = [song] if song else [info.track, *segments]
    pairs: list[tuple[str, str]] = []
    for pair in itertools.product(artists, songs):
        if pair[0] and pair[1] and normalize_name(pair[0]) != normalize_name(pair[1]) and pair not in pairs:
            pairs.append(pair)  # type: ignore[arg-type]
    return pairs


def normalize_name(name: str) -> str:
    """Spelling used to compare names: "The Smashing Pumpkins" ~ "smashing pumpkins", "Mr. Vain" ~ "mr vain"."""
    decomposed = unicodedata.normalize("NFKD", name.casefold().replace("&", " and "))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    words = re.sub(r"[^\w\s]|_", " ", re.sub(r"['’‘`]", "", plain)).split()
    return " ".join(words[1:] if words[:1] == ["the"] and len(words) > 1 else words)


def without_brackets(name: str) -> str:
    """Drop bracketed parts: "In Your Eyes (version from 'Say Anything')" -> "In Your Eyes"."""
    return _BRACKETS.sub("", name).strip() or name

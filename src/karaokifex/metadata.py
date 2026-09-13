"""Working out artist and song name for a video (pure functions)."""

from __future__ import annotations

import re

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

"""Lyrics lookup on lrclib.net and LRC parsing."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import requests

from karaokifex import __version__
from karaokifex.models import LyricLine, Lyrics

log = logging.getLogger(__name__)

SEARCH_URL = "https://lrclib.net/api/search"
USER_AGENT = f"karaokifex/{__version__}"
DURATION_TOLERANCE = 10.0  # seconds; within this, synced lyrics beat a closer plain-text match

_TIMESTAMP = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
_WORD_TIMESTAMP = re.compile(r"<\d+:\d+(?:[.:]\d+)?>")  # enhanced LRC per-word tags

HttpGet = Callable[..., Any]


def parse_lrc(text: str) -> list[LyricLine]:
    """Parse synced LRC lyrics; lines with several timestamps (repeated choruses) are expanded."""
    lines: list[LyricLine] = []
    for raw in text.splitlines():
        raw = raw.strip()
        stamps: list[float] = []
        position = 0
        while match := _TIMESTAMP.match(raw, position):
            minutes, seconds = match.groups()
            stamps.append(int(minutes) * 60 + float(seconds.replace(":", ".")))
            position = match.end()
        lyric = _WORD_TIMESTAMP.sub("", raw[position:]).strip()
        if stamps and lyric:
            lines.extend(LyricLine(stamp, lyric) for stamp in stamps)
    return sorted(lines, key=lambda line: line.start)  # type: ignore[arg-type, return-value]


def parse_plain(text: str) -> list[LyricLine]:
    return [LyricLine(None, line.strip()) for line in text.splitlines() if line.strip()]


def choose_best(candidates: list[dict[str, Any]], duration: float | None) -> dict[str, Any] | None:
    """Closest duration wins, but synced lyrics are preferred while within DURATION_TOLERANCE."""
    usable = [c for c in candidates if not c.get("instrumental") and (c.get("syncedLyrics") or c.get("plainLyrics"))]
    if not usable:
        return None

    def rank(candidate: dict[str, Any]) -> tuple[bool, bool, float]:
        if duration and candidate.get("duration"):
            difference = abs(candidate["duration"] - duration)
        else:
            difference = DURATION_TOLERANCE
        too_far = difference > DURATION_TOLERANCE
        return too_far, not too_far and not candidate.get("syncedLyrics"), difference

    return min(usable, key=rank)


def search_lrclib(artist: str, song: str, *, get: HttpGet = requests.get, timeout: float = 20) -> list[dict[str, Any]]:
    """Field search first (precise), then free-text search (forgiving)."""
    queries = [{"track_name": song, "artist_name": artist}, {"q": f"{artist} {song}"}]
    for params in queries:
        log.debug("lrclib search %s", params)
        response = get(SEARCH_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        if results := response.json():
            return results
    return []


def fetch_lyrics(artist: str, song: str, duration: float | None, *, get: HttpGet = requests.get) -> Lyrics | None:
    candidates = search_lrclib(artist, song, get=get)
    best = choose_best(candidates, duration)
    if best is None:
        return None
    lines = parse_lrc(best["syncedLyrics"]) if best.get("syncedLyrics") else []
    synced = bool(lines)
    if not synced:
        lines = parse_plain(best.get("plainLyrics") or "")
    return Lyrics(
        lrclib_id=best.get("id"),
        artist=best.get("artistName", artist),
        track=best.get("trackName", song),
        album=best.get("albumName"),
        duration=best.get("duration"),
        synced=synced,
        lines=tuple(lines),
    )


def save_lyrics(lyrics: Lyrics | None, path: Path) -> None:
    """Also records a miss, so a re-run doesn't query lrclib again."""
    data = {"found": False} if lyrics is None else {"found": True, **lyrics.to_dict()}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_lyrics(path: Path) -> Lyrics | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.pop("found"):
        return None
    return Lyrics.from_dict(data)

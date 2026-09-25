"""Lyrics lookup on lrclib.net and LRC parsing."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import requests

from karaokifex import __version__
from karaokifex.models import LyricLine, Lyrics
from karaokifex.timing import normalize, split_words
from karaokifex.workspace import partial_path

log = logging.getLogger(__name__)

SEARCH_URL = "https://lrclib.net/api/search"
USER_AGENT = f"karaokifex/{__version__}"
DURATION_TOLERANCE = 10.0  # seconds; within this, synced lyrics beat a closer plain-text match
MAX_CANDIDATES = 3  # lyrics versions kept; the subtitles step picks the one that aligns best
PROMPT_CHARS = 800  # whisper's prompt holds ~220 tokens
MIN_LANGUAGE_PROBABILITY = 0.7
ATTEMPTS = 5  # waiting 2, 4, 8, 16 s in between: lrclib is busy now and then (503) and back soon
RETRY_WAIT = 2.0
_RETRY = frozenset({502, 503, 504})

_TIMESTAMP = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
_WORD_TIMESTAMP = re.compile(r"<(\d+):(\d+(?:[.:]\d+)?)>")  # enhanced LRC per-word tags

HttpGet = Callable[..., Any]


def _seconds(minutes: str, seconds: str) -> float:
    return int(minutes) * 60 + float(seconds.replace(":", "."))


def parse_lrc(text: str) -> list[LyricLine]:
    """Parse synced LRC lyrics; lines with several timestamps (repeated choruses) are expanded."""
    lines: list[LyricLine] = []
    for raw in text.splitlines():
        raw = raw.strip()
        stamps: list[float] = []
        position = 0
        while match := _TIMESTAMP.match(raw, position):
            stamps.append(_seconds(*match.groups()))
            position = match.end()
        body = raw[position:]
        lyric = re.sub(r"\s+", " ", _WORD_TIMESTAMP.sub("", body)).strip()
        if stamps and lyric:
            word_starts, end = _word_tags(body, lyric)
            lines.extend(LyricLine(stamp, lyric, word_starts, end) for stamp in stamps)
    return sorted(lines, key=lambda line: line.start)  # type: ignore[arg-type, return-value]


def _word_tags(body: str, lyric: str) -> tuple[tuple[float | None, ...] | None, float | None]:
    """Per-word start times from enhanced LRC tags, aligned with split_words(lyric)."""
    pieces = _WORD_TIMESTAMP.split(body)  # text, minutes, seconds, text, minutes, seconds, ...
    if len(pieces) == 1:
        return None, None
    starts: list[float | None] = [None] * len(split_words(pieces[0]))
    end: float | None = None
    for index in range(1, len(pieces), 3):
        time = _seconds(pieces[index], pieces[index + 1])
        words = split_words(pieces[index + 2])
        if words:
            starts += [time] + [None] * (len(words) - 1)
        elif index + 3 >= len(pieces) and starts:
            end = time  # a closing tag after the last word
    if len(starts) != len(split_words(lyric)) or all(start is None for start in starts):
        return None, None
    return tuple(starts), end


def parse_plain(text: str) -> list[LyricLine]:
    return [LyricLine(None, line.strip()) for line in text.splitlines() if line.strip()]


def rank_candidates(candidates: list[dict[str, Any]], duration: float | None) -> list[dict[str, Any]]:
    """Closest duration first, but synced lyrics are preferred while within DURATION_TOLERANCE.

    Candidates whose text is identical to a better-ranked one are dropped.
    """
    usable = [c for c in candidates if not c.get("instrumental") and (c.get("syncedLyrics") or c.get("plainLyrics"))]

    def rank(candidate: dict[str, Any]) -> tuple[bool, bool, float]:
        if duration and candidate.get("duration"):
            difference = abs(candidate["duration"] - duration)
        else:
            difference = DURATION_TOLERANCE
        too_far = difference > DURATION_TOLERANCE
        return too_far, not too_far and not candidate.get("syncedLyrics"), difference

    ranked: list[dict[str, Any]] = []
    seen: set[tuple[bool, str]] = set()
    for candidate in sorted(usable, key=rank):
        key = (bool(candidate.get("syncedLyrics")),
               " ".join(normalize(w) for w in (candidate.get("syncedLyrics") or candidate.get("plainLyrics")).split()))
        if key not in seen:
            seen.add(key)
            ranked.append(candidate)
    return ranked


def choose_best(candidates: list[dict[str, Any]], duration: float | None) -> dict[str, Any] | None:
    ranked = rank_candidates(candidates, duration)
    return ranked[0] if ranked else None


def search_lrclib(artist: str, song: str, *, get: HttpGet = requests.get, timeout: float = 20) -> list[dict[str, Any]]:
    """Field search first (precise), then free-text search (forgiving)."""
    queries = [{"track_name": song, "artist_name": artist}, {"q": f"{artist} {song}"}]
    for params in queries:
        log.debug("lrclib search %s", params)
        if results := _ask(params, get=get, timeout=timeout):
            return results
    return []


def _ask(params: dict[str, str], *, get: HttpGet, timeout: float) -> list[dict[str, Any]]:
    """One search; a busy answer, a gateway error or a timeout is asked again, waiting twice as long each time."""
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(RETRY_WAIT * 2 ** (attempt - 1))
        try:
            response = get(SEARCH_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == ATTEMPTS - 1:
                raise
            log.info("lrclib didn't answer (%s), asking again", error)
            continue
        if response.status_code in _RETRY and attempt < ATTEMPTS - 1:
            log.info("lrclib answered %d, asking again", response.status_code)
            continue
        response.raise_for_status()
        return response.json()
    raise AssertionError("unreachable")


def _to_lyrics(candidate: dict[str, Any], artist: str, song: str) -> Lyrics:
    lines = parse_lrc(candidate["syncedLyrics"]) if candidate.get("syncedLyrics") else []
    synced = bool(lines)
    if not synced:
        lines = parse_plain(candidate.get("plainLyrics") or "")
    return Lyrics(
        lrclib_id=candidate.get("id"),
        artist=candidate.get("artistName", artist),
        track=candidate.get("trackName", song),
        album=candidate.get("albumName"),
        duration=candidate.get("duration"),
        synced=synced,
        lines=tuple(lines),
    )


def fetch_lyrics(artist: str, song: str, duration: float | None, *, get: HttpGet = requests.get,
                 limit: int = MAX_CANDIDATES) -> list[Lyrics]:
    """The best-ranked lyrics versions (empty on a miss)."""
    ranked = rank_candidates(search_lrclib(artist, song, get=get), duration)
    return [lyrics for c in ranked[:limit] if (lyrics := _to_lyrics(c, artist, song)).lines]


def describe(lyrics: Lyrics) -> str:
    kind = "synced" if lyrics.synced else "plain"
    return f"lrclib #{lyrics.lrclib_id}: {lyrics.artist} – {lyrics.track} ({kind})"


def save_lyrics(candidates: Sequence[Lyrics], path: Path) -> None:
    """Also records a miss, so a re-run doesn't query lrclib again."""
    data = {"found": bool(candidates), "candidates": [lyrics.to_dict() for lyrics in candidates]}
    partial = partial_path(path)
    partial.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    partial.replace(path)


def load_lyrics(path: Path) -> list[Lyrics]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.pop("found"):
        return []
    if "candidates" not in data:  # written by an older version: a single match
        return [Lyrics.from_dict(data)]
    return [Lyrics.from_dict(candidate) for candidate in data["candidates"]]


def prompt_text(lines: Sequence[LyricLine], limit: int = PROMPT_CHARS) -> str:
    """The lyrics as a whisper prompt: each distinct line once, cut at a word boundary."""
    seen: set[str] = set()
    distinct = [line.text for line in lines if not (line.text in seen or seen.add(line.text))]
    text = " ".join(distinct)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


def guess_language(lines: Sequence[LyricLine]) -> str | None:
    """ISO 639-1 code of the lyrics' language, or None when unsure."""
    text = " ".join(line.text for line in lines)
    if len(text.split()) < 5:
        return None
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0  # deterministic results
    try:
        best = detect_langs(text)[0]
    except LangDetectException:
        return None
    if best.prob < MIN_LANGUAGE_PROBABILITY:
        return None
    return best.lang.split("-")[0]  # "zh-cn" -> "zh"

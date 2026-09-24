"""Canonical artist and song names from MusicBrainz (recording search of the ws/2 JSON API).

One search asks for every (artist, song) guess at once. A recording confirms a guess when both
names match it, spelling aside; the names of the guess most recordings confirm win, in the
spelling most of them use. The artist is the one credited on the recordings ("Paul McCartney &
Wings", not the band's entry "Wings"), but in its own entry's spelling when that is merely
another way to write it ("The Smashing Pumpkins" for "Smashing Pumpkins"). No confirmation
means no confident match, and the guess stays.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import requests

from karaokifex import __version__
from karaokifex.metadata import normalize_name, without_brackets

log = logging.getLogger(__name__)

SEARCH_URL = "https://musicbrainz.org/ws/2/recording"
# MusicBrainz asks every client for a User-Agent with a way to contact its author.
USER_AGENT = f"karaokifex/{__version__} ( https://github.com/samluescher/karaokifex )"
MIN_INTERVAL = 1.0  # seconds between requests: MusicBrainz allows one per second
RESULTS = 100  # the most one search returns
ATTEMPTS = 6  # waiting 1, 2, 4 ... 32 s in between: MusicBrainz is often busy for several seconds
_RETRY = frozenset({502, 503, 504})

# MusicBrainz spells with typographic punctuation; file names and lrclib searches get the plain kind.
_PLAIN_PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‐": "-", "‑": "-"})

HttpGet = Callable[..., Any]

_lock = threading.Lock()
_last_request = 0.0


@dataclass(frozen=True)
class Match:
    artist: str
    song: str
    recordings: int  # how many recordings confirmed these names
    guess: tuple[str, str]  # the (artist, song) guess they confirmed


def lookup(guesses: Sequence[tuple[str, str]], text: str | None = None, *, get: HttpGet = requests.get,
           timeout: float = 15) -> Match | None:
    """The canonical names for the best-confirmed guess; `text` is a whole title that may hold both names."""
    if not guesses and not text:
        return None
    return best_match(search(build_query(guesses, text), get=get, timeout=timeout), guesses, text)


def build_query(guesses: Sequence[tuple[str, str]], text: str | None = None) -> str:
    """Lucene: each guess as a phrase pair, and `text` as loose words in both the title and the artist."""
    clauses = [f'(recording:"{_terms(song)}" AND artist:"{_terms(artist)}")'
               for artist, song in guesses if _terms(song) and _terms(artist)]
    if text and _terms(text):
        clauses.append(f"(recording:({_terms(text)}) AND artist:({_terms(text)}))")
    return " OR ".join(clauses)


def search(query: str, *, get: HttpGet = requests.get, timeout: float = 15) -> list[dict[str, Any]]:
    """Recordings matching `query`, at most one request per MIN_INTERVAL.

    MusicBrainz answers 503 when all its clients together ask too much (or one too fast), so busy
    answers, gateway errors and timeouts are retried, waiting twice as long each time.
    """
    global _last_request
    for attempt in range(ATTEMPTS):
        with _lock:
            time.sleep(max(0.0, _last_request + MIN_INTERVAL * 2 ** attempt - time.monotonic()))
            _last_request = time.monotonic()
        log.debug("MusicBrainz search %s", query)
        try:
            response = get(SEARCH_URL, params={"query": query, "fmt": "json", "limit": RESULTS},
                           headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == ATTEMPTS - 1:
                raise
            log.debug("MusicBrainz didn't answer (%s), trying again", error)
            continue
        if response.status_code in _RETRY and attempt < ATTEMPTS - 1:
            log.debug("MusicBrainz answered %d, trying again", response.status_code)
            continue
        response.raise_for_status()
        return response.json().get("recordings", [])
    raise AssertionError("unreachable")


def best_match(recordings: Sequence[dict[str, Any]], guesses: Sequence[tuple[str, str]],
               text: str | None = None) -> Match | None:
    confirmed: dict[tuple[str, str], list[tuple[str, str]]] = {}  # guess -> (artist, title) of each recording
    for recording in recordings:
        names = _names(recording)
        if names is None:
            continue
        artist, title, canonical, credited = names
        for guess in guesses:
            if _same_artist(guess[0], canonical, credited) and _same_song(guess[1], title):
                confirmed.setdefault(guess, []).append((artist, title))
                break
        else:
            if text and _covers(text, credited, title):
                confirmed.setdefault(("", text), []).append((artist, title))
    if not confirmed:
        return None
    # The most confirmed guess wins; on a tie, the likelier (earlier) guess.
    order = [*guesses, ("", text)]
    guess, found = max(confirmed.items(), key=lambda item: (len(item[1]), -order.index(item[0])))
    artist = Counter(artist for artist, _ in found).most_common(1)[0][0]
    song = _spelling([title for _, title in found], guess[1])
    return Match(artist.translate(_PLAIN_PUNCTUATION), song.translate(_PLAIN_PUNCTUATION), len(found), guess)


def _names(recording: dict[str, Any]) -> tuple[str, str, str, str] | None:
    """(artist to show, title, the artists' own names, the artist as credited on this recording)."""
    credits = recording.get("artist-credit") or []
    title = recording.get("title")
    if not credits or not title:
        return None
    canonical = "".join((c.get("artist") or {}).get("name", c.get("name", "")) + c.get("joinphrase", "")
                        for c in credits).strip()
    credited = "".join(c.get("name", "") + c.get("joinphrase", "") for c in credits).strip()
    shown = canonical if normalize_name(canonical) == normalize_name(credited) else credited
    return shown, title.strip(), canonical, credited


def _same_artist(guess: str, canonical: str, credited: str) -> bool:
    return normalize_name(guess) in (normalize_name(canonical), normalize_name(credited))


def _same_song(guess: str, title: str) -> bool:
    return normalize_name(guess) == normalize_name(title) or \
        normalize_name(without_brackets(guess)) == normalize_name(without_brackets(title))


def _covers(text: str, artist: str, title: str) -> bool:
    """Whether artist and title together are exactly the words of `text` (in any order)."""
    words = set(normalize_name(text).split())
    return words == set(normalize_name(artist).split()) | set(normalize_name(without_brackets(title)).split())


def _spelling(titles: list[str], guess: str) -> str:
    """The title as most recordings spell it, preferring spellings of the guess itself over ones with
    additions like "(from Apocalypse Now)"."""
    exact = [title for title in titles if normalize_name(title) == normalize_name(guess)]
    return Counter(exact or [without_brackets(title) for title in titles]).most_common(1)[0][0]


def _terms(text: str) -> str:
    """Words safe inside a Lucene query: letters, digits and apostrophes (MusicBrainz indexes "o'connor")."""
    return " ".join(re.sub(r"[^\w']|_", " ", text.translate(_PLAIN_PUNCTUATION).casefold()).split())

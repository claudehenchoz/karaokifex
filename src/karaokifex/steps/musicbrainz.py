"""Canonical artist and song names from MusicBrainz (recording search of the ws/2 JSON API).

One search asks for every (artist, song) guess at once. A recording confirms a guess when both
names match it, spelling aside; the names of the guess most recordings confirm win, in the
spelling most of them use. The artist is the one credited on the recordings ("Paul McCartney &
Wings", not the band's entry "Wings"), but in its own entry's spelling when that is merely
another way to write it ("The Smashing Pumpkins" for "Smashing Pumpkins"). No confirmation
means no confident match, and the guess stays.

`describe` then tells what MusicBrainz knows of the song: the album it first came out on and
the year it did, its genres, its writers, the language it is sung in and where its artist is
from. The recordings it confirms (both names match, as above) are the song; the album is the
earliest official one of the artist's own that isn't a compilation or a live record, and the
recording on it is the one asked about its genres and its work -- whose writers are the
song's (composer, lyricist, writer) and whose language is the song's.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import requests

from karaokifex import __version__
from karaokifex.metadata import normalize_name, without_brackets

log = logging.getLogger(__name__)

API = "https://musicbrainz.org/ws/2"
SEARCH_URL = f"{API}/recording"
# MusicBrainz asks every client for a User-Agent with a way to contact its author.
USER_AGENT = f"karaokifex/{__version__} ( https://github.com/samluescher/karaokifex )"
MIN_INTERVAL = 1.0  # seconds between requests: MusicBrainz allows one per second
RESULTS = 100  # the most one search returns
ATTEMPTS = 6  # waiting 1, 2, 4 ... 32 s in between: MusicBrainz is often busy for several seconds
_RETRY = frozenset({502, 503, 504})

# MusicBrainz spells with typographic punctuation; file names and lrclib searches get the plain kind.
_PLAIN_PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‐": "-", "‑": "-"})

# the work's relations that make a writer of the song
WRITER_ROLES = ("writer", "composer", "lyricist", "librettist")
MAX_GENRES = 5
VARIOUS_ARTISTS = "89ad4ac3-39f7-470e-963a-56509c546377"
# MusicBrainz names languages in ISO 639-3; the lyrics' guess and whisper speak ISO 639-1
_LANGUAGES = {"eng": "en", "deu": "de", "ger": "de", "gsw": "de", "fra": "fr", "fre": "fr", "ita": "it", "spa": "es",
              "por": "pt", "nld": "nl", "dut": "nl", "swe": "sv", "nor": "no", "nob": "no", "nno": "no", "dan": "da",
              "fin": "fi", "isl": "is", "pol": "pl", "ces": "cs", "hun": "hu", "rus": "ru", "ukr": "uk", "ell": "el",
              "tur": "tr", "jpn": "ja", "kor": "ko", "zho": "zh", "cmn": "zh", "yue": "zh", "hin": "hi", "ara": "ar",
              "heb": "he", "gle": "ga", "cym": "cy", "lat": "la", "ron": "ro", "hrv": "hr", "srp": "sr", "slv": "sl"}

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
    """The canonical names for the best-confirmed guess; `text` is a whole title that may hold both names.

    When none is confirmed, the guesses are asked about again without what their song adds in
    brackets ("99 Luftballons [1983]"), which MusicBrainz's titles seldom have.
    """
    if not guesses and not text:
        return None
    match = best_match(search(build_query(guesses, text), get=get, timeout=timeout), guesses, text)
    bare = list(dict.fromkeys((artist, without_brackets(song)) for artist, song in guesses
                              if without_brackets(song) and without_brackets(song) != song))
    if match is None and bare:
        match = best_match(search(build_query(bare), get=get, timeout=timeout), bare)
    return match


def build_query(guesses: Sequence[tuple[str, str]], text: str | None = None) -> str:
    """Lucene: each guess as a phrase pair, and `text` as loose words in both the title and the artist."""
    clauses = [f'(recording:"{_terms(song)}" AND artist:"{_terms(artist)}")'
               for artist, song in guesses if _terms(song) and _terms(artist)]
    if text and _terms(text):
        clauses.append(f"(recording:({_terms(text)}) AND artist:({_terms(text)}))")
    return " OR ".join(clauses)


def search(query: str, *, get: HttpGet = requests.get, timeout: float = 15) -> list[dict[str, Any]]:
    """Recordings matching `query`."""
    log.debug("MusicBrainz search %s", query)
    return _request(SEARCH_URL, {"query": query, "limit": RESULTS}, get=get, timeout=timeout).get("recordings", [])


def entity(kind: str, mbid: str, inc: Sequence[str] = (), *, get: HttpGet = requests.get,
           timeout: float = 15) -> dict[str, Any]:
    """One entry ('recording', 'work', 'release-group', 'artist') with the `inc` it is asked for."""
    log.debug("MusicBrainz %s %s (%s)", kind, mbid, "+".join(inc))
    return _request(f"{API}/{kind}/{mbid}", {"inc": "+".join(inc)} if inc else {}, get=get, timeout=timeout)


def _request(url: str, params: dict[str, Any], *, get: HttpGet, timeout: float) -> dict[str, Any]:
    """At most one request per MIN_INTERVAL.

    MusicBrainz answers 503 when all its clients together ask too much (or one too fast), so busy
    answers, gateway errors and timeouts are retried, waiting twice as long each time.
    """
    global _last_request
    for attempt in range(ATTEMPTS):
        with _lock:
            time.sleep(max(0.0, _last_request + MIN_INTERVAL * 2 ** attempt - time.monotonic()))
            _last_request = time.monotonic()
        try:
            response = get(url, params={**params, "fmt": "json"},
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
        return response.json()
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


# --- what MusicBrainz knows of a song --------------------------------------------------


@dataclass(frozen=True)
class Writer:
    name: str
    roles: tuple[str, ...]  # "writer", "composer", "lyricist", "librettist"


@dataclass(frozen=True)
class Details:
    album: str | None = None
    year: int | None = None
    genres: tuple[str, ...] = ()
    writers: tuple[Writer, ...] = ()
    language: str | None = None  # ISO 639-1 where there is one ("de"), else MusicBrainz's own code
    country: str | None = None  # where the artist is from (ISO 3166, "NO")
    ids: dict[str, str] | None = None  # the MusicBrainz ids of the recording, release group, work and artist

    def to_dict(self) -> dict[str, Any]:
        return {"album": self.album, "year": self.year, "genres": list(self.genres),
                "writers": [{"name": w.name, "roles": list(w.roles)} for w in self.writers],
                "language": self.language, "country": self.country, "musicbrainz": self.ids or {}}


def describe(artist: str, song: str, *, get: HttpGet = requests.get, timeout: float = 15) -> Details | None:
    """What MusicBrainz knows of `artist` - `song`, or None when it knows no recording of it.

    Six requests at most (a second apart): the search -- again with the artist's words loose and
    the song's without what it adds in brackets, when the names as a phrase find nothing ("Paul
    McCartney & Wings", "Total Eclipse of the Heart (Turn Around)") --, the recording, its work,
    its album and its artist.
    """
    found = confirmed(search(build_query([(artist, song)]), get=get, timeout=timeout), artist, song)
    title = _terms(without_brackets(song)) or _terms(song)
    if not found and _terms(artist) and title:
        loose = f'recording:"{title}" AND artist:({_terms(artist)})'
        found = confirmed(search(loose, get=get, timeout=timeout), artist, song)
    if not found:
        return None
    album = first_album(found)
    recording = album[0] if album else max(found, key=lambda r: len(r.get("releases") or []))
    ask = partial(entity, get=get, timeout=timeout)
    full = ask("recording", recording["id"], ("work-rels", "genres", "artist-credits"))
    works = [r["work"] for r in full.get("relations") or [] if r.get("type") == "performance" and r.get("work")]
    work = ask("work", works[0]["id"], ("artist-rels",)) if works else {}
    group = ask("release-group", album[1]["id"], ("genres",)) if album else {}
    credits = full.get("artist-credit") or recording.get("artist-credit") or []
    artist_id = (credits[0].get("artist") or {}).get("id") if credits else None
    performer = ask("artist", artist_id, ("genres",)) if artist_id else {}
    ids = {"recording": recording["id"], "release-group": album[1]["id"] if album else None,
           "work": work.get("id"), "artist": artist_id}
    return Details(album=album[1]["title"].translate(_PLAIN_PUNCTUATION) if album else None,
                   year=first_year(found),
                   genres=top_genres(full.get("genres"), group.get("genres"), performer.get("genres")),
                   writers=writers_of(work), language=language_of(work), country=performer.get("country"),
                   ids={k: v for k, v in ids.items() if v})


def confirmed(recordings: Sequence[dict[str, Any]], artist: str, song: str) -> list[dict[str, Any]]:
    """The recordings that are this song by this artist, spelling aside."""
    out = []
    for recording in recordings:
        names = _names(recording)
        if names and _same_artist(artist, names[2], names[3]) and _same_song(song, names[1]):
            out.append(recording)
    return out


def first_album(recordings: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(recording, release group) of the earliest official album of the artist's own with the song on
    it: no compilation, live record or soundtrack, and credited to one of the song's own artists."""
    best = None
    for recording in recordings:
        own = _artist_ids(recording.get("artist-credit"))
        for release in recording.get("releases") or []:
            group = release.get("release-group") or {}
            if release.get("status") != "Official" or group.get("primary-type") != "Album" or group.get("secondary-types"):
                continue
            by = _artist_ids(release.get("artist-credit"))
            if by and not (by & own) or VARIOUS_ARTISTS in by:
                continue
            date = release.get("date") or "9999"
            if best is None or date < best[0]:
                best = (date, recording, group)
    return (best[1], best[2]) if best else None


def _artist_ids(credits: Sequence[dict[str, Any]] | None) -> set[str]:
    return {c["artist"]["id"] for c in credits or [] if (c.get("artist") or {}).get("id")}


def first_year(recordings: Sequence[dict[str, Any]]) -> int | None:
    """The year the song first came out, on any of its recordings."""
    years = [int(d[:4]) for r in recordings if (d := r.get("first-release-date") or "")[:4].isdigit()]
    return min(years) if years else None


def top_genres(*lists: Sequence[dict[str, Any]] | None) -> tuple[str, ...]:
    """The genres voted most for the recording, then its album, then its artist (each weighed less):
    those with at least a sixth of the leader's votes, MAX_GENRES at most."""
    votes: Counter[str] = Counter()
    for weight, genres in zip((3, 2, 1), lists):
        for genre in genres or []:
            votes[genre["name"]] += weight * int(genre.get("count") or 1)
    if not votes:
        return ()
    ranked = votes.most_common()
    return tuple(name for name, n in ranked if n * 6 >= ranked[0][1])[:MAX_GENRES]


def writers_of(work: dict[str, Any]) -> tuple[Writer, ...]:
    """Who wrote the song, in the work's order, each once with every role they had."""
    roles: dict[str, list[str]] = {}
    for relation in work.get("relations") or []:
        name = (relation.get("artist") or {}).get("name")
        if relation.get("type") in WRITER_ROLES and name and relation["type"] not in roles.setdefault(name, []):
            roles[name].append(relation["type"])
    return tuple(Writer(name.translate(_PLAIN_PUNCTUATION), tuple(r)) for name, r in roles.items())


def language_of(work: dict[str, Any]) -> str | None:
    """The language the song is sung in; None for none ("zxx") or unknown."""
    codes = work.get("languages") or ([work["language"]] if work.get("language") else [])
    codes = [c for c in codes if c != "zxx"]
    if not codes:
        return None
    return "mul" if len(codes) > 1 else _LANGUAGES.get(codes[0], codes[0])

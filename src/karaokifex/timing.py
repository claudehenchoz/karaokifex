"""Merge lyric lines (lrclib) with word timestamps (whisperx) — pure functions, no I/O.

The lyrics decide *what* is displayed and how it is split into lines; whisperx
decides *when* each word is sung. Words are paired up by a dynamic-programming
sequence alignment of their normalised spellings (only allowing pairs that fit
lrclib's line timestamps, so a repeated chorus is matched to the right
occurrence). Lyric words whisperx did not hear get times interpolated from
their neighbours or, failing that, from lrclib's line timestamps.
"""

from __future__ import annotations

import difflib
import functools
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass
from itertools import groupby
from typing import Sequence

from karaokifex.models import LyricLine, TimedLine, TimedWord

MATCH_WINDOW = 5.0  # seconds a matched word may lie outside its lrclib line window
MIN_OFFSET_SAMPLES = 3  # heard line starts needed before trusting an estimated timing offset
FUZZY_RATIO = 0.6  # minimum spelling similarity to pair two non-identical words
LAST_LINE_WINDOW = 15.0  # assumed length of the final lrclib line
SECONDS_PER_CHAR = 0.08  # natural word duration estimate, clamped to the range below
MIN_WORD, MAX_WORD = 0.2, 0.8
MIN_DURATION = 0.05

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})

Window = tuple[float, float]


@dataclass(frozen=True)
class Alignment:
    lines: list[TimedLine]
    matched: int  # lyric words that got their timing from whisperx
    total: int

    @property
    def match_ratio(self) -> float:
        return self.matched / self.total if self.total else 0.0


@dataclass(frozen=True)
class _Token:
    line: int
    text: str
    norm: str

    @property
    def natural(self) -> float:
        """Rough time it takes to sing this word."""
        return min(max(len(self.norm) * SECONDS_PER_CHAR, MIN_WORD), MAX_WORD)


def normalize(word: str) -> str:
    """Spelling used for comparing words: "Don’t!" -> "don't", "Café" -> "cafe"."""
    decomposed = unicodedata.normalize("NFKD", word.translate(_APOSTROPHES))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^\w']|_", "", stripped.casefold()).strip("'")


def split_words(text: str) -> list[str]:
    """Split a lyric line into words, gluing punctuation-only tokens ("-", "...") onto a neighbour."""
    words: list[str] = []
    leading = ""
    for token in text.split():
        if normalize(token):
            words.append(leading + token)
            leading = ""
        elif words:
            words[-1] += " " + token
        else:
            leading += token + " "
    return words


def align_lyrics(lines: Sequence[LyricLine], heard: Sequence[TimedWord], *, window: float = MATCH_WINDOW) -> Alignment:
    """Time every word of `lines` using the words whisperx `heard`."""
    kept = [(line, words) for line in lines if (words := split_words(line.text))]
    tokens = [_Token(index, word, normalize(word)) for index, (_, words) in enumerate(kept) for word in words]
    if not tokens:
        return Alignment([], 0, 0)
    line_starts = [line.start for line, _ in kept]
    heard = [word for word in heard if normalize(word.text)]
    if offset := _estimate_offset(tokens, heard, line_starts, window):
        line_starts = [None if start is None else start + offset for start in line_starts]

    times: list[tuple[float, float] | None] = [None] * len(tokens)
    for i, j in _match_pairs(tokens, heard, _line_windows(line_starts), window):
        times[i] = (heard[j].start, heard[j].end)
    _drop_non_monotonic(times)
    matched = sum(t is not None for t in times)

    _time_unheard_synced_lines(tokens, times, line_starts)
    _fill_gaps(tokens, times)
    final = _make_monotonic(times)  # type: ignore[arg-type]  # every slot is filled by now

    timed_lines = [
        TimedLine(tuple(TimedWord(tokens[k].text, *final[k]) for k, _ in group))
        for _, group in groupby(enumerate(tokens), key=lambda item: item[1].line)
    ]
    return Alignment(timed_lines, matched, len(tokens))


def lines_from_words(words: Sequence[TimedWord], *, max_words: int = 8, max_pause: float = 0.8) -> list[TimedLine]:
    """Fallback when there are no lyrics: build display lines straight from the transcription."""
    lines: list[TimedLine] = []
    current: list[TimedWord] = []

    def flush() -> None:
        if current:
            lines.append(TimedLine(tuple(current)))
            current.clear()

    for word in words:
        if not normalize(word.text):
            continue
        if current and (word.start - current[-1].end > max_pause or len(current) >= max_words):
            flush()
        current.append(TimedWord(word.text.strip(), word.start, word.end))
        if len(current) >= 3 and word.text.rstrip().endswith((".", "?", "!")):
            flush()
    flush()
    return lines


def _line_windows(starts: Sequence[float | None]) -> list[Window | None]:
    """(start, end) of each lyric line according to lrclib, or None when unsynced."""
    windows: list[Window | None] = []
    for index, start in enumerate(starts):
        if start is None:
            windows.append(None)
            continue
        following = next((s for s in starts[index + 1:] if s is not None), None)
        windows.append((start, following if following is not None else start + LAST_LINE_WINDOW))
    return windows


def _estimate_offset(tokens: Sequence[_Token], heard: Sequence[TimedWord], line_starts: Sequence[float | None],
                     slack: float) -> float:
    """How far the video's timing is shifted against lrclib's (e.g. a music video with a longer intro).

    Median over lines whose first word was heard, using an alignment that ignores the timestamps.
    """
    if all(start is None for start in line_starts):
        return 0.0
    first_words = {k for k in range(len(tokens)) if k == 0 or tokens[k - 1].line != tokens[k].line}
    differences = [
        heard[j].start - start
        for i, j in _match_pairs(tokens, heard, [None] * len(line_starts), slack)
        if i in first_words and (start := line_starts[tokens[i].line]) is not None
    ]
    return statistics.median(differences) if len(differences) >= MIN_OFFSET_SAMPLES else 0.0


@functools.lru_cache(maxsize=65536)
def _similarity(lyric: str, heard: str) -> float:
    """1 for identical words, a bit less for near-misses ("colour"/"color"), 0 otherwise."""
    if lyric == heard:
        return 1.0
    ratio = difflib.SequenceMatcher(None, lyric, heard).ratio()
    return ratio * 0.9 if ratio >= FUZZY_RATIO else 0.0


def _match_pairs(tokens: Sequence[_Token], heard: Sequence[TimedWord], windows: Sequence[Window | None],
                 slack: float) -> list[tuple[int, int]]:
    """Order-preserving pairing (lyric index, heard index) maximising total word similarity.

    A classic LCS-style dynamic programme; a pair is only allowed when the heard
    word lies within its lyric line's lrclib window (± slack).
    """
    heard_norm = [normalize(word.text) for word in heard]
    rows = len(tokens) + 1
    columns = len(heard) + 1
    score = [[0.0] * columns for _ in range(rows)]
    for i in range(1, rows):
        token = tokens[i - 1]
        window = windows[token.line]
        low, high = (window[0] - slack, window[1] + slack) if window else (-math.inf, math.inf)
        previous, row = score[i - 1], score[i]
        for j in range(1, columns):
            best = previous[j] if previous[j] >= row[j - 1] else row[j - 1]
            if low <= heard[j - 1].start <= high:
                similarity = _similarity(token.norm, heard_norm[j - 1])
                if similarity and previous[j - 1] + similarity > best:
                    best = previous[j - 1] + similarity
            row[j] = best

    pairs: list[tuple[int, int]] = []
    i, j = rows - 1, columns - 1
    while i > 0 and j > 0:
        if score[i][j] == score[i - 1][j]:
            i -= 1
        elif score[i][j] == score[i][j - 1]:
            j -= 1
        else:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
    return pairs[::-1]


def _drop_non_monotonic(times: list[tuple[float, float] | None]) -> None:
    """Safety net in case whisperx reports words out of order."""
    last = -math.inf
    for index, value in enumerate(times):
        if value is None:
            continue
        if value[0] < last:
            times[index] = None
        else:
            last = value[0]


def _time_unheard_synced_lines(tokens: list[_Token], times: list[tuple[float, float] | None],
                               line_starts: Sequence[float | None]) -> None:
    """Lines whisperx missed entirely but lrclib has a timestamp for start at that timestamp."""
    for line, group in groupby(range(len(tokens)), key=lambda k: tokens[k].line):
        indices = list(group)
        start = line_starts[line]
        if start is None or any(times[k] is not None for k in indices):
            continue
        first, last = indices[0], indices[-1]
        previous_end = next((t[1] for k in range(first - 1, -1, -1) if (t := times[k]) is not None), 0.0)
        next_start = next((t[0] for k in range(last + 1, len(times)) if (t := times[k]) is not None), math.inf)
        next_line = next((s for s in line_starts[line + 1:] if s is not None), math.inf)
        natural = sum(tokens[k].natural for k in indices)
        begin = max(start, previous_end)
        finish = min(next_line - 0.1, next_start, begin + natural * 2.5)
        _pack(times, indices, [tokens[k].natural for k in indices], begin, max(finish, begin + natural))


def _fill_gaps(tokens: list[_Token], times: list[tuple[float, float] | None]) -> None:
    """Interpolate every remaining run of untimed words between its timed neighbours."""
    count, i = len(times), 0
    while i < count:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < count and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else None  # type: ignore[index]
        right = times[j][0] if j < count else None  # type: ignore[index]
        _fill_run(tokens, times, i, j, left, right)
        i = j


def _fill_run(tokens: list[_Token], times: list[tuple[float, float] | None], i: int, j: int,
              left: float | None, right: float | None) -> None:
    run = list(range(i, j))
    natural = sum(tokens[k].natural for k in run)
    if left is None and right is None:
        left, right = 0.0, natural
    elif left is None:
        left = max(0.0, right - natural)  # type: ignore[operator]
    elif right is None:
        right = left + natural
    assert left is not None and right is not None
    right = max(right, left)

    previous_line = tokens[i - 1].line if i > 0 else None
    next_line = tokens[j].line if j < len(tokens) else None
    if right - left <= natural * 1.5 or previous_line == next_line:
        _pack(times, run, [tokens[k].natural for k in run], left, right)
        return

    # A long gap (instrumental break, unheard lines): keep words close to the line they belong to.
    head = [k for k in run if tokens[k].line == previous_line]
    tail = [k for k in run if tokens[k].line == next_line]
    middle = [k for k in run if k not in head and k not in tail]
    if head:
        end = left + sum(tokens[k].natural for k in head)
        _pack(times, head, [tokens[k].natural for k in head], left, end)
        left = end
    if tail:
        start = right - sum(tokens[k].natural for k in tail)
        _pack(times, tail, [tokens[k].natural for k in tail], start, right)
        right = start
    groups = [list(g) for _, g in groupby(middle, key=lambda k: tokens[k].line)]
    if groups:
        slot = (right - left) / len(groups)
        for number, group in enumerate(groups):
            start = left + number * slot
            length = min(slot, sum(tokens[k].natural for k in group))
            _pack(times, group, [tokens[k].natural for k in group], start, start + length)


def _pack(times: list[tuple[float, float] | None], indices: list[int], weights: list[float],
          start: float, end: float) -> None:
    """Spread consecutive words over [start, end], proportionally to their weights."""
    total = sum(weights) or 1.0
    cursor = start
    for index, weight in zip(indices, weights):
        length = (end - start) * weight / total
        times[index] = (cursor, cursor + length)
        cursor += length


def _make_monotonic(times: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    previous_start = 0.0
    for start, end in times:
        start = max(start, previous_start)
        result.append((start, max(end, start + MIN_DURATION)))
        previous_start = start
    for index in range(len(result) - 1):
        start, end = result[index]
        next_start = result[index + 1][0]
        if end > next_start:
            result[index] = (start, max(next_start, start + MIN_DURATION))
    return result

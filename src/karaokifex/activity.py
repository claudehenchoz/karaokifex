"""When the lead vocals are audible, from the RMS envelope of the lead-vocal stem (numpy only, no I/O).

The karaoke stem is clean, so plain energy thresholding is a reliable voice
activity detector for it. The timing code uses it as a veto: words don't start
in silence, interpolated words land where someone sings, held notes last until
the energy drops, and lines placed on silence were cut from the video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOP = 0.02  # seconds per frame
FLOOR_MARGIN = 15.0  # dB above the noise floor (10th percentile) that counts as voiced...
PEAK_RANGE = 40.0  # ...unless that is more than this far below the loud parts (99th percentile)
HYSTERESIS = 6.0  # dB; once voiced, a frame stays voiced down to threshold - this
MIN_GAP = 0.15  # seconds; shorter silences inside singing are filled (breaths, consonants)
MIN_BLIP = 0.10  # seconds; shorter voiced bits are dropped (clicks, bleed)


@dataclass(frozen=True)
class Activity:
    db: np.ndarray  # frame energy in dB
    voiced: np.ndarray  # bool per frame
    hop: float = HOP

    @classmethod
    def compute(cls, samples: np.ndarray, sample_rate: int, hop: float = HOP) -> Activity:
        mono = samples.mean(axis=1) if samples.ndim == 2 else samples
        size = max(1, round(hop * sample_rate))
        count = len(mono) // size
        frames = np.asarray(mono[: count * size], dtype=np.float64).reshape(count, size)
        db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)
        if count == 0:
            return cls(db, np.zeros(0, dtype=bool), hop)
        high = max(np.percentile(db, 10) + FLOOR_MARGIN, np.percentile(db, 99) - PEAK_RANGE)
        voiced = _hysteresis(db, high, high - HYSTERESIS)
        voiced = _close_runs(voiced, value=False, shorter_than=round(MIN_GAP / hop))
        voiced = _close_runs(voiced, value=True, shorter_than=round(MIN_BLIP / hop))
        return cls(db, voiced, hop)

    @property
    def duration(self) -> float:
        return len(self.voiced) * self.hop

    def _frame(self, t: float) -> int:
        return int(min(max(t / self.hop, 0.0), len(self.voiced)))  # clamp before int(): callers pass ±inf

    def is_voiced(self, t: float) -> bool:
        frame = self._frame(t)
        return frame < len(self.voiced) and bool(self.voiced[frame])

    def voiced_fraction(self, start: float, end: float) -> float:
        a, b = self._frame(start), self._frame(end)
        if b <= a:
            return 1.0 if self.is_voiced(start) else 0.0
        return float(self.voiced[a:b].mean())

    def voiced_intervals(self, start: float, end: float) -> list[tuple[float, float]]:
        """Voiced stretches inside [start, end]."""
        a, b = self._frame(start), self._frame(end)
        intervals: list[tuple[float, float]] = []
        for run_start, run_end in _runs(self.voiced[a:b]):
            intervals.append((max(start, (a + run_start) * self.hop), min(end, (a + run_end) * self.hop)))
        return intervals

    def next_voiced(self, t: float, limit: float) -> float | None:
        """First voiced moment in [t, limit], or None."""
        a, b = self._frame(t), self._frame(limit)
        hits = np.flatnonzero(self.voiced[a:b])
        return None if not len(hits) else max(t, (a + hits[0]) * self.hop)

    def onset_near(self, t: float, radius: float) -> float | None:
        """The silence→voice transition closest to t, within ±radius."""
        a, b = self._frame(t - radius), self._frame(t + radius)
        onsets = [a + i for i in range(max(a, 1) - a, b - a) if self.voiced[a + i] and not self.voiced[a + i - 1]]
        if not onsets:
            return None
        return min((frame * self.hop for frame in onsets), key=lambda time: abs(time - t))

    def phrase_end(self, t: float, limit: float) -> float:
        """Where the singing going on at t stops (at most `limit`); t itself when silent there."""
        if not self.is_voiced(t):
            return t
        a, b = self._frame(t), self._frame(limit)
        silent = np.flatnonzero(~self.voiced[a:b])
        return limit if not len(silent) else max(t, (a + silent[0]) * self.hop)

    def onset_envelope(self) -> np.ndarray:
        """Rises in vocal energy per frame, emphasised where a phrase starts after a pause (like a lyric line)."""
        if not len(self.db):
            return self.db
        floor = np.percentile(self.db, 10) + FLOOR_MARGIN - HYSTERESIS
        padded = np.pad(np.maximum(self.db, floor), 1, mode="edge")  # zero padding would fake a rise at the end
        smooth = np.convolve(padded, np.ones(3) / 3, mode="valid")
        rises = np.maximum(np.diff(smooth, prepend=smooth[0]), 0.0)
        weight = np.full(len(rises), 0.25)
        for start, _ in _runs(self.voiced):
            weight[max(0, start - 2): start + 3] = 1.0
        return rises * weight

    def save(self, path: Path) -> None:
        with path.open("wb") as file:
            np.savez_compressed(file, db=self.db, voiced=self.voiced, hop=np.array(self.hop))

    @classmethod
    def load(cls, path: Path) -> Activity:
        with np.load(path) as data:
            return cls(data["db"], data["voiced"].astype(bool), float(data["hop"]))


def _hysteresis(db: np.ndarray, high: float, low: float) -> np.ndarray:
    voiced = np.zeros(len(db), dtype=bool)
    on = False
    for index, value in enumerate(db):
        on = value >= low if on else value >= high
        voiced[index] = on
    return voiced


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) frame ranges where mask is True."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _close_runs(mask: np.ndarray, *, value: bool, shorter_than: int) -> np.ndarray:
    """Flip interior runs of `value` shorter than the given frame count."""
    result = mask.copy()
    target = mask if value else ~mask
    for start, end in _runs(target):
        interior = value or (start > 0 and end < len(mask))  # leading/trailing silence is real silence
        if end - start < shorter_than and interior:
            result[start:end] = not value
    return result

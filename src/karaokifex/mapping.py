"""How lrclib's timeline (the record) maps onto the video's (numpy only, no I/O).

Music videos are shifted (longer intros), sometimes sped up by a few percent,
and sometimes have a verse cut or a section added. The map is a line
`t_video = scale · t_lrc + offset`, fitted robustly to anchors (lyric lines whose
first word whisperx heard), plus piecewise-constant corrections for sections
that jumped. Before any anchors exist, the global offset comes from
cross-correlating lrclib's line starts with the vocal onsets.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

MAX_LAG = 90.0  # seconds the video may be shifted against the record
ONSET_SIGMA = 0.1  # seconds; smoothing of the onset envelope before correlating
SCALE_RANGE = (0.95, 1.05)  # plausible speed changes
INLIER = 0.4  # seconds an anchor may deviate from the fitted line
MIN_INLIERS = 5
MIN_PAIR_SPAN = 5.0  # seconds; anchor pairs closer than this give unreliable slopes
MIN_OFFSET_SAMPLES = 3  # anchors needed before trusting a plain median offset
JUMP = 1.0  # seconds; a section whose residuals move by more than this has its own shift...
MIN_SECTION = 3  # ...if at least this many consecutive anchors agree
RUNNING_MEDIAN = 5

Anchor = tuple[float, float]  # (lrclib time, video time)


@dataclass(frozen=True)
class TimeMap:
    scale: float = 1.0
    offset: float = 0.0
    corrections: tuple[tuple[float, float], ...] = ()  # (from lrclib time, extra shift), sorted

    def map(self, t: float) -> float:
        shift = 0.0
        for start, extra in self.corrections:
            if t >= start:
                shift = extra
        return self.scale * t + self.offset + shift

    def support(self, anchors: Sequence[Anchor], tolerance: float = INLIER) -> int:
        """How many anchors this map explains."""
        return sum(abs(self.map(lrc) - video) < tolerance for lrc, video in anchors)

    def describe(self) -> str:
        text = f"video = {self.scale:.4f} × lrclib {self.offset:+.2f} s"
        if self.corrections:
            text += ", sections: " + ", ".join(f"from {start:.1f} s {extra:+.1f} s" for start, extra in self.corrections)
        return text

    def to_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "offset": self.offset, "corrections": [list(c) for c in self.corrections]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeMap:
        return cls(data["scale"], data["offset"], tuple((a, b) for a, b in data["corrections"]))


def xcorr_offset(line_starts: Sequence[float | None], envelope: np.ndarray, hop: float, *,
                 max_lag: float = MAX_LAG) -> float | None:
    """The shift that best lines lrclib's line starts up with vocal onsets (None if undecidable)."""
    starts = np.array([s for s in line_starts if s is not None])
    if len(starts) < MIN_OFFSET_SAMPLES or not len(envelope) or not envelope.any():
        return None
    sigma = max(1, round(ONSET_SIGMA / hop))
    kernel = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    smooth = np.convolve(envelope, kernel, mode="same")
    lags = np.arange(-round(max_lag / hop), round(max_lag / hop) + 1)
    frames = np.round(starts / hop).astype(int)[:, None] + lags[None, :]
    valid = (frames >= 0) & (frames < len(smooth))
    scores = np.where(valid, smooth[np.clip(frames, 0, len(smooth) - 1)], 0.0).sum(axis=0)
    scores -= 1e-9 * np.abs(lags)  # ties: prefer the smaller shift
    return float(lags[np.argmax(scores)] * hop)


def fit_time_map(anchors: Sequence[Anchor], prior_offset: float | None = None) -> TimeMap:
    """Robust line fit through the anchors, plus corrections for sections that jumped."""
    if len(anchors) < MIN_INLIERS:
        return TimeMap(offset=_fallback_offset(anchors, prior_offset))
    lrc = np.array([a for a, _ in anchors])
    video = np.array([v for _, v in anchors])
    scale, offset = _ransac(lrc, video)
    if scale is None:
        return TimeMap(offset=_fallback_offset(anchors, prior_offset))
    return TimeMap(scale, offset, _section_corrections(lrc, video - (scale * lrc + offset)))


def _fallback_offset(anchors: Sequence[Anchor], prior_offset: float | None) -> float:
    if len(anchors) >= MIN_OFFSET_SAMPLES:
        return statistics.median(v - a for a, v in anchors)
    return prior_offset or 0.0


def _ransac(lrc: np.ndarray, video: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    """Deterministic RANSAC: every anchor pair (and every anchor at scale 1) is a hypothesis."""
    i, j = np.triu_indices(len(lrc), k=1)
    span = lrc[j] - lrc[i]
    keep = span > MIN_PAIR_SPAN
    i, j, span = i[keep], j[keep], span[keep]
    slopes = (video[j] - video[i]) / span
    plausible = (slopes >= SCALE_RANGE[0]) & (slopes <= SCALE_RANGE[1])
    scales = np.concatenate([slopes[plausible], np.ones(len(lrc))])
    offsets = np.concatenate([video[i[plausible]] - slopes[plausible] * lrc[i[plausible]], video - lrc])
    residuals = np.abs(video[None, :] - (scales[:, None] * lrc[None, :] + offsets[:, None]))
    counts = (residuals < INLIER).sum(axis=1)
    best = int(np.argmax(counts - 1e-6 * np.abs(scales - 1)))  # ties: prefer no speed change
    if counts[best] < MIN_INLIERS:
        return None, None
    inliers = residuals[best] < INLIER
    if np.ptp(lrc[inliers]) > MIN_PAIR_SPAN:
        scale, offset = np.polyfit(lrc[inliers], video[inliers], 1)
        if SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
            return float(scale), float(offset)
    return float(scales[best]), float(np.median(video[inliers] - scales[best] * lrc[inliers]))


def _section_corrections(lrc: np.ndarray, residuals: np.ndarray) -> tuple[tuple[float, float], ...]:
    order = np.argsort(lrc)
    lrc, residuals = lrc[order], residuals[order]
    half = RUNNING_MEDIAN // 2
    medians = np.array([np.median(residuals[max(0, k - half): k + half + 1]) for k in range(len(residuals))])
    corrections: list[tuple[float, float]] = []
    current, k = 0.0, 0
    while k < len(medians):
        if abs(medians[k] - current) > JUMP:
            end = k
            while end < len(medians) and abs(medians[end] - medians[k]) <= JUMP:
                end += 1
            if end - k >= MIN_SECTION:
                value = float(np.median(residuals[k:end]))
                start = k
                while start > 0 and abs(residuals[start - 1] - value) <= INLIER:
                    start -= 1  # the running median lags behind the jump by up to `half` anchors
                if corrections and start <= 0:
                    start = k
                corrections.append((float(lrc[start]), value))
                current, k = value, end
                continue
        k += 1
    return tuple(corrections)

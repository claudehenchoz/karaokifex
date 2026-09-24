"""The dominant colours of a video: k-means over the pixels of frames sampled across it (pure numpy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PALETTE_SIZE = 5
MAX_PIXELS = 50_000  # k-means on more pixels than this barely changes the result
BAR_LEVEL = 24  # rows and columns no brighter than this in every frame are black bars
_ITERATIONS = 30


@dataclass(frozen=True)
class Swatch:
    rgb: tuple[int, int, int]
    weight: float  # share of the sampled pixels closest to this colour

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)

    def to_dict(self) -> dict[str, Any]:
        return {"hex": self.hex, "rgb": list(self.rgb), "weight": round(self.weight, 4)}


def without_bars(frames: np.ndarray) -> np.ndarray:
    """Crop letterbox and pillarbox bars: rows and columns that are black in every frame (frames, height, width, 3).

    Film footage in a 16:9 video is mostly letterboxed, and its bars would otherwise be the most common colour.
    """
    dark = np.asarray(frames).max(axis=3) <= BAR_LEVEL
    rows, columns = ~dark.all(axis=(0, 2)), ~dark.all(axis=(0, 1))
    if not rows.any() or not columns.any():  # black throughout: nothing to crop to
        return frames
    return frames[:, rows][:, :, columns]


def dominant_colors(frames: np.ndarray, count: int = PALETTE_SIZE) -> list[Swatch]:
    """Up to `count` colours that best summarise the RGB `frames` (any shape ending in 3), most common first.

    Deterministic: pixels are thinned with a fixed stride and k-means++ starts from a fixed seed.
    Fewer colours come back when the frames hold fewer distinct ones.
    """
    pixels = np.asarray(frames, dtype=np.float64).reshape(-1, 3)
    if not len(pixels):
        return []
    pixels = pixels[:: -(-len(pixels) // MAX_PIXELS)]  # ceil division: at most MAX_PIXELS remain
    centers = _initial_centers(pixels, count, np.random.default_rng(0))
    for _ in range(_ITERATIONS):
        labels = _nearest(pixels, centers)
        moved = np.array([pixels[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                          for k in range(len(centers))])
        if np.allclose(moved, centers, atol=0.5):
            break
        centers = moved
    weights = np.bincount(_nearest(pixels, centers), minlength=len(centers)) / len(pixels)
    order = np.argsort(-weights, kind="stable")
    return [Swatch(tuple(int(v) for v in np.clip(np.rint(centers[k]), 0, 255)), float(weights[k]))  # type: ignore[arg-type]
            for k in order if weights[k] > 0]


def _nearest(pixels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def _initial_centers(pixels: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: each new centre is drawn with probability proportional to its squared distance from the others."""
    centers = [pixels[rng.integers(len(pixels))]]
    distances = ((pixels - centers[0]) ** 2).sum(axis=1)
    while len(centers) < count and distances.sum() > 0:
        centers.append(pixels[rng.choice(len(pixels), p=distances / distances.sum())])
        distances = np.minimum(distances, ((pixels - centers[-1]) ** 2).sum(axis=1))
    return np.array(centers)

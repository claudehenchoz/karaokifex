"""Stem separation with audio-separator, and the karaoke made from its lead vocals.

audio-separator scales the mix to a peak of 0.9 before separating, and each stem it writes again,
so its stems add up to the song only at gains that differ from song to song. A karaoke model's
"instrumental" came out a few dB quieter than the song across the board, the bass and the air
included, where there is next to no voice: a thinner copy of the song. `combine` fits each model's
stems back to the song's own level, averages the models' lead vocals (an ensemble, when there are
several), and makes the backing the song minus that lead: everything but the lead voice, at the
level and with the fullness the song has.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from karaokifex.gpu import free_gpu_memory
from karaokifex.workspace import partial_path

log = logging.getLogger(__name__)

# audio-separator's defaults for Roformer/MDXC models, minus the overlap we set ourselves.
_MDXC_PARAMS = {"segment_size": 256, "override_model_segment_size": False, "batch_size": None, "pitch_shift": 0}


def separate(audio: Path, *, model: str, stems: dict[str, Path], model_dir: Path, overlap: int, fp16: bool = True,
             verbose: bool = False, on_stage: Callable[[str], None] = lambda _: None) -> None:
    """Run `model` on `audio` and write the requested stems, e.g. {"vocals": Path(".../vocals.wav")}.

    Stem names are the model's own ("vocals", "instrumental"), matched case-insensitively.
    `overlap` is how many overlapping windows cover each sample: runtime grows linearly with it.
    Roformer configs typically ask for 8, but the difference to 2 is barely audible.
    `fp16` runs Roformers in half precision on CUDA (audio-separator ignores it where unsupported).
    """
    from audio_separator.separator import Separator  # heavy import (torch); only when actually separating

    output_dir = next(iter(stems.values())).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed on success, so an aborted run never looks finished.
    temporary = {stem: path.with_name(f"{path.stem}_partial{path.suffix}") for stem, path in stems.items()}

    separator = Separator(
        log_level=logging.INFO if verbose else logging.WARNING,
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format="WAV",
        normalization_threshold=1.0,  # as little rescaling as it allows; combine() undoes the rest
        use_native_fp16=fp16,
        mdxc_params={**_MDXC_PARAMS, "overlap": overlap},
    )
    on_stage("loading model (downloaded on first use)…")
    separator.load_model(model_filename=model)
    on_stage("separating stems…")
    produced = separator.separate(str(audio), custom_output_names={s: p.stem for s, p in temporary.items()})
    log.debug("%s produced %s", model, produced)

    for stem, path in stems.items():
        if not temporary[stem].exists():
            raise RuntimeError(f"{model} produced no {stem!r} stem (got: {', '.join(map(str, produced))})")
        temporary[stem].replace(path)
    del separator
    free_gpu_memory()


def fit_gains(mix: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """The gains (a, b) with which `a * first + b * second` comes closest to `mix` (least squares).

    Sums are taken a chunk at a time in float64: a song is some twenty million samples.
    """
    s = np.zeros((2, 2)), np.zeros(2)
    gram, target = s
    step = 1 << 20
    x, y, m = first.reshape(-1), second.reshape(-1), mix.reshape(-1)
    for i in range(0, len(m), step):
        a, b, c = (v[i:i + step].astype(np.float64) for v in (x, y, m))
        gram += [[a @ a, a @ b], [a @ b, b @ b]]
        target += [a @ c, b @ c]
    if abs(np.linalg.det(gram)) < 1e-9 * max(gram[0, 0] * gram[1, 1], 1e-30):
        return 1.0, 1.0  # one of the stems is silent: nothing to fit
    ga, gb = np.linalg.solve(gram, target)
    return float(ga), float(gb)


def combine(mix: Path, stems: Sequence[tuple[Path, Path]], *, backing: Path, lead: Path) -> list[float]:
    """Write the lead (the models' leads averaged, each at the song's own level) and the backing (the
    song minus that lead) from each model's (lead, backing) stems. Returns each lead's gain."""
    import soundfile as sf

    song, rate = sf.read(str(mix), dtype="float32", always_2d=True)
    leads, gains = [], []
    for lead_path, backing_path in stems:
        voice, _ = sf.read(str(lead_path), dtype="float32", always_2d=True)
        rest, _ = sf.read(str(backing_path), dtype="float32", always_2d=True)
        n = min(len(song), len(voice), len(rest))
        _, gain = fit_gains(song[:n], rest[:n], voice[:n])
        leads.append(voice[:n] * gain)
        gains.append(gain)
    n = min(len(v) for v in leads)
    voice = np.mean([v[:n] for v in leads], axis=0)
    for path, samples in ((lead, voice), (backing, song[:n] - voice)):
        partial = partial_path(path)
        sf.write(str(partial), samples, rate, subtype="FLOAT")
        partial.replace(path)
    return gains

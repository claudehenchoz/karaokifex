"""Stem separation with audio-separator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from karaokifex.gpu import free_gpu_memory

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

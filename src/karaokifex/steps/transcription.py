"""Word-level transcription of the vocals stem with whisperx."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from karaokifex.console import adopt_loggers
from karaokifex.gpu import free_gpu_memory
from karaokifex.models import TimedWord

log = logging.getLogger(__name__)

# Singing is more continuous than speech; lower voice-activity thresholds keep more of it.
VAD_OPTIONS = {"vad_onset": 0.35, "vad_offset": 0.25}
BATCH_SIZE = 8


def load_model(name: str, device: str, language: str | None = None) -> Any:
    import whisperx  # heavy import (torch, pyannote); only when actually transcribing

    try:
        import lightning  # noqa: F401  # pulled in by pyannote's VAD; attaches a console handler on import
    except ImportError:
        pass
    adopt_loggers("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning", "lightning_fabric")

    compute_type = "float16" if device == "cuda" else "int8"
    return whisperx.load_model(name, device, compute_type=compute_type, language=language, vad_options=VAD_OPTIONS)


def transcribe(model: Any, vocals: Path, device: str, *, language: str | None = None,
               on_stage: Callable[[str], None] = lambda _: None) -> tuple[list[TimedWord], str]:
    """Return every recognised word with start/end times, plus the (detected) language."""
    import whisperx

    audio = whisperx.load_audio(str(vocals))
    on_stage("transcribing…")
    detected = language is None
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)
    language = result["language"]
    if detected:
        log.info("auto-detected language: %s (use --language to override)", language)
    on_stage(f"aligning words ({language})…")
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    except ValueError as error:
        # Detection only listens to the first 30 s, and singing easily fools it (Björk → Welsh).
        hint = "was auto-detected, which is easily fooled by singing" if detected else "was requested"
        raise ValueError(f"whisperx has no word-alignment model for language {language!r}, which {hint}. "
                         f"Re-run with --language set to the song's language, e.g. --language en") from error
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    del align_model
    free_gpu_memory()

    words = [
        TimedWord(str(word["word"]).strip(), float(word["start"]), float(word["end"]))
        for word in aligned["word_segments"]
        if _is_time(word.get("start")) and _is_time(word.get("end"))
    ]
    return words, language


def _is_time(value: Any) -> bool:
    return value is not None and not math.isnan(float(value))


def save_transcript(words: list[TimedWord], language: str, path: Path) -> None:
    data = {"language": language, "words": [asdict(word) for word in words]}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load_transcript(path: Path) -> tuple[list[TimedWord], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TimedWord(**word) for word in data["words"]], data["language"]

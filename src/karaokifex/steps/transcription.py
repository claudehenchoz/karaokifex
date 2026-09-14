"""whisperx: word-level transcription of the vocals stem, and forced alignment of known lyrics."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Hashable, Sequence

from karaokifex.console import adopt_loggers
from karaokifex.gpu import free_gpu_memory
from karaokifex.models import TimedWord
from karaokifex.workspace import partial_path

log = logging.getLogger(__name__)

# Singing is more continuous than speech; lower voice-activity thresholds keep more of it.
VAD_OPTIONS = {"vad_onset": 0.35, "vad_offset": 0.25}
# Numbers come out spelled ("two", not "2"), like they are in lyrics.
ASR_OPTIONS = {"suppress_numerals": True}
BATCH_SIZE = 8

Window = tuple[float, float]


def load_model(name: str, device: str, language: str | None = None) -> Any:
    import whisperx  # heavy import (torch, pyannote); only when actually transcribing

    try:
        import lightning  # noqa: F401  # pulled in by pyannote's VAD; attaches a console handler on import
    except ImportError:
        pass
    adopt_loggers("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning", "lightning_fabric")

    compute_type = "float16" if device == "cuda" else "int8"
    return whisperx.load_model(name, device, compute_type=compute_type, language=language, vad_options=VAD_OPTIONS,
                               asr_options=ASR_OPTIONS)


def transcribe(model: Any, vocals: Path, device: str, *, language: str | None = None, prompt: str | None = None,
               on_stage: Callable[[str], None] = lambda _: None) -> tuple[list[TimedWord], str]:
    """Return every recognised word with start/end times and alignment score, plus the (detected) language.

    `prompt` (the expected lyrics) biases whisper's decoder towards those words.
    """
    import whisperx

    if hasattr(model, "options"):
        model.options = replace(model.options, initial_prompt=prompt or None)
    audio = whisperx.load_audio(str(vocals))
    on_stage("transcribing…")
    detected = language is None
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)
    language = result["language"]
    if detected:
        log.info("auto-detected language: %s (use --language to override)", language)
    on_stage(f"aligning words ({language})…")
    align_model, metadata = _load_align_model(whisperx, language, device, detected)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    del align_model
    free_gpu_memory()

    words = [
        TimedWord(str(word["word"]).strip(), float(word["start"]), float(word["end"]), _score(word))
        for word in aligned["word_segments"]
        if _is_time(word.get("start")) and _is_time(word.get("end"))
    ]
    return words, language


def force_align(requests: Sequence[tuple[Hashable, list[str], Window]], vocals: Path, language: str, device: str, *,
                on_stage: Callable[[str], None] = lambda _: None) -> dict[Hashable, list[TimedWord] | None]:
    """Place each line's known words inside its window (wav2vec2 CTC alignment).

    `requests` are (key, normalised words, window). Returns per key the timed words,
    or None where the aligner couldn't place every word. Empty when whisperx has no
    alignment model for the language. The model is loaded once for all requests.
    """
    import whisperx

    if not requests:
        return {}
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    except ValueError:
        log.warning("no forced-alignment model for language %r — timing from the transcription only", language)
        return {}
    audio = whisperx.load_audio(str(vocals))
    results: dict[int, list[TimedWord] | None] = {}
    try:
        for number, (index, words, (start, end)) in enumerate(requests, 1):
            on_stage(f"forced alignment: line {number}/{len(requests)}")
            segment = {"text": " ".join(words), "start": start, "end": end}
            aligned = whisperx.align([segment], align_model, metadata, audio, device, return_char_alignments=False)
            placed = aligned["word_segments"]
            if len(placed) != len(words) or not all(_is_time(w.get("start")) and _is_time(w.get("end")) for w in placed):
                results[index] = None
                continue
            results[index] = [TimedWord(word, float(w["start"]), float(w["end"]), _score(w))
                              for word, w in zip(words, placed)]
    finally:
        del align_model
        free_gpu_memory()
    return results


def _load_align_model(whisperx: Any, language: str, device: str, detected: bool) -> tuple[Any, Any]:
    try:
        return whisperx.load_align_model(language_code=language, device=device)
    except ValueError as error:
        # Detection only listens to the first 30 s, and singing easily fools it (Björk → Welsh).
        hint = "was auto-detected, which is easily fooled by singing" if detected else "was requested"
        raise ValueError(f"whisperx has no word-alignment model for language {language!r}, which {hint}. "
                         f"Re-run with --language set to the song's language, e.g. --language en") from error


def _is_time(value: Any) -> bool:
    return value is not None and not math.isnan(float(value))


def _score(word: dict[str, Any]) -> float | None:
    score = word.get("score")
    return float(score) if _is_time(score) else None


def save_transcript(words: list[TimedWord], language: str, path: Path) -> None:
    data = {"language": language, "words": [asdict(word) for word in words]}
    partial = partial_path(path)
    partial.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    partial.replace(path)


def load_transcript(path: Path) -> tuple[list[TimedWord], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TimedWord(**word) for word in data["words"]], data["language"]

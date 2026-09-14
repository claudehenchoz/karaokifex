import sys
import types
from dataclasses import dataclass

import pytest

from karaokifex.models import TimedWord
from karaokifex.steps import transcription


@dataclass
class Options:
    initial_prompt: str | None = None


class FakeModel:
    def __init__(self, language):
        self.language = language
        self.options = Options()

    def transcribe(self, audio, batch_size, language=None):
        return {"segments": [], "language": language or self.language}


def fake_align(segments, model, metadata, audio, device, return_char_alignments=False):
    """Spreads each segment's words evenly over its window; a word "drop" makes the aligner lose it."""
    if not segments:
        return {"word_segments": [{"word": "hi", "start": 1.0, "end": 1.5, "score": 0.9}]}
    placed = []
    for segment in segments:
        words = segment["text"].split()
        step = (segment["end"] - segment["start"]) / len(words)
        placed += [{"word": word, "start": segment["start"] + i * step, "end": segment["start"] + (i + 1) * step,
                    "score": 0.8} for i, word in enumerate(words) if word != "drop"]
    return {"word_segments": placed}


@pytest.fixture
def fake_whisperx(monkeypatch):
    """A stand-in whisperx that only has English alignment models."""
    module = types.ModuleType("whisperx")
    module.load_audio = lambda path: [0.0]

    def load_align_model(language_code, device):
        if language_code != "en":
            raise ValueError(f"No default align-model for language: {language_code}")
        return object(), {}

    module.load_align_model = load_align_model
    module.align = fake_align
    monkeypatch.setitem(sys.modules, "whisperx", module)
    monkeypatch.setattr(transcription, "free_gpu_memory", lambda: None)


def test_misdetected_language_explains_how_to_override(fake_whisperx, tmp_path):
    with pytest.raises(ValueError, match="auto-detected.*--language"):
        transcription.transcribe(FakeModel("cy"), tmp_path / "vocals.wav", "cpu")


def test_forced_language_is_used_and_scores_are_kept(fake_whisperx, tmp_path):
    words, language = transcription.transcribe(FakeModel("cy"), tmp_path / "vocals.wav", "cpu", language="en")
    assert language == "en"
    assert words == [TimedWord("hi", 1.0, 1.5, 0.9)]


def test_prompt_goes_to_the_decoder_options(fake_whisperx, tmp_path):
    model = FakeModel("en")
    transcription.transcribe(model, tmp_path / "vocals.wav", "cpu", prompt="one two three")
    assert model.options.initial_prompt == "one two three"
    transcription.transcribe(model, tmp_path / "vocals.wav", "cpu")
    assert model.options.initial_prompt is None


def test_force_align_places_each_line_in_its_window(fake_whisperx, tmp_path):
    requests = [(0, ["one", "two"], (1.0, 3.0)), (2, ["three", "drop"], (5.0, 6.0))]
    result = transcription.force_align(requests, tmp_path / "vocals.wav", "en", "cpu")
    assert [(w.text, w.start, w.end, w.score) for w in result[0]] == [("one", 1.0, 2.0, 0.8), ("two", 2.0, 3.0, 0.8)]
    assert result[2] is None


def test_force_align_without_a_model_for_the_language(fake_whisperx, tmp_path):
    assert transcription.force_align([(0, ["un"], (1.0, 2.0))], tmp_path / "v.wav", "cy", "cpu") == {}


def test_transcript_roundtrip_reads_old_files(tmp_path):
    path = tmp_path / "transcript.json"
    transcription.save_transcript([TimedWord("hi", 1.0, 1.5, 0.7)], "en", path)
    assert transcription.load_transcript(path) == ([TimedWord("hi", 1.0, 1.5, 0.7)], "en")
    path.write_text('{"language": "en", "words": [{"text": "hi", "start": 1.0, "end": 1.5}]}', encoding="utf-8")
    assert transcription.load_transcript(path)[0] == [TimedWord("hi", 1.0, 1.5)]

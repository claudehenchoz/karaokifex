import sys
import types

import pytest

from karaokifex.steps import transcription


class FakeModel:
    def __init__(self, language):
        self.language = language

    def transcribe(self, audio, batch_size, language=None):
        return {"segments": [], "language": language or self.language}


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
    module.align = lambda *args, **kwargs: {"word_segments": [{"word": "hi", "start": 1.0, "end": 1.5}]}
    monkeypatch.setitem(sys.modules, "whisperx", module)
    monkeypatch.setattr(transcription, "free_gpu_memory", lambda: None)


def test_misdetected_language_explains_how_to_override(fake_whisperx, tmp_path):
    with pytest.raises(ValueError, match="auto-detected.*--language"):
        transcription.transcribe(FakeModel("cy"), tmp_path / "vocals.wav", "cpu")


def test_forced_language_is_used(fake_whisperx, tmp_path):
    words, language = transcription.transcribe(FakeModel("cy"), tmp_path / "vocals.wav", "cpu", language="en")
    assert language == "en"
    assert [w.text for w in words] == ["hi"]

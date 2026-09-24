"""Run configuration, built from the command line."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_KARAOKE_MODEL = "mel_band_roformer_karaoke_gabox.ckpt"
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_SEPARATION_OVERLAP = 2
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "karaokifex" / "models"
DEFAULT_RESOLUTION = 1080


@dataclass(frozen=True)
class Config:
    url: str
    artist: str | None = None
    song: str | None = None
    language: str | None = None
    karaoke_model: str = DEFAULT_KARAOKE_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    separation_overlap: int = DEFAULT_SEPARATION_OVERLAP
    fp16: bool = True
    device: str = "auto"
    ffmpeg: str | None = None  # None: pick automatically from PATH
    gpu_jobs: int = 1
    burn_lyrics: bool = True  # False: the video keeps its picture; the lyrics files are written either way
    darken: float = 0.08
    resolution: int = DEFAULT_RESOLUTION
    lead_volume: float = 0.0
    output_dir: Path = Path(".")
    model_dir: Path = DEFAULT_MODEL_DIR
    mix_vote: bool = False  # also transcribe the full mix and let both transcriptions vote
    debug_ass: bool = False  # render with words coloured by timing source
    keep_temp: bool = False  # keep every temporary file (they are deleted after a successful run)
    keep_source: bool = False  # keep the original download when the temporary files are deleted
    force: bool = False
    verbose: bool = False

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch  # heavy import, only needed once we actually run

        return "cuda" if torch.cuda.is_available() else "cpu"

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
    musicbrainz: bool = True  # canonical artist and song names from MusicBrainz
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
    upscale: bool | None = None  # scale a source below `resolution` up to it; None: only with burned-in lyrics
    lead_volume: float = 0.0
    browser_friendly: bool = False  # MP4 with H.264 + AAC; the video is copied when it already is H.264
    output_dir: Path = Path(".")
    model_dir: Path = DEFAULT_MODEL_DIR
    mix_vote: bool = False  # also transcribe the full mix and let both transcriptions vote
    palette: bool = False  # find the video's dominant colours and write them to metadata.json
    describe: bool = False  # the song's album, year, genres, writers and language, from MusicBrainz, in song.json
    quality: bool = False  # the source's and the renders' resolution, frame rate, codecs and bitrates, in quality.json
    debug_ass: bool = False  # render with words coloured by timing source
    keep_temp: bool = False  # keep every temporary file (they are deleted after a successful run)
    keep_source: bool = False  # also render the original video with its own sound, like the karaoke video
    force: bool = False
    verbose: bool = False

    @property
    def target_height(self) -> int | None:
        """The height a smaller source is scaled up to, or None to keep every source's own size.

        Upscaling is for burned-in lyrics, which need the lines to render them sharply. Without
        them the picture is left as it is -- copied where it can be -- and the player scales it.
        """
        upscale = self.burn_lyrics if self.upscale is None else self.upscale
        return self.resolution if upscale else None

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch  # heavy import, only needed once we actually run

        return "cuda" if torch.cuda.is_available() else "cpu"

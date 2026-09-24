"""The per-song working folder: every intermediate file and final artifact lives here."""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def sanitize_name(name: str, max_length: int = 100) -> str:
    """Make `name` safe to use as a file or folder name on every OS (Windows being the strictest)."""
    cleaned = re.sub(r"\s+", " ", _ILLEGAL_CHARS.sub("", name)).strip()
    cleaned = cleaned[:max_length].rstrip(" .")
    if not cleaned:
        return "untitled"
    if cleaned.upper() in _RESERVED_NAMES:
        return f"_{cleaned}"
    return cleaned


def partial_path(path: Path) -> Path:
    """Where to write `path` while it is still being produced ("audio.wav" -> "audio.partial.wav").

    Steps write here and rename on success, so a crashed run never leaves a
    half-written file that a later run would mistake for a finished one.
    """
    return path.with_name(f"{path.stem}.partial{path.suffix}")


@dataclass(frozen=True)
class Workspace:
    root: Path
    title: str

    @classmethod
    def create(cls, parent: Path, title: str) -> Workspace:
        workspace = cls(parent / sanitize_name(title), title)
        workspace.stems_dir.mkdir(parents=True, exist_ok=True)
        return workspace

    # --- download & extraction -------------------------------------------------
    @property
    def info_json(self) -> Path:
        return self.root / "info.json"

    @property
    def source(self) -> Path:
        return self.root / "source.mkv"

    @property
    def video(self) -> Path:
        return self.root / "video.mkv"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    # --- stem separation ---------------------------------------------------------
    @property
    def stems_dir(self) -> Path:
        return self.root / "stems"

    @property
    def karaoke_backing(self) -> Path:
        """Instrumental plus backing vocals — the karaoke audio track."""
        return self.stems_dir / "karaoke_backing.wav"

    @property
    def karaoke_lead(self) -> Path:
        """Lead vocals only — what whisperx transcribes."""
        return self.stems_dir / "karaoke_lead.wav"

    @property
    def lead_activity(self) -> Path:
        """When the lead vocals are audible (see activity.py)."""
        return self.stems_dir / "lead_activity.npz"

    # --- lyrics & subtitles ------------------------------------------------------
    @property
    def lyrics_json(self) -> Path:
        return self.root / "lyrics.json"

    @property
    def transcript_json(self) -> Path:
        return self.root / "transcript.json"

    @property
    def transcript_mix_json(self) -> Path:
        """Transcription of the full mix (--mix-vote)."""
        return self.root / "transcript_mix.json"

    @property
    def forced_json(self) -> Path:
        """Time map, line windows and forced-alignment results per lyrics candidate."""
        return self.root / "forced.json"

    @property
    def timings_json(self) -> Path:
        """Every displayed word with its time, score and source (for karaokifex-eval)."""
        return self.root / "timings.json"

    @property
    def subtitles(self) -> Path:
        return self.root / "lyrics.ass"

    @property
    def debug_subtitles(self) -> Path:
        """Like lyrics.ass, but each word coloured by what timed it."""
        return self.root / "lyrics.debug.ass"

    @property
    def metadata_json(self) -> Path:
        """What karaokifex finds out about the video itself (--palette: its dominant colours)."""
        return self.root / "metadata.json"

    @property
    def final_video(self) -> Path:
        return self.root / f"{sanitize_name(self.title)} (Karaoke).mkv"

    @property
    def debug_video(self) -> Path:
        return self.root / f"{sanitize_name(self.title)} (Karaoke debug).mkv"

    @property
    def plain_video(self) -> Path:
        """The karaoke audio with the untouched picture: no lyrics burned in (--no-burn-lyrics)."""
        return self.root / f"{sanitize_name(self.title)} (Karaoke, no lyrics).mkv"

    @property
    def original_video(self) -> Path:
        """The video with its own sound, the song as released, made like the karaoke video (--keep-source)."""
        return self.root / f"{sanitize_name(self.title)} (Original).mkv"

    # --- cleanup -----------------------------------------------------------------
    def artifacts(self) -> frozenset[Path]:
        """Files worth keeping after a successful run — including videos rendered by earlier runs.

        Besides the videos and the karaoke audio, that is the data describing them: the lyrics,
        their word timings and the video's metadata. The download itself is temporary; the
        original survives as `original_video` (--keep-source).
        """
        videos = (self.final_video, self.debug_video, self.plain_video, self.original_video)
        earlier_renders = [path for video in videos for path in self.root.glob(f"{glob.escape(video.stem)}.*")]
        keep = {self.final_video, self.subtitles, self.karaoke_backing, self.lyrics_json, self.timings_json,
                self.info_json, self.metadata_json}
        return frozenset(keep | {p for p in earlier_renders if ".partial." not in p.name})

    def temp_files(self) -> list[Path]:
        """Every file in the workspace that is not an artifact."""
        keep = {path.resolve() for path in self.artifacts()}
        return sorted(p for p in self.root.rglob("*") if p.is_file() and p.resolve() not in keep)

    def cleanup(self) -> list[Path]:
        """Delete all temp files (and folders left empty); returns what was removed."""
        removed = self.temp_files()
        for path in removed:
            path.unlink(missing_ok=True)
        directories = sorted((d for d in self.root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True)
        for directory in directories:
            if not any(directory.iterdir()):
                directory.rmdir()
        return removed

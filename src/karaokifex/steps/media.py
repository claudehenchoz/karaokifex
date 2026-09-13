"""ffmpeg: splitting the download into audio and video, and rendering the karaoke video."""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import ffmpeg

from karaokifex.workspace import partial_path

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float | None], None]


@dataclass(frozen=True)
class Encoder:
    codec: str
    options: dict[str, Any] = field(hash=False)
    gpu: bool

    @property
    def label(self) -> str:
        return f"{self.codec} ({'GPU' if self.gpu else 'CPU'})"


# p5 is one of NVENC's modern presets (ffmpeg >= 4.3); current NVIDIA drivers reject the legacy ones.
NVENC = Encoder("h264_nvenc", {"preset": "p5", "tune": "hq", "rc": "vbr", "cq": 19, "b:v": 0}, gpu=True)
X264 = Encoder("libx264", {"preset": "medium", "crf": 18}, gpu=False)


@dataclass(frozen=True)
class FfmpegBinary:
    """The ffmpeg executable to run, and the video encoder it can drive."""

    path: str = "ffmpeg"
    encoder: Encoder = X264


class FfmpegError(RuntimeError):
    pass


@functools.cache
def find_ffmpeg(explicit: str | None = None) -> FfmpegBinary:
    """Pick the ffmpeg to use: the first one on PATH that burns in subtitles and encodes on the GPU.

    PATH often holds several builds (ImageMagick ships an old one), and old builds can't drive
    current NVIDIA drivers, so each candidate is test-driven with a tiny encode.
    """
    candidates = [explicit] if explicit else _ffmpegs_on_path()
    usable = [candidate for candidate in candidates if _has_libass(candidate)]
    if not usable:
        raise FfmpegError(f"no ffmpeg with libass (needed to burn in subtitles) among: {', '.join(candidates)}")
    for candidate in usable:
        if _can_encode(candidate, NVENC):
            return FfmpegBinary(candidate, NVENC)
    log.warning("no ffmpeg that can encode on the GPU (NVENC) found — rendering on the CPU will be slow")
    return FfmpegBinary(usable[0], X264)


def _ffmpegs_on_path() -> list[str]:
    found: dict[str, str] = {}
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory and (path := shutil.which("ffmpeg", path=directory)):
            found.setdefault(os.path.normcase(os.path.realpath(path)), path)
    return list(found.values()) or ["ffmpeg"]


def _has_libass(binary: str) -> bool:
    try:
        output = subprocess.run([binary, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == ["subtitles"] for line in output.splitlines())


def _can_encode(binary: str, encoder: Encoder) -> bool:
    args = [binary, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2",
            "-c:v", encoder.codec, *_as_args(encoder.options), "-f", "null", "-"]
    try:
        return subprocess.run(args, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _as_args(options: dict[str, Any]) -> list[str]:
    return [item for key, value in options.items() for item in (f"-{key}", str(value))]


def extract_audio(source: Path, target: Path, *, binary: str = "ffmpeg", duration: float | None = None,
                  on_progress: ProgressCallback | None = None) -> None:
    """Decode the audio to 44.1 kHz stereo WAV, the input format of the separation models."""
    partial = partial_path(target)
    stream = ffmpeg.input(str(source)).output(str(partial), vn=None, acodec="pcm_s16le", ar=44100, ac=2)
    run(stream, binary=binary, duration=duration, on_progress=on_progress)
    partial.replace(target)


def extract_video(source: Path, target: Path, *, binary: str = "ffmpeg", duration: float | None = None,
                  on_progress: ProgressCallback | None = None) -> None:
    """Copy the video stream without its audio (no re-encoding)."""
    partial = partial_path(target)
    stream = ffmpeg.input(str(source)).output(str(partial), an=None, vcodec="copy")
    run(stream, binary=binary, duration=duration, on_progress=on_progress)
    partial.replace(target)


def render(video: Path, audio: Path, subtitles: Path, target: Path, *, tool: FfmpegBinary, darken: float,
           duration: float | None = None, on_progress: ProgressCallback | None = None) -> str:
    """Darken the video, burn in the subtitles and pair it with the karaoke audio.

    Returns a label for the encoder that was used. Falls back to x264 if the GPU encoder fails.
    """
    encoders = [tool.encoder] if tool.encoder == X264 else [tool.encoder, X264]
    for index, encoder in enumerate(encoders):
        try:
            _render(tool.path, encoder, video, audio, subtitles, target,
                    darken=darken, duration=duration, on_progress=on_progress)
            return encoder.label
        except FfmpegError as error:
            if index == len(encoders) - 1:
                raise
            log.warning("%s failed, falling back to %s: %s", encoder.label, encoders[index + 1].label, error)
    raise AssertionError("unreachable")


def _render(binary: str, encoder: Encoder, video: Path, audio: Path, subtitles: Path, target: Path, *,
            darken: float, duration: float | None, on_progress: ProgressCallback | None) -> None:
    # The subtitles filter chokes on Windows drive letters ("C:"), so ffmpeg runs
    # inside the output folder and gets every path relative to it.
    folder = target.parent
    partial = partial_path(target)

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, folder)).as_posix()

    # With a GPU encoder, decode on the GPU as well (ffmpeg falls back to the CPU if it can't).
    # Frames come back to system memory for the eq and subtitles filters, which only exist on the CPU.
    decode = {"hwaccel": "cuda"} if encoder.gpu else {}
    picture = (
        ffmpeg.input(relative(video), **decode).video
        .filter("eq", brightness=-darken)
        .filter("subtitles", relative(subtitles))
    )
    sound = ffmpeg.input(relative(audio)).audio
    stream = ffmpeg.output(
        picture, sound, relative(partial),
        vcodec=encoder.codec, acodec="aac", audio_bitrate="320k", pix_fmt="yuv420p", movflags="+faststart",
        shortest=None, **encoder.options,
    )
    run(stream, binary=binary, cwd=folder, duration=duration, on_progress=on_progress)
    partial.replace(target)


def run(stream: Any, *, binary: str = "ffmpeg", cwd: Path | None = None, duration: float | None = None,
        on_progress: ProgressCallback | None = None) -> None:
    """Run an ffmpeg-python stream, reporting progress (0..1) from ffmpeg's `-progress` output."""
    args = [binary, "-hide_banner", "-nostats", "-loglevel", "error", "-progress", "pipe:1", "-y",
            *ffmpeg.compile(stream)[1:]]
    log.debug("$ %s", subprocess.list2cmdline(args))
    # Below-normal priority keeps the desktop responsive while ffmpeg crunches.
    priority = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    process = subprocess.Popen(args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                               creationflags=priority)
    errors: list[str] = []
    reader = threading.Thread(target=lambda: errors.extend(process.stderr), daemon=True)  # type: ignore[arg-type]
    reader.start()
    assert process.stdout is not None
    for line in process.stdout:
        key, _, value = line.strip().partition("=")
        # Both keys are in microseconds (ffmpeg < 4.3 only has the misnamed "out_time_ms").
        if key in ("out_time_us", "out_time_ms") and value.isdigit() and duration and on_progress:
            on_progress(min(int(value) / 1_000_000 / duration, 1.0))
    code = process.wait()
    reader.join()
    if code != 0:
        details = "".join(errors).strip()[-2000:] or "no error output (was it killed?)"
        raise FfmpegError(f"ffmpeg exited with code {code}: {details}")

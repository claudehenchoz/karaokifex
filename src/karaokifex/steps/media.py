"""ffmpeg: splitting the download into audio and video, and rendering the karaoke video."""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import ffmpeg
import numpy as np

from karaokifex.workspace import partial_path

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float | None], None]

# Output video formats, most efficient first, with their encoder on the GPU (NVENC) and on the CPU.
FORMATS = ("av1", "hevc", "h264")
GPU_ENCODERS = {"av1": "av1_nvenc", "hevc": "hevc_nvenc", "h264": "h264_nvenc"}
CPU_ENCODERS = {"av1": "libsvtav1", "hevc": "libx265", "h264": "libx264"}
# Roughly how many bits each format needs for the same picture quality, relative to AV1.
BITRATE_FACTOR = {"av1": 1.0, "vp9": 1.3, "hevc": 1.3, "h264": 1.8}
# Audio encoder and bitrate per source audio codec; anything else becomes AAC.
AUDIO_ENCODERS = {"opus": ("libopus", "160k"), "aac": ("aac", "256k")}
# H.264 that every browser's <video> plays (with AAC audio, in an MP4): 8-bit 4:2:0 in one of these profiles.
BROWSER_PROFILES = frozenset({"Constrained Baseline", "Baseline", "Main", "High"})

_SOFTWARE_ENCODERS = frozenset({*CPU_ENCODERS.values(), "libopus", "aac"})
# Below-normal priority keeps the desktop responsive while ffmpeg crunches.
_BELOW_NORMAL = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0


@dataclass(frozen=True)
class Encoder:
    format: str  # "av1" | "hevc" | "h264"
    gpu: bool

    @property
    def codec(self) -> str:
        return (GPU_ENCODERS if self.gpu else CPU_ENCODERS)[self.format]

    @property
    def label(self) -> str:
        return f"{self.codec} ({'GPU' if self.gpu else 'CPU'})"

    def options(self, bitrate: int | None) -> dict[str, Any]:
        """Encoder options: an average bitrate when known, constant quality otherwise."""
        if self.gpu:
            if bitrate:
                rate: dict[str, Any] = {"b:v": bitrate, "maxrate": 2 * bitrate, "bufsize": 4 * bitrate}
            else:
                rate = {"cq": 23, "b:v": 0}
            # p4 is NVENC's balanced preset. The modern p1-p7 presets need ffmpeg >= 4.3;
            # current drivers reject the legacy ones.
            return {"preset": "p4", "tune": "hq", "rc": "vbr", **rate}
        preset = {"av1": 8, "hevc": "fast", "h264": "medium"}[self.format]
        rate = {"b:v": bitrate} if bitrate else {"crf": {"av1": 32, "hevc": 22, "h264": 20}[self.format]}
        return {"preset": preset, **rate}


@dataclass(frozen=True)
class FfmpegBinary:
    """The ffmpeg executable to run, and what it can encode on this machine."""

    path: str = "ffmpeg"
    gpu_formats: frozenset[str] = frozenset()
    software: frozenset[str] = frozenset({"libx264", "aac"})

    @property
    def gpu(self) -> bool:
        return bool(self.gpu_formats)

    @property
    def ffprobe(self) -> str:
        sibling = Path(self.path).with_name(Path(self.path).name.lower().replace("ffmpeg", "ffprobe"))
        return str(sibling) if sibling.is_file() else "ffprobe"

    def describe(self) -> str:
        if not self.gpu:
            return "no GPU encoder — CPU only"
        return "GPU encoders: " + ", ".join(f for f in FORMATS if f in self.gpu_formats)


@dataclass(frozen=True)
class SourceInfo:
    video_format: str | None = None
    video_bitrate: int | None = None
    audio_format: str | None = None
    video_width: int | None = None
    video_height: int | None = None
    pixel_format: str | None = None
    profile: str | None = None

    @property
    def browser_ready(self) -> bool:
        """Whether every browser plays this video stream as it is (--browser-friendly copies it then)."""
        return self.video_format == "h264" and self.pixel_format == "yuv420p" and self.profile in BROWSER_PROFILES


class FfmpegError(RuntimeError):
    pass


# --- choosing ffmpeg and encoders ------------------------------------------------------


@functools.cache
def find_ffmpeg(explicit: str | None = None) -> FfmpegBinary:
    """Pick the ffmpeg to use: the first one on PATH that burns in subtitles and encodes on the GPU.

    PATH often holds several builds (ImageMagick ships an old one), and old builds can't drive
    current NVIDIA drivers, so each candidate is test-driven with tiny encodes.
    """
    candidates = [explicit] if explicit else _ffmpegs_on_path()
    usable = [candidate for candidate in candidates if _has_libass(candidate)]
    if not usable:
        raise FfmpegError(f"no ffmpeg with libass (needed to burn in subtitles) among: {', '.join(candidates)}")
    for candidate in usable:
        if gpu_formats := frozenset(f for f in FORMATS if _can_encode(candidate, GPU_ENCODERS[f])):
            return FfmpegBinary(candidate, gpu_formats, _software_encoders(candidate))
    log.warning("no ffmpeg that can encode on the GPU (NVENC) found — rendering on the CPU will be slow")
    return FfmpegBinary(usable[0], frozenset(), _software_encoders(usable[0]))


def choose_encoder(source_format: str | None, tool: FfmpegBinary, *, gpu: bool = True,
                   formats: tuple[str, ...] = FORMATS) -> Encoder:
    """The source's own format if possible, else the most efficient one; the GPU beats the CPU.

    `formats` limits the choice, most efficient first (--browser-friendly allows only H.264).
    """
    order = sorted(formats, key=lambda f: f != source_format)  # stable sort: source format first
    if gpu:
        for fmt in order:
            if fmt in tool.gpu_formats:
                return Encoder(fmt, gpu=True)
    for fmt in order:
        if CPU_ENCODERS[fmt] in tool.software:
            return Encoder(fmt, gpu=False)
    raise FfmpegError(f"{tool.path} has no usable video encoder")


def target_bitrate(source: SourceInfo, fmt: str) -> int | None:
    """The source's bitrate, scaled for how efficient the output format is compared to the source's."""
    if not source.video_bitrate or source.video_format not in BITRATE_FACTOR:
        return source.video_bitrate
    return round(source.video_bitrate * BITRATE_FACTOR[fmt] / BITRATE_FACTOR[source.video_format])


def scale_filter(source_height: int | None, target_height: int) -> str | None:
    """Return a proportional upscale filter when the source is below the target height."""
    return f"scale=-2:{target_height}" if source_height and source_height < target_height else None


def audio_encoder(source_format: str | None, tool: FfmpegBinary, *, browser: bool = False) -> tuple[str, str]:
    """The source's audio codec if this ffmpeg can encode it, else AAC; always AAC for browsers."""
    codec, bitrate = AUDIO_ENCODERS.get(source_format or "", AUDIO_ENCODERS["aac"])
    return (codec, bitrate) if codec in tool.software and not browser else AUDIO_ENCODERS["aac"]


def probe_source(video: Path, original: Path | None, *, ffprobe: str = "ffprobe") -> SourceInfo:
    """Format and bitrate of the (video-only) extract, plus the original download's audio codec."""
    info = ffmpeg.probe(str(video), cmd=ffprobe)
    stream = next((s for s in info["streams"] if s.get("codec_type") == "video"), {})
    bitrate = int(stream.get("bit_rate") or 0)
    duration = float(info.get("format", {}).get("duration") or 0)
    if not bitrate and duration:  # MKV rarely records per-stream bitrates; the file holds only this stream
        bitrate = int(video.stat().st_size * 8 / duration)
    audio_format = None
    if original is not None and original.exists():
        streams = ffmpeg.probe(str(original), cmd=ffprobe)["streams"]
        audio_format = next((s.get("codec_name") for s in streams if s.get("codec_type") == "audio"), None)
    return SourceInfo(stream.get("codec_name"), bitrate or None, audio_format,
                      stream.get("width"), stream.get("height"), stream.get("pix_fmt"), stream.get("profile"))


def _ffmpegs_on_path() -> list[str]:
    found: dict[str, str] = {}
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory and (path := shutil.which("ffmpeg", path=directory)):
            found.setdefault(os.path.normcase(os.path.realpath(path)), path)
    return list(found.values()) or ["ffmpeg"]


def _output(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _has_libass(binary: str) -> bool:
    lines = _output([binary, "-hide_banner", "-filters"]).splitlines()
    return any(line.split()[1:2] == ["subtitles"] for line in lines)


def _software_encoders(binary: str) -> frozenset[str]:
    lines = _output([binary, "-hide_banner", "-encoders"]).splitlines()
    return frozenset(parts[1] for line in lines if len(parts := line.split()) > 1) & _SOFTWARE_ENCODERS


def _can_encode(binary: str, codec: str) -> bool:
    args = [binary, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2",
            "-c:v", codec, "-preset", "p4", "-f", "null", "-"]
    try:
        return subprocess.run(args, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def format_bitrate(bitrate: int | None) -> str:
    return f"{bitrate / 1_000_000:.1f} Mbit/s" if bitrate else "constant quality"


# --- the ffmpeg jobs -----------------------------------------------------------------------


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


def sample_frames(video: Path, *, binary: str = "ffmpeg", ffprobe: str = "ffprobe", count: int = 32,
                  size: tuple[int, int] = (64, 36), duration: float | None = None) -> np.ndarray:
    """`count` small RGB frames spread evenly over the video, as an array (frames, height, width, 3).

    Each frame is a fast seek to the keyframe nearest its time, so only `count` frames get decoded
    (some decoders, dav1d among them, ignore ffmpeg's keyframes-only switch).
    """
    if not duration:
        duration = float(ffmpeg.probe(str(video), cmd=ffprobe)["format"]["duration"])
    width, height = size

    def grab(time: float) -> bytes:
        stream = (ffmpeg.input(str(video), ss=round(time, 3), noaccurate_seek=None).video
                  .filter("scale", width, height)
                  .output("pipe:", vframes=1, format="rawvideo", pix_fmt="rgb24"))
        args = [binary, "-hide_banner", "-loglevel", "error", *ffmpeg.compile(stream)[1:]]
        result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=120,
                                creationflags=_BELOW_NORMAL)
        if result.returncode != 0:
            details = result.stderr.decode("utf-8", "replace").strip()[-2000:]
            raise FfmpegError(f"ffmpeg exited with code {result.returncode}: {details}")
        return result.stdout

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = pool.map(grab, [(index + 0.5) / count * duration for index in range(count)])
        data = b"".join(frame for frame in frames if len(frame) == width * height * 3)
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, height, width, 3)


def render(video: Path, audio: Path, subtitles: Path | None, target: Path, *, tool: FfmpegBinary,
           source: SourceInfo, lead: Path | None = None, lead_volume: float = 0.0, darken: float = 0.08,
           target_height: int = 1080, browser: bool = False, copy_audio: bool = False,
           duration: float | None = None, on_progress: ProgressCallback | None = None) -> str:
    """Darken the video, burn in the subtitles and pair it with the karaoke audio.

    Encodes to the source's video format at a comparable bitrate when the hardware allows, and
    falls back to the CPU if the GPU encoder fails. Without subtitles the picture is left as it
    is: the video stream is copied, unless it must be upscaled. `browser` makes an MP4 every
    browser plays: H.264 High (copied when the source already is such H.264), AAC, fast start.
    `copy_audio` passes `audio` (the source's own track) through when the output takes its codec.
    Returns a description of the encoding.
    """
    if copy_audio and (not browser or source.audio_format == "aac"):
        audio_codec, audio_bitrate = "copy", None
    else:
        audio_codec, audio_bitrate = audio_encoder(source.audio_format, tool, browser=browser)
    sound = f"{source.audio_format or 'audio'} copied" if audio_codec == "copy" else audio_codec
    options: dict[str, Any] = dict(lead=lead, lead_volume=lead_volume, darken=darken, target_height=target_height,
                                   source_height=source.video_height, browser=browser, duration=duration,
                                   on_progress=on_progress)
    untouched = subtitles is None and not scale_filter(source.video_height, target_height)
    if untouched and (source.browser_ready or not browser):
        log.info("source %s copied as it is, audio %s", source.video_format or "video", sound)
        _render(tool.path, None, None, (audio_codec, audio_bitrate), video, audio, None, target, **options)
        return f"{source.video_format or 'video'} copied + {sound}"
    formats = ("h264",) if browser else FORMATS
    attempts = [choose_encoder(source.video_format, tool, formats=formats)]
    if attempts[0].gpu:
        attempts.append(choose_encoder(source.video_format, tool, gpu=False, formats=formats))
    for index, encoder in enumerate(attempts):
        bitrate = target_bitrate(source, encoder.format)
        log.info("source %s at %s → %s at %s, audio %s", source.video_format or "unknown",
                 format_bitrate(source.video_bitrate), encoder.label, format_bitrate(bitrate), sound)
        try:
            _render(tool.path, encoder, bitrate, (audio_codec, audio_bitrate), video, audio, subtitles, target,
                    **options)
            return f"{encoder.label} at {format_bitrate(bitrate)} + {sound}"
        except FfmpegError as error:
            if index == len(attempts) - 1:
                raise
            log.warning("%s failed, falling back to %s: %s", encoder.label, attempts[index + 1].label, error)
    raise AssertionError("unreachable")


def _render(binary: str, encoder: Encoder | None, bitrate: int | None, audio_encoding: tuple[str, str | None],
            video: Path, audio: Path, subtitles: Path | None, target: Path, *, darken: float,
            duration: float | None, lead: Path | None, lead_volume: float, target_height: int,
            source_height: int | None, browser: bool, on_progress: ProgressCallback | None) -> None:
    """One ffmpeg run. `encoder` None copies the video stream (no filters then); `subtitles` None burns in nothing."""
    # The subtitles filter chokes on Windows drive letters ("C:"), so ffmpeg runs
    # inside the output folder and gets every path relative to it.
    folder = target.parent
    partial = partial_path(target)

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, folder)).as_posix()

    # With a GPU encoder, decode on the GPU as well (ffmpeg falls back to the CPU if it can't).
    # Frames come back to system memory for the eq and subtitles filters, which only exist on the CPU.
    decode = {"hwaccel": "cuda"} if encoder is not None and encoder.gpu else {}
    picture = ffmpeg.input(relative(video), **decode).video
    if encoder is None:
        video_options: dict[str, Any] = {"vcodec": "copy"}
    else:
        if scale_filter(source_height, target_height):
            picture = picture.filter("scale", -2, target_height)
        if subtitles is not None:
            picture = picture.filter("eq", brightness=-darken).filter("subtitles", relative(subtitles))
        video_options = {"vcodec": encoder.codec, "pix_fmt": "yuv420p", **encoder.options(bitrate)}
        if browser:
            video_options["profile:v"] = "high"
    if browser:
        video_options["movflags"] = "+faststart"  # the index goes first, so playback starts while loading
    sound = ffmpeg.input(relative(audio)).audio
    if lead is not None and lead_volume:
        lead_stream = ffmpeg.input(relative(lead)).audio.filter("volume", lead_volume)
        sound = ffmpeg.filter([sound, lead_stream], "amix", inputs=2, duration="first", dropout_transition=0)
    audio_codec, audio_bitrate = audio_encoding
    audio_options = {"acodec": audio_codec, **({"audio_bitrate": audio_bitrate} if audio_bitrate else {})}
    stream = ffmpeg.output(picture, sound, relative(partial), shortest=None, **audio_options, **video_options)
    run(stream, binary=binary, cwd=folder, duration=duration, on_progress=on_progress)
    partial.replace(target)


def run(stream: Any, *, binary: str = "ffmpeg", cwd: Path | None = None, duration: float | None = None,
        on_progress: ProgressCallback | None = None) -> None:
    """Run an ffmpeg-python stream, reporting progress (0..1) from ffmpeg's `-progress` output."""
    args = [binary, "-hide_banner", "-nostats", "-loglevel", "error", "-progress", "pipe:1", "-y",
            *ffmpeg.compile(stream)[1:]]
    log.debug("$ %s", subprocess.list2cmdline(args))
    process = subprocess.Popen(args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                               creationflags=_BELOW_NORMAL)
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

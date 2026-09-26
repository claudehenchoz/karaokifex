"""Metadata lookup and download with yt-dlp."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yt_dlp

from karaokifex.models import VideoInfo

log = logging.getLogger(__name__)

# Best video + best audio (merged), or the best single file if that's all there is.
FORMAT = "bv*+ba/b"
# For --browser-friendly: H.264 wins when it comes at the best resolution and frame rate on offer,
# so the render can often copy the video instead of encoding it.
PREFER_H264 = ["res", "fps", "vcodec:h264"]

ProgressCallback = Callable[[float | None, str], None]


class _YtdlpLogger:
    """Routes yt-dlp's console output into our logging (and thus the task-tagged log)."""

    def debug(self, message: str) -> None:
        log.debug(message.removeprefix("[debug] "))

    def info(self, message: str) -> None:
        log.debug(message)

    def warning(self, message: str) -> None:
        log.warning(message)

    def error(self, message: str) -> None:
        log.error(message)


def _options(**extra: Any) -> dict[str, Any]:
    return {
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "logger": _YtdlpLogger(),
        "format": FORMAT,
        # YouTube needs a JavaScript runtime; yt-dlp only enables deno by default.
        "js_runtimes": {"deno": {}, "node": {}},
        **extra,
    }


def probe(url: str) -> VideoInfo:
    """Fetch metadata (title, duration, resolution, …) without downloading anything."""
    with yt_dlp.YoutubeDL(_options()) as ydl:
        info = ydl.extract_info(url, download=False)
        return VideoInfo.from_ytdlp(ydl.sanitize_info(info))


def download(url: str, target: Path, on_progress: ProgressCallback, *, prefer_h264: bool = False) -> Path:
    """Download best video + best audio into `target` (always an .mkv)."""

    def progress_hook(status: dict[str, Any]) -> None:
        if status["status"] == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            fraction = status.get("downloaded_bytes", 0) / total if total else None
            stream = "video" if (status.get("info_dict") or {}).get("vcodec", "none") != "none" else "audio"
            speed = status.get("speed")
            speed_text = f" · {speed / 1_048_576:.1f} MiB/s" if speed else ""
            on_progress(fraction, f"{stream} stream{speed_text}")
        elif status["status"] == "finished":
            on_progress(1.0, "stream downloaded")

    def postprocessor_hook(status: dict[str, Any]) -> None:
        if status["status"] == "started":
            on_progress(None, f"post-processing ({status.get('postprocessor', 'ffmpeg')})…")

    options = _options(
        outtmpl=str(target.with_suffix("")) + ".%(ext)s",
        merge_output_format="mkv",
        postprocessors=[{"key": "FFmpegVideoRemuxer", "preferedformat": "mkv"}],
        progress_hooks=[progress_hook],
        postprocessor_hooks=[postprocessor_hook],
        **({"format_sort": PREFER_H264} if prefer_h264 else {}),
    )
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    if not target.exists():
        raise RuntimeError(f"yt-dlp finished but {target.name} is missing")
    return target

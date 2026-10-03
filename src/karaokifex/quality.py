"""How good a song's picture and sound are: quality.json.

    karaokifex-quality <song folder>... [--force]

The pipeline's `quality` step (--quality) writes it from the download itself, before the
download is deleted: its resolution, frame rate, codecs and bitrates (`source`), and the same for
each render (`karaoke`, `original`). Songs made before it get it from what is left: their renders,
and info.json's resolution for the source (what YouTube's best format had). The original is the
download's own picture copied whenever it could be, so where it has the source's size and codec,
its bitrate is the source's too (`source.from` says which).

`summary()` reads one file; `write_quality()` puts a song's together.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import ffmpeg

log = logging.getLogger("karaokifex")


def _fps(rate: str | None) -> float | None:
    try:
        num, den = (rate or "").split("/")
        return round(int(num) / int(den), 3) if int(den) else None
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def summary(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """A media file's container, size, duration and overall bitrate, and its first video and audio streams."""
    info = ffmpeg.probe(str(path), cmd=ffprobe)
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or 0) or None
    size = _int(fmt.get("size")) or path.stat().st_size
    out: dict[str, Any] = {"container": fmt.get("format_name"), "size": size, "duration": duration,
                           "bitrate": _int(fmt.get("bit_rate")) or (round(size * 8 / duration) if duration else None)}
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video:
        out["video"] = {"codec": video.get("codec_name"), "profile": video.get("profile"), "width": video.get("width"),
                        "height": video.get("height"), "fps": _fps(video.get("avg_frame_rate")),
                        "bitrate": _int(video.get("bit_rate")), "pix_fmt": video.get("pix_fmt")}
    if audio:
        out["audio"] = {"codec": audio.get("codec_name"), "bitrate": _int(audio.get("bit_rate")),
                        "sample_rate": _int(audio.get("sample_rate")), "channels": audio.get("channels")}
    return out


def write_quality(target: Path, *, source: dict[str, Any] | None, renders: dict[str, Path],
                  ffprobe: str = "ffprobe") -> dict[str, Any]:
    """quality.json: the source's summary (or what stands in for it) and each render's."""
    out: dict[str, Any] = {"source": source}
    for name, path in renders.items():
        if path.exists():
            out[name] = summary(path, ffprobe=ffprobe)
    src_height = ((source or {}).get("video") or {}).get("height")
    heights = [(out[n].get("video") or {}).get("height") for n in renders if n in out]
    out["upscaled"] = bool(src_height and any(h and h > src_height for h in heights))
    partial = target.with_name(f"{target.stem}.partial{target.suffix}")
    partial.write_text(json.dumps(out, indent=1), encoding="utf-8")
    partial.replace(target)
    return out


def after_the_fact(folder: Path, renders: dict[str, Path], *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """The source of a song whose download is gone: info.json's resolution, and the original's streams
    wherever the original is the source's picture copied (the same size and codec it was downloaded in)."""
    info_file = folder / "info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    source: dict[str, Any] = {"from": "info.json"}
    if info.get("width") and info.get("height"):
        source["video"] = {"width": info["width"], "height": info["height"]}
    original = renders.get("original")
    if original and original.exists():
        o = summary(original, ffprobe=ffprobe)
        ov = o.get("video") or {}
        if ov.get("height") == info.get("height") and ov.get("width") == info.get("width"):
            source = {**o, "from": "the original, the download's picture copied"}
    return source


@click.command(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100})
@click.argument("folders", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Write quality.json again where it is there already.")
@click.option("--ffprobe", default="ffprobe", show_default=True, help="The ffprobe to use.")
def main(folders: tuple[Path, ...], force: bool, ffprobe: str) -> None:
    """Write quality.json into each song folder made before --quality, from what is left of the song."""
    from karaokifex.console import setup_logging
    from karaokifex.workspace import Workspace

    setup_logging(False)
    for folder in folders:
        ws = Workspace(folder, folder.name)
        if ws.quality_json.exists() and not force:
            continue
        renders = {name: path for name, path in (("karaoke", _first(folder, "(Karaoke")), ("original", _first(folder, "(Original)")))
                   if path}
        try:
            source = after_the_fact(folder, renders, ffprobe=ffprobe)
            q = write_quality(ws.quality_json, source=source, renders=renders, ffprobe=ffprobe)
        except ffmpeg.Error as error:
            log.warning("%s: could not be read (%s)", folder.name, error)
            continue
        v = (q.get("source") or {}).get("video") or {}
        log.info("%s: source %sx%s%s", folder.name, v.get("width"), v.get("height"), ", upscaled" if q["upscaled"] else "")


def _first(folder: Path, marker: str) -> Path | None:
    """The folder's video with `marker` in its name, the browser-friendly MP4 first."""
    found = sorted((p for p in folder.iterdir() if marker in p.name and p.suffix in (".mp4", ".mkv")),
                   key=lambda p: p.suffix != ".mp4")
    return found[0] if found else None


if __name__ == "__main__":
    main()

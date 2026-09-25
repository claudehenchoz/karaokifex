"""Command-line entry point."""

from __future__ import annotations

import os

# The live task board replaces the libraries' own progress bars, which would garble it.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402

from karaokifex import __version__  # noqa: E402
from karaokifex.config import (  # noqa: E402
    DEFAULT_KARAOKE_MODEL,
    DEFAULT_MODEL_DIR,
    DEFAULT_SEPARATION_OVERLAP,
    DEFAULT_RESOLUTION,
    DEFAULT_WHISPER_MODEL,
    Config,
)
from karaokifex.console import console, print_summary, setup_logging  # noqa: E402
from karaokifex.pipeline import PipelineResult, run_pipeline  # noqa: E402
from karaokifex.runner import format_duration  # noqa: E402
from karaokifex.workspace import Workspace  # noqa: E402

log = logging.getLogger("karaokifex")


@click.command(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100})
@click.argument("url")
@click.option("-a", "--artist", help="Artist name. Default: taken from the video metadata or title.")
@click.option("-s", "--song", help="Song name. Default: taken from the video metadata or title.")
@click.option("--musicbrainz/--no-musicbrainz", default=True, show_default=True,
              help="Look the song up on MusicBrainz and use its artist and song names when it is sure; "
                   "names given with --artist/--song win.")
@click.option("-l", "--language",
              help="Language code, e.g. 'en'. Default: detected from the lyrics, else from the singing.")
@click.option("--karaoke-model", default=DEFAULT_KARAOKE_MODEL, show_default=True,
              help="audio-separator model splitting lead vocals from the rest (backing vocals included).")
@click.option("--whisper-model", default=DEFAULT_WHISPER_MODEL, show_default=True, help="whisperx model.")
@click.option("--overlap", "separation_overlap", type=click.IntRange(min=1), default=DEFAULT_SEPARATION_OVERLAP,
              show_default=True, help="Stem separation overlap: higher is marginally cleaner, proportionally slower.")
@click.option("--fp16/--fp32", default=True, show_default=True,
              help="Run stem separation in half precision on the GPU (much faster).")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", show_default=True,
              help="Device for whisperx.")
@click.option("--ffmpeg", envvar="KARAOKIFEX_FFMPEG", type=click.Path(dir_okay=False),
              help="ffmpeg executable (env: KARAOKIFEX_FFMPEG). Default: the first one on PATH that can "
                   "burn in subtitles and encode on the GPU.")
@click.option("--gpu-jobs", type=click.IntRange(min=1), default=1, show_default=True,
              help="How many GPU-heavy steps may run at the same time.")
@click.option("--burn-lyrics/--no-burn-lyrics", default=True, show_default=True,
              help="Burn the lyrics into the video. Without, the picture stays as it is (the video is copied "
                   "unless it must be upscaled) and the lyrics are only written to lyrics.ass and timings.json.")
@click.option("--darken", type=click.FloatRange(0, 1), default=0.08, show_default=True,
              help="How much darker the video gets behind the burned-in lyrics (brightness offset).")
@click.option("--resolution", type=click.IntRange(min=144), default=DEFAULT_RESOLUTION, show_default=True,
              help="Minimum output height in pixels; smaller sources are upscaled proportionally.")
@click.option("--lead-volume", type=click.FloatRange(0, 1), default=0.0, show_default=True,
              help="Mix the isolated lead vocal back into the karaoke audio (0 = no voice, 1 = full volume).")
@click.option("--browser-friendly", is_flag=True,
              help="Write an MP4 that every browser plays: H.264, AAC, fast start. The download prefers H.264, "
                   "and with --no-burn-lyrics such a video is copied instead of re-encoded (upscaling to "
                   "--resolution still re-encodes).")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("."),
              show_default=True, help="Where the per-song folder is created.")
@click.option("--model-dir", type=click.Path(file_okay=False, path_type=Path), default=DEFAULT_MODEL_DIR,
              show_default=True, help="Cache folder for separation models.")
@click.option("--mix-vote", is_flag=True,
              help="Also transcribe the full mix and let both transcriptions vote on word times (slower).")
@click.option("--palette", is_flag=True,
              help="Find the video's dominant colours (5, each with its share of the picture) and write them "
                   "to metadata.json.")
@click.option("--describe", is_flag=True,
              help="Look up what MusicBrainz knows of the song -- the album it first came out on, the year, its "
                   "genres, its writers, its language and where its artist is from -- and write it to song.json.")
@click.option("--quality", is_flag=True,
              help="Write quality.json: the download's resolution, frame rate, codecs and bitrates, read before it "
                   "is deleted, and the same for each render.")
@click.option("--debug-ass", is_flag=True,
              help="Render '(Karaoke debug).mkv' with each word coloured by what timed it "
                   "(green forced, cyan whisper, violet LRC tag, orange LRC line, red interpolated).")
@click.option("--keep-source", is_flag=True,
              help="Also keep the original video with its own sound (vocals included) as '(Original).mkv', "
                   "made like the karaoke video: the same format, resolution and --browser-friendly MP4.")
@click.option("--keep-temp", is_flag=True,
              help="Keep all temporary files (e.g. for karaokifex-eval --recompute). By default they are "
                   "deleted after a successful run.")
@click.option("--autodelete", is_flag=True, hidden=True, expose_value=False,
              help="No effect: temporary files are now deleted by default.")
@click.option("--force", is_flag=True, help="Redo every step, even if its output already exists.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output, including the libraries' logs.")
@click.version_option(__version__, "-V", "--version")
def main(url: str, **options: object) -> None:
    """Turn the YouTube video at URL into a karaoke video with word-by-word highlighted lyrics."""
    config = Config(url=url, **options)  # type: ignore[arg-type]
    if config.debug_ass and not config.burn_lyrics:
        raise click.UsageError("--debug-ass burns in the lyrics; it can't be combined with --no-burn-lyrics.")
    setup_logging(config.verbose)
    try:
        result = run_pipeline(config)
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/] Finished steps are kept; run the same command again to resume.")
        os._exit(130)  # worker threads may be stuck in native code; don't wait for them
    except Exception as error:
        log.error("%s: %s", type(error).__name__, error, exc_info=config.verbose)
        raise SystemExit(1) from error

    show_result(result)
    if not result.ok:
        console.print("Temporary files were kept, so running the same command again resumes where it stopped.")
        raise SystemExit(1)
    clean_up(result.job.workspace, keep_temp=config.keep_temp)


def show_result(result: PipelineResult) -> None:
    ws = result.job.workspace
    outcomes = result.report.outcomes
    details = [("Song", result.job.title), ("Folder", str(ws.root.resolve()))]
    if result.ok:
        details.insert(0, ("Video", str(result.job.output_video.resolve())))
        if result.job.config.keep_source:
            details.insert(1, ("Original", str(result.job.original_video.resolve())))
        chosen = outcomes["subtitles"].result if "subtitles" in outcomes else None
        lyrics_source = chosen or outcomes["lyrics"].result or "lrclib (from an earlier run)"
        details.append(("Lyrics", lyrics_source))
    for name, error in result.report.failures.items():
        details.append((f"✖ {name}", f"{type(error).__name__}: {error}"))
    total = sum(outcome.elapsed for outcome in outcomes.values())
    details.append(("Compute time", f"{format_duration(total)} (summed over all tasks)"))
    print_summary(result.ok, details)


def clean_up(workspace: Workspace, *, keep_temp: bool) -> None:
    """Delete the temporary files of a successful run, unless --keep-temp asks to keep them."""
    temp_files = workspace.temp_files()
    if not temp_files:
        return
    total = human_size(sum(path.stat().st_size for path in temp_files))
    if keep_temp:
        console.print(f"Kept {len(temp_files)} temporary files ({total}) in {workspace.root}.")
        return
    removed = workspace.cleanup()
    console.print(f"🧹 Removed {len(removed)} temporary files ({total}).")


def human_size(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")

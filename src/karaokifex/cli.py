"""Command-line entry point."""

from __future__ import annotations

import os

# The live task board replaces the libraries' own progress bars, which would garble it.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402
from rich.table import Table  # noqa: E402

from karaokifex import __version__  # noqa: E402
from karaokifex.config import (  # noqa: E402
    DEFAULT_KARAOKE_MODEL,
    DEFAULT_MODEL_DIR,
    DEFAULT_SEPARATION_OVERLAP,
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
@click.option("-l", "--language", help="Language code for transcription, e.g. 'en'. Default: auto-detect.")
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
@click.option("--darken", type=click.FloatRange(0, 1), default=0.08, show_default=True,
              help="How much darker the video gets (brightness offset).")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("."),
              show_default=True, help="Where the per-song folder is created.")
@click.option("--model-dir", type=click.Path(file_okay=False, path_type=Path), default=DEFAULT_MODEL_DIR,
              show_default=True, help="Cache folder for separation models.")
@click.option("--autodelete", is_flag=True, help="Delete temporary files at the end without asking.")
@click.option("--force", is_flag=True, help="Redo every step, even if its output already exists.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output, including the libraries' logs.")
@click.version_option(__version__, "-V", "--version")
def main(url: str, **options: object) -> None:
    """Turn the YouTube video at URL into a karaoke video with word-by-word highlighted lyrics."""
    config = Config(url=url, **options)  # type: ignore[arg-type]
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
    offer_cleanup(result.job.workspace, autodelete=config.autodelete)


def show_result(result: PipelineResult) -> None:
    ws = result.job.workspace
    outcomes = result.report.outcomes
    details = [("Song", result.job.title), ("Folder", str(ws.root.resolve()))]
    if result.ok:
        details.insert(0, ("Video", str(ws.final_video.resolve())))
        lyrics_source = outcomes["lyrics"].result or "lrclib (from an earlier run)"
        details.append(("Lyrics", lyrics_source))
    for name, error in result.report.failures.items():
        details.append((f"✖ {name}", f"{type(error).__name__}: {error}"))
    total = sum(outcome.elapsed for outcome in outcomes.values())
    details.append(("Compute time", f"{format_duration(total)} (summed over all tasks)"))
    print_summary(result.ok, details)


def offer_cleanup(workspace: Workspace, *, autodelete: bool) -> None:
    temp_files = workspace.temp_files()
    if not temp_files:
        return
    table = Table(title="Temporary files", title_justify="left", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="grey70")
    table.add_column(justify="right", style="grey50")
    for path in temp_files:
        table.add_row(str(path.relative_to(workspace.root)), human_size(path.stat().st_size))
    total = sum(path.stat().st_size for path in temp_files)
    console.print(table)
    if autodelete or click.confirm(f"Delete these {len(temp_files)} temporary files ({human_size(total)})?",
                                   default=True):
        removed = workspace.cleanup()
        console.print(f"🧹 Removed {len(removed)} temporary files.")
    else:
        console.print(f"Kept temporary files in {workspace.root}.")


def human_size(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")

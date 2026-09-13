"""Terminal output: the shared rich console, task-tagged logging and the live task board.

Every log record is tagged with the pipeline task that emitted it (the runner
tracks that in a ContextVar inside each worker thread), each task has its own
colour, so interleaved output from parallel tasks stays readable:

    12:03:04  download          video stream 12.4 MiB/s
    12:03:04  lyrics            found synced lyrics: Queen – Bohemian Rhapsody
"""

from __future__ import annotations

import logging
import threading
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from karaokifex.runner import Status, current_task, format_duration

console = Console(highlight=False)

_PALETTE = (
    "deep_sky_blue1", "magenta", "green3", "gold1", "orchid", "turquoise2",
    "orange1", "spring_green2", "hot_pink", "cornflower_blue", "khaki1", "medium_purple1",
)
_colours: dict[str, str] = {"main": "bright_white", "probe": "bright_white"}
_tag_width = 8

# Third-party loggers that are chatty at INFO level.
_LIBRARY_LOGGERS = (
    "audio_separator", "faster_whisper", "pyannote", "lightning", "lightning_fabric", "pytorch_lightning",
    "speechbrain", "urllib3", "httpx", "huggingface_hub", "numba", "matplotlib", "fsspec", "filelock", "PIL",
)
_LEVEL_STYLES = {logging.DEBUG: "grey50", logging.WARNING: "yellow", logging.ERROR: "bold red", logging.CRITICAL: "bold red"}

_STATUS_LOOK = {  # icon, style, label
    Status.WAITING: ("·", "grey50", "waiting"),
    Status.RUNNING: ("", "", "running"),
    Status.DONE: ("✔", "green", "done"),
    Status.CACHED: ("↺", "cyan", "cached"),
    Status.FAILED: ("✖", "bold red", "failed"),
    Status.SKIPPED: ("⊘", "grey50", "skipped"),
}


def register_tasks(names: Iterable[str]) -> None:
    """Give every task a stable colour and size the log tag column."""
    global _tag_width
    for name in names:
        if name not in _colours:
            _colours[name] = _PALETTE[(len(_colours) - 2) % len(_PALETTE)]
        _tag_width = max(_tag_width, len(name))


def colour_for(task: str) -> str:
    return _colours.get(task) or _PALETTE[zlib.crc32(task.encode()) % len(_PALETTE)]


class TaskLogHandler(logging.Handler):
    """Prints records as `time  task  message`, with wrapped lines indented under the message."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            task = current_task.get()
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + logging.Formatter().formatException(record.exc_info)
            prefix = Text.assemble(
                (datetime.fromtimestamp(record.created).strftime("%H:%M:%S"), "grey50"),
                "  ",
                (task.ljust(_tag_width), f"bold {colour_for(task)}"),
            )
            line = Table.grid(padding=(0, 2))
            line.add_column(no_wrap=True)
            line.add_column(overflow="fold")
            line.add_row(prefix, Text(message, style=_LEVEL_STYLES.get(record.levelno, "")))
            console.print(line)
        except Exception:
            self.handleError(record)


class _DropKnownNoise(logging.Filter):
    """Library messages that are expected and harmless in our setup."""

    # We only run PyTorch-based Roformer models, so ONNX Runtime acceleration is irrelevant.
    MESSAGES = ("CUDAExecutionProvider not available in ONNXruntime",)

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(text in record.getMessage() for text in self.MESSAGES)


def setup_logging(verbose: bool = False) -> None:
    handler = TaskLogHandler()
    handler.addFilter(_DropKnownNoise())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger("karaokifex").setLevel(logging.DEBUG if verbose else logging.INFO)

    library_level = logging.INFO if verbose else logging.WARNING
    for name in _LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(library_level)
    # whisperx installs its own stdout handler unless its logger already has one.
    whisperx_logger = logging.getLogger("whisperx")
    whisperx_logger.handlers[:] = [handler]
    whisperx_logger.propagate = False
    whisperx_logger.setLevel(library_level)

    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING if verbose else logging.ERROR)


def adopt_loggers(*names: str) -> None:
    """Route loggers that installed their own console handler on import through ours instead."""
    level = logging.INFO if logging.getLogger("karaokifex").isEnabledFor(logging.DEBUG) else logging.WARNING
    for name in names:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)


@dataclass
class _Row:
    name: str
    description: str
    spinner: Spinner
    status: Status = Status.WAITING
    progress: float | None = None
    note: str = ""
    started: float | None = None
    finished: float | None = None

    def elapsed(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or time.monotonic()) - self.started


class TaskBoard:
    """Live view of the pipeline, one row per task. Implements the runner's RunObserver."""

    def __init__(self, title: str, tasks: Iterable[tuple[str, str]]) -> None:
        self.title = title
        self._lock = threading.Lock()
        self._rows = {
            name: _Row(name, description, Spinner("dots", style=colour_for(name)))
            for name, description in tasks
        }

    def task_status(self, name: str, status: Status) -> None:
        with self._lock:
            row = self._rows[name]
            row.status = status
            if status is Status.RUNNING:
                row.started = time.monotonic()
            elif row.started is not None and row.finished is None:
                row.finished = time.monotonic()
            if status is not Status.RUNNING:
                row.progress = None
            if status.succeeded:
                row.note = ""

    def task_progress(self, name: str, fraction: float | None) -> None:
        with self._lock:
            self._rows[name].progress = None if fraction is None else min(max(fraction, 0.0), 1.0)

    def task_note(self, name: str, note: str) -> None:
        with self._lock:
            self._rows[name].note = note

    def __rich__(self) -> Panel:
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=1, no_wrap=True)
        table.add_column(width=_tag_width, no_wrap=True)
        table.add_column(width=20, no_wrap=True)
        table.add_column(width=4, justify="right", no_wrap=True)
        table.add_column(width=7, justify="right", no_wrap=True)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        with self._lock:
            rows = list(self._rows.values())
            for row in rows:
                table.add_row(*self._cells(row))
        finished = sum(row.status.succeeded for row in rows)
        return Panel(
            table,
            title=f"🎤 [bold]karaokifex[/] · {escape(self.title)}",
            title_align="left",
            subtitle=f"{finished}/{len(rows)} done",
            subtitle_align="right",
            border_style="grey42",
        )

    @staticmethod
    def _cells(row: _Row) -> tuple:
        colour = colour_for(row.name)
        elapsed = row.elapsed()
        timing = Text(format_duration(elapsed) if elapsed is not None else "", style="grey62")
        note = Text(row.note, style="grey78") if row.note else Text(row.description, style="grey42")
        name = Text(row.name, style=f"bold {colour}")
        if row.status is Status.RUNNING:
            if row.progress is None:
                return row.spinner, name, Text("running…", style=colour), Text(""), timing, note
            bar = ProgressBar(total=1.0, completed=row.progress, width=20, complete_style=colour, finished_style=colour)
            return row.spinner, name, bar, Text(f"{row.progress * 100:3.0f}%"), timing, note
        icon, style, label = _STATUS_LOOK[row.status]
        if row.status in (Status.WAITING, Status.SKIPPED):
            name = Text(row.name, style="grey50")
        return Text(icon, style=style), name, Text(label, style=style), Text(""), timing, note


def print_summary(ok: bool, details: Iterable[tuple[str, str]]) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    for key, value in details:
        grid.add_row(key, value)
    title = "✨ [bold green]Karaoke video ready[/]" if ok else "💥 [bold red]Pipeline failed[/]"
    console.print(Panel(grid, title=title, title_align="left", border_style="green" if ok else "red"))

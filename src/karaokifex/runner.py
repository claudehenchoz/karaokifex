"""A tiny dependency-graph task runner.

Tasks declare which tasks they depend on. Every task whose dependencies have
finished is started immediately on a thread pool, so independent work always
runs in parallel. GPU-heavy tasks additionally share a semaphore so they don't
fight over video memory; with a shared GPU lock file, the tasks that run a model
also take turns with other runs on the same machine.

Tasks may declare output files: when those already exist (and everything the
task depends on was reused too) the task is skipped as "cached", which makes
re-running after a failure cheap.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence

from karaokifex.gpu import SharedGpu

log = logging.getLogger("karaokifex")

#: Name of the task running in the current thread — used to tag log output.
current_task: ContextVar[str] = ContextVar("current_task", default="main")


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"
    CACHED = "cached"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def succeeded(self) -> bool:
        return self in (Status.DONE, Status.CACHED)

    @property
    def blocked(self) -> bool:
        """True if tasks depending on this one can never run."""
        return self in (Status.FAILED, Status.SKIPPED)


class RunObserver(Protocol):
    """Receives live updates about tasks (implemented by the terminal task board)."""

    def task_status(self, name: str, status: Status) -> None: ...

    def task_progress(self, name: str, fraction: float | None) -> None: ...

    def task_note(self, name: str, note: str) -> None: ...


class _NullObserver:
    def task_status(self, name: str, status: Status) -> None:
        pass

    def task_progress(self, name: str, fraction: float | None) -> None:
        pass

    def task_note(self, name: str, note: str) -> None:
        pass


@dataclass
class TaskContext:
    """Handed to every task function: upstream results plus progress reporting."""

    name: str
    _results: MutableMapping[str, Any]
    _observer: RunObserver
    _outcomes: Mapping[str, TaskOutcome] = field(default_factory=dict)

    def result(self, task: str) -> Any:
        """Return value of an upstream task (None if it was cached)."""
        return self._results.get(task)

    def take(self, task: str) -> Any:
        """Like result(), but the runner forgets the value, so it can be freed (e.g. a model in GPU memory)."""
        if (outcome := self._outcomes.get(task)) is not None:
            outcome.result = None
        return self._results.pop(task, None)

    def progress(self, fraction: float | None) -> None:
        self._observer.task_progress(self.name, fraction)

    def note(self, text: str) -> None:
        self._observer.task_note(self.name, text)


@dataclass(frozen=True)
class Task:
    name: str
    run: Callable[[TaskContext], Any]
    deps: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    gpu: bool = False
    model: bool = False  # runs a model on the GPU: takes the shared GPU lock, when there is one
    description: str = ""

    def outputs_exist(self) -> bool:
        return bool(self.outputs) and all(path.exists() for path in self.outputs)


@dataclass
class TaskOutcome:
    status: Status = Status.WAITING
    elapsed: float = 0.0
    result: Any = None
    error: BaseException | None = None


@dataclass
class RunReport:
    outcomes: dict[str, TaskOutcome] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(outcome.status.succeeded for outcome in self.outcomes.values())

    @property
    def failures(self) -> dict[str, BaseException]:
        return {name: o.error for name, o in self.outcomes.items() if o.error is not None}


class TaskRunner:
    def __init__(
        self,
        tasks: Sequence[Task],
        *,
        gpu_slots: int = 1,
        gpu_lock: Path | None = None,
        force: bool = False,
        observer: RunObserver | None = None,
    ) -> None:
        self.tasks: dict[str, Task] = {}
        for task in tasks:
            if task.name in self.tasks:
                raise ValueError(f"Duplicate task name {task.name!r}")
            self.tasks[task.name] = task
        for task in tasks:
            for dep in task.deps:
                if dep not in self.tasks:
                    raise ValueError(f"Task {task.name!r} depends on unknown task {dep!r}")
        _check_acyclic(self.tasks)
        self.force = force
        self.observer: RunObserver = observer or _NullObserver()
        self._gpu = threading.Semaphore(max(1, gpu_slots))
        self._shared = gpu_lock

    def run(self) -> RunReport:
        report = RunReport({name: TaskOutcome() for name in self.tasks})
        results: dict[str, Any] = {}
        waiting = list(self.tasks)
        running: dict[Future, str] = {}
        pool = ThreadPoolExecutor(max_workers=max(1, len(self.tasks)), thread_name_prefix="task")
        try:
            self._schedule(waiting, running, report, results, pool)
            while running:
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    name = running.pop(future)
                    outcome = report.outcomes[name]
                    error = future.exception()
                    if error is None:
                        outcome.result = results[name] = future.result()
                        self._set_status(name, outcome, Status.DONE)
                        _log_as(name, logging.INFO, "✔ done in %s", format_duration(outcome.elapsed))
                    else:
                        outcome.error = error
                        self._set_status(name, outcome, Status.FAILED)
                self._schedule(waiting, running, report, results, pool)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return report

    def _schedule(self, waiting: list[str], running: dict[Future, str], report: RunReport,
                  results: dict[str, Any], pool: ThreadPoolExecutor) -> None:
        """Start (or skip, or mark cached) every waiting task whose dependencies are settled."""
        progressed = True
        while progressed:
            progressed = False
            for name in list(waiting):
                task = self.tasks[name]
                dep_status = [report.outcomes[dep].status for dep in task.deps]
                if any(status.blocked for status in dep_status):
                    waiting.remove(name)
                    self._set_status(name, report.outcomes[name], Status.SKIPPED)
                    _log_as(name, logging.WARNING, "skipped because an upstream task failed")
                    progressed = True
                elif all(status.succeeded for status in dep_status):
                    waiting.remove(name)
                    progressed = True
                    upstream_cached = all(status is Status.CACHED for status in dep_status)
                    if not self.force and upstream_cached and task.outputs_exist():
                        self._set_status(name, report.outcomes[name], Status.CACHED)
                        _log_as(name, logging.INFO, "↺ reusing existing output")
                    else:
                        running[pool.submit(self._execute, task, report.outcomes, results)] = name

    def _execute(self, task: Task, outcomes: Mapping[str, TaskOutcome], results: MutableMapping[str, Any]) -> Any:
        token = current_task.set(task.name)
        outcome = outcomes[task.name]
        context = TaskContext(task.name, results, self.observer, outcomes)
        try:
            if not task.gpu:
                return self._run_timed(task, context, outcome)
            if not self._gpu.acquire(blocking=False):
                context.note("waiting for a free GPU slot…")
                self._gpu.acquire()
                context.note("")
            try:
                if not (task.model and self._shared):
                    return self._run_timed(task, context, outcome)
                context.note("waiting for the GPU (another song is using it)…")
                with SharedGpu(self._shared):
                    context.note("")
                    return self._run_timed(task, context, outcome)
            finally:
                self._gpu.release()
        finally:
            current_task.reset(token)

    def _run_timed(self, task: Task, context: TaskContext, outcome: TaskOutcome) -> Any:
        self.observer.task_status(task.name, Status.RUNNING)
        started = time.monotonic()
        try:
            return task.run(context)
        except Exception as error:
            log.error("✖ %s: %s", type(error).__name__, error, exc_info=log.isEnabledFor(logging.DEBUG))
            raise
        finally:
            outcome.elapsed = time.monotonic() - started

    def _set_status(self, name: str, outcome: TaskOutcome, status: Status) -> None:
        outcome.status = status
        self.observer.task_status(name, status)


def _log_as(task: str, level: int, message: str, *args: Any) -> None:
    token = current_task.set(task)
    try:
        log.log(level, message, *args)
    finally:
        current_task.reset(token)


def _check_acyclic(tasks: Mapping[str, Task]) -> None:
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in done:
            return
        if name in visiting:
            raise ValueError(f"Task dependency cycle: {' -> '.join((*path, name))}")
        visiting.add(name)
        for dep in tasks[name].deps:
            visit(dep, (*path, name))
        visiting.discard(name)
        done.add(name)

    for name in tasks:
        visit(name, ())


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m{secs:02d}s"

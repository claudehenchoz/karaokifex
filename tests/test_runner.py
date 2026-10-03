import threading
import time

import pytest

from karaokifex.runner import Status, Task, TaskRunner


def noop(ctx):
    return None


def test_dependencies_run_in_order_and_pass_results():
    order = []

    def first(ctx):
        order.append("a")
        return 21

    def second(ctx):
        order.append("b")
        return ctx.result("a") * 2

    report = TaskRunner([Task("b", second, deps=("a",)), Task("a", first)]).run()
    assert order == ["a", "b"]
    assert report.ok and report.outcomes["b"].result == 42


def test_take_hands_over_a_result_and_forgets_it():
    seen = {}

    def consumer(ctx):
        seen["taken"] = ctx.take("a")
        seen["after"] = ctx.result("a")

    report = TaskRunner([Task("a", lambda ctx: "model"), Task("b", consumer, deps=("a",))]).run()
    assert seen == {"taken": "model", "after": None}
    assert report.outcomes["a"].result is None  # nothing keeps the value alive any more


def test_independent_tasks_run_concurrently():
    barrier = threading.Barrier(2, timeout=5)  # only passes if both tasks are running at once
    report = TaskRunner([Task("x", lambda ctx: barrier.wait()), Task("y", lambda ctx: barrier.wait())]).run()
    assert report.ok


def test_failure_skips_dependents_but_not_independent_tasks():
    def boom(ctx):
        raise ValueError("nope")

    report = TaskRunner([
        Task("a", boom),
        Task("b", noop, deps=("a",)),
        Task("c", noop, deps=("b",)),
        Task("d", noop),
    ]).run()
    statuses = {name: outcome.status for name, outcome in report.outcomes.items()}
    assert statuses == {"a": Status.FAILED, "b": Status.SKIPPED, "c": Status.SKIPPED, "d": Status.DONE}
    assert not report.ok
    assert isinstance(report.failures["a"], ValueError)


def test_existing_outputs_are_reused(tmp_path):
    existing = tmp_path / "out.txt"
    existing.write_text("x")
    calls = []
    report = TaskRunner([
        Task("a", lambda ctx: calls.append("a"), outputs=(existing,)),
        Task("b", lambda ctx: calls.append("b"), deps=("a",), outputs=(tmp_path / "missing",)),
    ]).run()
    assert report.outcomes["a"].status is Status.CACHED
    assert report.outcomes["b"].status is Status.DONE
    assert calls == ["b"]


def test_force_reruns_everything(tmp_path):
    existing = tmp_path / "out.txt"
    existing.write_text("x")
    calls = []
    TaskRunner([Task("a", lambda ctx: calls.append("a"), outputs=(existing,))], force=True).run()
    assert calls == ["a"]


def test_outputs_are_recomputed_when_an_upstream_task_reran(tmp_path):
    existing = tmp_path / "out.txt"
    existing.write_text("x")
    report = TaskRunner([Task("a", noop), Task("b", noop, deps=("a",), outputs=(existing,))]).run()
    assert report.outcomes["b"].status is Status.DONE


def test_gpu_tasks_share_the_gpu_slots():
    lock = threading.Lock()
    active = peak = 0

    def gpu_work(ctx):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    TaskRunner([Task(f"g{i}", gpu_work, gpu=True) for i in range(3)], gpu_slots=1).run()
    assert peak == 1


def test_more_gpu_slots_allow_parallel_gpu_tasks():
    barrier = threading.Barrier(2, timeout=5)
    tasks = [Task(name, lambda ctx: barrier.wait(), gpu=True) for name in ("x", "y")]
    assert TaskRunner(tasks, gpu_slots=2).run().ok


def test_model_tasks_of_two_runs_take_turns_with_a_shared_gpu_lock(tmp_path):
    lock = threading.Lock()
    active = peak = 0
    both = threading.Barrier(2, timeout=5)

    def model_work(ctx):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    def run():
        # the other run's steps without a model (a download, a render) still overlap
        TaskRunner([Task("download", lambda ctx: both.wait()),
                    Task("separate", model_work, deps=("download",), gpu=True, model=True)],
                   gpu_lock=tmp_path / "gpu.lock").run()

    runs = [threading.Thread(target=run) for _ in range(2)]
    for thread in runs:
        thread.start()
    for thread in runs:
        thread.join()
    assert peak == 1


def test_observer_sees_the_lifecycle():
    events = []

    class Recorder:
        def task_status(self, name, status):
            events.append((name, status))

        def task_progress(self, name, fraction):
            events.append((name, fraction))

        def task_note(self, name, note):
            events.append((name, note))

    def work(ctx):
        ctx.note("working")
        ctx.progress(0.5)

    TaskRunner([Task("a", work)], observer=Recorder()).run()
    assert events == [("a", Status.RUNNING), ("a", "working"), ("a", 0.5), ("a", Status.DONE)]


def test_invalid_graphs_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        TaskRunner([Task("a", noop, deps=("zzz",))])
    with pytest.raises(ValueError, match="cycle"):
        TaskRunner([Task("a", noop, deps=("b",)), Task("b", noop, deps=("a",))])
    with pytest.raises(ValueError, match="Duplicate"):
        TaskRunner([Task("a", noop), Task("a", noop)])

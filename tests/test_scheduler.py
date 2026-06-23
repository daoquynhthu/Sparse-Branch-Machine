#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_hardware import (  # noqa: E402
    HardwareSnapshot,
    ResourcePolicy,
    parse_linux_meminfo,
    snapshot_hardware,
)
from sbm_scheduler import (  # noqa: E402
    ResourceScheduler,
    RunEstimate,
    ScheduledRun,
    estimate_worker_memory,
)


def test_policy_budgets() -> None:
    snapshot = HardwareSnapshot(cpu_count=20, total_memory_bytes=1000, available_memory_bytes=600)
    policy = ResourcePolicy(
        cpu_fraction=0.5,
        memory_fraction=0.9,
        minimum_free_memory_bytes=100,
    )
    assert policy.cpu_budget(snapshot) == 10
    assert policy.dispatchable_memory(snapshot) == 500

    fraction_limited = ResourcePolicy(
        cpu_fraction=1.0,
        memory_fraction=0.5,
        minimum_free_memory_bytes=10,
    )
    assert fraction_limited.dispatchable_memory(snapshot) == 300


def test_snapshot_injection() -> None:
    snapshot = snapshot_hardware(
        cpu_count_provider=lambda: 12,
        memory_provider=lambda: (64 << 30, 48 << 30),
    )
    assert snapshot.cpu_count == 12
    assert snapshot.total_memory_bytes == 64 << 30
    assert snapshot.available_memory_bytes == 48 << 30


def test_linux_meminfo_parser() -> None:
    total, available = parse_linux_meminfo(
        "MemTotal:       16384 kB\nMemFree: 1024 kB\nMemAvailable: 12288 kB\n"
    )
    assert total == 16384 * 1024
    assert available == 12288 * 1024


class FakeWorker:
    def __init__(self, returncode: int, polls: int = 1, on_finish=lambda: None) -> None:
        self.returncode = returncode
        self.polls = polls
        self.on_finish = on_finish

    def poll(self) -> int | None:
        if self.polls > 0:
            self.polls -= 1
            return None
        self.on_finish()
        self.on_finish = lambda: None
        return self.returncode


def test_dual_constraint_fifo_and_release() -> None:
    state = {"available": 700, "launched": []}

    def profile() -> HardwareSnapshot:
        return HardwareSnapshot(4, 1000, state["available"])

    def launch(run: ScheduledRun) -> FakeWorker:
        state["launched"].append(run.run_id)
        state["available"] -= run.estimate.memory_bytes
        return FakeWorker(
            1 if run.run_id == "first" else 0,
            on_finish=lambda: state.__setitem__(
                "available", state["available"] + run.estimate.memory_bytes
            ),
        )

    transitions: list[tuple[str, str]] = []
    scheduler = ResourceScheduler(
        policy=ResourcePolicy(0.5, 1.0, 100),
        snapshot_provider=profile,
        launcher=launch,
        emit=lambda transition: transitions.append((transition.run_id, transition.state)),
        sleep=lambda _: None,
    )
    results = scheduler.run([
        ScheduledRun("first", RunEstimate(cpu_slots=2, memory_bytes=500), None),
        ScheduledRun("second", RunEstimate(cpu_slots=1, memory_bytes=300), None),
    ])
    assert state["launched"] == ["first", "second"]
    assert results == {"first": 1, "second": 0}
    assert transitions == [
        ("first", "queued"),
        ("second", "queued"),
        ("first", "started"),
        ("first", "failed"),
        ("second", "started"),
        ("second", "completed"),
    ]


def test_memory_limit_blocks_launch() -> None:
    snapshot = HardwareSnapshot(8, 1000, 300)
    scheduler = ResourceScheduler(
        policy=ResourcePolicy(1.0, 0.9, 100),
        snapshot_provider=lambda: snapshot,
        launcher=lambda _: (_ for _ in ()).throw(AssertionError("must not launch")),
        emit=lambda _: None,
        sleep=lambda _: None,
        maximum_idle_polls=2,
    )
    try:
        scheduler.run([ScheduledRun("large", RunEstimate(1, 250), None)])
    except RuntimeError as error:
        assert "cannot fit" in str(error)
    else:
        raise AssertionError("unschedulable run was not rejected")


def test_worker_memory_estimate_uses_growth_and_margin() -> None:
    assert estimate_worker_memory(
        dataset_bytes=100,
        model_bytes=200,
        observed_growth_bytes=300,
        process_overhead_bytes=400,
        margin=1.5,
    ) == 1200


if __name__ == "__main__":
    test_policy_budgets()
    test_snapshot_injection()
    test_linux_meminfo_parser()
    test_dual_constraint_fifo_and_release()
    test_memory_limit_blocks_launch()
    test_worker_memory_estimate_uses_growth_and_margin()
    print("Scheduler tests passed")

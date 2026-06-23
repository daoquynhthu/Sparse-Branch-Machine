"""FIFO process scheduling under live CPU and available-memory constraints."""

from __future__ import annotations

import subprocess
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from sbm_hardware import HardwareSnapshot, ResourcePolicy, snapshot_hardware


@dataclass(frozen=True)
class RunEstimate:
    cpu_slots: int
    memory_bytes: int

    def __post_init__(self) -> None:
        if self.cpu_slots < 1:
            raise ValueError("cpu_slots must be positive")
        if self.memory_bytes < 0:
            raise ValueError("memory_bytes must be non-negative")


@dataclass(frozen=True)
class ScheduledRun:
    run_id: str
    estimate: RunEstimate
    payload: Any


@dataclass(frozen=True)
class Transition:
    run_id: str
    state: str
    returncode: int | None = None


class Worker(Protocol):
    def poll(self) -> int | None: ...


@dataclass
class Reservation:
    run: ScheduledRun
    worker: Worker


def launch_subprocess(run: ScheduledRun) -> subprocess.Popen[bytes]:
    if not isinstance(run.payload, (list, tuple)):
        raise TypeError("subprocess payload must be an argument sequence")
    return subprocess.Popen(run.payload, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compact_transition(transition: Transition) -> None:
    suffix = "" if transition.returncode is None else f" rc={transition.returncode}"
    print(f"scheduler {transition.state} {transition.run_id}{suffix}", flush=True)


class ResourceScheduler:
    def __init__(
        self,
        policy: ResourcePolicy,
        snapshot_provider: Callable[[], HardwareSnapshot] = snapshot_hardware,
        launcher: Callable[[ScheduledRun], Worker] = launch_subprocess,
        emit: Callable[[Transition], None] = compact_transition,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = 0.1,
        maximum_idle_polls: int = 100,
    ) -> None:
        self.policy = policy
        self.snapshot_provider = snapshot_provider
        self.launcher = launcher
        self.emit = emit
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.maximum_idle_polls = maximum_idle_polls

    def run(self, runs: Iterable[ScheduledRun]) -> dict[str, int]:
        pending = deque(runs)
        running: list[Reservation] = []
        results: dict[str, int] = {}
        idle_polls = 0

        for run in pending:
            self.emit(Transition(run.run_id, "queued"))

        while pending or running:
            progressed = False
            survivors: list[Reservation] = []
            for reservation in running:
                returncode = reservation.worker.poll()
                if returncode is None:
                    survivors.append(reservation)
                    continue
                results[reservation.run.run_id] = returncode
                state = "completed" if returncode == 0 else "failed"
                self.emit(Transition(reservation.run.run_id, state, returncode))
                progressed = True
            running = survivors

            if pending:
                snapshot = self.snapshot_provider()
                cpu_used = sum(item.run.estimate.cpu_slots for item in running)
                run = pending[0]
                worker_limit = self.policy.cpu_budget(snapshot)
                fits_cpu = cpu_used + run.estimate.cpu_slots <= worker_limit
                fits_workers = (
                    self.policy.max_workers is None
                    or len(running) < self.policy.max_workers
                )
                fits_memory = (
                    run.estimate.memory_bytes <= self.policy.dispatchable_memory(snapshot)
                )
                if fits_cpu and fits_workers and fits_memory:
                    pending.popleft()
                    try:
                        worker = self.launcher(run)
                    except Exception:
                        results[run.run_id] = -1
                        self.emit(Transition(run.run_id, "failed", -1))
                    else:
                        running.append(Reservation(run, worker))
                        self.emit(Transition(run.run_id, "started"))
                    progressed = True

            if progressed:
                idle_polls = 0
            elif pending and not running:
                idle_polls += 1
                if idle_polls >= self.maximum_idle_polls:
                    run = pending[0]
                    raise RuntimeError(
                        f"run {run.run_id!r} cannot fit current CPU/memory limits"
                    )

            if pending or running:
                self.sleep(self.poll_interval_seconds)

        return results

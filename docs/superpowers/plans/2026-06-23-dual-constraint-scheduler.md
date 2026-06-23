# Dual-Constraint Hardware Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dynamically dispatch independent experiments up to configurable CPU and currently available-memory fractions without changing per-run learning semantics.

**Architecture:** Keep each online model in an isolated process. A Python scheduler profiles host resources, reserves conservative per-run CPU/memory estimates, launches only within both limits, refreshes available memory under pressure and emits compact state transitions.

**Tech Stack:** Python standard library, Windows `GlobalMemoryStatusEx`, Linux `/proc/meminfo`, multiprocessing/subprocess, existing C ABI runner.

---

## File structure

- Create `python/sbm_hardware.py`: cross-platform immutable hardware snapshots.
- Create `python/sbm_scheduler.py`: resource policy, reservations and process lifecycle.
- Create `tests/test_scheduler.py`: deterministic fake-profile scheduling tests.
- Modify `scripts/tune.py`: use scheduler instead of fixed job count when auto mode is enabled.
- Modify `BUILDING.md`, `API.md`, `RESEARCH_LOG.md`: configuration and measured utilization.

### Task 1: Hardware profile and policy

**Files:**
- Create: `python/sbm_hardware.py`
- Create: `tests/test_scheduler.py`

- [x] Write RED tests for CPU fraction, available-memory budget formula and
Windows/Linux snapshot parsing using injected providers.
- [x] Run `python tests/test_scheduler.py`; expect missing module failure.
- [x] Implement `HardwareSnapshot`, `ResourcePolicy` and
`dispatchable_memory = max(0,min(fraction*available,available-minimum_free))`.
- [x] Verify tests pass and commit `feat: profile schedulable hardware resources`.

### Task 2: Deterministic reservation scheduler

**Files:**
- Create: `python/sbm_scheduler.py`
- Modify: `tests/test_scheduler.py`

- [x] Write RED tests proving a worker is not launched above CPU or memory
budget, reservations release after failure, and pending order remains FIFO.
- [x] Implement `RunEstimate`, `Reservation`, `WorkerState` and a scheduler whose
resource provider and process launcher are injected for tests.
- [x] Emit only `queued`, `started`, `completed` and `failed` compact transitions.
- [x] Verify tests and commit `feat: schedule experiments under dual limits`.

### Task 3: Integrate automated experiment dispatch

**Files:**
- Modify: `scripts/tune.py`
- Modify: `tests/test_scheduler.py`

- [x] Write RED integration tests comparing fixed one-worker results with auto
multi-worker results after removing timing/process metadata.
- [x] Add CLI options `--cpu-fraction`, `--memory-fraction`,
`--minimum-free-memory`, `--max-workers` and `--fixed-jobs`.
- [x] Estimate fixed model/index/output bytes from config and update growth from
an optional calibration run; apply a conservative margin before reservation.
- [x] Verify identical result ordering and commit `feat: auto-size experiment workers`.

### Task 4: Stress verification and documentation

**Files:**
- Modify: `BUILDING.md`
- Modify: `API.md`
- Modify: `RESEARCH_LOG.md`

- [x] Run fake-provider boundary tests at 0.5, 0.9 and 1.0 fractions.
- [x] Run a real local multi-seed smoke with both fractions at 0.9; record peak
committed memory, average CPU, worker count and failure cleanup.
- [x] Run full CTest and Python compile/tests.
- [x] Document behavior and commit `docs: record dual-constraint scheduler results`.

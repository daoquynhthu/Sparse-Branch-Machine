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


if __name__ == "__main__":
    test_policy_budgets()
    test_snapshot_injection()
    test_linux_meminfo_parser()
    print("Scheduler tests passed")

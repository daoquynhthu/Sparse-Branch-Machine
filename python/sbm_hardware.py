"""Cross-platform hardware snapshots and schedulable resource budgets."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class HardwareSnapshot:
    cpu_count: int
    total_memory_bytes: int
    available_memory_bytes: int
    captured_at: float = 0.0

    def __post_init__(self) -> None:
        if self.cpu_count < 1:
            raise ValueError("cpu_count must be positive")
        if self.total_memory_bytes < 1:
            raise ValueError("total_memory_bytes must be positive")
        if not 0 <= self.available_memory_bytes <= self.total_memory_bytes:
            raise ValueError("available memory must be within total memory")


@dataclass(frozen=True)
class ResourcePolicy:
    cpu_fraction: float = 0.9
    memory_fraction: float = 0.9
    minimum_free_memory_bytes: int = 1 << 30
    max_workers: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.cpu_fraction <= 1.0:
            raise ValueError("cpu_fraction must be in (0, 1]")
        if not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError("memory_fraction must be in (0, 1]")
        if self.minimum_free_memory_bytes < 0:
            raise ValueError("minimum_free_memory_bytes must be non-negative")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be positive")

    def cpu_budget(self, snapshot: HardwareSnapshot) -> int:
        budget = max(1, int(snapshot.cpu_count * self.cpu_fraction))
        if self.max_workers is not None:
            budget = min(budget, self.max_workers)
        return budget

    def dispatchable_memory(self, snapshot: HardwareSnapshot) -> int:
        fraction_limit = int(snapshot.available_memory_bytes * self.memory_fraction)
        reserve_limit = snapshot.available_memory_bytes - self.minimum_free_memory_bytes
        return max(0, min(fraction_limit, reserve_limit))


def parse_linux_meminfo(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].endswith(":"):
            values[fields[0][:-1]] = int(fields[1]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    if total <= 0 or available < 0:
        raise RuntimeError("invalid /proc/meminfo")
    return total, min(available, total)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _system_memory() -> tuple[int, int]:
    if sys.platform == "win32":
        status = _MemoryStatusEx()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return int(status.total_physical), int(status.available_physical)
    if sys.platform.startswith("linux"):
        return parse_linux_meminfo(Path("/proc/meminfo").read_text(encoding="ascii"))
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = page_size * os.sysconf("SC_PHYS_PAGES")
    available = page_size * os.sysconf("SC_AVPHYS_PAGES")
    return int(total), int(available)


def snapshot_hardware(
    cpu_count_provider: Callable[[], int | None] = os.cpu_count,
    memory_provider: Callable[[], tuple[int, int]] = _system_memory,
) -> HardwareSnapshot:
    cpu_count = cpu_count_provider() or 1
    total, available = memory_provider()
    return HardwareSnapshot(
        cpu_count=cpu_count,
        total_memory_bytes=total,
        available_memory_bytes=available,
        captured_at=time.time(),
    )

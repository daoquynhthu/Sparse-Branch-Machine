#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_presets import apply_preset  # noqa: E402


def main() -> None:
    r3 = apply_preset("r3-baseline")
    assert r3["adaptive_topology"] is True
    assert r3["address_lags"] == 1

    upgrade = apply_preset("upgrade-v1")
    assert upgrade["use_momentum"] is True
    assert upgrade["beam_width_min"] == 2
    assert upgrade["beam_width"] == 8
    assert upgrade["max_refinement_rounds"] == 2

    overridden = apply_preset("upgrade-v1", {"beam_width": 12})
    assert overridden["beam_width"] == 12
    assert overridden["use_momentum"] is True

    try:
        apply_preset("nonexistent")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass

    print("preset tests passed")


if __name__ == "__main__":
    main()

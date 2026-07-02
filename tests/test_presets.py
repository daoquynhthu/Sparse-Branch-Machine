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
    assert "beam_width_min" not in upgrade

    adaptive = apply_preset("upgrade-v1-adaptive")
    assert adaptive["use_momentum"] is True
    assert adaptive["beam_width_min"] == 2
    assert adaptive["beam_width"] == 8
    assert adaptive["max_refinement_rounds"] == 2


    shadow = apply_preset("gpaf-shadow-v1")
    assert shadow["gpaf_shadow_observation"] is True
    assert shadow["gpaf_slots"] == 1024
    assert shadow["gpaf_residents_per_slot"] == 4
    assert "gpaf_candidate_retrieval" not in shadow

    retrieval = apply_preset("gpaf-retrieval-v1")
    assert retrieval["gpaf_shadow_observation"] is True
    assert retrieval["gpaf_candidate_retrieval"] is True
    assert retrieval["gpaf_query_keys_per_step"] == 4
    assert retrieval["gpaf_slots"] == 1024

    overridden = apply_preset("upgrade-v1-adaptive", {"beam_width": 12})
    assert overridden["beam_width"] == 12
    assert overridden["use_momentum"] is True

    shadow = apply_preset("gpaf-shadow-v1")
    assert shadow["gpaf_shadow_observation"] is True
    assert shadow["gpaf_slots"] == 1024
    assert shadow["gpaf_residents_per_slot"] == 4
    assert "gpaf_candidate_retrieval" not in shadow

    retrieval = apply_preset("gpaf-retrieval-v1")
    assert retrieval["gpaf_shadow_observation"] is True
    assert retrieval["gpaf_candidate_retrieval"] is True
    assert retrieval["gpaf_query_keys_per_step"] == 4
    assert retrieval["gpaf_slots"] == 1024

    try:
        apply_preset("nonexistent")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass

    print("preset tests passed")


if __name__ == "__main__":
    main()

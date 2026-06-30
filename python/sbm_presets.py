"""Named configuration presets for SPM experiments.

Presets are declarative dictionaries of runtime parameter overrides. They are
applied on top of the library defaults, so adding a new preset never requires
code changes outside this file unless the parameter itself is new.
"""

from __future__ import annotations

from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "r3-baseline": {
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
    },
    "upgrade-v1": {
        # Proven 10M improvement over r3-baseline: -0.111 nats/token.
        # Adam-like momentum on sparse decisions with bias correction.
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
        "use_momentum": True,
    },
    "upgrade-v1-adaptive": {
        # Experimental: adds adaptive beam width and iterative refinement.
        # 10M result is slightly worse than upgrade-v1 but still beats baseline.
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
        "use_momentum": True,
        "beam_width": 8,
        "beam_width_min": 2,
        "confidence_threshold": 0.7,
        "max_refinement_rounds": 2,
        "refinement_confidence_threshold": 0.5,
    },
}


def apply_preset(preset_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a config dictionary for the named preset with optional overrides.

    Later overrides take precedence over preset values.
    """
    if preset_name not in PRESETS:
        raise KeyError(f"unknown preset {preset_name!r}; available: {sorted(PRESETS)}")
    result = dict(PRESETS[preset_name])
    if overrides:
        result.update(overrides)
    return result

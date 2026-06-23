#!/usr/bin/env python3
"""Reproduce the sparse-address-program theory ablation through the shared API.

The script compares three architectures on exactly the same generated token
streams for every seed:

1. adaptive sparse programs with arity <= 2;
2. adaptive singleton lag channels only;
3. fixed singleton channels [1, 2, 4].

It intentionally contains no task-specific parameter search. Hyperparameter
search remains the responsibility of tune.py; this script is a deterministic
architecture comparison and records all raw runs in one JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def compact_run(result: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "eval_cross_entropy": float(result["eval_cross_entropy"]),
        "steps_per_second": float(result["steps_per_second"]),
        "live_nodes": int(result["live_nodes"]),
        "avg_active": float(result["avg_active"]),
        "avg_candidates": float(result["avg_candidates"]),
        "learned_address_programs": result.get("learned_address_programs", []),
        "multiscale_baseline_eval_cross_entropy": float(
            result["multiscale_baseline_eval_cross_entropy"]
        ),
        "oracle_cross_entropy": float(result["oracle_cross_entropy"]),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": runs,
        "mean_eval_cross_entropy": statistics.fmean(
            run["eval_cross_entropy"] for run in runs
        ),
        "mean_steps_per_second": statistics.fmean(
            run["steps_per_second"] for run in runs
        ),
        "median_steps_per_second": statistics.median(
            run["steps_per_second"] for run in runs
        ),
        "mean_live_nodes": statistics.fmean(run["live_nodes"] for run in runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="Path to sbm_api shared library")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("7,11,19"))
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=80_000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--interaction-strength", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default="research_results/theory_alignment_v10_ablation.json",
    )
    args = parser.parse_args()

    if args.sequences < 1 or args.sequence_length < 2 or args.vocab_size < 2:
        parser.error("dataset dimensions must be positive and vocab size >= 2")

    runtime = Runtime(args.library)
    modes: dict[str, dict[str, Any]] = {
        "adaptive_programs": {
            "adaptive_topology": True,
            "address_lags": [1],
            "topology_max_arity": 2,
        },
        "adaptive_singletons": {
            "adaptive_topology": True,
            "address_lags": [1],
            "topology_max_arity": 1,
        },
        "fixed_124": {
            "adaptive_topology": False,
            "address_lags": [1, 2, 4],
        },
    }

    raw: dict[str, list[dict[str, Any]]] = {name: [] for name in modes}
    for seed in args.seeds:
        with runtime.generate_math_token_dataset(
            args.sequences,
            args.sequence_length,
            args.vocab_size,
            seed,
            args.temperature,
            args.interaction_strength,
        ) as dataset:
            for name, values in modes.items():
                with runtime.config({**values, "seed": seed}) as config:
                    result = dataset.run(config, args.warmup, strict_freeze=True)
                raw[name].append(compact_run(result, seed))

    output: dict[str, Any] = {
        "api_version": runtime.version,
        "dataset": {
            "sequences": args.sequences,
            "sequence_length": args.sequence_length,
            "vocab_size": args.vocab_size,
            "warmup": args.warmup,
            "temperature": args.temperature,
            "interaction_strength": args.interaction_strength,
            "seeds": args.seeds,
        },
        "meta_rule": (
            "propose address programs containing at most two history offsets in "
            "deterministic complexity order; accept or retire them using frozen "
            "counterfactual cross-entropy credit"
        ),
    }
    for name, runs in raw.items():
        output[name] = summarize(runs)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: {
                    "mean_eval_cross_entropy": output[name]["mean_eval_cross_entropy"],
                    "median_steps_per_second": output[name]["median_steps_per_second"],
                }
                for name in modes
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

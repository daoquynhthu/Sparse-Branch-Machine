#!/usr/bin/env python3
"""Reproducible v11 ablation for computable addresses and sparse token output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sbm_runtime import Runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=80000)
    parser.add_argument("--output", default="research_results/theory_alignment_v11_ablation.json")
    args = parser.parse_args()

    runtime = Runtime(args.library)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    variants = {
        "sparse_computable": {
            "sparse_token_output": True,
            "topology_enable_delta": True,
        },
        "sparse_tuple_only": {
            "sparse_token_output": True,
            "topology_enable_delta": False,
        },
        "dense_computable": {
            "sparse_token_output": False,
            "topology_enable_delta": True,
        },
    }
    output: dict[str, object] = {
        "api_version": runtime.version,
        "seeds": seeds,
        "variants": {},
    }
    for name, values in variants.items():
        runs = []
        for seed in seeds:
            with runtime.generate_math_token_dataset(
                sequence_count=args.sequences,
                sequence_length=args.sequence_length,
                vocab_size=args.vocab_size,
                seed=seed,
                temperature=0.8,
                interaction_strength=0.2,
            ) as dataset:
                with runtime.config({**values, "seed": seed}) as config:
                    result = dataset.run(config, warmup=args.warmup, strict_freeze=True)
                    runs.append(result)
        output["variants"][name] = {
            "runs": runs,
            "mean_eval_cross_entropy": sum(r["eval_cross_entropy"] for r in runs) / len(runs),
            "mean_steps_per_second": sum(r["steps_per_second"] for r in runs) / len(runs),
            "mean_estimated_bytes": sum(r["estimated_bytes"] for r in runs) / len(runs),
            "mean_sparse_output_entries": sum(r.get("sparse_output_entries", 0) for r in runs) / len(runs),
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

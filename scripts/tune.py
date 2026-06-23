#!/usr/bin/env python3
"""Automatic hyperparameter search over the shared-library API.

The search space is discovered from sbm_parameter_schema_json().  Default mode
uses repeated-seed successive halving: many candidates are evaluated on one
seed, weaker candidates are removed, and survivors are evaluated on additional
seeds.  No recompilation or subprocess-per-trial is required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime, SBMError  # noqa: E402


def parse_assignment(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, value = text.split("=", 1)
    if not name or not value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name, value


def parse_seed_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def cast_override(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if "," in value:
        return [int(item) for item in value.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def sample_parameter(rng: random.Random, descriptor: Mapping[str, Any]) -> Any:
    kind = descriptor["type"]
    minimum = float(descriptor["minimum"])
    maximum = float(descriptor["maximum"])
    scale = descriptor.get("scale", "linear")
    if scale == "log":
        value = math.exp(rng.uniform(math.log(minimum), math.log(maximum)))
    else:
        value = rng.uniform(minimum, maximum)
    if kind in {"uint32", "uint64"}:
        return max(int(minimum), min(int(maximum), int(round(value))))
    return value


def fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def score_result(result: Mapping[str, Any], node_penalty: float) -> float:
    quality = float(result["eval_r2"])
    if node_penalty <= 0.0:
        return quality
    nodes = max(1.0, float(result["live_nodes"]))
    return quality - node_penalty * math.log1p(nodes) / math.log(100_000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="Path to sbm_api shared library")
    parser.add_argument("--trials", type=int, default=27)
    parser.add_argument("--eta", type=int, default=3, help="Successive-halving reduction factor")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=20260623)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("7,11,19"))
    parser.add_argument("--length", type=int, default=120_000)
    parser.add_argument("--warmup", type=int, default=40_000)
    parser.add_argument("--alphabet", type=int, default=64)
    parser.add_argument("--vector-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--noise-std", type=float, default=0.035)
    parser.add_argument("--include", help="Comma-separated parameter names; default uses schema search_default")
    parser.add_argument("--set", action="append", type=parse_assignment, default=[])
    parser.add_argument("--node-penalty", type=float, default=0.0)
    parser.add_argument("--output", default="tuning_results.json")
    parser.add_argument("--best-config", default="best_config.json")
    args = parser.parse_args()

    if args.trials < 1 or args.eta < 2 or args.jobs < 1:
        parser.error("trials/jobs must be positive and eta must be at least 2")

    runtime = Runtime(args.library)
    schema = runtime.parameter_schema()
    descriptors = {item["name"]: item for item in schema["parameters"]}
    if args.include:
        search_names = [name.strip() for name in args.include.split(",") if name.strip()]
    else:
        search_names = [item["name"] for item in schema["parameters"] if item.get("search_default")]
    unknown = [name for name in search_names if name not in descriptors]
    if unknown:
        parser.error(f"unknown search parameters: {', '.join(unknown)}")
    unusable = [name for name in search_names if not descriptors[name].get("tunable")]
    if unusable:
        parser.error(f"parameters are not marked tunable: {', '.join(unusable)}")

    base = {name: cast_override(value) for name, value in args.set}
    rng = random.Random(args.search_seed)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(candidates) < args.trials:
        values = dict(base)
        for name in search_names:
            values[name] = sample_parameter(rng, descriptors[name])
        key = fingerprint(values)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"id": key, "config": values, "runs": {}, "status": "active"})

    datasets = {
        seed: runtime.generate_dataset(
            args.length,
            args.alphabet,
            args.vector_dim,
            args.hidden_dim,
            seed,
            args.noise_std,
        )
        for seed in args.seeds
    }

    def evaluate(candidate: dict[str, Any], seed: int) -> tuple[str, int, dict[str, Any] | None, str | None]:
        try:
            values = dict(candidate["config"])
            values["seed"] = seed
            with runtime.config(values) as config:
                result = datasets[seed].run(config, args.warmup, strict_freeze=True)
            return candidate["id"], seed, result, None
        except Exception as error:  # keep the search running on invalid combinations
            return candidate["id"], seed, None, str(error)

    active = candidates
    stages: list[dict[str, Any]] = []
    for stage_index, seed in enumerate(args.seeds):
        if not active:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(evaluate, candidate, seed) for candidate in active]
            for future in concurrent.futures.as_completed(futures):
                candidate_id, run_seed, result, error = future.result()
                candidate = next(item for item in active if item["id"] == candidate_id)
                candidate["runs"][str(run_seed)] = {"result": result, "error": error}

        for candidate in active:
            valid = [
                item["result"]
                for item in candidate["runs"].values()
                if item["result"] is not None
            ]
            scores = [score_result(result, args.node_penalty) for result in valid]
            candidate["mean_score"] = statistics.fmean(scores) if scores else None
            candidate["mean_eval_r2"] = (
                statistics.fmean(float(result["eval_r2"]) for result in valid)
                if valid
                else None
            )

        score_key = lambda item: item.get("mean_score") if item.get("mean_score") is not None else float("-inf")
        active.sort(key=score_key, reverse=True)
        keep = len(active) if stage_index + 1 == len(args.seeds) else max(1, math.ceil(len(active) / args.eta))
        survivors = active[:keep]
        survivor_ids = {item["id"] for item in survivors}
        for candidate in active:
            if candidate["id"] not in survivor_ids:
                candidate["status"] = f"eliminated_stage_{stage_index + 1}"
        stages.append({
            "stage": stage_index + 1,
            "seed": seed,
            "evaluated": len(active),
            "survivors": [item["id"] for item in survivors],
        })
        checkpoint = {
            "api_version": runtime.version,
            "search_parameters": search_names,
            "base_config": base,
            "stages": stages,
            "candidates": candidates,
            "incomplete": True,
        }
        Path(args.output).write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        active = survivors

    score_key = lambda item: item.get("mean_score") if item.get("mean_score") is not None else float("-inf")
    finalists = list(active) if active else candidates
    best = max(finalists, key=score_key)
    ranked = sorted(candidates, key=score_key, reverse=True)
    best["status"] = "best"
    output = {
        "api_version": runtime.version,
        "search_parameters": search_names,
        "base_config": base,
        "dataset": {
            "length": args.length,
            "warmup": args.warmup,
            "alphabet": args.alphabet,
            "vector_dim": args.vector_dim,
            "hidden_dim": args.hidden_dim,
            "noise_std": args.noise_std,
            "seeds": args.seeds,
        },
        "objective": "mean eval_r2" if args.node_penalty == 0 else "mean eval_r2 minus node penalty",
        "stages": stages,
        "best": {"id": best["id"], "config": best["config"], "mean_eval_r2": best.get("mean_eval_r2")},
        "candidates": ranked,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    Path(args.best_config).write_text(json.dumps(best["config"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["best"], indent=2, sort_keys=True))

    for dataset in datasets.values():
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

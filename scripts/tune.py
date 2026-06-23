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
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime, SBMError  # noqa: E402
from sbm_hardware import ResourcePolicy  # noqa: E402
from sbm_scheduler import (  # noqa: E402
    ResourceScheduler,
    RunEstimate,
    ScheduledRun,
    estimate_worker_memory,
)


def execute_worker_request(request_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    output_path = Path(request["output"])
    try:
        runtime = Runtime(request.get("library"))
        dataset_spec = request["dataset"]
        seed = int(request["seed"])
        if dataset_spec["task"] == "token-ce":
            dataset = runtime.generate_math_token_dataset(
                dataset_spec["sequences"],
                dataset_spec["sequence_length"],
                dataset_spec["vocab_size"],
                seed,
                dataset_spec["math_temperature"],
                dataset_spec["interaction_strength"],
            )
        else:
            dataset = runtime.generate_dataset(
                dataset_spec["length"],
                dataset_spec["alphabet"],
                dataset_spec["vector_dim"],
                dataset_spec["hidden_dim"],
                seed,
                dataset_spec["noise_std"],
            )
        try:
            values = dict(request["config"])
            values["seed"] = seed
            with runtime.config(values) as config:
                result = dataset.run(
                    config,
                    dataset_spec["warmup"],
                    strict_freeze=True,
                )
            payload = {"result": result, "error": None}
        finally:
            dataset.close()
    except Exception as error:
        payload = {"result": None, "error": str(error)}
    output_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return 0


def dataset_spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": args.task,
        "sequences": args.sequences,
        "sequence_length": args.sequence_length,
        "vocab_size": args.vocab_size,
        "math_temperature": args.math_temperature,
        "interaction_strength": args.interaction_strength,
        "length": args.length,
        "warmup": args.warmup,
        "alphabet": args.alphabet,
        "vector_dim": args.vector_dim,
        "hidden_dim": args.hidden_dim,
        "noise_std": args.noise_std,
    }


def dataset_memory_bytes(args: argparse.Namespace) -> int:
    if args.task == "token-ce":
        return args.sequences * args.sequence_length * 4 + (args.sequences + 1) * 8
    return args.length * (4 + args.vector_dim * 4)


def evaluate_isolated(
    active: Sequence[dict[str, Any]],
    seed: int,
    args: argparse.Namespace,
    observed_model_bytes: int,
) -> list[tuple[str, int, dict[str, Any] | None, str | None]]:
    policy = ResourcePolicy(
        cpu_fraction=args.cpu_fraction,
        memory_fraction=args.memory_fraction,
        minimum_free_memory_bytes=args.minimum_free_memory_mib << 20,
        max_workers=args.max_workers,
    )
    estimate_bytes = estimate_worker_memory(
        dataset_bytes=dataset_memory_bytes(args),
        model_bytes=args.initial_model_memory_mib << 20,
        observed_growth_bytes=observed_model_bytes,
        margin=args.memory_estimate_margin,
    )
    dataset_spec = dataset_spec_from_args(args)
    with tempfile.TemporaryDirectory(prefix="sbm-tune-") as directory:
        root = Path(directory)
        scheduled: list[ScheduledRun] = []
        outputs: dict[str, Path] = {}
        for index, candidate in enumerate(active):
            run_id = f"s{seed}-{index}-{candidate['id']}"
            request_path = root / f"{run_id}.request.json"
            output_path = root / f"{run_id}.result.json"
            request = {
                "library": args.library,
                "dataset": dataset_spec,
                "seed": seed,
                "config": candidate["config"],
                "output": str(output_path),
            }
            request_path.write_text(json.dumps(request), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-request", str(request_path)]
            scheduled.append(ScheduledRun(run_id, RunEstimate(1, estimate_bytes), command))
            outputs[run_id] = output_path

        scheduler = ResourceScheduler(policy=policy)
        returncodes = scheduler.run(scheduled)
        completed: list[tuple[str, int, dict[str, Any] | None, str | None]] = []
        for candidate, run in zip(active, scheduled):
            output_path = outputs[run.run_id]
            if returncodes.get(run.run_id) != 0 or not output_path.exists():
                completed.append((candidate["id"], seed, None, "worker process failed"))
                continue
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            completed.append((candidate["id"], seed, payload["result"], payload["error"]))
        return completed


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
    quality = (
        -float(result["eval_cross_entropy"])
        if "eval_cross_entropy" in result
        else float(result["eval_r2"])
    )
    if node_penalty <= 0.0:
        return quality
    nodes = max(1.0, float(result["live_nodes"]))
    return quality - node_penalty * math.log1p(nodes) / math.log(100_000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-request", help=argparse.SUPPRESS)
    parser.add_argument("--library", help="Path to sbm_api shared library")
    parser.add_argument("--trials", type=int, default=27)
    parser.add_argument("--eta", type=int, default=3, help="Successive-halving reduction factor")
    parser.add_argument("--fixed-jobs", type=int, help="Disable auto scheduling and use this thread count")
    parser.add_argument("--jobs", type=int, dest="fixed_jobs", help=argparse.SUPPRESS)
    parser.add_argument("--cpu-fraction", type=float, default=0.9)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=1024)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--initial-model-memory-mib", type=int, default=256)
    parser.add_argument("--memory-estimate-margin", type=float, default=1.35)
    parser.add_argument("--search-seed", type=int, default=20260623)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("7,11,19"))
    parser.add_argument("--task", choices=["token-ce", "vector"], default="token-ce")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--math-temperature", type=float, default=0.8)
    parser.add_argument("--interaction-strength", type=float, default=0.2)
    parser.add_argument("--length", type=int, default=120_000)
    parser.add_argument("--warmup", type=int, default=80_000)
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

    if args.worker_request:
        return execute_worker_request(args.worker_request)

    if args.trials < 1 or args.eta < 2:
        parser.error("trials must be positive and eta must be at least 2")
    if args.fixed_jobs is not None and args.fixed_jobs < 1:
        parser.error("fixed jobs must be positive")
    try:
        ResourcePolicy(
            args.cpu_fraction,
            args.memory_fraction,
            args.minimum_free_memory_mib << 20,
            args.max_workers,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.initial_model_memory_mib < 1 or args.memory_estimate_margin < 1.0:
        parser.error("memory estimate inputs must be positive and margin at least one")

    runtime = Runtime(args.library)
    schema = runtime.parameter_schema()
    descriptors = {item["name"]: item for item in schema["parameters"]}
    if args.include:
        search_names = [name.strip() for name in args.include.split(",") if name.strip()]
    else:
        search_names = [
            item["name"]
            for item in schema["parameters"]
            if item.get("search_default") and args.task in item.get("tasks", [args.task])
        ]
    unknown = [name for name in search_names if name not in descriptors]
    if unknown:
        parser.error(f"unknown search parameters: {', '.join(unknown)}")
    unusable = [name for name in search_names if not descriptors[name].get("tunable")]
    if unusable:
        parser.error(f"parameters are not marked tunable: {', '.join(unusable)}")

    base = {name: cast_override(value) for name, value in args.set}
    rng = random.Random(args.search_seed)
    baseline_id = fingerprint(base)
    candidates: list[dict[str, Any]] = [
        {"id": baseline_id, "config": dict(base), "runs": {}, "status": "active"}
    ]
    seen: set[str] = {baseline_id}
    while len(candidates) < args.trials:
        values = dict(base)
        for name in search_names:
            values[name] = sample_parameter(rng, descriptors[name])
        key = fingerprint(values)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"id": key, "config": values, "runs": {}, "status": "active"})

    if args.fixed_jobs is not None and args.task == "token-ce":
        datasets = {
            seed: runtime.generate_math_token_dataset(
                args.sequences,
                args.sequence_length,
                args.vocab_size,
                seed,
                args.math_temperature,
                args.interaction_strength,
            )
            for seed in args.seeds
        }
    elif args.fixed_jobs is not None:
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
    else:
        datasets = {}

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
    observed_model_bytes = 0
    for stage_index, seed in enumerate(args.seeds):
        if not active:
            break
        if args.fixed_jobs is None:
            stage_results = evaluate_isolated(active, seed, args, observed_model_bytes)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.fixed_jobs) as executor:
                futures = [executor.submit(evaluate, candidate, seed) for candidate in active]
                stage_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        for candidate_id, run_seed, result, error in stage_results:
            candidate = next(item for item in active if item["id"] == candidate_id)
            candidate["runs"][str(run_seed)] = {"result": result, "error": error}
            if result is not None:
                observed_model_bytes = max(
                    observed_model_bytes, int(result.get("estimated_bytes", 0))
                )

        for candidate in active:
            valid = [
                item["result"]
                for item in candidate["runs"].values()
                if item["result"] is not None
            ]
            scores = [score_result(result, args.node_penalty) for result in valid]
            candidate["mean_score"] = statistics.fmean(scores) if scores else None
            if valid and "eval_cross_entropy" in valid[0]:
                candidate["mean_eval_cross_entropy"] = statistics.fmean(
                    float(result["eval_cross_entropy"]) for result in valid
                )
            else:
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
            "task": args.task,
            "length": args.length if args.task == "vector" else None,
            "sequences": args.sequences if args.task == "token-ce" else None,
            "sequence_length": args.sequence_length if args.task == "token-ce" else None,
            "vocab_size": args.vocab_size if args.task == "token-ce" else None,
            "warmup": args.warmup,
            "alphabet": args.alphabet if args.task == "vector" else None,
            "vector_dim": args.vector_dim if args.task == "vector" else None,
            "hidden_dim": args.hidden_dim if args.task == "vector" else None,
            "noise_std": args.noise_std if args.task == "vector" else None,
            "math_temperature": args.math_temperature if args.task == "token-ce" else None,
            "interaction_strength": args.interaction_strength if args.task == "token-ce" else None,
            "seeds": args.seeds,
        },
        "objective": (
            "minimize mean eval_cross_entropy"
            if args.task == "token-ce"
            else "maximize mean eval_r2"
        ),
        "stages": stages,
        "best": {
            "id": best["id"],
            "config": best["config"],
            "mean_eval_cross_entropy": best.get("mean_eval_cross_entropy"),
            "mean_eval_r2": best.get("mean_eval_r2"),
        },
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

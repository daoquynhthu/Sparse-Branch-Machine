#!/usr/bin/env python3
"""Run resumable mapped-corpus token training with model and cursor checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime  # noqa: E402


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sbm-corpus-manifest":
        raise ValueError("unsupported corpus manifest schema")
    train = manifest.get("splits", {}).get("train")
    if not train or not train.get("shards"):
        raise ValueError("manifest has no train shards")
    return manifest


def _train_shards(manifest_path: Path, manifest: dict[str, Any]) -> list[Path]:
    result: list[Path] = []
    for shard in manifest["splits"]["train"]["shards"]:
        shard_path = (manifest_path.parent / shard["path"]).resolve()
        if shard_path.parent != manifest_path.parent:
            raise ValueError("shard path escapes the manifest directory")
        result.append(shard_path)
    return result


def _state_template(
    args: argparse.Namespace,
    manifest_path: Path,
    shard_paths: list[Path],
    config_values: dict[str, Any],
) -> dict[str, Any]:
    model_path = args.checkpoint.with_suffix(args.checkpoint.suffix + ".model.sbc")
    return {
        "schema": "sbm-resumable-corpus-run",
        "manifest": str(manifest_path),
        "output": str(args.output.resolve()),
        "model_checkpoint": str(model_path.resolve()),
        "seed": args.seed,
        "config": config_values,
        "train_shards": [str(path) for path in shard_paths],
        "current_shard": 0,
        "cursor": [0, 0, 0],
        "examples": 0,
        "cross_entropy_sum": 0.0,
        "target_probability_sum": 0.0,
        "top1": 0,
        "top5": 0,
        "last_checkpoint_examples": 0,
        "completed": False,
    }


def _save_state(runtime: Runtime, machine: Any, state: dict[str, Any], path: Path) -> None:
    model_path = Path(state["model_checkpoint"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    machine.save_checkpoint(model_path)
    state["saved_at_unix"] = time.time()
    _atomic_write_json(path, state)


def _result_payload(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    examples = int(state["examples"])
    cross_entropy = (
        float(state["cross_entropy_sum"]) / examples if examples else math.nan
    )
    mean_probability = (
        float(state["target_probability_sum"]) / examples if examples else math.nan
    )
    return {
        "schema": "sbm-resumable-corpus-training-result",
        "manifest": state["manifest"],
        "seed": state["seed"],
        "config": state["config"],
        "checkpoint": str(args.checkpoint.resolve()),
        "train_examples": examples,
        "train_cross_entropy": cross_entropy,
        "train_bits_per_token": cross_entropy / math.log(2.0),
        "train_perplexity": math.exp(min(cross_entropy, 80.0)),
        "train_mean_target_probability": mean_probability,
        "train_top1_accuracy": float(state["top1"]) / examples if examples else 0.0,
        "train_top5_accuracy": float(state["top5"]) / examples if examples else 0.0,
        "completed": bool(state["completed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bucket-bits", type=int, default=14)
    parser.add_argument("--max-train-examples", type=int, required=True)
    parser.add_argument("--checkpoint-every-examples", type=int, default=100000)
    parser.add_argument("--stop-after-examples", type=int)
    parser.add_argument("--adaptive-topology", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    if args.max_train_examples <= 0:
        parser.error("--max-train-examples must be positive")
    if args.checkpoint_every_examples <= 0:
        parser.error("--checkpoint-every-examples must be positive")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output already exists: {args.output}")

    runtime = Runtime(args.library)
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    shard_paths = _train_shards(manifest_path, manifest)
    if not shard_paths:
        raise ValueError("no train shards")

    with runtime.open_token_shard(shard_paths[0], shard_index=0) as first_shard:
        vocabulary = first_shard.vocab_size

    config_values: dict[str, Any] = {
        "seed": args.seed,
        "bucket_bits": args.bucket_bits,
        "token_alphabet": vocabulary,
        "vector_dim": vocabulary,
        "sparse_token_output": True,
        "adaptive_topology": args.adaptive_topology,
    }
    for override in args.overrides:
        if "=" not in override:
            parser.error(f"--set requires NAME=VALUE, got {override!r}")
        name, value = override.split("=", 1)
        config_values[name] = value

    if args.resume:
        state = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if state.get("schema") != "sbm-resumable-corpus-run":
            raise ValueError("unsupported checkpoint schema")
        if state.get("manifest") != str(manifest_path):
            raise ValueError("checkpoint manifest does not match")
        machine = runtime.load_machine_checkpoint(state["model_checkpoint"])
    else:
        with runtime.config(config_values) as config:
            machine = runtime.machine(config)
        state = _state_template(args, manifest_path, shard_paths, config_values)

    try:
        next_checkpoint = (
            int(state["last_checkpoint_examples"]) + args.checkpoint_every_examples
        )
        stop_after = args.stop_after_examples
        current_shard = int(state["current_shard"])
        for shard_position in range(current_shard, len(shard_paths)):
            with runtime.open_token_shard(
                shard_paths[shard_position], shard_index=shard_position
            ) as shard:
                if shard_position == current_shard:
                    shard.seek(tuple(int(value) for value in state["cursor"]))
                while int(state["examples"]) < args.max_train_examples:
                    before = shard.cursor
                    example = shard.next_example()
                    if example is None:
                        break
                    after = shard.cursor
                    if before[2] == 0 or before[1] != after[1]:
                        machine.reset_sequence()
                    stats = machine.step_token(example[0], example[1], learn=True)
                    state["current_shard"] = shard_position
                    state["cursor"] = list(after)
                    state["examples"] = int(state["examples"]) + 1
                    state["cross_entropy_sum"] = (
                        float(state["cross_entropy_sum"]) + stats["cross_entropy"]
                    )
                    state["target_probability_sum"] = (
                        float(state["target_probability_sum"])
                        + stats["target_probability"]
                    )
                    state["top1"] = int(state["top1"]) + int(stats["top1_correct"])
                    state["top5"] = int(state["top5"]) + int(stats["top5_correct"])

                    if int(state["examples"]) >= next_checkpoint:
                        state["last_checkpoint_examples"] = int(state["examples"])
                        _save_state(runtime, machine, state, args.checkpoint)
                        next_checkpoint += args.checkpoint_every_examples

                    if stop_after is not None and int(state["examples"]) >= stop_after:
                        _save_state(runtime, machine, state, args.checkpoint)
                        _atomic_write_json(args.output, _result_payload(args, state))
                        return 0
            state["current_shard"] = shard_position + 1
            state["cursor"] = [shard_position + 1, 0, 0]

        state["completed"] = int(state["examples"]) >= args.max_train_examples
        _save_state(runtime, machine, state, args.checkpoint)
        _atomic_write_json(args.output, _result_payload(args, state))
        return 0
    finally:
        machine.close()


if __name__ == "__main__":
    raise SystemExit(main())

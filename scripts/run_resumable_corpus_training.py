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
    return _split_shards(manifest_path, manifest, "train")


def _split_shards(
    manifest_path: Path,
    manifest: dict[str, Any],
    split_name: str,
) -> list[Path]:
    result: list[Path] = []
    split = manifest.get("splits", {}).get(split_name)
    if not split or not split.get("shards"):
        raise ValueError(f"manifest has no shards for {split_name!r}")
    for shard in split["shards"]:
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
        "eval_split": args.eval_split,
        "eval_shards": [],
        "phase": "train",
        "current_shard": 0,
        "cursor": [0, 0, 0],
        "eval_current_shard": 0,
        "eval_cursor": [0, 0, 0],
        "examples": 0,
        "eval_examples": 0,
        "cross_entropy_sum": 0.0,
        "eval_cross_entropy_sum": 0.0,
        "target_probability_sum": 0.0,
        "eval_target_probability_sum": 0.0,
        "top1": 0,
        "top5": 0,
        "eval_top1": 0,
        "eval_top5": 0,
        "unigram_counts": [],
        "unigram_total": 0,
        "current_table": {},
        "pair_table": {},
        "lag2_table": {},
        "lag4_table": {},
        "eval_unigram_cross_entropy_sum": 0.0,
        "eval_unigram_probability_sum": 0.0,
        "eval_current_cross_entropy_sum": 0.0,
        "eval_current_probability_sum": 0.0,
        "eval_pair_cross_entropy_sum": 0.0,
        "eval_pair_probability_sum": 0.0,
        "eval_interpolated_cross_entropy_sum": 0.0,
        "eval_interpolated_probability_sum": 0.0,
        "seed_only_cross_entropy_sum": 0.0,
        "active_only_cross_entropy_sum": 0.0,
        "content_only_cross_entropy_sum": 0.0,
        "tuple_only_cross_entropy_sum": 0.0,
        "channel_subset_examples": 0,
        "eval_channel_credit": {},
        "eval_program_attribution": {},
        "eval_channel_responsibility": {},
        "train_history": [],
        "eval_history": [],
        "topology_frozen": False,
        "last_checkpoint_examples": 0,
        "completed": False,
    }


def _save_state(runtime: Runtime, machine: Any, state: dict[str, Any], path: Path) -> None:
    model_path = Path(state["model_checkpoint"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    machine.save_checkpoint(model_path)
    state["saved_at_unix"] = time.time()
    _atomic_write_json(path, state)


def _metric(prefix: str, state: dict[str, Any], examples: int) -> dict[str, Any]:
    if prefix == "train":
        ce_key = "cross_entropy_sum"
        probability_key = "target_probability_sum"
    else:
        ce_key = f"{prefix}_cross_entropy_sum"
        probability_key = f"{prefix}_target_probability_sum"
    if not examples:
        return {
            f"{prefix}_cross_entropy": None,
            f"{prefix}_bits_per_token": None,
            f"{prefix}_perplexity": None,
            f"{prefix}_mean_target_probability": None,
        }
    ce = float(state[ce_key]) / examples
    prob = float(state[probability_key]) / examples
    return {
        f"{prefix}_cross_entropy": ce,
        f"{prefix}_bits_per_token": ce / math.log(2.0),
        f"{prefix}_perplexity": math.exp(min(ce, 80.0)),
        f"{prefix}_mean_target_probability": prob,
    }


def _ce_metric(prefix: str, ce_sum: float, examples: int) -> dict[str, Any]:
    if not examples:
        return {
            f"{prefix}_cross_entropy": None,
            f"{prefix}_bits_per_token": None,
            f"{prefix}_perplexity": None,
            f"{prefix}_top1_accuracy": None,
            f"{prefix}_top5_accuracy": None,
            f"{prefix}_mean_target_probability": None,
        }
    ce = ce_sum / examples
    return {
        f"{prefix}_cross_entropy": ce,
        f"{prefix}_bits_per_token": ce / math.log(2.0),
        f"{prefix}_perplexity": math.exp(min(ce, 80.0)),
        f"{prefix}_top1_accuracy": None,
        f"{prefix}_top5_accuracy": None,
        f"{prefix}_mean_target_probability": math.exp(-min(ce, 80.0)),
    }


def _result_payload(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    examples = int(state["examples"])
    eval_examples = int(state["eval_examples"])
    train = _metric("train", state, examples)
    eval_metric = _metric("eval", state, eval_examples)
    summary = state.get("machine_summary", {})
    diagnostics = state.get("machine_diagnostics", {})
    result: dict[str, Any] = {
        "task": "real_corpus_next_token_cross_entropy",
        "objective": "token_cross_entropy",
        "vocab_size": int(state.get("vocab_size", 0)),
        "sequence_count": None,
        "train_examples": examples,
        "eval_examples": eval_examples,
        "address_lags": state["config"].get("address_lags", [1]),
        "learned_address_lags": summary.get("learned_address_lags", []),
        "learned_address_programs": summary.get("learned_address_programs", []),
        "learned_address_operations": summary.get("learned_address_operations", []),
        "learned_channel_credit": summary.get("learned_channel_credit", []),
        "learned_channel_phase": summary.get("learned_channel_phase", []),
        "eval_channel_attribution": _finish_channel_attribution(state, summary),
        "eval_channel_mean_responsibility": _finish_channel_responsibility(state),
        "eval_program_attribution": _finish_program_attribution(state, summary),
        "topology_events": summary.get("topology_events", []),
        "strict_freeze": True,
        "dataset_hash": None,
        "completed": bool(state["completed"]),
    }
    result.update(diagnostics)
    result.update(train)
    result.update(eval_metric)
    result["train_top1_accuracy"] = float(state["top1"]) / examples if examples else 0.0
    result["train_top5_accuracy"] = float(state["top5"]) / examples if examples else 0.0
    result["eval_top1_accuracy"] = (
        float(state["eval_top1"]) / eval_examples if eval_examples else 0.0
    )
    result["eval_top5_accuracy"] = (
        float(state["eval_top5"]) / eval_examples if eval_examples else 0.0
    )
    result.update(_baseline_metrics(state, eval_examples))
    return {
        "schema": "sbm-resumable-corpus-training-result",
        "manifest": state["manifest"],
        "seed": state["seed"],
        "eval_split": state["eval_split"],
        "config": state["config"],
        "checkpoint": str(args.checkpoint.resolve()),
        "train_examples": examples,
        "eval_examples": eval_examples,
        "train_cross_entropy": train["train_cross_entropy"],
        "train_bits_per_token": train["train_bits_per_token"],
        "train_perplexity": train["train_perplexity"],
        "train_mean_target_probability": train["train_mean_target_probability"],
        "train_top1_accuracy": result["train_top1_accuracy"],
        "train_top5_accuracy": result["train_top5_accuracy"],
        "eval_cross_entropy": eval_metric["eval_cross_entropy"],
        "eval_bits_per_token": eval_metric["eval_bits_per_token"],
        "eval_perplexity": eval_metric["eval_perplexity"],
        "eval_mean_target_probability": eval_metric["eval_mean_target_probability"],
        "eval_top1_accuracy": result["eval_top1_accuracy"],
        "eval_top5_accuracy": result["eval_top5_accuracy"],
        "completed": bool(state["completed"]),
        "result": result,
    }


def _finish_channel_attribution(
    state: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    programs = summary.get("learned_address_programs", [])
    ops = summary.get("learned_address_operations", [])
    phases = summary.get("learned_channel_phase", [])
    result = []
    for key, value in sorted(state.get("eval_channel_credit", {}).items()):
        channel = int(key)
        observations = int(value.get("observations", 0))
        result.append({
            "channel": channel,
            "lags": programs[channel] if channel < len(programs) else [],
            "op": ops[channel] if channel < len(ops) else 0,
            "phase": phases[channel] if channel < len(phases) else 0,
            "eval_observations": observations,
            "eval_positive": int(value.get("positive", 0)),
            "eval_documents": 0,
            "eval_positive_documents": 0,
            "eval_credit_sum": float(value.get("credit_sum", 0.0)),
            "eval_mean_credit": (
                float(value.get("credit_sum", 0.0)) / observations
                if observations else 0.0
            ),
            "eval_positive_fraction": (
                float(value.get("positive", 0)) / observations
                if observations else 0.0
            ),
            "eval_positive_document_fraction": 0.0,
        })
    return result


def _finish_channel_responsibility(state: dict[str, Any]) -> list[float]:
    credit = state.get("eval_channel_credit", {})
    responsibility = state.get("eval_channel_responsibility", {})
    result = []
    for key in sorted(credit, key=lambda item: int(item)):
        observations = int(credit[key].get("observations", 0))
        result.append(float(responsibility.get(key, 0.0)) / observations if observations else 0.0)
    return result


def _finish_program_attribution(
    state: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    programs = summary.get("learned_address_programs", [])
    ops = summary.get("learned_address_operations", [])
    result = []
    for key, value in sorted(state.get("eval_program_attribution", {}).items()):
        channel_text, dependency_text = key.split(":", 1)
        channel = int(channel_text)
        observations = int(value.get("observations", 0))
        credit_sum = float(value.get("credit_sum", 0.0))
        description_cost = float(value.get("description_cost", 0.0))
        execution_cost = float(value.get("execution_cost", 0.0))
        result.append({
            "channel": channel,
            "lags": programs[channel] if channel < len(programs) else [],
            "op": ops[channel] if channel < len(ops) else 0,
            "dependency": int(dependency_text),
            "observations": observations,
            "positive": int(value.get("positive", 0)),
            "credit_sum": credit_sum,
            "mean_credit": credit_sum / observations if observations else 0.0,
            "positive_fraction": (
                float(value.get("positive", 0)) / observations
                if observations else 0.0
            ),
            "description_cost": description_cost,
            "execution_cost": execution_cost,
            "structural_value": credit_sum - description_cost,
        })
    return result


def _ensure_baseline_state(state: dict[str, Any], vocabulary: int) -> None:
    if not state.get("unigram_counts"):
        state["unigram_counts"] = [0 for _ in range(vocabulary)]


def _observe_table(table: dict[str, Any], key: int, target: int) -> None:
    row = table.setdefault(str(key), {"total": 0, "counts": {}})
    row["total"] = int(row["total"]) + 1
    counts = row["counts"]
    counts[str(target)] = int(counts.get(str(target), 0)) + 1


def _table_probability(
    table: dict[str, Any],
    key: int,
    target: int,
    fallback: float,
    smoothing: float = 0.5,
) -> float:
    row = table.get(str(key))
    if row is None:
        return fallback
    total = int(row.get("total", 0))
    observed = int(row.get("counts", {}).get(str(target), 0))
    return (observed + smoothing * fallback) / (total + smoothing)


def _history_lag(history: list[int], lag: int) -> int:
    if not history:
        return 0
    if len(history) > lag:
        return history[-1 - lag]
    return history[0]


def _pair_key(left: int, right: int) -> int:
    return (int(left) << 32) | int(right)


def _observe_training_baselines(
    state: dict[str, Any],
    history: list[int],
    current: int,
    target: int,
) -> None:
    counts = state["unigram_counts"]
    counts[target] = int(counts[target]) + 1
    state["unigram_total"] = int(state["unigram_total"]) + 1
    lag1 = _history_lag(history, 1)
    lag2 = _history_lag(history, 2)
    lag4 = _history_lag(history, 4)
    _observe_table(state["current_table"], current, target)
    _observe_table(state["pair_table"], _pair_key(lag1, current), target)
    _observe_table(state["lag2_table"], _pair_key(lag2, current), target)
    _observe_table(state["lag4_table"], _pair_key(lag4, current), target)


def _unigram_probability(state: dict[str, Any], target: int) -> float:
    counts = state["unigram_counts"]
    vocabulary = max(1, len(counts))
    total = int(state.get("unigram_total", 0))
    prior = 0.5
    return (int(counts[target]) + prior) / (total + prior * vocabulary)


def _add_probability_metric(state: dict[str, Any], prefix: str, probability: float) -> None:
    bounded = max(probability, 1e-12)
    state[f"{prefix}_cross_entropy_sum"] = (
        float(state.get(f"{prefix}_cross_entropy_sum", 0.0)) - math.log(bounded)
    )
    state[f"{prefix}_probability_sum"] = (
        float(state.get(f"{prefix}_probability_sum", 0.0)) + bounded
    )


def _baseline_metrics(state: dict[str, Any], eval_examples: int) -> dict[str, Any]:
    result = {}
    for prefix in [
        "unigram_baseline_eval",
        "current_token_baseline_eval",
        "pair_context_baseline_eval",
        "interpolated_multiscale_baseline_eval",
    ]:
        ce_sum = float(state.get(f"{prefix}_cross_entropy_sum", 0.0))
        result.update(_ce_metric(prefix, ce_sum, eval_examples))
    result["multiscale_baseline_eval_cross_entropy"] = result[
        "interpolated_multiscale_baseline_eval_cross_entropy"
    ]
    result["multiscale_baseline_eval_bits_per_token"] = result[
        "interpolated_multiscale_baseline_eval_bits_per_token"
    ]
    result["multiscale_baseline_eval_perplexity"] = result[
        "interpolated_multiscale_baseline_eval_perplexity"
    ]
    result["multiscale_baseline_eval_top1_accuracy"] = None
    result["multiscale_baseline_eval_top5_accuracy"] = None
    result["multiscale_baseline_eval_mean_target_probability"] = result[
        "interpolated_multiscale_baseline_eval_mean_target_probability"
    ]
    subset_examples = int(state.get("channel_subset_examples", 0))
    for prefix, key in [
        ("seed_only_eval", "seed_only_cross_entropy_sum"),
        ("active_channels_only_eval", "active_only_cross_entropy_sum"),
        ("content_channels_only_eval", "content_only_cross_entropy_sum"),
        ("tuple_channels_only_eval", "tuple_only_cross_entropy_sum"),
    ]:
        result.update(_ce_metric(prefix, float(state.get(key, 0.0)), subset_examples))
    return result


def _observe_eval_baselines(
    state: dict[str, Any],
    history: list[int],
    current: int,
    target: int,
) -> None:
    unigram = _unigram_probability(state, target)
    current_probability = _table_probability(
        state["current_table"], current, target, unigram)
    lag1 = _history_lag(history, 1)
    lag2 = _history_lag(history, 2)
    lag4 = _history_lag(history, 4)
    pair_probability = _table_probability(
        state["pair_table"], _pair_key(lag1, current), target, current_probability)
    lag2_probability = _table_probability(
        state["lag2_table"], _pair_key(lag2, current), target, current_probability)
    lag4_probability = _table_probability(
        state["lag4_table"], _pair_key(lag4, current), target, current_probability)
    _add_probability_metric(state, "unigram_baseline_eval", unigram)
    _add_probability_metric(state, "current_token_baseline_eval", current_probability)
    _add_probability_metric(state, "pair_context_baseline_eval", pair_probability)
    _add_probability_metric(
        state,
        "interpolated_multiscale_baseline_eval",
        (pair_probability + lag2_probability + lag4_probability) / 3.0,
    )


def _observe_eval_attribution(state: dict[str, Any], stats: dict[str, Any]) -> None:
    count = int(stats.get("channel_credit_count", 0))
    if count == 0:
        return
    for channel in range(count):
        credit = float(stats["channel_credit"][channel])
        key = str(channel)
        row = state["eval_channel_credit"].setdefault(
            key, {"credit_sum": 0.0, "observations": 0, "positive": 0})
        row["credit_sum"] = float(row["credit_sum"]) + credit
        row["observations"] = int(row["observations"]) + 1
        row["positive"] = int(row["positive"]) + (1 if credit > 0.0 else 0)
        state["eval_channel_responsibility"][key] = (
            float(state["eval_channel_responsibility"].get(key, 0.0))
            + float(stats["channel_responsibility_mass"][channel])
        )
        dependency = int(stats["channel_dependency"][channel])
        program_key = f"{channel}:{dependency}"
        program = state["eval_program_attribution"].setdefault(
            program_key,
            {
                "credit_sum": 0.0,
                "description_cost": 0.0,
                "execution_cost": 0.0,
                "observations": 0,
                "positive": 0,
            },
        )
        program["credit_sum"] = float(program["credit_sum"]) + credit
        program["description_cost"] = (
            float(program["description_cost"])
            + float(stats["channel_description_cost"][channel])
        )
        program["execution_cost"] = (
            float(program["execution_cost"])
            + float(stats["channel_execution_cost"][channel])
        )
        program["observations"] = int(program["observations"]) + 1
        program["positive"] = int(program["positive"]) + (1 if credit > 0.0 else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bucket-bits", type=int, default=14)
    parser.add_argument("--max-train-examples", type=int, required=True)
    parser.add_argument("--max-eval-examples", type=int, required=True)
    parser.add_argument("--checkpoint-every-examples", type=int, default=100000)
    parser.add_argument("--stop-after-examples", type=int)
    parser.add_argument("--stop-after-eval-examples", type=int)
    parser.add_argument("--adaptive-topology", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    if args.max_train_examples <= 0:
        parser.error("--max-train-examples must be positive")
    if args.max_eval_examples <= 0:
        parser.error("--max-eval-examples must be positive")
    if args.checkpoint_every_examples <= 0:
        parser.error("--checkpoint-every-examples must be positive")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output already exists: {args.output}")

    runtime = Runtime(args.library)
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    shard_paths = _train_shards(manifest_path, manifest)
    eval_shard_paths = _split_shards(manifest_path, manifest, args.eval_split)
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
        state["eval_shards"] = [str(path) for path in eval_shard_paths]
        state["vocab_size"] = vocabulary
        _ensure_baseline_state(state, vocabulary)

    try:
        next_checkpoint = (
            int(state["last_checkpoint_examples"]) + args.checkpoint_every_examples
        )
        stop_after = args.stop_after_examples
        if state.get("phase", "train") == "train":
            current_shard = int(state["current_shard"])
            history = [int(value) for value in state.get("train_history", [])]
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
                            history = [int(example[0])]
                        elif not history:
                            history = [int(example[0])]
                        stats = machine.step_token(example[0], example[1], learn=True)
                        _observe_training_baselines(
                            state, history, int(example[0]), int(example[1]))
                        history.append(int(example[1]))
                        state["train_history"] = history[-8:]
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
            if int(state["examples"]) >= args.max_train_examples:
                state["phase"] = "eval"
                state["eval_current_shard"] = 0
                state["eval_cursor"] = [0, 0, 0]
                state["eval_history"] = []

        if state.get("phase") == "eval":
            if not bool(state.get("topology_frozen", False)):
                machine.freeze_topology()
                state["topology_frozen"] = True
            eval_history = [int(value) for value in state.get("eval_history", [])]
            current_eval_shard = int(state["eval_current_shard"])
            for shard_position in range(current_eval_shard, len(eval_shard_paths)):
                with runtime.open_token_shard(
                    eval_shard_paths[shard_position], shard_index=shard_position
                ) as shard:
                    if shard_position == current_eval_shard:
                        shard.seek(tuple(int(value) for value in state["eval_cursor"]))
                    while int(state["eval_examples"]) < args.max_eval_examples:
                        before = shard.cursor
                        example = shard.next_example()
                        if example is None:
                            break
                        after = shard.cursor
                        if before[2] == 0 or before[1] != after[1]:
                            machine.reset_sequence()
                            eval_history = [int(example[0])]
                        elif not eval_history:
                            eval_history = [int(example[0])]
                        stats = machine.step_token(example[0], example[1], learn=False)
                        _observe_eval_baselines(
                            state, eval_history, int(example[0]), int(example[1]))
                        _observe_eval_attribution(state, stats)
                        if stats["channel_subset_available"]:
                            state["channel_subset_examples"] = (
                                int(state["channel_subset_examples"]) + 1
                            )
                            state["seed_only_cross_entropy_sum"] = (
                                float(state["seed_only_cross_entropy_sum"])
                                + stats["seed_only_cross_entropy"]
                            )
                            state["active_only_cross_entropy_sum"] = (
                                float(state["active_only_cross_entropy_sum"])
                                + stats["active_only_cross_entropy"]
                            )
                            state["content_only_cross_entropy_sum"] = (
                                float(state["content_only_cross_entropy_sum"])
                                + stats["content_only_cross_entropy"]
                            )
                            state["tuple_only_cross_entropy_sum"] = (
                                float(state["tuple_only_cross_entropy_sum"])
                                + stats["tuple_only_cross_entropy"]
                            )
                        eval_history.append(int(example[1]))
                        state["eval_history"] = eval_history[-8:]
                        state["eval_current_shard"] = shard_position
                        state["eval_cursor"] = list(after)
                        state["eval_examples"] = int(state["eval_examples"]) + 1
                        state["eval_cross_entropy_sum"] = (
                            float(state["eval_cross_entropy_sum"]) + stats["cross_entropy"]
                        )
                        state["eval_target_probability_sum"] = (
                            float(state["eval_target_probability_sum"])
                            + stats["target_probability"]
                        )
                        state["eval_top1"] = (
                            int(state["eval_top1"]) + int(stats["top1_correct"])
                        )
                        state["eval_top5"] = (
                            int(state["eval_top5"]) + int(stats["top5_correct"])
                        )
                        if (
                            args.stop_after_eval_examples is not None
                            and int(state["eval_examples"]) >= args.stop_after_eval_examples
                        ):
                            state["machine_summary"] = machine.summary()
                            state["machine_diagnostics"] = machine.diagnostics()
                            _save_state(runtime, machine, state, args.checkpoint)
                            _atomic_write_json(args.output, _result_payload(args, state))
                            return 0
                state["eval_current_shard"] = shard_position + 1
                state["eval_cursor"] = [shard_position + 1, 0, 0]

        state["completed"] = (
            int(state["examples"]) >= args.max_train_examples
            and int(state["eval_examples"]) >= args.max_eval_examples
        )
        state["machine_summary"] = machine.summary()
        state["machine_diagnostics"] = machine.diagnostics()
        _save_state(runtime, machine, state, args.checkpoint)
        _atomic_write_json(args.output, _result_payload(args, state))
        return 0
    finally:
        machine.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime, SBMError  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
args = parser.parse_args()

runtime = Runtime(args.library)
assert runtime.version.startswith("9.")
schema = runtime.parameter_schema()
assert any(item["name"] == "edge_score_weight" for item in schema["parameters"])
assert any(item["name"] == "classification_learning_rate" for item in schema["parameters"])
assert any(item["name"] == "output_tree_seed" for item in schema["parameters"])
assert any(
    item["name"] == "topology_accept_uses_structural_value"
    for item in schema["parameters"]
)

with runtime.config() as registry_config:
    for descriptor in schema["parameters"]:
        registry_config.set(descriptor["name"], descriptor["default"])
with runtime.config({"output_tree_seed": 31}) as output_seed_config:
    assert output_seed_config.as_dict()["output_tree_seed"] == 31
with runtime.config({"topology_accept_uses_structural_value": True}) as value_config:
    assert value_config.as_dict()["topology_accept_uses_structural_value"] is True

with runtime.generate_dataset(8000, 32, 16, 20, 9) as dataset:
    assert dataset.kind == "vector_regression"
    with runtime.config({"seed": 9, "exact_region_mass": 0.88}) as config:
        result = dataset.run(config, warmup=3000)
assert result["strict_freeze"] is True
assert math.isfinite(result["eval_r2"])

with runtime.generate_math_token_dataset(8, 512, 16, 13, 0.8, 0.2) as dataset:
    assert dataset.kind == "token_cross_entropy"
    assert dataset.sequence_count == 8
    assert dataset.example_count == 4088
    with runtime.config({"seed": 13, "bucket_bits": 8}) as config:
        token_result = dataset.run(config, warmup=2500)
assert math.isfinite(token_result["eval_cross_entropy"])
assert token_result["objective"] == "token_cross_entropy"
assert token_result["unigram_baseline_eval_top1_accuracy"] is None
assert math.isfinite(token_result["interpolated_multiscale_baseline_eval_cross_entropy"])
assert token_result["baseline_elapsed_seconds"] >= 0.0

with tempfile.TemporaryDirectory() as directory:
    checkpoint_path = Path(directory) / "machine.sbc"
    with runtime.config({"seed": 13, "bucket_bits": 8, "vector_dim": 16}) as config:
        left = runtime.machine(config)
        source = runtime.machine(config)
        try:
            for step in range(128):
                assert left.step_token(step % 16, (step + 1) % 16)["live_nodes"] > 0
                source.step_token(step % 16, (step + 1) % 16)
            source.save_checkpoint(checkpoint_path)
            resumed = runtime.load_machine_checkpoint(checkpoint_path)
            try:
                for step in range(128, 192):
                    left_stats = left.step_token(step % 16, (step + 1) % 16)
                    resumed_stats = resumed.step_token(step % 16, (step + 1) % 16)
                    assert left_stats["predicted_token"] == resumed_stats["predicted_token"]
                    assert left_stats["live_nodes"] == resumed_stats["live_nodes"]
                    assert math.isclose(
                        left_stats["cross_entropy"],
                        resumed_stats["cross_entropy"],
                        abs_tol=1e-6,
                )
                resumed.freeze_topology()
                assert resumed.retire_channel(0, "RecoverableRetire") is False
                assert resumed.restore_channel(0) is False
                assert resumed.restore_dependency_closure(0) == 0
                eval_stats = resumed.step_token(1, 3, learn=False)
                assert "channel_credit" in eval_stats
                assert "channel_binding_kind" in eval_stats
                assert "channel_input_state" in eval_stats
                assert "channel_output_state" in eval_stats
                assert "channel_required_dependency_binding" in eval_stats
                assert "channel_binding_distance" in eval_stats
                assert "channel_binding_pattern_span" in eval_stats
                assert "channel_binding_key" in eval_stats
                assert "channel_call_key" in eval_stats
                assert "channel_dependency_binding_key" in eval_stats
                assert "channel_subset_available" in eval_stats
                assert resumed.diagnostics()["steps"] == 193
                assert "dependency_blocked_channels" in resumed.diagnostics()
                summary = resumed.summary()
                assert "learned_address_programs" in summary
                assert "learned_channel_phase" in summary
                assert "learned_channel_effective_enabled" in summary
            finally:
                resumed.close()
        finally:
            left.close()
            source.close()

with runtime.token_dataset_from_ids([1, 2, 3, 1, 2, 4], 8, [0, 3, 6]) as external:
    assert external.sequence_count == 2
    assert external.example_count == 4
    with tempfile.TemporaryDirectory() as directory:
        shard_path = Path(directory) / "python_api.sbt"
        external.write_token_shard(shard_path)
        with runtime.open_token_shard(shard_path, shard_index=5) as shard:
            assert shard.vocab_size == 8
            assert shard.token_count == 6
            assert shard.sequence_count == 2
            assert shard.next_example() == (1, 2)
            resume = shard.cursor
            assert shard.next_example() == (2, 3)
            shard.seek((5, 0, 2**64 - 1))
            try:
                shard.next_example()
                raise AssertionError("malformed cursor was accepted")
            except SBMError as error:
                assert "token_offset" in str(error)
            assert shard.cursor == (5, 0, 2**64 - 1)
            shard.seek((5, 2, 0))
            assert shard.next_example() is None
            with runtime.config({"bucket_bits": 6}) as shard_config:
                shard_result = shard.run(shard_config, warmup=2)
            assert shard_result["task"] == "real_corpus_next_token_cross_entropy"
            assert shard_result["eval_examples"] == 2
        with runtime.open_token_shard(shard_path, shard_index=5) as replay:
            replay.seek(resume)
            assert replay.next_example() == (2, 3)
        with runtime.token_corpus() as corpus:
            corpus.add_shard(shard_path, "train", 0)
            corpus.add_shard(shard_path, "eval", 1)
            with runtime.config({"bucket_bits": 6}) as corpus_config:
                corpus_result = corpus.run(corpus_config)
            assert corpus_result["train_examples"] == 4
            assert corpus_result["eval_examples"] == 4
            with runtime.config({"bucket_bits": 6}) as limited_config:
                limited_result = corpus.run(
                    limited_config,
                    max_train_examples=2,
                    max_eval_examples=2,
                )
            assert limited_result["train_examples"] == 2
            assert limited_result["eval_examples"] == 2

print("Python API tests passed")

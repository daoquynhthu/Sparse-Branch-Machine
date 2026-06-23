#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
args = parser.parse_args()

runtime = Runtime(args.library)
assert runtime.version.startswith("4.")
schema = runtime.parameter_schema()
assert any(item["name"] == "edge_score_weight" for item in schema["parameters"])
assert any(item["name"] == "classification_learning_rate" for item in schema["parameters"])

with runtime.config() as registry_config:
    for descriptor in schema["parameters"]:
        registry_config.set(descriptor["name"], descriptor["default"])

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
        with runtime.open_token_shard(shard_path, shard_index=5) as replay:
            replay.seek(resume)
            assert replay.next_example() == (2, 3)

print("Python API tests passed")

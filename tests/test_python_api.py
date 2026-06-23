#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
args = parser.parse_args()

runtime = Runtime(args.library)
schema = runtime.parameter_schema()
assert any(item["name"] == "edge_score_weight" for item in schema["parameters"])

with runtime.config() as registry_config:
    for descriptor in schema["parameters"]:
        registry_config.set(descriptor["name"], descriptor["default"])

with runtime.generate_dataset(8000, 32, 16, 20, 9) as dataset:
    with runtime.config({"seed": 9, "exact_region_mass": 0.88}) as config:
        result = dataset.run(config, warmup=3000)

assert result["strict_freeze"] is True
assert math.isfinite(result["eval_r2"])
assert result["dataset_hash"] != 0
print("Python API tests passed")

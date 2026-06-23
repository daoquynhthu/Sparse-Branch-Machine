import argparse
import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_naime_dataset.py"
SPEC = importlib.util.spec_from_file_location("convert_naime_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main():
    rows = [[10, 11, 12], [20, 21]]
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        first = Path(first_dir) / "fixture.sbt"
        second = Path(second_dir) / "fixture.sbt"
        first_info = MODULE.write_shard(first, rows, 32)
        second_info = MODULE.write_shard(second, rows, 32)
        assert first.read_bytes() == second.read_bytes()
        assert first_info == second_info
        assert first_info["tokens"] == 5
        assert first_info["sequences"] == 2
        assert first_info["examples"] == 3
        try:
            MODULE.write_shard(first, rows, 32)
        except FileExistsError:
            pass
        else:
            raise AssertionError("writer overwrote an existing shard")

    with tempfile.TemporaryDirectory() as output_dir:
        shards = MODULE.convert_split(
            rows * 3, Path(output_dir), "train", 32, shard_tokens=5, max_tokens=10
        )
        assert len(shards) == 2
        assert sum(item["tokens"] for item in shards) == 10
        assert sum(item["sequences"] for item in shards) == 4

    limits = argparse.Namespace(
        max_tokens_per_split=10,
        max_train_tokens=20,
        max_validation_tokens=30,
    )
    assert MODULE.split_token_limit(limits, "train") == 20
    assert MODULE.split_token_limit(limits, "validation") == 30
    assert MODULE.split_token_limit(limits, "test") == 10


if __name__ == "__main__":
    main()

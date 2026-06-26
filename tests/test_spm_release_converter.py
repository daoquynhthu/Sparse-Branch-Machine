import importlib.util
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_spm_release_views.py"
SPEC = importlib.util.spec_from_file_location("convert_spm_release_views", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_view(path: Path, rows: list[list[int]], split: str) -> None:
    table = pa.table(
        {
            "doc_id": pa.array(
                [bytes([index + 1]) * 32 for index in range(len(rows))],
                type=pa.binary(32),
            ),
            "normalized_sha256": pa.array(
                [bytes([index + 11]) * 32 for index in range(len(rows))],
                type=pa.binary(32),
            ),
            "split": pa.array([split] * len(rows)),
            "rank": pa.array(
                [index.to_bytes(16, "big") for index in range(len(rows))],
                type=pa.binary(16),
            ),
            "token_ids": pa.array(rows, type=pa.list_(pa.uint32())),
            "token_count": pa.array([len(row) for row in rows], type=pa.uint32()),
        }
    )
    pq.write_table(table, path)


def main():
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        views = root / "views"
        tokenizer = root / "tokenizer"
        output = root / "out"
        reports = root / "reports"
        views.mkdir()
        tokenizer.mkdir()
        reports.mkdir()
        _write_view(views / "train-10000000.parquet", [[1, 2, 3], [4, 5]], "train")
        _write_view(views / "validation.parquet", [[6, 7, 8]], "validation")
        _write_view(views / "test.parquet", [[9, 10]], "test")
        (views / "views-manifest.json").write_text(
            json.dumps({"train_documents": 2, "budgets": []}), encoding="utf-8"
        )
        (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
        (tokenizer / "tokenizer-manifest.json").write_text(
            json.dumps({"train_documents": 2}), encoding="utf-8"
        )
        report = reports / "validation.json"
        report.write_text(json.dumps({"ok": True, "issues": []}), encoding="utf-8")

        shards = MODULE.convert_view(
            views / "train-10000000.parquet",
            output,
            "train",
            vocab_size=32,
            shard_tokens=8,
            max_tokens=None,
        )
        assert len(shards) == 1
        assert shards[0]["tokens"] == 5
        assert shards[0]["sequences"] == 2

        args = [
            "--views",
            str(views),
            "--tokenizer",
            str(tokenizer),
            "--validation-report",
            str(report),
            "--output",
            str(root / "manifest_out"),
            "--vocab-size",
            "32",
        ]
        old_argv = __import__("sys").argv
        try:
            __import__("sys").argv = [str(SCRIPT), *args]
            MODULE.main()
        finally:
            __import__("sys").argv = old_argv
        manifest = json.loads((root / "manifest_out" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "admissible_document_level"
        assert manifest["splits"]["train"]["tokens"] == 5
        assert manifest["splits"]["validation"]["tokens"] == 3
        assert manifest["splits"]["test"]["tokens"] == 2


if __name__ == "__main__":
    main()

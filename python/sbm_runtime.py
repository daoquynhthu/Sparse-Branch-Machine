"""ctypes binding for the stable Sparse Branch Machine C API.

The wrapper intentionally depends only on the Python standard library.  It keeps
all model parameters in the shared library's registry, so Python scripts do not
need to mirror C++ Config fields.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


class SBMError(RuntimeError):
    pass


class _TokenShardCursor(ctypes.Structure):
    _fields_ = [
        ("shard_index", ctypes.c_uint64),
        ("sequence_index", ctypes.c_uint64),
        ("token_offset", ctypes.c_uint64),
    ]


class _StepStats(ctypes.Structure):
    _fields_ = [
        ("cross_entropy", ctypes.c_float),
        ("target_probability", ctypes.c_float),
        ("active_nodes", ctypes.c_uint32),
        ("candidates_examined", ctypes.c_uint32),
        ("live_nodes", ctypes.c_uint32),
        ("predicted_token", ctypes.c_uint32),
        ("top1_correct", ctypes.c_int),
        ("top5_correct", ctypes.c_int),
        ("ranking_available", ctypes.c_int),
        ("channel_credit_count", ctypes.c_uint8),
        ("channel_credit", ctypes.c_float * 8),
        ("channel_responsibility_mass", ctypes.c_float * 8),
        ("channel_dependency", ctypes.c_uint32 * 8),
        ("channel_parent_channel", ctypes.c_uint8 * 8),
        ("channel_dependency_channel", ctypes.c_uint8 * 8),
        ("channel_input_state", ctypes.c_uint8 * 8),
        ("channel_output_state", ctypes.c_uint8 * 8),
        ("channel_required_dependency_binding", ctypes.c_uint8 * 8),
        ("channel_caller_removed_credit", ctypes.c_float * 8),
        ("channel_dependency_retained_credit", ctypes.c_float * 8),
        ("channel_dependency_removed_credit", ctypes.c_float * 8),
        ("channel_binding_kind", ctypes.c_uint8 * 8),
        ("channel_binding_matched", ctypes.c_uint8 * 8),
        ("channel_binding_current_token", ctypes.c_uint32 * 8),
        ("channel_binding_matched_token", ctypes.c_uint32 * 8),
        ("channel_binding_successor", ctypes.c_uint32 * 8),
        ("channel_binding_distance", ctypes.c_uint32 * 8),
        ("channel_binding_pattern_span", ctypes.c_uint32 * 8),
        ("channel_binding_key", ctypes.c_uint64 * 8),
        ("channel_dependency_binding_key", ctypes.c_uint64 * 8),
        ("channel_call_key", ctypes.c_uint64 * 8),
        ("channel_description_cost", ctypes.c_float * 8),
        ("channel_execution_cost", ctypes.c_float * 8),
        ("channel_subset_available", ctypes.c_int),
        ("seed_only_cross_entropy", ctypes.c_float),
        ("active_only_cross_entropy", ctypes.c_float),
        ("content_only_cross_entropy", ctypes.c_float),
        ("tuple_only_cross_entropy", ctypes.c_float),
    ]


def _candidate_library_names() -> list[str]:
    system = platform.system()
    if system == "Windows":
        return ["sbm_api.dll", "libsbm_api.dll"]
    if system == "Darwin":
        return ["libsbm_api.dylib"]
    return ["libsbm_api.so"]


def find_library(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    env = os.environ.get("SBM_LIBRARY")
    if env:
        path = Path(env).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    root = Path(__file__).resolve().parents[1]
    roots = [
        root / "lib",
        root / "bin",
        root / "build",
        root / "build" / "Release",
        root / "build-shared",
        root / "build-shared" / "Release",
        root / "build_modular",
        root / "build_modular" / "Release",
    ]
    for directory in roots:
        for name in _candidate_library_names():
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()

    searched = ", ".join(str(path) for path in roots)
    raise FileNotFoundError(
        f"Could not find sbm_api shared library. Set SBM_LIBRARY. Searched: {searched}"
    )


class Runtime:
    def __init__(self, library: Optional[str] = None):
        self.path = find_library(library)
        self._dll_directories = []
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            search_directories = [self.path.parent]
            for entry in os.environ.get("PATH", "").split(os.pathsep):
                directory = Path(entry) if entry else None
                if directory is None or not directory.is_dir():
                    continue
                has_libstdcpp = (directory / "libstdc++-6.dll").is_file()
                has_libgcc = any(directory.glob("libgcc_s_*-1.dll"))
                if has_libstdcpp and has_libgcc:
                    search_directories.append(directory)
                    break
            seen = set()
            for directory in search_directories:
                try:
                    resolved = directory.expanduser().resolve()
                except OSError:
                    continue
                key = os.path.normcase(str(resolved))
                if key in seen or not resolved.is_dir():
                    continue
                seen.add(key)
                try:
                    self._dll_directories.append(os.add_dll_directory(str(resolved)))
                except OSError:
                    continue
        self.lib = ctypes.CDLL(str(self.path))
        self._bind()

    def _bind(self) -> None:
        lib = self.lib
        lib.sbm_api_version.restype = ctypes.c_uint32
        lib.sbm_api_version_string.restype = ctypes.c_char_p
        lib.sbm_last_error.restype = ctypes.c_char_p
        lib.sbm_parameter_schema_json.restype = ctypes.c_char_p

        lib.sbm_config_create.restype = ctypes.c_void_p
        lib.sbm_config_clone.argtypes = [ctypes.c_void_p]
        lib.sbm_config_clone.restype = ctypes.c_void_p
        lib.sbm_config_destroy.argtypes = [ctypes.c_void_p]
        lib.sbm_config_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.sbm_config_set.restype = ctypes.c_int
        lib.sbm_config_get_json.argtypes = [ctypes.c_void_p]
        lib.sbm_config_get_json.restype = ctypes.c_void_p

        lib.sbm_machine_create.argtypes = [ctypes.c_void_p]
        lib.sbm_machine_create.restype = ctypes.c_void_p
        lib.sbm_machine_load_checkpoint.argtypes = [ctypes.c_char_p]
        lib.sbm_machine_load_checkpoint.restype = ctypes.c_void_p
        lib.sbm_machine_save_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.sbm_machine_save_checkpoint.restype = ctypes.c_int
        lib.sbm_machine_destroy.argtypes = [ctypes.c_void_p]
        lib.sbm_machine_step_token.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(_StepStats),
        ]
        lib.sbm_machine_step_token.restype = ctypes.c_int
        lib.sbm_machine_reset_sequence.argtypes = [ctypes.c_void_p]
        lib.sbm_machine_freeze_topology.argtypes = [ctypes.c_void_p]
        lib.sbm_machine_diagnostics_json.argtypes = [ctypes.c_void_p]
        lib.sbm_machine_diagnostics_json.restype = ctypes.c_void_p
        lib.sbm_machine_summary_json.argtypes = [ctypes.c_void_p]
        lib.sbm_machine_summary_json.restype = ctypes.c_void_p

        lib.sbm_dataset_generate.argtypes = [
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_float,
        ]
        lib.sbm_dataset_generate.restype = ctypes.c_void_p
        lib.sbm_token_dataset_generate_math.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.sbm_token_dataset_generate_math.restype = ctypes.c_void_p
        lib.sbm_token_dataset_from_ids.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        lib.sbm_token_dataset_from_ids.restype = ctypes.c_void_p
        lib.sbm_dataset_load.argtypes = [ctypes.c_char_p]
        lib.sbm_dataset_load.restype = ctypes.c_void_p
        lib.sbm_dataset_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.sbm_dataset_save.restype = ctypes.c_int
        lib.sbm_dataset_destroy.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_kind_of.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_kind_of.restype = ctypes.c_uint32
        lib.sbm_dataset_hash.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_hash.restype = ctypes.c_uint64
        lib.sbm_dataset_length.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_length.restype = ctypes.c_size_t
        lib.sbm_dataset_vector_dim.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_vector_dim.restype = ctypes.c_uint32
        lib.sbm_dataset_alphabet.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_alphabet.restype = ctypes.c_uint32
        lib.sbm_dataset_vocab_size.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_vocab_size.restype = ctypes.c_uint32
        lib.sbm_dataset_sequence_count.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_sequence_count.restype = ctypes.c_size_t
        lib.sbm_dataset_example_count.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_example_count.restype = ctypes.c_size_t

        lib.sbm_token_shard_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.sbm_token_shard_write.restype = ctypes.c_int
        lib.sbm_token_shard_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.sbm_token_shard_open.restype = ctypes.c_void_p
        lib.sbm_token_shard_destroy.argtypes = [ctypes.c_void_p]
        lib.sbm_token_shard_vocab_size.argtypes = [ctypes.c_void_p]
        lib.sbm_token_shard_vocab_size.restype = ctypes.c_uint32
        lib.sbm_token_shard_token_count.argtypes = [ctypes.c_void_p]
        lib.sbm_token_shard_token_count.restype = ctypes.c_uint64
        lib.sbm_token_shard_sequence_count.argtypes = [ctypes.c_void_p]
        lib.sbm_token_shard_sequence_count.restype = ctypes.c_uint64
        lib.sbm_token_shard_dataset_hash.argtypes = [ctypes.c_void_p]
        lib.sbm_token_shard_dataset_hash.restype = ctypes.c_uint64
        lib.sbm_token_shard_next.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_TokenShardCursor),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.sbm_token_shard_next.restype = ctypes.c_int
        lib.sbm_run_token_shard_experiment_json.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.sbm_run_token_shard_experiment_json.restype = ctypes.c_void_p
        lib.sbm_token_corpus_create.restype = ctypes.c_void_p
        lib.sbm_token_corpus_destroy.argtypes = [ctypes.c_void_p]
        lib.sbm_token_corpus_add_shard.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.sbm_token_corpus_add_shard.restype = ctypes.c_int
        lib.sbm_run_token_corpus_experiment_json.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.sbm_run_token_corpus_experiment_json.restype = ctypes.c_void_p
        lib.sbm_run_token_corpus_experiment_limited_json.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.sbm_run_token_corpus_experiment_limited_json.restype = ctypes.c_void_p

        lib.sbm_run_experiment_json.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.sbm_run_experiment_json.restype = ctypes.c_void_p
        lib.sbm_string_free.argtypes = [ctypes.c_void_p]

    def _error(self, action: str) -> SBMError:
        raw = self.lib.sbm_last_error()
        detail = raw.decode("utf-8", errors="replace") if raw else "unknown error"
        return SBMError(f"{action}: {detail}")

    def _take_json(self, pointer: int, action: str) -> Dict[str, Any]:
        if not pointer:
            raise self._error(action)
        try:
            raw = ctypes.string_at(pointer).decode("utf-8")
            return json.loads(raw)
        finally:
            self.lib.sbm_string_free(pointer)

    @property
    def version(self) -> str:
        return self.lib.sbm_api_version_string().decode("ascii")

    def parameter_schema(self) -> Dict[str, Any]:
        return json.loads(self.lib.sbm_parameter_schema_json().decode("utf-8"))

    def config(self, values: Optional[Mapping[str, Any]] = None) -> "Config":
        result = Config(self)
        if values:
            result.update(values)
        return result

    def generate_dataset(
        self,
        length: int = 120_000,
        alphabet: int = 64,
        vector_dim: int = 16,
        hidden_dim: int = 24,
        seed: int = 7,
        noise_std: float = 0.035,
    ) -> "Dataset":
        pointer = self.lib.sbm_dataset_generate(
            length, alphabet, vector_dim, hidden_dim, seed, noise_std
        )
        if not pointer:
            raise self._error("generate dataset")
        return Dataset(self, pointer)

    def generate_math_token_dataset(
        self,
        sequence_count: int = 64,
        sequence_length: int = 2048,
        vocab_size: int = 128,
        seed: int = 7,
        temperature: float = 1.0,
        interaction_strength: float = 0.35,
    ) -> "Dataset":
        pointer = self.lib.sbm_token_dataset_generate_math(
            sequence_count,
            sequence_length,
            vocab_size,
            seed,
            temperature,
            interaction_strength,
        )
        if not pointer:
            raise self._error("generate mathematical token dataset")
        return Dataset(self, pointer)

    def token_dataset_from_ids(
        self,
        tokens: Iterable[int],
        vocab_size: int,
        sequence_offsets: Optional[Iterable[int]] = None,
    ) -> "Dataset":
        token_values = [int(token) for token in tokens]
        if not token_values:
            raise ValueError("tokens must be non-empty")
        token_array = (ctypes.c_uint32 * len(token_values))(*token_values)
        if sequence_offsets is None:
            offset_pointer = None
            offset_count = 0
        else:
            offset_values = [int(offset) for offset in sequence_offsets]
            offset_array = (ctypes.c_uint64 * len(offset_values))(*offset_values)
            offset_pointer = offset_array
            offset_count = len(offset_values)
        pointer = self.lib.sbm_token_dataset_from_ids(
            token_array,
            len(token_values),
            vocab_size,
            offset_pointer,
            offset_count,
        )
        if not pointer:
            raise self._error("create token dataset from IDs")
        return Dataset(self, pointer)

    def load_dataset(self, path: str | os.PathLike[str]) -> "Dataset":
        pointer = self.lib.sbm_dataset_load(os.fsencode(path))
        if not pointer:
            raise self._error("load dataset")
        return Dataset(self, pointer)

    def open_token_shard(
        self,
        path: str | os.PathLike[str],
        shard_index: int = 0,
        verify_payload: bool = True,
    ) -> "TokenShard":
        pointer = self.lib.sbm_token_shard_open(
            os.fsencode(path), shard_index, int(verify_payload)
        )
        if not pointer:
            raise self._error("open token shard")
        return TokenShard(self, pointer, shard_index)

    def token_corpus(self) -> "TokenCorpus":
        return TokenCorpus(self)

    def machine(self, config: "Config") -> "Machine":
        pointer = self.lib.sbm_machine_create(config.pointer)
        if not pointer:
            raise self._error("create machine")
        return Machine(self, pointer)

    def load_machine_checkpoint(self, path: str | os.PathLike[str]) -> "Machine":
        pointer = self.lib.sbm_machine_load_checkpoint(os.fsencode(path))
        if not pointer:
            raise self._error("load machine checkpoint")
        return Machine(self, pointer)

    def open_token_corpus(
        self,
        manifest_path: str | os.PathLike[str],
        eval_split: str = "validation",
        verify_payload: bool = True,
    ) -> "TokenCorpus":
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "sbm-corpus-manifest":
            raise ValueError("unsupported corpus manifest schema")
        corpus = self.token_corpus()
        try:
            for split_name, split_kind in (("train", "train"), (eval_split, "eval")):
                split = manifest.get("splits", {}).get(split_name)
                if not split or not split.get("shards"):
                    raise ValueError(f"manifest has no shards for {split_name!r}")
                for index, shard in enumerate(split["shards"]):
                    shard_path = (path.parent / shard["path"]).resolve()
                    if shard_path.parent != path.parent:
                        raise ValueError("shard path escapes the manifest directory")
                    corpus.add_shard(
                        shard_path, split_kind, index, verify_payload=verify_payload
                    )
            return corpus
        except BaseException:
            corpus.close()
            raise


class Config:
    def __init__(self, runtime: Runtime, pointer: Optional[int] = None):
        self.runtime = runtime
        self.pointer = pointer or runtime.lib.sbm_config_create()
        if not self.pointer:
            raise runtime._error("create config")

    def close(self) -> None:
        if self.pointer:
            self.runtime.lib.sbm_config_destroy(self.pointer)
            self.pointer = None

    def __enter__(self) -> "Config":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _encode_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value)
        return str(value)

    def set(self, name: str, value: Any) -> "Config":
        status = self.runtime.lib.sbm_config_set(
            self.pointer,
            name.encode("utf-8"),
            self._encode_value(value).encode("utf-8"),
        )
        if status != 0:
            raise self.runtime._error(f"set parameter {name}")
        return self

    def update(self, values: Mapping[str, Any]) -> "Config":
        for name, value in values.items():
            self.set(name, value)
        return self

    def clone(self) -> "Config":
        pointer = self.runtime.lib.sbm_config_clone(self.pointer)
        if not pointer:
            raise self.runtime._error("clone config")
        return Config(self.runtime, pointer)

    def as_dict(self) -> Dict[str, Any]:
        pointer = self.runtime.lib.sbm_config_get_json(self.pointer)
        return self.runtime._take_json(pointer, "serialize config")


class Dataset:
    def __init__(self, runtime: Runtime, pointer: int):
        self.runtime = runtime
        self.pointer = pointer

    def close(self) -> None:
        if self.pointer:
            self.runtime.lib.sbm_dataset_destroy(self.pointer)
            self.pointer = None

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def kind(self) -> str:
        value = int(self.runtime.lib.sbm_dataset_kind_of(self.pointer))
        if value == 1:
            return "vector_regression"
        if value == 2:
            return "token_cross_entropy"
        return "unknown"

    @property
    def hash(self) -> int:
        return int(self.runtime.lib.sbm_dataset_hash(self.pointer))

    @property
    def length(self) -> int:
        return int(self.runtime.lib.sbm_dataset_length(self.pointer))

    @property
    def vector_dim(self) -> int:
        return int(self.runtime.lib.sbm_dataset_vector_dim(self.pointer))

    @property
    def alphabet(self) -> int:
        return int(self.runtime.lib.sbm_dataset_alphabet(self.pointer))

    @property
    def vocab_size(self) -> int:
        return int(self.runtime.lib.sbm_dataset_vocab_size(self.pointer))

    @property
    def sequence_count(self) -> int:
        return int(self.runtime.lib.sbm_dataset_sequence_count(self.pointer))

    @property
    def example_count(self) -> int:
        return int(self.runtime.lib.sbm_dataset_example_count(self.pointer))

    def save(self, path: str | os.PathLike[str]) -> None:
        if self.runtime.lib.sbm_dataset_save(self.pointer, os.fsencode(path)) != 0:
            raise self.runtime._error("save dataset")

    def write_token_shard(self, path: str | os.PathLike[str]) -> None:
        if self.runtime.lib.sbm_token_shard_write(self.pointer, os.fsencode(path)) != 0:
            raise self.runtime._error("write token shard")

    def run(
        self,
        config: Config,
        warmup: int = 40_000,
        strict_freeze: bool = True,
        prefill: int = 0,
        prune_interval: int = 0,
        merge_interval: int = 0,
    ) -> Dict[str, Any]:
        pointer = self.runtime.lib.sbm_run_experiment_json(
            self.pointer,
            warmup,
            config.pointer,
            int(strict_freeze),
            prefill,
            prune_interval,
            merge_interval,
        )
        return self.runtime._take_json(pointer, "run experiment")


class Machine:
    def __init__(self, runtime: Runtime, pointer: int):
        self.runtime = runtime
        self.pointer = pointer

    def close(self) -> None:
        if self.pointer:
            self.runtime.lib.sbm_machine_destroy(self.pointer)
            self.pointer = None

    def __enter__(self) -> "Machine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def step_token(
        self,
        input_token: int,
        target_token: int,
        learn: bool = True,
    ) -> Dict[str, Any]:
        stats = _StepStats()
        status = self.runtime.lib.sbm_machine_step_token(
            self.pointer,
            int(input_token),
            int(target_token),
            int(learn),
            ctypes.byref(stats),
        )
        if status != 0:
            raise self.runtime._error("step machine token")
        return {
            "cross_entropy": float(stats.cross_entropy),
            "target_probability": float(stats.target_probability),
            "active_nodes": int(stats.active_nodes),
            "candidates_examined": int(stats.candidates_examined),
            "live_nodes": int(stats.live_nodes),
            "predicted_token": int(stats.predicted_token),
            "top1_correct": bool(stats.top1_correct),
            "top5_correct": bool(stats.top5_correct),
            "ranking_available": bool(stats.ranking_available),
            "channel_credit_count": int(stats.channel_credit_count),
            "channel_credit": [
                float(stats.channel_credit[i]) for i in range(int(stats.channel_credit_count))
            ],
            "channel_responsibility_mass": [
                float(stats.channel_responsibility_mass[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_dependency": [
                int(stats.channel_dependency[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_parent_channel": [
                int(stats.channel_parent_channel[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_dependency_channel": [
                int(stats.channel_dependency_channel[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_input_state": [
                int(stats.channel_input_state[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_output_state": [
                int(stats.channel_output_state[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_required_dependency_binding": [
                int(stats.channel_required_dependency_binding[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_caller_removed_credit": [
                float(stats.channel_caller_removed_credit[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_dependency_retained_credit": [
                float(stats.channel_dependency_retained_credit[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_dependency_removed_credit": [
                float(stats.channel_dependency_removed_credit[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_kind": [
                int(stats.channel_binding_kind[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_matched": [
                bool(stats.channel_binding_matched[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_current_token": [
                int(stats.channel_binding_current_token[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_matched_token": [
                int(stats.channel_binding_matched_token[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_successor": [
                int(stats.channel_binding_successor[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_distance": [
                int(stats.channel_binding_distance[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_pattern_span": [
                int(stats.channel_binding_pattern_span[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_binding_key": [
                int(stats.channel_binding_key[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_dependency_binding_key": [
                int(stats.channel_dependency_binding_key[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_call_key": [
                int(stats.channel_call_key[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_description_cost": [
                float(stats.channel_description_cost[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_execution_cost": [
                float(stats.channel_execution_cost[i])
                for i in range(int(stats.channel_credit_count))
            ],
            "channel_subset_available": bool(stats.channel_subset_available),
            "seed_only_cross_entropy": float(stats.seed_only_cross_entropy),
            "active_only_cross_entropy": float(stats.active_only_cross_entropy),
            "content_only_cross_entropy": float(stats.content_only_cross_entropy),
            "tuple_only_cross_entropy": float(stats.tuple_only_cross_entropy),
        }

    def reset_sequence(self) -> None:
        self.runtime.lib.sbm_machine_reset_sequence(self.pointer)

    def freeze_topology(self) -> None:
        self.runtime.lib.sbm_machine_freeze_topology(self.pointer)

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        status = self.runtime.lib.sbm_machine_save_checkpoint(
            self.pointer, os.fsencode(path)
        )
        if status != 0:
            raise self.runtime._error("save machine checkpoint")

    def diagnostics(self) -> Dict[str, Any]:
        pointer = self.runtime.lib.sbm_machine_diagnostics_json(self.pointer)
        return self.runtime._take_json(pointer, "machine diagnostics")

    def summary(self) -> Dict[str, Any]:
        pointer = self.runtime.lib.sbm_machine_summary_json(self.pointer)
        return self.runtime._take_json(pointer, "machine summary")


class TokenShard:
    def __init__(self, runtime: Runtime, pointer: int, shard_index: int):
        self.runtime = runtime
        self.pointer = pointer
        self._cursor = _TokenShardCursor(shard_index, 0, 0)

    def close(self) -> None:
        if self.pointer:
            self.runtime.lib.sbm_token_shard_destroy(self.pointer)
            self.pointer = None

    def __enter__(self) -> "TokenShard":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def vocab_size(self) -> int:
        return int(self.runtime.lib.sbm_token_shard_vocab_size(self.pointer))

    @property
    def token_count(self) -> int:
        return int(self.runtime.lib.sbm_token_shard_token_count(self.pointer))

    @property
    def sequence_count(self) -> int:
        return int(self.runtime.lib.sbm_token_shard_sequence_count(self.pointer))

    @property
    def dataset_hash(self) -> int:
        return int(self.runtime.lib.sbm_token_shard_dataset_hash(self.pointer))

    @property
    def cursor(self) -> tuple[int, int, int]:
        return (
            int(self._cursor.shard_index),
            int(self._cursor.sequence_index),
            int(self._cursor.token_offset),
        )

    def seek(self, cursor: tuple[int, int, int]) -> None:
        self._cursor = _TokenShardCursor(*cursor)

    def next_example(self) -> Optional[tuple[int, int]]:
        input_token = ctypes.c_uint32()
        target_token = ctypes.c_uint32()
        status = self.runtime.lib.sbm_token_shard_next(
            self.pointer,
            ctypes.byref(self._cursor),
            ctypes.byref(input_token),
            ctypes.byref(target_token),
        )
        if status < 0:
            raise self.runtime._error("read token shard")
        if status == 0:
            return None
        return int(input_token.value), int(target_token.value)

    def run(
        self,
        config: Config,
        warmup: int,
        strict_freeze: bool = True,
        prefill: int = 0,
        prune_interval: int = 0,
        merge_interval: int = 0,
    ) -> Dict[str, Any]:
        pointer = self.runtime.lib.sbm_run_token_shard_experiment_json(
            self.pointer,
            warmup,
            config.pointer,
            int(strict_freeze),
            prefill,
            prune_interval,
            merge_interval,
        )
        return self.runtime._take_json(pointer, "run token shard experiment")


class TokenCorpus:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.pointer = runtime.lib.sbm_token_corpus_create()
        if not self.pointer:
            raise runtime._error("create token corpus")

    def close(self) -> None:
        if self.pointer:
            self.runtime.lib.sbm_token_corpus_destroy(self.pointer)
            self.pointer = None

    def __enter__(self) -> "TokenCorpus":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def add_shard(
        self,
        path: str | os.PathLike[str],
        split: str,
        shard_index: int,
        verify_payload: bool = True,
    ) -> None:
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")
        status = self.runtime.lib.sbm_token_corpus_add_shard(
            self.pointer,
            os.fsencode(path),
            0 if split == "train" else 1,
            shard_index,
            int(verify_payload),
        )
        if status != 0:
            raise self.runtime._error("add token corpus shard")

    def run(
        self,
        config: Config,
        strict_freeze: bool = True,
        prefill: int = 0,
        prune_interval: int = 0,
        merge_interval: int = 0,
        max_train_examples: Optional[int] = None,
        max_eval_examples: Optional[int] = None,
    ) -> Dict[str, Any]:
        if max_train_examples is None and max_eval_examples is None:
            pointer = self.runtime.lib.sbm_run_token_corpus_experiment_json(
                self.pointer,
                config.pointer,
                int(strict_freeze),
                prefill,
                prune_interval,
                merge_interval,
            )
        else:
            if max_train_examples is None or max_eval_examples is None:
                raise ValueError("both max_train_examples and max_eval_examples are required")
            pointer = self.runtime.lib.sbm_run_token_corpus_experiment_limited_json(
                self.pointer,
                max_train_examples,
                max_eval_examples,
                config.pointer,
                int(strict_freeze),
                prefill,
                prune_interval,
                merge_interval,
            )
        return self.runtime._take_json(pointer, "run token corpus experiment")

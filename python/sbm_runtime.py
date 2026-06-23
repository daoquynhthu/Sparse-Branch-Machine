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

        lib.sbm_dataset_generate.argtypes = [
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_float,
        ]
        lib.sbm_dataset_generate.restype = ctypes.c_void_p
        lib.sbm_dataset_load.argtypes = [ctypes.c_char_p]
        lib.sbm_dataset_load.restype = ctypes.c_void_p
        lib.sbm_dataset_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.sbm_dataset_save.restype = ctypes.c_int
        lib.sbm_dataset_destroy.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_hash.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_hash.restype = ctypes.c_uint64
        lib.sbm_dataset_length.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_length.restype = ctypes.c_size_t
        lib.sbm_dataset_vector_dim.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_vector_dim.restype = ctypes.c_uint32
        lib.sbm_dataset_alphabet.argtypes = [ctypes.c_void_p]
        lib.sbm_dataset_alphabet.restype = ctypes.c_uint32

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

    def load_dataset(self, path: str | os.PathLike[str]) -> "Dataset":
        pointer = self.lib.sbm_dataset_load(os.fsencode(path))
        if not pointer:
            raise self._error("load dataset")
        return Dataset(self, pointer)


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

    def save(self, path: str | os.PathLike[str]) -> None:
        if self.runtime.lib.sbm_dataset_save(self.pointer, os.fsencode(path)) != 0:
            raise self.runtime._error("save dataset")

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

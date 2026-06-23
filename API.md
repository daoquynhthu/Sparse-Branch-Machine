# Stable API contract

## C ABI

The public ABI is declared in `include/sbm/api.h`. It intentionally exposes only
opaque handles and plain C scalar types. C++ classes, STL containers and compiler
ABI details do not cross the boundary.

Main ownership rules:

- `sbm_config_create` / `sbm_config_destroy` own configuration handles;
- `sbm_dataset_generate` or `sbm_dataset_load` / `sbm_dataset_destroy` own datasets;
- `sbm_run_experiment_json` and `sbm_config_get_json` return strings released by
  `sbm_string_free`;
- `sbm_parameter_schema_json` returns static read-only storage and must not be
  freed;
- `sbm_last_error` is thread-local and valid until the next API call on that
  thread.

The ABI version is available through `sbm_api_version()`. Additive functions may
be introduced without changing the major ABI version; incompatible handle or
calling-convention changes require a new major version.

## Runtime parameter registry

`sbm_parameter_schema_json()` is the authoritative parameter-discovery source.
Each entry includes:

- name and runtime type;
- current default;
- whether it is tunable;
- whether automatic search includes it by default;
- optional minimum, maximum and sampling scale;
- whether the dataset overrides the value;
- a short semantic description.

All current `sbm::Config` fields are accepted by `sbm_config_set`. The CLI and
Python code do not maintain a second list of setters.

## Python binding

`python/sbm_runtime.py` loads `sbm_api` through `ctypes` and provides three small
ownership wrappers:

```python
from sbm_runtime import Runtime

runtime = Runtime("build/libsbm_api.so")
with runtime.generate_dataset(seed=7) as dataset:
    with runtime.config({"exact_region_mass": 0.88}) as config:
        result = dataset.run(config, warmup=40_000)
        print(result["eval_r2"])
```

The Python layer receives result JSON from C++ and does not duplicate metric or
model logic.

# Build and linkage layout

The project now separates research components into shared libraries. On Windows
these targets produce DLLs; on Linux they produce `.so` files; on macOS they
produce `.dylib` files.

| Target | Responsibility | Typical consumer |
|---|---|---|
| `sbm_core` | SIMD vector kernels and address signatures | internal libraries |
| `sbm_machine` | node storage, routing, learning and maintenance | C++ embedding / experiment layer |
| `sbm_dataset` | task generation, hashing and binary persistence | experiment layer |
| `sbm_experiment` | baselines, metrics and experiment runner | stable API layer |
| `sbm_api` | versioned C ABI, parameter registry and opaque handles | CLI and Python |
| `sbm_cli` | argument parsing, file output and API calls only | shell users |

The CLI links directly only to `sbm_api`. It does not include model, dataset or
experiment C++ headers. Python uses the same C ABI through `ctypes`; there is no
separate Python implementation of the algorithm.

## Configure and build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The build produces a shared-library chain similar to:

```text
sbm_cli
  -> sbm_api
      -> sbm_experiment
          -> sbm_machine
          -> sbm_dataset
              -> sbm_core
```

Build only the model after changing a learning rule:

```bash
cmake --build build --target sbm_machine -j2
```

Build only the stable API and its dependencies:

```bash
cmake --build build --target sbm_api -j2
```

Disable executable and tests for library-only embedding:

```bash
cmake -S . -B build-library \
  -DSBM_BUILD_CLI=OFF \
  -DSBM_BUILD_TESTS=OFF
cmake --build build-library -j2
```

## Thin CLI

The CLI has no hand-maintained hyperparameter option list. It discovers the
runtime registry exposed by `sbm_api` and accepts generic assignments:

```bash
./build/sbm_cli --describe-parameters
./build/sbm_cli --dump-config \
  --set exact_region_mass=0.88 \
  --set edge_score_weight=0.32

./build/sbm_cli --length 120000 --warmup 40000 --seed 7 \
  --set exact_region_mass=0.88 \
  --output results.json
```

`--set` is intended for reproducing a known configuration or a single diagnostic
ablation. Repeated calibration should use the automatic search script below.

## Python API

Set the library path explicitly when it is outside the default build folders:

```bash
export SBM_LIBRARY="$PWD/build/libsbm_api.so"
python scripts/run_python_api.py --length 20000 --warmup 7000
```

Windows example:

```powershell
$env:SBM_LIBRARY = "$PWD\build\Release\sbm_api.dll"
python scripts/run_python_api.py
```

The binding is standard-library-only and keeps dataset/config ownership explicit.
A generated dataset can be reused for many in-process trials without serialization
or process startup overhead.

## Automatic hyperparameter search

Do not tune by repeatedly editing headers or manually launching parameter grids.
Use the schema-driven successive-halving search:

```bash
python scripts/tune.py \
  --library build/libsbm_api.so \
  --trials 27 \
  --seeds 7,11,19 \
  --jobs 3 \
  --output tuning_results.json \
  --best-config best_config.json
```

By default the script searches parameters marked `search_default` by the shared
library. A deliberate broader study can select other registered parameters:

```bash
python scripts/tune.py \
  --include beam_width,bucket_scan_limit,edge_score_weight,exact_region_mass \
  --trials 36
```

Each stage writes a checkpoint to the output path. Invalid configurations are
recorded rather than terminating the search. The final choice is based only on
candidates that survive all configured seeds, avoiding the bias of selecting a
one-seed outlier.

Use the selected configuration directly from Python:

```bash
python scripts/run_python_api.py \
  --config best_config.json \
  --length 120000 \
  --warmup 40000
```

## Installation

```bash
cmake --install build --prefix install
```

This installs shared libraries, the C/C++ headers, the CLI, the Python wrapper
and the automation scripts. Build and install RPATHs use the library directory,
so binaries do not depend on the source-tree build layout.

# Build and linkage layout

The project is separated into shared libraries.  Windows produces DLLs, Linux
`.so` files and macOS `.dylib` files.

| Target | Responsibility |
|---|---|
| `sbm_core` | SIMD vector kernels and address signatures |
| `sbm_machine` | storage, routing, vector learning and token cross-entropy |
| `sbm_dataset` | vector/math-token generation and binary persistence |
| `sbm_experiment` | baselines, metrics and experiment runners |
| `sbm_api` | versioned C ABI, parameter registry and opaque handles |
| `sbm_cli` | argument parsing and API calls only |

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Build only the learning library after changing an objective:

```bash
cmake --build build --target sbm_machine -j2
```

Build only the API and dependencies:

```bash
cmake --build build --target sbm_api -j2
```

## Token cross-entropy CLI

```bash
./build/sbm_cli \
  --task token-ce \
  --sequences 64 \
  --sequence-length 2048 \
  --vocab-size 32 \
  --math-temperature 0.8 \
  --interaction-strength 0.2 \
  --warmup 80000 \
  --output results.json
```

Save and reload exactly the same mathematical token stream:

```bash
./build/sbm_cli --task token-ce \
  --write-dataset math_tokens.bin \
  --output first.json

./build/sbm_cli --dataset math_tokens.bin \
  --warmup 80000 \
  --output replay.json
```

The file magic identifies vector versus token datasets automatically.

## Python in-process execution

```bash
export SBM_LIBRARY="$PWD/build/libsbm_api.so"
python scripts/run_python_api.py \
  --task token-ce \
  --sequences 16 \
  --sequence-length 1024 \
  --vocab-size 32 \
  --warmup 10000
```

## Automated search

```bash
python scripts/tune.py \
  --library build/libsbm_api.so \
  --task token-ce \
  --trials 27 \
  --seeds 7,11,19 \
  --eta 3 \
  --jobs 3 \
  --output tuning_token.json \
  --best-config best_token.json
```

The tuner:

- discovers ranges from the shared library;
- filters parameters by task applicability;
- always evaluates the unchanged default configuration;
- minimizes frozen cross-entropy for token tasks;
- maximizes frozen R2 for vector tasks;
- uses multi-seed successive halving;
- writes checkpoints without recompilation.

## Install

```bash
cmake --install build --prefix install
```

The installed CLI uses a relative `../lib` RPATH.  The installed Python wrapper
also searches the installation prefix's `lib` directory.

# Build and linkage layout

> **Document role:** This document is authoritative only for build, test, install and linkage procedures. Research plans belong in `ROADMAP_REAL_DATA.md`; interface semantics belong in `API.md`.


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
  --cpu-fraction 0.9 \
  --memory-fraction 0.9 \
  --minimum-free-memory-mib 1024 \
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

Automatic mode runs every candidate in an isolated process. It admits one
worker at a time after refreshing current available memory and checks both the
CPU-slot and memory budgets. The reservation combines dataset bytes, model
state, fixed process overhead and a configurable growth margin; observed
`estimated_bytes` raises the estimate for later halving stages. Use
`--fixed-jobs N` only to request the legacy in-process thread executor.

## Install

```bash
cmake --install build --prefix install
```

The installed CLI uses a relative `../lib` RPATH.  The installed Python wrapper
also searches the installation prefix's `lib` directory.


## Build-time discipline

Use target-level incremental builds during research. Do not clean and rebuild every shared library after a local algorithm edit. Build directories and generated libraries are not source artifacts and must not be committed or included in source packages.

Real-corpus tooling should be separate targets (for example acquisition/tokenization utilities and shard readers) so that model edits do not rebuild data tooling and data-format edits do not rebuild the learning core.

## Existing NAIME data compatibility conversion

The converter requires Hugging Face `datasets` and writes large artifacts
outside the repository:

```bash
python scripts/convert_naime_dataset.py \
  --dataset /data/fineweb_edu_1b_ctx1024 \
  --output /data/sbm/fineweb_edu_compat \
  --vocab-size 50257 \
  --shard-tokens 16000000
```

Use `--max-tokens-per-split` for a deterministic smoke subset, or
`--max-train-tokens` and `--max-validation-tokens` when split budgets differ.
The command emits
one compact JSON status line; detailed provenance, hashes, counts and known data
limitations are stored in `manifest.json`. Existing non-empty output directories
and existing shard files are rejected.

`Runtime.open_token_shard(...).run(...)` is available for a small single-shard
smoke. It splits that shard by example count and is not the corpus experiment
interface for separate train and validation manifests.

For a split-correct mapped run:

```python
with runtime.open_token_corpus("/data/sbm/fineweb_edu_compat/manifest.json") as corpus:
    with runtime.config({"sparse_token_output": True}) as config:
        result = corpus.run(config, strict_freeze=True)
```

The corpus handle consumes all manifest train shards before freezing once and
evaluating validation shards. Exact model checkpoint/resume is still pending.

Multi-seed corpus runs use the same dual-constraint scheduler:

```bash
python scripts/run_corpus_batch.py \
  --library build/libsbm_api.so \
  --manifest /data/sbm/fineweb_edu_compat/manifest.json \
  --seeds 7,11,19 \
  --output-dir /data/sbm/runs/medium \
  --cpu-fraction 0.9 \
  --memory-fraction 0.9
```

Console output is limited to scheduler state transitions and one final compact
summary. Full per-seed metrics are written under the output directory.

Long mapped-corpus runs emit compact native progress lines to stderr for large
corpora. Each line reports phase, processed examples, examples/s, ETA, running
train/eval NLL, live nodes and estimated state MB. Redirect stderr to a log file
for unattended runs; stdout remains reserved for explicit script output and the
full result JSON is still written only to `--output`.

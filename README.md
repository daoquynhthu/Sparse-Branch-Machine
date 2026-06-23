# Sparse Branch Machine C++ v2

Research prototype for a CPU-first learning machine based on stable addressing, dynamic branch traversal, sparse activation, and local updates.

## What changed from v1

- Stable logical `NodeId` separated from movable physical slots.
- Structure-of-arrays node metadata and flattened output statistics.
- Real pruning with swap-removal and index rebuilding.
- Stale-reference-safe buckets and edges.
- Fixed-budget candidate deduplication without per-step hash-table allocation.
- Deterministic SplitMix64-derived dataset generation.
- Portable binary datasets with version, magic, and checksum.
- Capacity-stress mode through `--prefill` distractor nodes.
- Memory estimate, pruning, stale-reference, and logical-ID diagnostics.
- Strict structure freeze during evaluation is now the default.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run

```bash
./build/sbm_benchmark --length 100001 --warmup 20000 --output results.json
```

Create and reuse an identical dataset:

```bash
./build/sbm_benchmark --length 100001 --write-dataset fsm.bin
./build/sbm_benchmark --dataset fsm.bin --warmup 20000
```

Stress the addressing system with one million irrelevant nodes:

```bash
./build/sbm_benchmark --prefill 1000000 --length 100001 --warmup 20000
```

Enable periodic pruning during training:

```bash
./build/sbm_benchmark --prefill 100000 --prune-interval 5000
```

## Remaining research limitations

This is still a finite-context prediction model. Stable storage and scalable addressing are now testable, but abstraction, compositional transfer, counterfactual routing credit, and long-horizon learning remain unresolved.

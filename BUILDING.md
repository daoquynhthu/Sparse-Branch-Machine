# Modular build layout

The project is split by research responsibility rather than by file size alone.

| Target | Sources | Rebuild when changing |
|---|---|---|
| `sbm_core` | SIMD vector kernels, signatures | vector math or address encoding |
| `sbm_machine` | storage, routing, learning, maintenance | model architecture and learning rules |
| `sbm_dataset` | generator, hashing, persistence | synthetic task definition |
| `sbm_experiment` | baselines, metrics, experiment runner | evaluation methodology |
| `sbm_benchmark` | CLI only | command-line behavior |
| `sbm_tests` | regression tests | test cases |

`sbm` remains an interface compatibility target, so external code can continue
linking the same target and including `sparse_branch_machine.hpp`.

## Fast build commands

Build only the model library after changing a learning rule:

```bash
cmake --build build --target sbm_machine -j2
```

Build the task generator without compiling the benchmark or tests:

```bash
cmake -S . -B build-task -DSBM_BUILD_BENCHMARK=OFF -DSBM_BUILD_TESTS=OFF
cmake --build build-task --target sbm_dataset -j2
```

Build and run tests:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sbm_tests -j2
ctest --test-dir build --output-on-failure
```

The module boundary is intended to preserve algorithm iteration speed. Do not
move task generation, benchmark baselines or JSON formatting into the machine
library merely for convenience.

## Runtime parameter sweeps

Mixture and local-learning calibration no longer requires editing `types.hpp`.
The benchmark accepts runtime overrides:

```bash
./build/sbm_benchmark --exact-region-mass 0.88 \
  --residual-gain 1.0 \
  --residual-pseudocount 0.75 \
  --edge-score-weight 0.32
```

This avoids header-triggered full rebuilds during parameter sweeps. Structural
changes still belong in the relevant module and should be covered by tests.

# Sparse Branch Machine Research

CPU-first sparse learning research platform. The current `vector-prediction-v4`
branch replaces categorical hidden-FSM prediction with token-conditioned
continuous-vector prediction while preserving the v2/v3 addressing, lifecycle,
logical-ID and sparse-control-flow architecture.

## Current task

Each step provides a discrete token and asks the model to predict a 16-dimensional
continuous vector. The target is generated from a stationary nonlinear relation
of the current token, several delayed tokens, a history-derived regime, pairwise
feature interactions and weak Gaussian observation noise. There is no hidden
random target driver: the noise-free relation is recoverable from observed token
history.

Default dataset:

- 64 token values
- 16-dimensional prediction target
- 24-dimensional synthetic feature bank
- 120,000 steps
- first 40,000 steps learn
- remaining 80,000 steps strictly frozen

Metrics are normalized RMSE, cosine similarity and R2 relative to the training-set
mean-vector baseline.

## Architecture

The model still performs:

1. recent-token locality-preserving signature construction;
2. exact address-region lookup;
3. sparse learned edge traversal;
4. fixed-width candidate competition;
5. weighted local vector aggregation;
6. local prototype-vector updates and lifecycle credit.

There is no tensor framework, global dense matrix, automatic differentiation or
backpropagation.

## SIMD

AVX2/FMA kernels are selected at runtime on supported x86-64 CPUs for:

- vector dot products;
- squared Euclidean distance;
- weighted aggregation (AXPY);
- local centroid updates.

A scalar fallback is retained for portability.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/sbm_benchmark --length 120000 --warmup 40000 \
  --seed 7 --output results_vector_v4.json
```

Dataset persistence remains supported:

```bash
./build/sbm_benchmark --write-dataset vector_task.bin
./build/sbm_benchmark --dataset vector_task.bin
```

## Baseline result, seed 7

- live nodes: 256
- mean active nodes: 5.9995
- mean candidates: 35.28
- frozen evaluation NRMSE: 0.7625
- frozen evaluation cosine: 0.7664
- frozen evaluation R2: 0.3995
- SIMD runtime path: enabled

The task is deliberately not saturated. The present model captures a meaningful
part of the relation but leaves substantial residual error for later algorithmic
work.

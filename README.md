# Sparse Branch Machine Research

CPU-first sparse learning research platform. The current implementation predicts
continuous vectors from discrete token streams without dense global matrices,
autodiff or backpropagation. It retains stable logical node addresses, bounded
candidate search, sparse branch edges and local updates.

## Current task

Each step provides a token from a 64-symbol alphabet and asks the model to predict
a 16-dimensional vector. The stationary target depends on current and delayed
tokens, a history-derived regime, nonlinear interactions and weak observation
noise. The first 40,000 of 120,000 steps are learned; the remaining 80,000 are
strictly frozen.

Simple frozen-evaluation controls on seed 7 are:

- global mean: R2 = 0;
- current-token centroid: R2 = 0.303;
- previous/current pair centroid: R2 = 0.520;
- pair plus lag-2 residual table: R2 = 0.706;
- fixed lag-1/2/4 residual tables: R2 = 0.824.

The last control is intentionally strong: it distinguishes a working sparse
machine from a task-specific collection of conditional means.

## Current architecture

The model now uses three generic temporal address views with lags 1, 2 and 4.
Each view has an independent logical address namespace:

1. lag-1 nodes provide the primary vector prediction;
2. lag-2 nodes store an additive residual;
3. lag-4 nodes store the remaining additive residual.

The prediction is therefore

```text
anchor(lag 1) + residual(lag 2) + residual(lag 4)
```

rather than a convex average of unrelated centroids. Every step still activates
at most six nodes from roughly twelve thousand stored nodes.

Exact address residents receive 88% of each channel's routing mass. Learned
control edges and nearby regions share the remaining 12%. Removing this branch
share reduced frozen R2 by roughly 0.006 on the seed-7 calibration, so the sparse
route is useful but not yet the dominant source of capability.

## Learning rules

Prediction is fixed before the current target may alter state. New nodes cannot
improve the score of the sample that created them.

Each exact address node learns an online local mean. Residual stages are trained
sequentially:

```text
lag-1 target = y
lag-2 target = y - lag-1 prediction
lag-4 target = y - lag-1 prediction - lag-2 prediction
```

The lag-1 mean uses the unbiased `1/n` update. Later residual means use a small
recency pseudocount because their upstream anchor estimates are still moving.
Nodes recalled through foreign control edges are read-only; their own local
vectors are not overwritten by another address context. Only edge credit learns
from such retrieval.

## Current results

Seeds 7, 11 and 19, each with 40,000 learning steps and 80,000 strict-freeze
steps:

| Metric | Mean |
|---|---:|
| Model frozen-eval R2 | 0.8076 |
| Current-token baseline R2 | 0.3031 |
| Token-pair baseline R2 | 0.5210 |
| Fixed multiscale table R2 | 0.8246 |
| Seen three-token context R2 | 0.8244 |
| Unseen three-token context R2 | 0.8048 |
| Live nodes | 12,287 |
| Active nodes per step | 5.97 |

The model is now close to the fixed multiscale conditional-table control and
substantially exceeds the pair baseline. It still does not surpass the fixed
1/2/4 residual table, so the result is evidence for a functioning sparse
multiscale architecture, not yet for superior abstraction.

## SIMD

AVX2/FMA kernels are selected at runtime for local dot products, squared
distance, aggregation and vector updates. A scalar fallback remains available.
SIMD accelerates only the small active vectors; the architecture remains driven
by addressing and branch control rather than dense matrix throughput.

## Engineering interface

The implementation is distributed as a shared-library stack rather than one
monolithic executable:

```text
sbm_core -> sbm_machine / sbm_dataset -> sbm_experiment -> sbm_api
                                                        -> sbm_cli
                                                        -> Python ctypes
```

`sbm_api` is a versioned C ABI with opaque configuration and dataset handles.
The command-line program links only this API and performs no model computation
itself. Python calls the same library in-process, so a dataset can be generated
once and reused across automated trials.

Build and test:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Run from the thin CLI:

```bash
./build/sbm_cli --length 120000 --warmup 40000 --seed 7 \
  --set exact_region_mass=0.88 \
  --output results.json
```

Discover every runtime parameter and its search metadata:

```bash
./build/sbm_cli --describe-parameters
```

Run through Python without spawning the CLI:

```bash
export SBM_LIBRARY="$PWD/build/libsbm_api.so"
python scripts/run_python_api.py --length 20000 --warmup 7000
```

Repeated hyperparameter tuning is automated rather than performed by editing
headers or manually launching grids:

```bash
python scripts/tune.py \
  --library build/libsbm_api.so \
  --trials 27 \
  --seeds 7,11,19 \
  --jobs 3 \
  --best-config best_config.json
```

The tuner reads its ranges from `sbm_parameter_schema_json()`, uses repeated-seed
successive halving, records invalid configurations, and checkpoints after every
stage. See `BUILDING.md` and `API.md` for the full contract.

## Current limits

- The three temporal views are fixed meta-structure rather than learned address
  projections.
- Control edges currently provide routing priors only; they do not yet carry a
  learned vector transformation or residual payload.
- Conservative specialization rarely triggers because each exact pair receives
  few samples. Aggressive splitting previously improved seen contexts while
  harming unseen contexts.
- The current task does not test variable binding, reusable subprograms,
  recursive execution or systematic extrapolation.

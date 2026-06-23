# Sparse Branch Machine Research

CPU-first sparse learning research platform. The current branch,
`vector-prediction-v5`, keeps the v2/v3 logical-ID, sparse-addressing and local
learning core, but fixes several correctness and experimental-design defects in
v4.

## Current task

Each step provides a discrete token and asks the model to predict a
16-dimensional continuous vector. The target is a stationary nonlinear function
of the current token, several delayed tokens, a history-derived regime and
multiplicative feature interactions, plus weak observation noise.

The token stream now has broad transition support. The earlier v4 generator was
nearly determined by the current token: a token-conditional mean baseline reached
R2 above 0.95. That task was rejected. In v5, the default baselines are roughly:

- global mean: R2 = 0;
- current-token mean: R2 = 0.30;
- previous/current-token pair mean: R2 = 0.52.

The task is therefore neither saturated nor dominated by a single token lookup.

Default dataset:

- 64 token values;
- 16-dimensional prediction target;
- 24-dimensional synthetic feature bank;
- 120,000 steps;
- first 40,000 steps learn;
- remaining 80,000 steps strictly frozen.

## v5 corrections

### No target leakage

Prediction metrics are fixed before the target may initialize a new node or
change an existing vector. v4 recomputed the current training prediction after a
new node had been initialized from the target, which made training metrics
optimistic.

### Compact token addressing

A 64-token alphabet needs six bits per symbol. v4 stored recent symbols as
8-bit bytes, so a 12-bit bucket represented the current token and only two useful
bits of the previous token. v5 packs symbols using `ceil(log2(alphabet))` bits;
the default 12-bit region therefore identifies the complete previous/current
pair.

### Hierarchical responsibility

Exact-region nodes receive a reserved share of prediction mass. Sparse edge and
nearby-region candidates retain a bounded exploratory share instead of diluting
content-addressed nodes equally.

### Local counterfactual credit

For each active node, v5 computes the prediction loss with that node removed.
The difference is used as a local contribution estimate. Node updates and edge
reinforcement are weighted by responsibility and positive contribution rather
than rank alone.

### Address-local specialization evidence

Total node visits no longer trigger structural splitting. v5 separately tracks
visits and residual loss obtained while a node is serving its own address region.
This removed thousands of false specializations caused by incidental edge or
neighbor retrieval.

### Bounded edge semantics

Edge weights use bounded EMA updates, decay when their source is revisited, and
are created only from positive counterfactual contribution. Edge strength now
enters candidate scoring. A pure exact-address ablation performs worse, indicating
that the sparse route contribution is useful rather than decorative.

### Stronger baselines and diagnostics

The benchmark reports:

- global-mean baseline;
- current-token mean baseline;
- previous/current-token pair baseline;
- model metrics on seen and unseen three-token contexts.

## Default results

Across seeds 7, 11 and 19:

| Metric | Mean |
|---|---:|
| Model frozen-eval R2 | 0.4805 |
| Current-token baseline R2 | 0.3031 |
| Token-pair baseline R2 | 0.5210 |
| Seen three-token context R2 | 0.5178 |
| Unseen three-token context R2 | 0.4744 |
| Live nodes | 4096 |
| Mean sparse edges | 72,402 |

The model now substantially exceeds the single-token baseline and approaches the
full token-pair lookup baseline. It does not yet beat the pair baseline, so the
current result is not evidence of superior abstraction. The seen/unseen context
gap also remains real.

## Architecture

The model performs:

1. compact recent-token signature construction;
2. exact address-region lookup;
3. sparse learned edge retrieval;
4. fixed-width candidate competition;
5. hierarchical responsibility assignment;
6. local vector aggregation;
7. counterfactual local credit;
8. local prototype, lifecycle and sparse-edge updates.

There is no tensor framework, dense global matrix, automatic differentiation or
backpropagation.

## SIMD

AVX2/FMA kernels are selected at runtime on supported x86-64 CPUs for vector dot
products, squared distance, weighted aggregation and local centroid updates. A
scalar fallback remains available.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/sbm_benchmark --length 120000 --warmup 40000 \
  --seed 7 --output results_vector_v5.json
```

Dataset persistence remains supported:

```bash
./build/sbm_benchmark --write-dataset vector_task.bin
./build/sbm_benchmark --dataset vector_task.bin
```

## Current limits

- The default node population still behaves largely as a learned pair-address
  table with sparse historical correction.
- Conservative multi-prototype splitting is implemented, but the default task
  does not provide enough repeated evidence per exact pair for it to help.
- An aggressive split calibration increased nodes from 4096 to 8325 and reduced
  frozen R2 from about 0.480 to 0.463; it improved seen contexts while harming
  unseen contexts, so it was rejected as overfitting.
- The system has not yet demonstrated variable binding, reusable subprograms or
  systematic extrapolation.

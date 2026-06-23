# Sparse Branch Machine research repository

CPU-first learning architecture based on stable addressing, dynamic branch traversal, sparse activation, persistent local state, and local structural learning.

## Repository workflow

- `main`: preserved v2 baseline.
- `architecture-v3`: incremental architecture and learning changes.
- The initial v2 import is retained as a Git commit; no rewrite replaced it.
- Experimental regressions and rejected semantics are recorded in `RESEARCH_LOG.md`.

## Architecture v3 additions

- finite route traces with decayed node-level credit;
- explicit cold/warm/mature/dormant node lifecycle;
- parent-linked specialization on repeated prediction conflict;
- edge eligibility state without unjustified blanket edge punishment;
- conservative merge of nodes that are close in both address and output distribution;
- separate split, merge, lifecycle, and stale-reference diagnostics.

The underlying v2 storage remains intact: stable logical IDs, movable physical slots, SoA metadata, hot/cold address buckets, sparse edge traversal, strict evaluation freeze, binary datasets, pruning, and capacity stress tests.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run

```bash
./build/sbm_benchmark --length 100001 --warmup 20000 --output results_v3.json
```

Periodic structural maintenance:

```bash
./build/sbm_benchmark \
  --length 100001 \
  --warmup 20000 \
  --prune-interval 5000 \
  --merge-interval 5000
```

## Current status

The architectural branch is deliberately not declared superior to v2. It preserves sparse execution and introduces auditable lifecycle and structural-learning semantics, but currently reaches 93.75% frozen evaluation accuracy versus 96.875% for v2 on the default deterministic FSM task. See `RESEARCH_LOG.md` for the retained negative experiment and interpretation.

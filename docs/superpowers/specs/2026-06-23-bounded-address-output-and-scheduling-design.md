# Bounded Addressing, Vocabulary Output, and Hardware Scheduling

## Status and scope

This design addresses four coupled scaling failures without changing the Sparse
Branch Machine's architectural identity:

1. address lookup latency must not grow with stored-node count;
2. address-index memory must not grow with the unused logical address space;
3. token-output storage and work must not grow linearly per node with vocabulary;
4. experiment execution must use available CPU and memory deliberately, while
   baseline evaluation must stop dominating wall time.

Persistent addressable nodes, learned address programs, bounded sparse routing,
local updates, hierarchical token output, strict evaluation freeze and ordered
online learning remain unchanged. Dense attention, batched-token learning and
global backpropagation are outside this design.

## Current root causes

### Address latency

`select_route` limits ordinary bucket and edge expansion, but token-node creation
still scans every member of the exact bucket to count residents and find the
nearest prototype. An overcrowded bucket therefore creates a hidden
stored-capacity-dependent path. Cold candidate sampling is bounded, but the
creation path is not.

### Address-index memory

The constructor allocates separate `buckets_`, `hot_buckets_` and split-step
arrays for every `(channel, 2^bucket_bits)` pair. Most entries are empty. Raising
address precision therefore allocates millions of empty `std::vector` objects
before the model learns a node.

### Vocabulary storage

Per-node dense logits are already avoided when sparse token output is enabled.
However, the implementation materializes a global binary tree, an offset for
every token and every token's full path. This costs `O(V log V)` fixed storage.
Each node's sparse decision list also uses a linear scan and has no explicit
capacity bound.

### Baseline wall time

The real-corpus evaluator materializes several `V`-length probability arrays
and scans them for every validation token. On the 50,257-token smoke run, model
time was 4.62 seconds while end-to-end time was about 80 seconds. Moving this
unchanged work to a background thread would hide latency but not remove CPU and
memory cost.

## Design 1: sparse bounded address index

Replace the dense parallel bucket arrays with a lazy directory keyed by the
existing flattened `(channel, bucket)` identifier. Only occupied buckets have a
`BucketState`.

Each state contains:

- exact resident count;
- all resident logical IDs for maintenance and rebuild operations;
- a bounded hot candidate set;
- a deterministic rotating cold-sample cursor;
- last specialization step.

Per-token operations have explicit limits:

- combined hot and cold lookup: at most `bucket_scan_limit` IDs per enabled
  channel, with hot entries consuming the budget first;
- edge expansion: at most `beam_width * edge_scan_limit` edges;
- neighboring probes: the existing fixed radius and remaining hard budget;
- specialization search: at most `bucket_scan_limit` prototypes, while the
  exact resident count is read in `O(1)`.

The full resident list remains available because persistent nodes must not be
discarded merely to accelerate routing. Full scans are allowed only in explicit
maintenance operations such as rebuild, prune or offline diagnostics, never in
the token hot path.

Hot-set replacement is deterministic. A new hot node replaces the weakest
indexed entry by phase, visits, utility and logical ID tie-break. Cold sampling
starts from a deterministic cursor derived from the bucket and machine RNG,
then inspects a bounded contiguous wraparound window. Sample order is stable for
the same seed and operation sequence.

## Design 2: implicit hierarchical token output

The hierarchical binary objective remains, but its tree becomes implicit.

For vocabulary size `V`, derive a keyed affine permutation:

```
rank(token) = (a * token + b) mod V
token(rank) = a_inverse * (rank - b) mod V
```

`a` is selected deterministically from the model seed and incremented until it
is coprime with `V`; `b` is seed-derived. The mapping is bijective and requires
constant storage.

The implicit tree recursively partitions rank intervals `[lo, hi)` at
`mid = lo + (hi - lo) / 2`. Every internal decision is identified by the unique
split boundary `mid - 1`, which lies in `[0, V - 2]`. Target paths are generated
from the permuted rank in `O(log V)`. Beam decoding stores `(lo, hi,
log_probability)` and converts a terminal rank through the inverse permutation.

This removes:

- `output_tree_`;
- `token_path_offsets_`;
- `token_path_steps_`;
- vocabulary-sized leaf-order sorting.

The objective remains an exact normalized hierarchical softmax. The keyed
permutation prevents numeric token-ID neighborhoods from becoming a manual
linguistic prior.

## Design 3: bounded local decision storage

After the implicit tree is accepted, replace each node's unsorted linear sparse
decision list with a sorted compact vector:

- lookup uses binary search;
- updates preserve sorted order;
- capacity is bounded by runtime parameter
  `max_sparse_decisions_per_node`;
- when full, an unseen decision replaces the entry with minimum absolute logit;
- ties are resolved by decision ID;
- zero or near-zero entries may be erased during maintenance.

The default capacity will be selected by an automated quality/memory ablation,
not manually assumed. Candidate defaults are 64, 128 and 256. This keeps learned
output memory at `O(nodes * fixed_capacity)` rather than allowing one broad
context node to accumulate vocabulary-scale state.

Because insertion is `O(capacity)`, the cap is also a hard latency bound. A flat
hash table is not the initial choice: at these capacities its allocation and
metadata overhead are likely larger than binary-search cost. Profiling may
justify a later representation-only change under identical semantics.

## Design 4: baseline evaluation without vocabulary scans

Synchronous experiment output must retain primary held-out NLL controls, but
need not compute exact top-k ranks for every baseline on every token.

The synchronous path will use sparse count queries:

- unigram target probability: direct count lookup;
- current-token and pair target probability: direct conditional-row lookup with
  the existing smoothing/backoff formula;
- replace the geometric multiscale full-distribution product with a documented
  interpolated/backoff n-gram control whose target probability and normalizer
  are available in `O(1)` from sparse counts.

Primary NLL, bits/token and perplexity remain exact for the declared baseline.
Baseline top-1/top-5 metrics become optional post-hoc diagnostics. When enabled,
they run in a bounded worker pool after primary metrics are fixed and are timed
separately. They may not delay checkpoint writes or be included in model
throughput.

This is preferable to merely making the current dense ranking asynchronous:
algorithmic work is removed first, and optional ranking cannot silently consume
the CPU budget needed by training.

## Design 5: dual-constraint hardware scheduler

The online model remains sequential across tokens. Hardware utilization is
therefore managed at two levels.

### Hardware profile

At startup, collect:

- logical and physical CPU count;
- SIMD capabilities already detected by the core;
- process affinity and NUMA information when available;
- current available physical memory;
- total physical memory and page size.

No synthetic benchmark changes model parameters. A short optional calibration
measures baseline preprocessing throughput and per-run state-growth rate.

### Resource policy

Add runtime scheduler settings with defaults:

```
cpu_fraction = 0.90
memory_fraction = 0.90
minimum_free_memory_bytes = platform-safe reserve
max_workers = auto
```

The memory limit is based on currently available memory at dispatch time, not
theoretical installed capacity. A run receives a conservative memory estimate
from fixed structures, current model bytes, configured node/output bounds,
baseline tables and a calibrated growth margin.

The dispatchable memory budget is
`max(0, min(memory_fraction * available, available - minimum_free_memory_bytes))`.
The scheduler launches another isolated worker only when both are true:

- assigned CPU slots remain within `floor(available_logical_cpus *
  cpu_fraction)`;
- predicted committed memory remains below the smaller of the fractional limit
  and the minimum-free-memory reserve.

It periodically refreshes available memory. Pressure stops new dispatches; it
does not kill or reorder active runs. CPU utilization is a target, not a reason
to oversubscribe memory or change sample order.

### Parallelism boundary

- one model preserves ordered token updates;
- independent seeds/configurations run in separate worker processes;
- shard count preprocessing and optional ranking may use bounded threads;
- reductions merge in deterministic shard order;
- candidate scoring is parallelized only if profiling proves work exceeds thread
  launch/synchronization overhead and exact ordering is preserved.

## Instrumentation and acceptance gates

### Addressing

Measure prefilled models with 0, 100,000 and 1,000,000 stored nodes.

- candidate checks remain within the configured analytical bound;
- p95 token latency from 100,000 to 1,000,000 nodes grows by at most 15%;
- no token-path bucket operation scans resident count;
- `bucket_bits=10,16,20` empty-index bytes differ by at most 5%, excluding hash
  table minimum allocation;
- predictive results and freeze invariants remain within declared deterministic
  equivalence or an approved quality tolerance.

### Vocabulary output

Measure `V=4,096`, `50,257` and `250,000` before training.

- fixed output-structure bytes grow by less than 1 MiB across the range;
- target NLL work follows tree depth, not `V`;
- decoder work is bounded by `beam_width * ceil(log2(V))`;
- per-node decision entries never exceed the configured cap;
- three-seed held-out NLL regression versus the current sparse output is no
  worse than 1% unless separately reviewed and accepted.

### Baselines

- unigram/current/pair NLL matches the old formulas within `1e-7` on fixtures;
- the replacement multiscale baseline has a separately named JSON field and is
  compared against the old control before removal;
- on the 50,257-word 10k/10k smoke, synchronous baseline wall time is no more
  than twice measured model time;
- optional ranking time is reported separately.

### Scheduler

- no new worker starts above either configured threshold;
- observed committed memory remains below the 90% policy in stress tests;
- one-worker and multi-worker runs produce identical per-run result JSON except
  timing and process metadata;
- worker failure releases reservations and cannot corrupt another run;
- console output remains one compact progress line per worker state transition.

## Implementation and commit order

1. Add scaling benchmarks and diagnostics that reproduce all four failures.
2. Replace dense address arrays with the lazy bounded bucket directory.
3. Replace materialized vocabulary trees and paths with the implicit keyed tree.
4. Add bounded sorted local decision storage and run the capacity ablation.
5. Replace synchronous dense baseline ranking with sparse target-probability
   evaluation; retain optional post-hoc ranking.
6. Add hardware profiling, memory estimation and the dual-constraint process
   scheduler.
7. Run the full correctness suite, scaling gates, fixed real-corpus smoke and
   automated multi-seed quality checks.

Each numbered implementation stage is a separate commit. A failed quality or
latency gate rolls back that stage without retaining dormant complexity.

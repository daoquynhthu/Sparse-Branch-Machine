# Sparse Branch Machine Research

Sparse Branch Machine is a CPU-first research platform for learning with
persistent addressable state, bounded sparse execution, local updates and
explicit structural lifecycle. It is not a Transformer implementation and it is
not yet a trained language model.

## Current status

The repository currently supports:

- mathematical and externally supplied token streams;
- next-token categorical cross-entropy;
- a retained continuous-vector regression objective;
- sparse content-addressed nodes and learned control edges;
- adaptive address programs with proposal, frozen validation, acceptance,
  auditing and retirement;
- experimental computable address variants;
- experimental hierarchical sparse token output;
- stable shared libraries, C ABI, thin CLI and Python `ctypes` API;
- runtime parameter discovery and automated multi-seed search;
- strict frozen evaluation and deterministic synthetic regression fixtures.

The workspace now has a versioned memory-mapped shard layer and a compatibility
converter for an existing tokenized FineWeb-Edu artifact. That artifact lacks
recoverable source-document boundaries, a pinned source revision and a tokenizer
artifact, so it does not pass the strict R0 reproducibility gate. It can support
pipeline and compatibility experiments while acquisition of a fully traceable
document-level corpus remains required.

Read [`ROADMAP_REAL_DATA.md`](ROADMAP_REAL_DATA.md) before starting new model
work.

## Document map

Each document has one authoritative role:

| Document | Role |
|---|---|
| [`ISSUES.md`](ISSUES.md) | confirmed current implementation defects and experiment blockers |
| [`ROADMAP_REAL_DATA.md`](ROADMAP_REAL_DATA.md) | normative next-phase research and dataset plan |
| [`THEORY_ALIGNMENT.md`](THEORY_ALIGNMENT.md) | theoretical target, current alignment and open theory |
| [`DESIGN_NOTES.md`](DESIGN_NOTES.md) | implementation and architecture invariants |
| [`TOKEN_TASK.md`](TOKEN_TASK.md) | token objectives, synthetic fixture and real-data contract |
| [`API.md`](API.md) | stable C/Python interface and ownership rules |
| [`BUILDING.md`](BUILDING.md) | build, test, install and incremental compilation |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | chronological experimental record, including failures |
| [`Agent.md`](Agent.md) | mandatory constraints and workflow for future agents |

Do not treat `RESEARCH_LOG.md` as a current specification. When summaries
conflict with a canonical document, the canonical document wins.

## Architecture snapshot

A prediction step is conceptually:

```text
token/context
    -> bounded address-program execution
    -> small candidate set
    -> sparse active nodes and control edges
    -> local output aggregation
    -> prediction fixed
    -> target-dependent local update during training only
```

The project aims to decouple persistent capacity from per-token work:

\[
N_{stored}\gg N_{active},
\qquad
C_{step}\not\propto N_{stored}.
\]

Current address programs can select small combinations of historical positions
and include an experimental generic modular-difference form. They do not yet
provide content-conditioned variable binding or relation-following. Those remain
architecture research problems, not completed features.

## Current experimental boundary

The mathematical token task uses the same input/target/loss contract as
next-token training, but its source is a reproducible mathematical process rather
than text. It is useful for regression tests and controlled ablations.

The current branch also contains an experimental hierarchical output tree. It
reduces large-vocabulary storage and target-NLL work, but on the small synthetic
vocabulary it currently trails the dense output in quality. It is an enabling
experiment, not an accepted final output architecture.

See:

- [`TOKEN_TASK.md`](TOKEN_TASK.md) for the task contract;
- [`THEORY_ALIGNMENT.md`](THEORY_ALIGNMENT.md) for the theoretical interpretation;
- `research_results/` for archived run outputs.

## Build and test

```bash
cmake -S . -B build-fast -DCMAKE_BUILD_TYPE=Release
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests -j2
ctest --test-dir build-fast --output-on-failure
```

The project is split into shared libraries. Build only the affected target after
an incremental change. Full details are in [`BUILDING.md`](BUILDING.md).

## Run the mathematical token fixture

```bash
./build-fast/sbm_cli \
  --task token-ce \
  --sequences 64 \
  --sequence-length 2048 \
  --vocab-size 32 \
  --warmup 80000 \
  --output results.json
```

## Python API

```python
from sbm_runtime import Runtime

runtime = Runtime("build-fast/libsbm_api.so")

with runtime.token_dataset_from_ids(
    token_ids,
    vocab_size=tokenizer_vocab_size,
    sequence_offsets=document_offsets,
) as dataset:
    with runtime.config() as config:
        result = dataset.run(config, warmup=train_examples)
```

The in-memory API is ready for external token IDs, but real-corpus streaming,
versioned shards and checkpoint/resume are next-phase deliverables rather than
completed capabilities.

## Automated experiments

Do not calibrate accepted defaults by repeatedly editing source constants. Use
the runtime parameter registry and automated multi-seed scripts:

```bash
python scripts/tune.py \
  --library build-fast/libsbm_api.so \
  --task token-ce \
  --trials 27 \
  --seeds 7,11,19 \
  --jobs 3
```

Architecture ablations and negative results must be scripted and recorded in
`RESEARCH_LOG.md`.

## Non-claims

The repository does not currently establish:

- natural-language understanding;
- emergent syntax or semantics;
- content-conditioned variable binding;
- a task-independent topology criterion;
- a quality-matched CPU advantage over GPU models;
- competitive language-model perplexity.

Those claims require the real-data gates defined in `ROADMAP_REAL_DATA.md`.

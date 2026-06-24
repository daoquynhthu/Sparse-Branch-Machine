# Agent instructions

> **Purpose:** This file is the mandatory repository briefing and operating
> contract for coding or research agents. Read it before changing code,
> experiments, parameters or documentation.

## 1. Mission

This repository explores a CPU-first learning architecture based on:

- persistent addressable state;
- strong conditional control flow;
- bounded sparse activation;
- local learning and local structural change;
- model capacity that can grow without proportional per-token compute;
- auditable topology proposal, validation, acceptance and retirement.

It is intentionally not a conventional dense Transformer implementation. The
long-term research target is a self-organizing computation topology capable of
content-conditioned relation and binding, not merely a faster n-gram table.

## 2. Current repository state

The current research branch contains:

- token-ID next-token cross-entropy and retained vector regression objectives;
- shared libraries with a stable C ABI;
- a thin CLI and Python `ctypes` binding;
- runtime parameter discovery;
- automated multi-seed search;
- stable logical node IDs and movable physical storage;
- sparse content addressing and control edges;
- adaptive address-program lifecycle;
- experimental computable address variants;
- experimental hierarchical sparse token output;
- strict training/evaluation topology freeze;
- mathematical synthetic tasks for deterministic regression.

The current blocker is the absence of a real natural-language corpus in the
cloud workspace. The next accepted research phase is real-corpus acquisition,
tokenization, sharding, streaming and training as specified in
`ROADMAP_REAL_DATA.md`.

## 3. Canonical document map

Documents have non-overlapping responsibilities:

| Document | Authoritative responsibility |
|---|---|
| `ISSUES.md` | confirmed current implementation defects and experiment blockers |
| `README.md` | concise repository entry point, current status and navigation |
| `ROADMAP_REAL_DATA.md` | normative next-phase plan and research gates |
| `THEORY_ALIGNMENT.md` | theoretical target, current alignment and unresolved theory |
| `DESIGN_NOTES.md` | architecture invariants and implementation semantics |
| `TOKEN_TASK.md` | token objectives, mathematical fixture and real token data contract |
| `API.md` | public C/Python ABI and ownership contract |
| `BUILDING.md` | build, test, install and incremental compilation commands |
| `RESEARCH_LOG.md` | chronological facts, accepted and negative experiments; not normative |
| `Agent.md` | mandatory agent constraints and workflow |

When documents conflict, use this order:

1. explicit current user instruction;
2. `Agent.md` hard constraints;
3. `ISSUES.md` for confirmed current blockers;
4. `ROADMAP_REAL_DATA.md` for next work;
5. `THEORY_ALIGNMENT.md` for theory;
6. `DESIGN_NOTES.md` for implementation invariants;
7. `API.md` and `BUILDING.md` for interfaces and tooling;
8. `README.md` summaries;
9. `RESEARCH_LOG.md` only as historical evidence.

Do not copy a full result or specification into several documents. Put it in the
canonical document and link to it elsewhere.

## 4. Hard theoretical constraints

### 4.1 Do not silently return to dense LLM architecture

Forbidden as the central learning mechanism:

- dense global matrix layers;
- full-network backpropagation;
- attention over every stored state;
- scanning every node or every parameter per token;
- hiding a dense neural router inside a nominally sparse system.

Small local SIMD/vector operations are allowed. A dense baseline may be added for
comparison, but not substituted for the research machine.

### 4.2 Keep active work bounded

Every architecture change must report:

- persistent node and edge count;
- active nodes per token;
- candidates inspected per token;
- output decisions or vocabulary work per token;
- model bytes;
- training and frozen-evaluation throughput.

A quality gain obtained by making active work scale linearly with stored capacity
is not aligned with the project objective.

### 4.3 Keep meta-rules minimal

Do not add a growing library of hand-written linguistic or task-specific rules.
A new primitive requires:

- generic semantics;
- local proposal from an existing structure;
- explicit inputs and outputs;
- a tractable dependency graph;
- held-out validation;
- complete rollback/erasure;
- an ablation against a simpler structure.

Never introduce labels such as subject, object, entity, modifier or coreference
as supervision for the research machine.

### 4.4 Synthetic results are not language evidence

Mathematical token tasks remain regression fixtures. They may validate
correctness, complexity and lifecycle mechanics. They do not demonstrate syntax,
semantics, variable binding or natural-language generalization.

A theory claim about language requires real corpus evidence under
`ROADMAP_REAL_DATA.md`.

### 4.5 No target leakage

The scored prediction must be fixed before reading the target. The target may
then update local state and structure. A new node initialized from the target may
not alter the prediction or metrics of the same sample.

### 4.6 Strict evaluation freeze

During frozen validation/test the system must not:

- update node parameters or control edges;
- create, accept, reject, retire or prune programs;
- update topology credit or learning-only diagnostics;
- alter tokenizer, corpus order or split membership.

A test must verify this invariant after any lifecycle change.

### 4.7 Structure selection is not raw training-loss selection

Probes require a separate adaptation period and frozen validation period. Do not
accept structures from the same examples used to fit them.

The current criterion is known to be task-sensitive. New work should log
prequential codelength, structure description cost and execution cost. Do not
hide cross-task failure by adding objective-specific thresholds without a
separate, documented theory change.

## 5. Data constraints

### 5.1 Current priority

Acquire and use real natural-language data. Start with a small, pinned real
corpus, then a deterministic web sample. Do not spend the next cycle improving
only the mathematical benchmark.

### 5.2 Data hygiene

- Split by document before tokenization.
- Train tokenizers on the training split only.
- Preserve document boundaries in every shard.
- Deduplicate exact normalized content across splits.
- Pin source revision and preprocessing version.
- Store hashes and manifests.
- Never commit full corpora, large shards, credentials or host-specific paths.
- Do not use test data for parameter search or topology calibration.

### 5.3 Dataset claims

Tiny or synthetic story data may be used for pipeline smoke tests but does not
complete a real-data gate. Record corpus origin and license before downloading or
redistributing any text.

## 6. Engineering constraints

### 6.1 Modify incrementally

Do not rewrite the repository from scratch. Extend the current modules. Copy a
worktree before invasive changes if necessary. Preserve Git history.

Create a new branch only for a significant architecture change or when explicitly
requested. Ordinary fixes and experiments stay on the current branch.

### 6.2 Shared-library boundary

- `sbm_api` is the public ABI.
- CLI code must remain thin and call the C API.
- Python must call the shared library rather than reimplement model logic.
- C++ classes and STL containers must not cross the C ABI.
- Breaking ABI changes require a major API version bump.
- New runtime parameters must be added to the authoritative parameter registry,
  set/get handling, JSON schema and tests.

### 6.3 Runtime parameters, not recompilation

Expose experimental parameters through the runtime schema. Do not edit header
defaults repeatedly to perform a search.

Manual single-value runs are allowed for debugging or a declared ablation. Any
accepted calibration must use the automated multi-seed search scripts and must
include the unchanged default configuration as a control.

### 6.4 Build within the execution limit

Prefer target-level incremental builds:

```bash
cmake --build build-fast --target sbm_machine -j2
cmake --build build-fast --target sbm_api -j2
```

Do not trigger a clean full optimized rebuild after every edit. If build
dependencies cause the 120-second tool limit to be exceeded, fix the dependency
boundary rather than increasing monolithic compilation.

### 6.5 CPU-first implementation

Allowed and encouraged:

- predictable bounded loops;
- SIMD for local vectors and output decisions;
- cache-aware structure-of-arrays storage;
- stable logical IDs with movable physical slots;
- memory mapping and streaming;
- branch and cache profiling;
- fixed-budget candidate search.

Do not optimize an invalid algorithm. Correctness, no leakage and experimental
meaning precede throughput.

## 7. Required workflow

### Step 1 — inspect before editing

```bash
git status --short --branch
git log --oneline --decorate -8
```

Read at minimum:

- `README.md`;
- `ROADMAP_REAL_DATA.md`;
- `THEORY_ALIGNMENT.md`;
- `DESIGN_NOTES.md`;
- the relevant API/build document;
- recent `RESEARCH_LOG.md` entries.

Never discard unknown uncommitted changes.

### Step 2 — state one falsifiable hypothesis

Before implementation, define:

- the observed failure;
- the proposed minimal change;
- the expected metric and structural effect;
- the baseline and negative control;
- the rollback condition.

### Step 3 — make the smallest coherent change

Keep data, objective, routing, lifecycle and performance changes in separate
commits where possible. Do not combine a new task with a new model and a new
metric unless the dependency is unavoidable.

### Step 4 — run correctness tests

At minimum:

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests -j2
ctest --test-dir build-fast --output-on-failure
```

For dataset changes, also verify:

- deterministic shard hash;
- document boundary reset;
- split disjointness;
- save/reload equivalence;
- checkpoint continuation equivalence when implemented.

### Step 5 — run a fixed smoke experiment

Use a small fixed dataset and seed to catch crashes and semantic drift. A smoke
run is not evidence for acceptance.

### Step 6 — use automated experiments

Use multi-seed automated scripts for parameter or architecture decisions. Record:

- exact commit;
- dataset/tokenizer manifest hashes;
- configuration JSON;
- seed list;
- train/validation/test boundaries;
- full metrics and structural statistics.

### Step 7 — compare against strong controls

At least one simple statistical control must match the information available to
the proposed model. Beating a unigram is insufficient when a short-context table
explains the task.

### Step 8 — document negative results

Append material negative experiments to `RESEARCH_LOG.md`. Do not retain
ineffective complexity merely because it appears theoretically cleaner.

### Step 9 — update canonical docs only

- new next-phase requirement -> `ROADMAP_REAL_DATA.md`;
- theory change -> `THEORY_ALIGNMENT.md`;
- implementation invariant -> `DESIGN_NOTES.md`;
- interface change -> `API.md`;
- build change -> `BUILDING.md`;
- historical result -> `RESEARCH_LOG.md`;
- concise status/navigation -> `README.md`.

### Step 10 — commit and package

Commit only after tests pass. Do not commit build directories, corpora or large
generated shards. For handoff, create both a source ZIP and a Git bundle, then
verify the bundle by cloning and running at least the API test target.

## 8. Acceptance requirements by change type

### Algorithm change

- fixed-seed correctness run;
- multi-seed held-out result;
- strong baseline;
- active-work and memory report;
- no target leakage;
- strict-freeze test;
- negative result recorded if rejected.

### Performance change

- semantic result equality or declared numerical tolerance;
- before/after on the same dataset and commit configuration;
- training and evaluation measured separately;
- memory included;
- no reduced candidate budget hidden as an optimization.

### Dataset change

- source and license recorded;
- pinned revision;
- document-level split;
- deterministic tokenization and shard hash;
- no validation/test tokenizer leakage;
- boundary-reset test.

### API change

- C, C++ and Python tests;
- ownership documented;
- parameter schema synchronized;
- installed-tree test;
- API version updated when incompatible.

## 9. Known unresolved theory

Do not claim these are solved:

- content-conditioned variable binding;
- relation-following independent of absolute position;
- dependency-aware ablation for shared program fragments;
- task-comparable topology value;
- long-horizon credit without branch explosion;
- stable real-language topology across domains;
- quality-matched CPU advantage over neural baselines.

The present address language remains mostly positional. The experimental modular
difference operator does not solve relational addressing. The sparse output
prototype solves a scaling problem, not language structure.

## 10. Definition of an honest result

An honest result states:

- what was implemented;
- what data were used;
- what baseline was beaten or not beaten;
- what resource cost changed;
- which alternative explanation remains;
- whether the result transfers across seeds, splits, objectives or domains;
- what is still unknown.

Do not convert an interface milestone into an intelligence claim, an asymptotic
argument into a performance claim, or a synthetic result into a language claim.

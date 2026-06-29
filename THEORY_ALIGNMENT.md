# Theory alignment and unresolved architecture

> **Document role:** This document defines the theoretical target, states which
> parts of the current machine align with it, and records the unresolved theory.
> It does not define the next work schedule; use `ROADMAP_REAL_DATA.md` for that.
> It does not serve as an experiment diary; use `RESEARCH_LOG.md`.

## 1. The theoretical target

The project is investigating a learning machine whose dominant computation is:

\[
\text{address}
\rightarrow
\text{read local state}
\rightarrow
\text{branch}
\rightarrow
\text{small local computation}
\rightarrow
\text{local structural update}.
\]

The desired properties are:

- persistent capacity much larger than the active working set;
- content- and state-dependent execution paths;
- local learning without a dense global backward pass;
- structures that can be proposed, evaluated, retained and erased;
- a small set of auditable meta-rules rather than a hand-written cognitive
  ontology;
- eventual support for binding, relation-following and reusable computation;
- CPU suitability through bounded branches, irregular memory access and sparse
  work rather than dense matrix throughput.

The target is not merely a sparse neural network and not merely a faster lookup
table. The central theoretical question is whether reusable computation can
emerge from local prediction pressure and structural reuse.

## 2. What the current system has established

### 2.1 Sparse execution and persistent storage

The implementation has stable logical node IDs, movable physical storage,
bounded candidate selection, sparse active nodes and learned control edges. The
stored graph may grow while the active beam remains bounded.

This establishes an executable CPU-first sparse substrate. It does not by itself
establish abstraction or language capability.

### 2.2 Auditable structural lifecycle

An address-program candidate follows:

```text
proposal -> local adaptation -> frozen validation -> counterfactual ablation
         -> acceptance/rejection -> mature audit -> possible retirement
```

Rejected or retired structures can be physically reclaimed while unrelated
logical node IDs remain stable. Topology events are recorded in experiment
output.

This is a meaningful advance over fixed topology, but the current credit rule is
known to be sensitive to objective and data distribution.

### 2.3 Token-aligned objective

The model can consume token IDs, predict a categorical next-token distribution
and train under cross-entropy. This aligns the data/loss contract with language
training.

Token interface compatibility is not language competence. Until real text is
used, all language-level interpretations remain hypotheses.

### 2.4 Sparse output experiment

The current branch includes an experimental hierarchical token output. Target
NLL can be computed along a binary path and per-node output storage depends on
observed decisions rather than one full vocabulary vector.

This addresses a genuine large-vocabulary scaling problem, but it is not yet an
accepted final design:

- on the small mathematical vocabulary it currently loses quality relative to
  dense output;
- candidate top-k decoding is approximate under a fixed beam;
- the tree partition is generic rather than semantically learned;
- compatibility-corpus behavior is now measured, but document-level and
  provenance-complete behavior is not.

### 2.5 Computable address experiment

The current branch also contains an experimental generic modular-difference
address form. It asks whether selected token values can be transformed rather
than only copied into an address.

This does not solve content-conditioned addressing. It remains tied to selected
positions and cannot bind an item whose distance changes with sentence structure.

## 3. What the current address language cannot express

The accepted address-program core is still dominated by sparse selection of a
small number of historical positions. A program such as `[1,4]` represents a
joint positional condition. It does not express:

- “find the earlier item matching this content”;
- “bind the selected item and reuse it later”;
- “follow the relation learned between two persistent states”;
- “return to an event or entity regardless of its token distance”;
- “call a reusable subprogram with local bindings.”

Natural-language dependencies are often relations over content, not fixed lags.
For example, an agreement or reference relation may persist while arbitrary
material is inserted between the related items. Increasing positional arity or
adding arithmetic transforms does not resolve this limitation.

The next program-language redesign therefore requires a content-conditioned
selection and binding model. Candidate minimal concepts include `Bind`, `Match`
and `Follow`, but these are not yet approved instructions. The current bounded
`ContentMatch` and `ContentFollow` operators now provide explicit execution
frames, channel lineage and caller/prerequisite attribution; future operators
must extend that contract with typed state, reusable bindings and rollback
semantics rather than bypass it.
The implementation also emits an address dependency graph, so shared
prerequisites and multi-caller reuse can be audited at the channel level before
stronger call semantics are added.
Matched content bindings additionally have stable binding keys that ignore
absolute distance. This provides direct evidence about repeated reuse of the
same binding, but it is still an audit mechanism rather than full variable
binding or subprogram invocation.
Those keys are now retained in a bounded training-time reuse registry, so
repeated bindings survive beyond one step and can later be used by lifecycle
logic. The current implementation still does not route by that registry or call
subprograms through it.

## 4. Lifecycle challenge under compositional programs

Whole-channel ablation is tractable because a channel has a clear additive
output contribution. A compositional program is harder:

- one fragment may produce a binding consumed by several callers;
- deleting a prerequisite disables downstream fragments;
- observed loss change cannot be assigned entirely to one shared dependency;
- programs may have option value before they have direct predictive value;
- a fragment may be useful only through reuse in several contexts.

To preserve traceability, future programs should be created by local typed graph
edits, not arbitrary program search. Each fragment will need:

- explicit input and output state;
- parent/lineage information, at least as strong as the current channel-level
  lineage;
- caller and dependency tracking, at least as strong as the current
  caller/prerequisite ablations;
- a versioned execution meaning;
- rollback semantics;
- validation that separates direct contribution from dependent contribution.

Whether the present lifecycle can be generalized without combinatorial
explosion is unresolved.

## 5. Task-sensitive structure value

The same lifecycle parameters did not transfer from token cross-entropy to the
retained vector-regression task. This is not treated as a tuning bug. Raw credit
is affected by:

- loss scale and output dimension;
- source entropy and noise;
- event frequency;
- local learner convergence rate;
- validation-window length;
- structural sparsity and reuse.

Natural language is much more heterogeneous than either synthetic objective.
A fixed threshold on raw loss improvement cannot be assumed task-agnostic.

The intended direction is a codelength account:

\[
S(P)=\Delta C_{heldout}(P)-C_{describe}(P)-\lambda C_{execute}(P),
\]

where predictive gain and structural complexity are measured in comparable
units. Prequential nats or bits provide a common output metric for categorical
and probabilistic regression objectives.

This does not create a universal threshold automatically. It makes assumptions
and costs explicit and allows common, rare, fast-learning and slow-learning
structures to be compared more honestly.

## 6. Why real language data is now required

The mathematical generator is transparent and reproducible, but its dependency
structure is known and low-dimensional. It cannot reveal which content
relations real language actually requires, how often they occur, or whether a
candidate program transfers across documents and domains.

The repository now has compatibility-only FineWeb-Edu token blocks and a mapped
corpus runner, but lacks an admissible document-level corpus with source,
tokenizer and split provenance. Therefore:

- further synthetic improvements are engineering/regression evidence only;
- program-language expansion should pause unless needed for a concrete real-data
  diagnostic;
- lifecycle and output designs must be re-evaluated on real held-out text;
- accepted language claims remain governed by `ROADMAP_REAL_DATA.md`.

## 7. Current theoretical claims that are justified

It is justified to say that the repository contains:

- a working sparse addressable learning substrate;
- bounded dynamic execution over persistent state;
- local categorical and vector learning;
- a traceable lifecycle for relatively independent address programs;
- a tokenizer-aligned training contract;
- experimental sparse output and computable addressing.

It is not justified to say that it has demonstrated:

- emergent syntax or semantics;
- variable binding;
- content-dependent relation traversal;
- a task-independent topology criterion;
- general language modeling capability;
- CPU superiority at matched quality.

## 8. Next theoretical gates

Theory work should resume in this order after real data is available:

1. establish real-corpus baselines and failure diagnostics;
2. express structural value in prequential codelength and explicit complexity;
3. identify failures not explained by bounded positional programs;
4. propose one minimal content-conditioned primitive;
5. extend the implemented lineage and dependency-aware ablation contract to
   typed state, reusable caller graphs and rollback;
6. test transfer across documents, shards and seeds;
7. only then consider composition, calls or deeper program graphs.

The project should prefer one falsifiable primitive over a broad hand-written
instruction set.

# Design notes and architecture invariants

> **Document role:** This document records implementation semantics and invariants that code changes must preserve. It is not the next-phase plan and should not duplicate full experiment histories. Use `ROADMAP_REAL_DATA.md` for planned work and `RESEARCH_LOG.md` for chronological results.


## Architectural invariant

The machine is designed around:

- a potentially large persistent address space;
- a fixed candidate budget;
- a fixed active route width;
- local state and edge updates only;
- stable logical references despite physical compaction.

The intended scaling property is that total stored capacity may grow while average active work remains bounded.

## v2 storage model

`NodeId` is a monotonic logical address. `id_to_slot` maps it to a movable physical slot. Node metadata is stored in structure-of-arrays form. Pruning uses swap-removal, updates the moved logical mapping, and rebuilds bucket indices. Edges retain logical IDs, so physical movement does not rewrite the graph.

## Hierarchical hot/cold addressing

A flat address bucket was not robust: 100,000 irrelevant prefetched nodes reduced frozen evaluation accuracy to about 78%, although throughput remained stable. v2 therefore searches:

1. learned control edges;
2. hot bucket entries that have actually participated in learning;
3. the cold global bucket.

With this separation, the same 100,000-node stress test recovered 100% evaluation accuracy in the recorded run. At one million distractors, average candidates remained 24 and average active nodes remained 4; evaluation accuracy was 96.875%.

This is evidence that bounded work is feasible, not evidence of general intelligence or superiority over dense models.

## Hard unresolved problems

- Generalization beyond finite-context recurrence.
- Counterfactual credit for nodes that should have been selected but were not.
- Formation and reuse of compositional subprograms.
- Long-horizon routing without beam explosion.
- Learned physical locality and NUMA placement.
- Fair quality-to-energy comparison against neural baselines.

## v5 addressing and learning invariants

The address prefix must use the information capacity of the token alphabet rather
than a fixed byte representation. For alphabet size A, each recent token occupies
`ceil(log2(A))` bits in the locality prefix. Mixed lower bits retain longer-context
discrimination.

Prediction is causally ordered:

1. select route from information available before the target;
2. fix prediction and evaluation metrics;
3. compute target-dependent counterfactual credit;
4. update local state and optionally create structure.

Target-dependent creation must never alter the prediction being scored for the
same step.

Node learning uses two distinct statistics:

- global visits/loss for lifecycle and general utility;
- address-local visits/loss for structural specialization.

Only address-local evidence may justify splitting an address region.

Candidate aggregation is hierarchical. Exact content-addressed nodes receive a
reserved responsibility mass; control edges and neighboring regions receive a
bounded residual mass. This preserves exploration without allowing several weak
candidates to overwhelm one precise content match.

Every structural or routing mechanism must be judged against simple conditional
centroid baselines. A model score above the global mean is not sufficient when a
single-token or token-pair table explains most of the target.

## Adaptive multiscale addressing

A single convex mixture was structurally mismatched to the current tasks:
independent delayed factors must be added, not averaged. Earlier versions solved
this by fixing lag-1, lag-2 and lag-4 address namespaces. The theory-alignment
branch retains additive channels but no longer treats those lags as immutable
meta-structure.

`address_lags` now specifies only seed channels. In adaptive mode the machine
proposes additional temporal views under a bounded meta-rule. Each proposal has
its own namespace, nodes and local parameters. It first adapts, then freezes for
a validation tail. The decision criterion is exact whole-channel ablation on
that frozen tail, not training loss and not the sum of independent node scores.

Accepted channels remain subject to mature counterfactual auditing. A rejected
or persistently harmful channel is retired as a structural unit: all of its
nodes are erased, indexes rebuilt and route references cleaned while unrelated
logical node identities remain stable.

Responsibilities are normalized per enabled channel, so a busy address view
cannot erase another view. Exact-address nodes still receive a reserved mass;
control-edge and neighboring candidates receive only the bounded remainder.
The active work is limited by the beam even while persistent capacity grows.

Address-local vectors or logits remain immutable under foreign-context
retrieval. A control edge may alter selection probability, but it cannot rewrite
the destination's local knowledge from the source context.

For vector regression, exact nodes learn sequential local residual means. For
token cross-entropy, enabled channels add local logits before one softmax. The
fixed `[1,2,4]` model remains available with `adaptive_topology=false` as a
strong structural control. The adaptive model must be compared against it rather
than being credited merely for beating a unigram baseline.

## Token cross-entropy objective

Token alignment is defined at the data/loss boundary, not by declaring the
mathematical process to be language.  A token sequence supplies input `x_t` and
target `x_{t+1}`; the machine produces normalized categorical probabilities and
receives exact cross-entropy.

The node store is reused as local logits.  Active temporal channels add logits,
then one softmax is evaluated over the output vocabulary.  Exact-address nodes
receive the local cross-entropy gradient proportional to their routing
responsibility.  Foreign nodes remain read-only, preserving the separation
between address-local knowledge and control-edge credit.

Sequence boundaries reset only transient execution state.  This is required
before corpus integration: concatenating unrelated documents would otherwise
create false lag relations and false control edges.

The mathematical generator contains no hidden stochastic state.  Its systematic
conditional distribution is fully determined by visible token history; the only
irreducible uncertainty is the categorical sample.  Oracle NLL is stored so that
model error can be separated from source entropy.

The repository retains a dense full-vocabulary logit path as a correctness control and includes an experimental hierarchical sparse output path. The sparse path stores observed binary decisions and computes target NLL along a tree path, but it is not yet the accepted final output architecture. Output designs must be compared on held-out quality, bytes, target-NLL cost and candidate decoding cost, especially on real tokenized text.

## Global and local token output

Sparse token prediction decomposes every implicit-tree decision into a shared
base logit and bounded address-local residuals:

```text
logit(d) = global_count_logit(d) + sum_i responsibility_i * residual(i, d)
```

The global term uses Jeffreys-smoothed left/right counts and is updated only
after the current prediction, ranking and counterfactual credit are fixed. The
two count arrays and derived logit cache are global `O(V)` state; local output
remains capacity-bounded per node. Prediction and update touch only target or
beam paths and remain `O(log V)` for fixed beam width. Frozen evaluation does
not update global counts.

The implicit output decomposition is keyed by `output_tree_seed`, not model
`seed`. Multi-seed comparisons must keep `output_tree_seed` fixed.

Each local sparse decision owns its own visit count, coding-gain EMA and last
update step. Local learning-rate maturity is decision-local. At capacity,
probation-aware retention uses coding benefit and visit evidence; logit
magnitude is not an importance measure. Eviction/reconstruction and saturation
remain explicit diagnostics because fixed capacity can still produce churn.

For multiple represented address channels, raw seed/residual channel weights
are normalized to total mass one before exact/non-exact subdivision. Changing
channel count therefore cannot increase confidence merely by increasing total
responsibility mass.

Token residual logits are learned in channel-local coordinates. Inference still
aggregates local residuals with global node responsibility, but dense and sparse
token updates use responsibility normalized by the represented channel's active
mass. This prevents adding a validated channel from quadratically diluting the
seed channel's already learned short-context distribution.

## Sparse address programs

The adaptive topology object is an `AddressProgram`, not a task-specific lag
channel. A positional program contains one or two sorted positive history
offsets; current token inclusion is implicit. Proposal order expands by
temporal radius and then arity. The acceptance rule is unchanged: local
adaptation, frozen validation, exact whole-program ablation, then accept or
erase.

The two-offset address uses coarse-to-fine storage. The current token and first
offset define the coarse region; the second offset and longer context refine the
full prototype inside that region. A fully joint top-level hash was tested and
rejected because it produced sample-starved addresses.

`ContentMatch(max_lag)` is the first content-conditioned address primitive. It
searches only the retained history, bounded by `max_lag`, for the nearest prior
token equal to the current token. Its address key encodes the current token, a
match-exists bit and the historical successor following the matched position;
the matched distance is mixed into the residual signature bits.

`ContentFollow(pattern_lag, max_lag)` is the first bounded Bind/Follow variant.
It matches the current token plus the token at `pattern_lag` against a previous
occurrence inside `max_lag`, then follows the historical successor after the
matched occurrence. More generally, all lags before the final lag are local
pattern constraints and the final lag is the search radius. This is a generic
operation over token identity and sequence order. It does not use target tokens,
linguistic labels, document metadata or an unbounded program search, and it
remains subject to the same probe/validation/rollback lifecycle as other
address programs. These operations are prototypes rather than a full binding,
caller or dependency-tracking system.

Evaluation is strictly read-only. `freeze_topology()` rejects incomplete probes
at the training boundary, and counterfactual credit is not accumulated on
evaluation examples.

## Real-corpus invariants

The next phase is governed by `ROADMAP_REAL_DATA.md`. Any real-corpus implementation must preserve these design invariants:

- documents are explicit units and reset transient execution state;
- tokenization and split membership are immutable experiment inputs;
- data are streamed or memory-mapped rather than loaded as one monolithic vector;
- checkpoint state includes the exact corpus cursor and RNG state;
- stored capacity, active work and output work are reported separately;
- validation/test examples never update parameters, topology or credit;
- synthetic mathematical data remain a regression fixture, not evidence of language capability.

## Program-language boundary

Current programs remain primarily positional selectors, with experimental
generic modular-difference, bounded content-match and bounded content-follow
variants. `ContentMatch` and `ContentFollow` may be described as
content-conditioned address primitives, but not as complete binding, call or
attention mechanisms. Stronger relation primitives still require explicit
state, local lineage, dependency-aware ablation and complete rollback. They may
not encode linguistic labels or rely on an unbounded search over arbitrary
programs.

The next architecture milestone is not another address-operator sweep. The
machine must first promote address programs from signature recipes to explicit
execution objects with frames, bindings, lineage, dependency-aware ablation and
recoverable lifecycle state. Until that exists, experiments may diagnose but
may not claim that the neuron-like address mechanism is complete.

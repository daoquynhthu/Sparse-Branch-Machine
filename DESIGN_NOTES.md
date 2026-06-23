# Design notes and current findings

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

The current full-vocabulary local logit vector is intentionally a correctness
implementation.  It must not be mistaken for a final large-vocabulary output
architecture.  Hierarchical, adaptive or sampled normalization will be required
before 30k+ vocabularies are memory-efficient.

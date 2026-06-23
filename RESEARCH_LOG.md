# Research log

## Baseline

Imported v2 unchanged as the first Git commit. The baseline couples stable logical IDs to movable physical slots, uses hot/cold address buckets, and reaches 96.875% frozen evaluation accuracy on the default deterministic FSM dataset.

## Architecture v3

This branch adds architectural semantics without replacing the v2 storage or execution core:

- finite-horizon route traces and decayed node credit;
- edge eligibility state;
- explicit node lifecycle: cold, warm, mature, dormant;
- parent-linked specialization nodes;
- structural split diagnostics;
- conservative redundant-node merge;
- periodic merge control in the benchmark CLI.

### Negative result retained

An early revision treated a wrong final prediction as negative evidence against every traversed control edge. Frozen evaluation accuracy fell from 96.875% to 78.125%. The inference was invalid: outcome failure does not identify which traversed edge was causally wrong. That edge punishment was removed. Current code keeps delayed node-level utility credit but only reinforces traversed edges weakly after errors until a counterfactual routing test exists.

### Current default result

On the same deterministic dataset:

- train accuracy: 99.73%;
- frozen evaluation accuracy: 93.75%;
- 100 live nodes;
- 54 conflict-triggered specialization events;
- about 4 active nodes and 18.4 examined candidates per step.

This is still below the v2 accuracy baseline. The branch is therefore an architectural research branch, not a replacement release. The regression is explicit and should be resolved by improving specialization selection rather than reverting to unconstrained growth or hiding the result.

## 2026-06-23 — vector-prediction-v4

Replaced the categorical hidden-FSM benchmark with token-conditioned continuous
vector prediction. The change was made in-place on the existing repository and
preserves stable logical IDs, SoA node storage, hot/cold indexes, sparse control
edges and lifecycle machinery.

### Failed calibration A: hidden stochastic dynamics

The first vector generator included a hidden continuously evolving state and
random token innovations that influenced future state. With 40k learning steps,
the model created nearly 40k nodes and frozen R2 was negative. This task mixed
learnable structure with irreducible hidden randomness and was rejected.

### Failed calibration B: address under-segmentation

The first locality-preserving signature placed older tokens in the most
significant bits. Only 18 nodes formed and R2 remained near zero. The byte order
was corrected so the most recent token dominates the address region.

### Failed calibration C: edge candidates displaced exact-address candidates

Incoming edge candidates could fill the budget before exact bucket residents
were considered. This caused 25k+ nodes despite only a small number of useful
address regions. Candidate ordering now guarantees exact-region retrieval first,
then learned edges, then nearby-region exploration.

### Accepted baseline

The final generator is stationary and observationally learnable. Targets combine
current and delayed token embeddings, a history-derived regime, nonlinear feature
interactions and weak Gaussian noise. Structure growth is triggered by address
novelty rather than individual high-loss samples.

Seed 7, 120k steps, 40k learning / 80k strict freeze:

- 256 live nodes
- 8,192 sparse edges
- 5.9995 active nodes per step
- 35.28 candidates per step
- train R2 0.4009
- frozen evaluation R2 0.3995
- frozen evaluation cosine 0.7664
- frozen evaluation NRMSE 0.7625

AVX2/FMA runtime dispatch is active for vector dot, distance, aggregation and
local update kernels. Performance is recorded only as a diagnostic; v4 remains
an architecture/algorithm iteration, not a performance-optimization phase.

## 2026-06-23 — vector-prediction-v5

v5 was developed incrementally from `vector-prediction-v4`; no replacement
project was created.

### Correctness defect: target leakage

v4 initialized a newly created node from the current target, inserted it into the
active set, recomputed the prediction and then recorded the training metric. The
current target therefore improved its own prediction. v5 fixes the prediction
before any target-dependent creation or update. A regression test compares two
fresh models receiving opposite first targets and requires identical zero first
predictions.

### Experimental defect: trivial token baseline

The accepted v4 task was still too simple. On re-evaluation with proper
baselines, the current-token centroid reached R2 about 0.955 and the token-pair
centroid about 0.975. The reported model R2 therefore did not establish learning
of a medium-complexity relation.

The generator now uses a broad-support token process and balances current and
delayed token contributions. Seed-7 baseline R2 values are approximately 0.303
for current token and 0.520 for the previous/current pair.

### Address defect: wasted symbol bits

The v4 locality signature reserved eight bits per token. With a 64-symbol
alphabet, two bits per token were always zero. A 12-bit bucket consequently
encoded only the current token and the upper two meaningful bits of the previous
token. v5 packs symbols using the alphabet bit width; 12 bits now represent the
complete previous/current pair.

### Credit and mixture defects

v4 updated nearly every beam member toward the target, even when the member had
little responsibility or harmed the aggregate. It also averaged exact, edge and
neighbor candidates in one softmax. v5 adds:

- exact-region reserved responsibility mass;
- leave-one-out contribution estimates;
- responsibility-weighted local learning;
- no update for weak non-exact candidates without positive contribution;
- bounded positive edge reinforcement and source-local decay;
- edge strength as a routing prior.

A pure exact-region ablation (`exact_region_mass = 1`) produced seed-7 R2 about
0.460, versus about 0.480 at the accepted 0.86 mass. Sparse historical routes are
therefore useful in the current task, although they do not yet surpass the pair
centroid baseline.

### False structural splitting

Node visit counts included occasions when a node appeared through an incoming
edge or neighboring bucket. Those counts were incorrectly treated as evidence
that the node's own address region needed specialization. v5 adds address-local
visit and residual statistics. With conservative defaults, the seed-7 model
stabilizes at 4096 nodes rather than 5366 false specializations.

An aggressive local split trial (minimum six local visits and lower residual
threshold) created 8325 nodes. Seen-context R2 rose to about 0.506, but unseen-
context R2 fell to about 0.456 and overall R2 fell to about 0.463. The trial was
rejected and recorded as evidence that local multimodality must be justified by
out-of-sample gain, not training conflict alone.

### Accepted multi-seed result

Seeds 7, 11 and 19, each with 40k learning and 80k strict-freeze steps:

- mean model evaluation R2: 0.4805;
- mean current-token baseline R2: 0.3031;
- mean token-pair baseline R2: 0.5210;
- mean seen-three-token-context R2: 0.5178;
- mean unseen-three-token-context R2: 0.4744;
- 4096 live nodes for each seed;
- about 72.4k sparse edges on average.

The accepted interpretation is limited: v5 repairs several invalidating defects
and shows useful sparse historical correction, but it has not yet exceeded a full
pair lookup baseline or demonstrated compositional abstraction.

## 2026-06-23 — modular-build-v5

The v5 implementation was split into independently compiled targets without
changing model semantics. The previous single translation unit was divided into
SIMD/signature core, machine storage, routing, learning, maintenance, dataset and
experiment modules. Public declarations were also split under `include/sbm/`,
while `sparse_branch_machine.hpp` remains a compatibility umbrella.

A fixed 20,000-step, seed-7 equivalence run produced zero differences in all JSON
fields except wall-clock timing. Unit tests passed before and after the refactor.
Touching only `machine_learning.cpp` rebuilt one translation unit plus archive and
executable relinks; dataset, SIMD, signatures and experiment sources were not
recompiled. On the current workspace this incremental build completed in about
3.6 seconds, well below the 120-second execution ceiling.

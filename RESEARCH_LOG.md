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

## 2026-06-23 — multiscale additive vector architecture

Work continued directly on `modular-build-v5`; no new branch was created.

The v5 model used one pair-address channel and a convex mixture of candidate
vectors. That representation was structurally unable to add independent delayed
factors. Three independent temporal views were introduced at lags 1, 2 and 4:
the first stores the primary prediction, and the latter two store sequential
additive residuals.

A first gradient-style residual implementation reached seed-7 frozen R2 about
0.699. A strong fixed-table control using the same lag-1/2/4 decomposition reached
about 0.824, exposing underfitting rather than an addressing failure. Each node
saw only about ten local samples, so the inherited fixed EMA learning rate was
replaced by an online local mean. Residual stages receive a mild recency
pseudocount to track moving upstream means.

Another correctness issue was removed: a node reached through a foreign control
edge was being updated toward that foreign context's target, corrupting the
node's own address-local vector. Foreign nodes are now read-only and only the
control edge receives credit.

Accepted runtime calibration across seeds 7, 11 and 19:

- exact-region mass: 0.88;
- residual channel gain: 1.0;
- residual recency pseudocount: 0.75;
- edge score weight: 0.32.

Frozen evaluation R2 values are 0.8058, 0.8108 and 0.8062, mean 0.8076. The fixed
multiscale table scores 0.8240, 0.8273 and 0.8223. Unseen three-token-context R2
remains above 0.803 for all three seeds.

Negative routing experiments were retained:

- forcing exact-region mass to 1.0 reduced seed-7 R2 by roughly 0.006, showing a
  small positive contribution from sparse branch candidates;
- reinforcing all three channels with same-channel-only edges increased graph
  size and candidate count while reducing R2 to about 0.799;
- widening unrestricted reinforcement from two to three route positions roughly
  doubled edge count without measurable quality gain.

The accepted implementation therefore keeps bounded two-position edge
reinforcement. Runtime mixture parameters are now exposed through the CLI so
future sweeps do not trigger full header recompilation.

## 2026-06-23 — shared-library API and automated calibration hygiene

Work continued on the existing `modular-build-v5` branch. No additional branch
was created because this is an engineering-interface change rather than a model
architecture change.

The previous executable linked the C++ experiment stack directly and maintained
its own list of selected hyperparameter flags. It has been replaced by a thin
`sbm_cli` that includes only the stable C header and links directly only to
`sbm_api`. The runtime is now split into shared libraries for core SIMD/signature
operations, machine logic, datasets, experiments and the C ABI facade.

The C ABI uses opaque config and dataset handles. Python loads it with a
standard-library-only `ctypes` wrapper, allowing immutable datasets to be reused
across many in-process experiments. C++, C ABI and Python smoke/regression tests
all pass.

All current `Config` fields are exposed through a generic `name=value` registry.
The same registry returns JSON metadata containing defaults, types, ranges and
search policy. CLI and Python therefore no longer maintain separate parameter
setter lists.

Manual repeated calibration is replaced by `scripts/tune.py`. The script samples
ranges returned by the shared library, evaluates many candidates on one seed,
uses successive halving across additional seeds, checkpoints after each stage and
writes a directly reusable best-config JSON file. A short four-candidate/two-seed
smoke search completed through the in-process API.

A fixed 20,000-step seed-7 semantic regression was executed before and after the
DLL/API conversion. Ignoring only wall-clock fields, the result JSON had zero
field differences. `ldd` confirms that the CLI directly depends on `sbm_api`; the
remaining model libraries are transitive shared-library dependencies.

## 2026-06-23 — tokenizer-aligned mathematical cross-entropy task

Work continued on `modular-build-v5`; no new branch was created.

The primary task was changed from continuous-vector regression to a standard
next-token categorical contract.  The data source is still mathematical rather
than linguistic: an explicit autoregressive distribution combines lag-1,
lag-2, lag-4 and a weaker nonlinear interaction term, then samples the next
token.  The dataset records oracle negative log-likelihood for each generated
token.

A new `TokenDataset` stores flat token IDs plus sequence offsets.  Sequence
boundaries clear history, previous routes and delayed trace credit while
retaining learned nodes.  The C and Python APIs can construct the same dataset
from arbitrary external tokenizer IDs, so later corpus integration does not
require changing the model call.

The machine now has a token-cross-entropy objective.  Active node vectors are
interpreted as local logits, combined additively and normalized by softmax.
Exact-address nodes receive local cross-entropy gradients; foreign edge-recalled
nodes remain read-only.  Label smoothing, temperature, logit decay and token
learning rates are exposed through the runtime registry.

Default calibration uses a 32-token vocabulary, 64 sequences of length 2048 and
80,000 learning examples.  Across seeds 7, 11 and 19, frozen NLL values are
3.4125, 3.4100 and 3.4114 (mean 3.4113).  The unigram baseline is about 3.4658,
the fixed multiscale conditional-table control about 3.3920, and the generator
oracle about 2.9143.  Top-1 remains near 6.2%, which is expected for the broad
stochastic target distribution.

The automated tuner was updated to minimize cross-entropy for token tasks,
filter parameters by task applicability and always include the default
configuration.  A nine-candidate, three-seed successive-halving run retained the
default configuration as best; no manually selected override was accepted.

## 2026-06-23 — theory-alignment-v9: learned address topology

A new `theory-alignment-v9` branch was created from committed token-training
baseline `b47713b` in a separate Git worktree.  The existing dirty worktree was
left untouched.

The fixed `lag-1/2/4` topology was replaced in the default research mode by one
seed channel and an explicit topology lifecycle.  Candidate lag channels are
proposed one at a time, trained locally, frozen for an out-of-sample validation
tail and evaluated by exact whole-channel ablation.  Positive validation credit
accepts the channel; rejection deletes all of its nodes and rebuilds indexes.
Accepted channels receive continuing counterfactual credit and can be retired
when mature credit is strongly negative.

An earlier version selected structure on the same samples used to train the
probe.  It accepted spurious lag-5 and lag-9 channels on one seed.  This was
rejected as training-error model selection.  The accepted implementation adds a
frozen validation tail and records every proposal, acceptance, rejection and
retirement as a `TopologyEvent`.

Three-seed full-task results are intentionally below the fixed-topology control:

- adaptive seed `[1]`: mean frozen NLL 3.41847;
- fixed `[1,2,4]`: mean frozen NLL 3.41132.

All adaptive runs end with `[1,2]`.  In seed 7 the model proposes lag 4 at step
14,336 and rejects it at step 18,432 with validation credit -0.00405.  It briefly
accepts lag 5 and lag 6, then mature auditing retires both after their credit
falls below the conservative negative threshold.  This demonstrates a complete
and auditable structural lifecycle, while also showing that the current local
learner does not yet assign positive reusable value to the known lag-4 factor.

## 2026-06-23 — theory alignment v10: sparse address programs

Work continued on the existing `theory-alignment-v9` branch. No new branch was
created.

The lag-only topology object was generalized to an `AddressProgram`: a sparse
set of at most two history offsets with the current token implicit. Programs are
enumerated by increasing maximum lag and arity, then passed through the existing
adapt/freeze/ablate/accept/retire lifecycle. The proposal mechanism contains no
knowledge that the generator uses lags 1, 2 and 4.

Three full seeds produced frozen NLL 3.37074, 3.36783 and 3.36070, mean 3.36642.
The fixed multiscale conditional-table control averaged 3.39199; fixed singleton
channels `[1,2,4]` averaged 3.41819; adaptive singleton-only topology averaged
3.42692. This is the first learned-topology result that beats the strong fixed
statistical control.

A proposed address rewrite that hashed all program operands into the top-level
bucket was rejected. It increased sparsity and worsened mean NLL to 3.42766.
The accepted implementation preserves a coarse-to-fine address hierarchy.

Strict-freeze auditing found two leaks. Program retirement could occur after the
training boundary, and evaluation examples continued to update credit EMAs.
Both were fixed: unfinished probes are resolved at the boundary, all structural
and credit mutation stops during evaluation, and no learning-only
counterfactuals are computed there.

Performance changes were semantics-preserving: reusable buffers, contiguous
history, direct ablation log-sum-exp, channel/node credit reuse, SIMD logit
updates and skipped evaluation credit. A direct seed-7 before/after comparison
raised throughput from about 68k to about 89k steps/s with identical predictive
and graph metrics. Full-run throughput remains topology-dependent.

A 12-candidate three-seed automated search over topology lifecycle parameters
and maximum program arity retained the default configuration. No manual override
was adopted.

The retained vector objective exposes an unresolved transfer problem. With the
token-calibrated structural lifecycle, adaptive discovery keeps only `[1]` and
reaches about `R2=0.528`, whereas fixed `[1,2,4]` remains about `R2=0.805`. This
was not patched with objective-specific proposal priors or manually lowered
thresholds. The result is retained as evidence that topology credit is not yet
task-independent.

A deterministic `scripts/theory_ablation.py` runner now reproduces the three
accepted architecture controls through the shared C API, so future comparisons
do not depend on manually assembled CLI commands.

## 2026-06-23 — documentation consolidation and real-data transition

The project reached a data-availability boundary rather than a conclusion about
synthetic tasks. The current cloud workspace lacks a real natural-language
corpus, so mathematical token experiments can no longer support the next theory
claims. They remain deterministic regression fixtures.

Documentation responsibilities were consolidated:

- `README.md` is the entry point and current-status summary;
- `ROADMAP_REAL_DATA.md` is the normative plan for acquiring and training on real
  corpora;
- `THEORY_ALIGNMENT.md` contains theory and unresolved architecture;
- `DESIGN_NOTES.md` contains implementation invariants;
- `TOKEN_TASK.md` defines token objectives and data contracts;
- `API.md` and `BUILDING.md` contain interface and tooling contracts;
- `RESEARCH_LOG.md` remains chronological and non-normative;
- `Agent.md` defines mandatory agent constraints and workflow.

The next primary milestone is a pinned real corpus, tokenizer trained on the
training split only, versioned document-aware token shards, streaming dataset
access, exact checkpoint/resume, strong statistical baselines and held-out
language experiments. Further improvements limited to the mathematical token
benchmark are not considered language or theory milestones.

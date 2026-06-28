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

## 2026-06-23 — mapped shards and NAIME compatibility audit

A versioned little-endian token shard format, zero-copy memory-mapped reader and
explicit `(shard, sequence, token)` cursor were added. C++, C and Python tests
verify deterministic bytes, no cross-sequence targets and exact next-example
replay after reopening a shard. Full payload verification scans mapped memory
without materializing another token vector.

The existing `fineweb_edu_1b_ctx1024` artifact was audited. It contains 975,610
train rows and 8,908 validation rows of 1,025 tokens, with observed GPT-2-sized
token IDs. The preparation record identifies `HuggingFaceFW/fineweb-edu` and
`sample-10BT`, but the saved artifact does not retain source document IDs,
original boundaries, source revision, tokenizer files/checksum, license, or
cross-split deduplication evidence. It is therefore classified
`compatibility_only`, not accepted as Phase R0 or D1.

A 20,500-token-per-split smoke conversion was run twice in clean external
directories. Both runs produced manifest SHA-256
`9408851cc48f92ea19673ec73391d687e86fd93296aaf3943b83389de3a6eea7`, and every
shard hash matched byte-for-byte. The C++ shared library opened and payload-
verified all four Python-generated shards. This validates the conversion and
cross-language format contract; it is not a model-quality experiment.

The mapped training path was then exercised on one 10,250-token train shard with
8,000 training and 2,240 frozen-evaluation examples, fixed topology and sparse
token output. It completed without materializing the shard. Model-only
throughput was 5,443 token/s, with 602 live nodes, 5,000 edges, mean 5.99 active
nodes, 38.51 candidates and 1.53 MB estimated model state.

This smoke result is predictively poor: evaluation NLL was 10.7596 versus 8.5245
for the unigram control. It is not accepted evidence for the architecture. The
full command took about 26 seconds while measured model time was 1.88 seconds;
dense 50,257-way statistical baseline evaluation dominates current end-to-end
smoke latency and must not be confused with learner throughput. Proper corpus
experiments still require separate train/validation shard sets and persistent
multi-shard model state.

A split-correct mapped corpus runner was then added. It accepts non-contiguous
train and validation mappings, resets state at every stored sequence, trains on
all train shards, freezes topology once and performs validation without updates.
C++, C and Python tests cover independent split counts and the freeze boundary.

An actual one-train-shard/one-validation-shard run used 10,240 examples in each
split. It produced evaluation NLL 10.7673 versus unigram 8.5815, current-token
9.7295, pair 10.0390 and multiscale 9.8390. The machine retained 676 nodes and
5,920 edges, averaged 5.99 active nodes and 41.10 candidates, used 1.87 MB
estimated model state and reported 4,429 model token/s. The command took about
80 seconds versus 4.62 seconds of measured model time because exact dense
baseline ranking dominates the 50,257-way validation path. The architecture is
therefore still substantially behind even trivial language controls at this
training scale; the run validates split semantics, not model quality.

## 2026-06-23 — bounded address and sparse-output scaling

Release tests had been compiled with `NDEBUG`, so their C++ `assert` checks were
not executing. Test targets now explicitly enable assertions. This exposed and
corrected one impossible topology-test configuration whose 512-step warmup
exceeded its 48-step probe lifetime.

The dense empty-bucket directory was replaced by occupied-bucket states with
bounded hot and cold candidate indices. Empty address-index storage for
`bucket_bits=10,16,20` changed from 57,344, 3,670,016 and 58,720,256 bytes to 8
bytes in all three cases. A deliberately colliding 20,000-node bucket changed
from 10,138 inspected entries to the configured limit of 16. Batched p95 token
latency between 100,000 and 1,000,000 prefetched nodes varied by roughly
0.84x–1.14x after the bounded cold sample replaced full resident rotation.

The materialized vocabulary tree and per-token path table were replaced by a
keyed affine permutation and implicit balanced intervals. Fixed output-structure
storage for vocabularies 4,096, 50,257 and 250,000 is now 120, 152 and 168 bytes,
versus 442,372, 8,922,232 and 39,747,076 bytes before the change. Three small
fixed-seed comparisons against the materialized-tree commit changed held-out NLL
by +0.070%, -0.149% and +0.025%, within the declared 1% gate.

Per-node sparse decisions are now sorted, binary-searched and capacity-bounded.
On a 4,096-token broad-context fixture, the unbounded control stored 596–651
decisions per node and used 450–478 KB. Capacities 64, 128 and 256 were tested on
seeds 7, 11 and 19. Capacity 64 used 92–93 KB, approximately doubled measured
model throughput from 12k to 25k token/s, and did not worsen NLL on any seed, so
64 became the default hard bound.

The synchronous statistical baselines now query only the target probability;
they no longer allocate, normalize or rank 50,257-way distributions per
validation token. Small-corpus direct-query NLL matched the previous unigram,
current-token and pair controls within `4e-8`. The former geometric multiscale
control was replaced by the explicitly named arithmetic interpolation of the
lag-1, lag-2 and lag-4 target probabilities; on that comparison fixture its NLL
was 7.8259 versus 7.7703 for the old, different estimator. Ranking fields are
therefore reported as unavailable rather than zero.

On the existing 10,240-train/10,240-validation FineWeb-Edu compatibility smoke,
model work took 0.924 seconds and baseline work 0.016 seconds, with end-to-end
wall time 0.990 seconds and measured model throughput 22,172 token/s. This
removes the previous 80-second baseline bottleneck. Evaluation NLL remained
poor at 10.8682 versus unigram 8.5815; the performance repair is not evidence of
predictive quality.

The experiment tuner now defaults to isolated worker processes governed by
simultaneous live CPU and available-memory limits. At 0.9/0.9 on the local
32-logical-CPU host, a 24-candidate synthetic stress run launched 24 workers,
completed without failure in 15.2 seconds, reached 1.17 GiB peak aggregate
worker RSS and averaged 63.5% whole-system CPU. The 28-slot CPU ceiling was not
reached because only 24 candidates existed. Automatic and fixed one-worker
execution produced identical candidate ordering and per-candidate NLL in the
integration fixture.

## 2026-06-23 — medium FineWeb-Edu compatibility run

A deterministic compatibility subset was converted from the local NAIME
artifact with 999,375 stored training tokens and 99,425 validation tokens.
Because the source artifact still lacks original document identity, tokenizer
provenance and deduplication evidence, this run is a scaling and failure-
diagnosis experiment, not an admissible D2 language-quality result.

Three fixed-topology seeds (7, 11 and 19) completed concurrently under the
0.9/0.9 scheduler in 90.2 seconds. Each consumed 998,400 train examples and
99,328 validation examples. Mean model throughput was 12,841 token/s; mean
reported model time was 85.49 seconds and baseline time 2.23 seconds. The final
machines retained about 19,253 nodes, 329,565 edges, 1.13 million bounded sparse
output entries and 22.0 MB estimated state.

Predictive quality failed decisively. Mean validation NLL was 10.6696 with
seed standard deviation 0.0253, versus unigram 7.6826, current-token 8.3085,
pair-context 9.3042 and interpolated multiscale 8.6670. Training NLL was also
10.6169, so the gap is not explained by ordinary validation overfitting. The
bounded scaling implementation is operational, but the current learning rule
does not exploit even frequency structure at this corpus scale. Architecture
diagnosis, rather than larger training, is now the blocking research task.

## 2026-06-24 — global hierarchical output prior

The medium-corpus failure was traced first to the absence of a shared output
base distribution. A single Jeffreys-smoothed hierarchical count prior was
added, with bounded local node logits reinterpreted as residuals. The output
tree received an independent `output_tree_seed=7`, held fixed across model
seeds. Unit tests verify score-before-update behavior, analytic branch
probabilities, frozen count state, decoding through the prior and seed
decoupling.

On the original one-shard 10,240/10,240 comparison, prior-only validation NLL
was 8.5877, the full fixed-topology model reached 8.58149 and the unigram control
was 8.58146. This closes the previous 10.8682 failure without changing data or
routing. The cached prior used 1.01 MB and the full smoke ran at 16,819 token/s.

The 998,400-train/99,328-validation gate completed on seeds 7, 11 and 19. NLLs
were 7.60671, 7.60692 and 7.60718 (mean 7.60693, standard deviation 0.00023),
beating the common unigram control 7.68264 by 0.07571. Mean model state was
18.48 MB before the derived-logit cache accounting adjustment. The three-way
run measured 8,749 token/s before cache optimization, below the prior 12,841
token/s implementation; a post-cache one-shard smoke recovered from 15,291 to
16,819 token/s but does not fully remove the training-time count-update cost.

P0-1 and P0-4 are therefore closed. This result establishes shared frequency
learning and a small contextual residual gain on a compatibility-only corpus;
it does not validate adaptive topology or resolve per-decision learning and
eviction defects.

## 2026-06-24 — per-decision evidence and channel mass

Capacity diagnostics first measured the old magnitude-only policy on the
10,240/10,240 smoke: 304,608 insertions, 276,890 evictions and 265,892 probable
reconstructions for 27,718 retained entries. The apparent 64-entry bound was
therefore hiding severe reconstruction churn.

Sparse entries now maintain decision-local visits, coding-gain EMA and update
recency. Learning-rate maturity is per decision. Eviction uses positive coding
gain, visit evidence and a 32-step probation period with deterministic ties.
On the same smoke, NLL improved from 8.58149 to 8.57956; evictions fell to
220,507, estimated state rose from 1.64 MB to 2.12 MB and throughput fell from
16.46k to 14.11k token/s. The semantic repair reduces but does not eliminate
capacity churn.

The 998,400/99,328 three-seed gate produced validation NLL 7.54625, 7.54651 and
7.54658 (mean 7.54645, standard deviation 0.00017), improving by 0.06049 over
the global-prior stage and by 0.13620 over unigram. Mean throughput was 8,558
token/s and estimated state 33.53 MB. Capacity remained hard-bounded at 64, but
mean evictions were 23.22 million and 12,049 nodes ended saturated. P0-2/P0-3
are closed as learning/selection semantics; P1-1 remains open because the
capacity operating point is still churn-heavy.

Cross-channel raw weights are now normalized over represented channels before
within-channel routing. The regression fixture and a two-channel mathematical
calibration measured maximum responsibility-mass error near `1.1e-7`; topology
credit remained finite. This closes P0-5 correctness but is not evidence that
adaptive topology improves language modeling.

## 2026-06-24 — evidence-gated sparse admission

A controlled seed-7 capacity curve rejected cap growth as the P1-1 repair.
Increasing the per-node cap from 64 to 128 and 256 improved validation NLL from
7.54625 to 7.49217 and 7.45667, but retained 18.5 and 14.4 evictions per train
token, expanded estimated state from 33.5 MB to 52.0 and 77.4 MB and reduced
measured throughput to 4.56k and 4.25k steps/s.

Full nodes now place unseen decisions in a bounded eight-candidate admission
table and promote on a second observation. Existing decisions, non-full nodes
and merge preservation are unchanged. At the original cap 64, the same medium
gate reached NLL 7.51149 with 1.090 evictions and 0.864 probable
reconstructions per train token. It rejected 23.40 million one-hit admissions,
promoted 1.09 million candidates and used 35.74 MB. This closes the diagnostic
and pathological-churn blocker; the remaining saturated population is a
quality/performance tuning question rather than silent replacement behavior.

## 2026-06-24 — address capacity pressure

All objective paths now count specialization attempts blocked solely by the
per-bucket cap. Final diagnostics also expose occupied buckets, full buckets
and maximum residents. On the 20,480-example compatibility smoke, cap 4 had
3,510 occupied buckets, only 33 full buckets, but 3,401 blocked attempts. The
pressure is concentrated in repeated hotspot conflicts rather than globally
undersized storage.

A cap 4/8/16 ablation produced NLL 8.16936/8.17441/8.17892 and throughput
16.96k/10.15k/7.29k steps/s. Blocked attempts remained 3,401/2,841/2,190 while
maximum lookup candidates grew 8/16/32. The default remains 4: simple capacity
growth is slower, slightly worse in this gate and does not solve the underlying
hotspot behavior.

## 2026-06-24 — strict mapped-shard resume cursors

Mapped shard iteration now validates resume state before mutation. Out-of-range
sequence indices, impossible token offsets and nonzero offsets on the explicit
end cursor fail through the native exception, C `-1`/last-error and Python
`SBMError` contracts. `{shard_index, sequence_count, 0}` remains the canonical
end state. Regression tests also verify that rejected cursors are not advanced.

## 2026-06-24 — portable shard payload encoding

The C++ shard writer no longer dumps host-memory representations for offsets
and token IDs. It uses a buffered explicit little-endian encoder, with known
32-bit/64-bit byte fixtures plus complete deterministic-file and mapped-read
coverage. The zero-copy reader still rejects big-endian hosts because its
mapped typed spans require native little-endian layout.

## 2026-06-24 — canonical status reconciliation

The canonical documents now distinguish three facts consistently: mapped
manifest-owned corpus execution is implemented; the repaired compatibility run
passes the unigram predictive gate; and the available token-block artifact is
still inadmissible for R0 because source documents, tokenizer provenance and
cross-split deduplication evidence are missing. Exact data cursors are available,
but model-state checkpoint serialization remains unimplemented.

## 2026-06-26 — document-level FineWeb-Edu R1 smoke

The local release from the SPM data pipeline was converted into SBM mapped
token shards with source revision, tokenizer hash, document-level splits and
validation report recorded in the manifest. The first smoke corpus used
`train-10000000.parquet` capped to 998,833 train tokens and validation/test
capped to 999,359/999,423 tokens. The generated manifest hash was
`a2cd979af3e154a647ea440eea5aa9d77b9b7f2410813b146ab13a762da3b6a9`.

Fixed-topology sparse output completed seeds 7, 11 and 19. Validation NLLs were
7.27091, 7.27127 and 7.27070, with mean 7.27096 and population standard
deviation 0.00024. The unigram control was 7.54050, current-token control was
7.77112 on seed 7, pair-context control was 8.78053 and interpolated multiscale
control was 8.11875. Mean throughput was 16.19k token steps/s and mean
estimated state was 48.0 MB.

Adaptive topology on the same corpus and seeds also beat unigram but lost to
the fixed control: NLLs were 7.41534, 7.41547 and 7.41545, mean 7.41542 with
standard deviation 0.00006. Mean throughput fell to 10.78k token steps/s and
mean estimated state rose to 270.8 MB. This is a negative result for using the
current adaptive topology as the default path for larger R1 runs. The next run
should scale the fixed-topology control before revisiting adaptive proposals.

A larger fixed-topology seed-7 diagnostic used all 9,999,473 tokens from the
10M train view with validation/test capped near 1M tokens. Default capacity
reached validation NLL 7.11537 versus unigram 7.50852, but failed the R1
short-context gate: current-token control was 6.49976 and interpolated
multiscale control was 6.76884. Throughput was 7.66k token steps/s, estimated
state was 109.2 MB, sparse output retained 2.76M entries, evicted 15.13M entries
and rejected 225.17M one-hit admissions.

Raising `max_sparse_decisions_per_node` to 128 on the same seed improved NLL to
7.05836, with estimated state 171.9 MB and throughput 6.60k token steps/s.
This confirms output-capacity pressure as one contributor, but the gain is far
too small to explain the gap to the short-context controls. On the 1M smoke,
capacity 64/128/512 produced NLL 7.27091/7.23966/7.19522 while growing state
48.0/71.2/132.2 MB and reducing throughput 10.08k/9.56k/8.36k token steps/s.
A dense-output 16k-vocabulary control did not complete the 1M smoke within 15
minutes and was stopped; it is not currently a practical first-line diagnostic.

Two follow-up 1M probes clarified the limit of this local tuning direction.
Raising address specializations per bucket from 4 to 16 made quality worse:
NLL changed from 7.27091 to 7.29730 at default output capacity and from 7.19522
to 7.24182 at output cap 512, while increasing state and lowering throughput.
This rejects simple address-bucket expansion as the repair. At output cap 512,
raising token learning rates to 0.8/0.2 improved the 1M seed-7 NLL to 6.99106,
while lowering them to 0.15/0.04 worsened NLL to 7.35162. The high-rate result
confirms local output adaptation was underpowered, but it remains a better
short-context frequency fit, not evidence for the intended content-conditioned
attention/binding mechanism. Further cap/rate sweeps are therefore deprioritized
in favor of architecture work on content-conditioned selection.

## 2026-06-26 — bounded ContentMatch address primitive

The address-program language now includes `ContentMatch(max_lag)`, a bounded
content-conditioned Match/Follow primitive. It searches the retained history for
the nearest previous token equal to the current token, then encodes the current
token, a match-exists bit, the historical successor after the matched position
and the matched distance into the address signature. This uses only past input
tokens and the current input token. It does not use the next-token target,
linguistic labels, document metadata or unbounded search.

The primitive is exposed through adaptive topology proposal and can be disabled
with `topology_enable_content_match=false` for ablation. Regression coverage
checks that different matched successors produce different signatures and that
matched and unmatched contexts do not collapse to the same address.

A 1M FineWeb-Edu smoke with adaptive topology and default sparse output accepted
the new content channel: learned operations were `[0, 0, 0, 1, 1, 2]`, with
`op=2` accepted for lag 2 at step 30,720 and positive validation credit
0.00249. The run reached validation NLL 7.41302, train NLL 7.40241,
10.09k steps/s and 263.8 MB estimated state. This is essentially the same
quality band as the earlier adaptive-topology smoke without content matching
and remains worse than the fixed-topology 1M control and the high-rate capacity
diagnostic. The result proves the primitive is integrated and can receive local
credit, but it does not repair the R1 short-context-control failure.

## 2026-06-27 — bounded ContentFollow address primitive

The content-addressing path was expanded from a single match operator to a
small semantic family. `ContentFollow(pattern_lag, max_lag)` treats the final
lag as the bounded search radius and earlier lags as local pattern constraints.
For `[1, 2]`, the interpreter searches for a previous occurrence matching the
current token and the preceding token, then encodes the historical successor
after that occurrence. This is the first implemented Bind/Follow-like address
operation; it is still target-blind, label-free and bounded by retained history.

Regression coverage now verifies that `ContentFollow` separates equal-current
contexts whose matched historical successor differs, and separates matched from
unmatched local patterns. Adaptive topology proposal now includes
`ContentMatch(max_lag)` and two-lag `ContentFollow(pattern_lag, max_lag)`
programs under the same validation and rollback lifecycle.

A 1M FineWeb-Edu smoke disabled Delta proposals to bring content programs into
the early topology budget. Learned operations were `[0, 0, 0, 2, 3, 0]`.
`ContentMatch([2])` was accepted with credit 0.00838 and
`ContentFollow([1,2])` was accepted with credit 0.00349. Validation NLL was
7.41103, train NLL was 7.40035, throughput was 7.73k steps/s and estimated
state was 256.5 MB. This is a real semantic integration result, but not a
predictive breakthrough: quality remains in the adaptive-topology band and
still does not approach the fixed-topology/high-rate diagnostics.

## 2026-06-27 — multi-channel token residual diagnosis

The content-addressing failure was decomposed with fixed-topology controls.
The key question was whether content semantics were weak, or whether any added
channel damages the token residual/output path.

Fixed lag-1 on the 1M FineWeb-Edu smoke reached validation NLL 7.27091 at
`max_sparse_decisions_per_node=64`. Fixed `address_lags=1,2,4`, with no
adaptive topology and no content primitive, regressed to 7.37680 while growing
state from 48.0 MB to 138.2 MB and increasing admission rejections from 17.8M
to 29.5M. This isolates a multi-channel problem independent of content
semantics.

The same pattern held after removing the obvious output-capacity/learning-rate
constraint. Lag-1 with cap512 and high token learning rates reached NLL
6.99106. Fixed `1,2,4` with the same cap512/high-rate setting reached only
7.16674 while using 371.9 MB state and 10.54M sparse output entries. Thus the
extra channels remain harmful even when local output capacity is much larger.

Lowering `residual_channel_gain` to 0.25 partially recovered the multi-channel
runs: cap64 improved from 7.37680 to 7.32561, and cap512/high-rate improved
from 7.16674 to 7.08197. This is direct evidence that channel responsibility
mass and fusion policy are causal. It does not fully recover lag-1 quality,
which points to a second contributor: sparse-output entries and admissions are
fragmented across many more address nodes.

Code inspection found a concrete asymmetry. The vector residual path learns each
stage in channel-local coordinates, dividing the desired residual and within
channel update by channel mass. Token dense and sparse objectives instead scale
local logit updates directly by node responsibility, while inference also
multiplies the same local logits by responsibility. With multiple represented
channels this makes the aggregate-logit learning signal effectively too small,
and it weakens the seed channel when new channels are added. This explains why
adaptive/content channels can show positive ablation credit but still reduce
global held-out NLL relative to the single-channel control.

## 2026-06-27 — channel-local token residual repair

The token dense and sparse objectives now update local residual logits in
channel-local coordinates. Inference is unchanged: active nodes still contribute
`responsibility * local_logit`. During learning, however, the local update uses
`responsibility / active_channel_mass`, matching the coordinate system used by
the represented channel instead of applying global responsibility a second time.

Regression coverage was added to the channel scaling test. It trains a single
channel and a two-channel sparse-token model on the same rare-target update and
requires the two-channel local log-gain to remain within the same order as the
single-channel gain. This protects against the previous quadratic dilution.

The minimal fixed-topology reproductions were repaired on the 1M FineWeb-Edu
smoke. Fixed `address_lags=1,2,4` with cap64 improved from NLL 7.37680 before
the fix to 7.21146 after the fix, beating the lag-1 cap64 control at 7.27091.
With cap512 and high token learning rates, fixed `1,2,4` improved from 7.16674
to 6.91215, beating the lag-1 high-rate control at 6.99106.

The content-focused adaptive smoke also improved. With Delta proposals disabled
to bring content programs into the early budget, the run accepted the same
operations `[0, 0, 0, 2, 3, 0]` as before, including `ContentMatch([2])` and
`ContentFollow([1,2])`. Validation NLL improved from 7.41103 to 7.16205, train
NLL from 7.40035 to 7.12646, and accepted content-channel credit increased:
`ContentMatch([2])` proposal credit rose from 0.00838 to 0.02340 and
`ContentFollow([1,2])` from 0.00349 to 0.01073. This does not yet close the
larger 10M R1 short-context gate, but it resolves the diagnosed multi-channel
residual dilution and restores content-addressing experiments to a meaningful
state.

## 2026-06-27 — repaired 10M/1M R1 pass

After the channel-local token residual repair, the medium FineWeb-Edu R1 run was
rerun with fixed `address_lags=1,2,4`, `max_sparse_decisions_per_node=512`,
`classification_learning_rate=0.8` and
`classification_mature_learning_rate=0.2`. The run used the document-level
10M train / 1M validation manifest and strict frozen evaluation.

The repaired model reached validation NLL 6.47603 and train NLL 6.57198. It now
passes the R1 short-context gate: unigram was 7.50852, interpolated multiscale
was 6.76884 and the previous strongest current-token control was 6.49976. The
result is also substantially better than the old pre-repair fixed runs:
default cap64 reached 7.11537 and cap128 reached 7.05836 on the same split.

Resource metrics were: 4.35k examples/s, 2,528.9 model seconds, 1.20 GB
estimated state, 112,500 live nodes, 39.44M sparse output entries, 7.80M sparse
output evictions, 150.15M admission rejections and 11.10M address-capacity
blocked splits. These are acceptable for an R1 pass but remain the next obvious
scaling pressure before 100M-token R3 runs.

The first attempt at avoiding per-step decoder vector allocations was rejected:
the fixed-buffer variant preserved NLL but reduced 1M throughput from about
15.15k to 8.26k examples/s, likely because fixed arrays introduced larger
per-step copy/sort/cache costs. It was fully reverted. The useful operational
change was native progress logging: large corpus runs now emit compact stderr
lines with phase, processed examples, examples/s, ETA, running NLL, live nodes
and estimated state MB. The successful R1 run wrote regular progress lines and
ended at running eval NLL 6.48 with state about 1.14 GiB.

## 2026-06-27 — R2 fixed-structure diagnosis

An aggregate-only R2 diagnostic script was added and run on the repaired
10M/1M R1 corpus. It analyzes the fixed lag structures used by the successful
R1 run before any example inspection, with cuts by document position, target
train frequency, address-key reuse and cross-document codelength gain.

The structural-control result is negative for the fixed lag keys themselves.
The validation current-token conditional table reached NLL 6.40482, while
lag1/lag2/lag4 controls reached 7.18158, 7.35314 and 7.47228. The equal-weight
lag1/2/4 interpolation reached 6.56526, losing 0.16043 nats/token against the
current-token table. Across 877 validation documents, only 143 documents had
positive current-minus-interpolated gain; the mean document gain was -0.12826
nats/token.

The only clean positive aggregate bucket was early document position 1-3, where
interpolated lags beat the current-token table by 0.21772 nats/token. Most
later positions and most address-reuse buckets were negative, especially high
reuse buckets, which is consistent with sparse address-key memorization rather
than robust reusable structure.

This does not invalidate the repaired R1 pass: the trained model still beat the
frozen current-token control. It does mean that R1 cannot be interpreted as
evidence that fixed lag address keys are the causal mechanism. The R2 hard gate
also remains open because the successful R1 run used fixed programs `[1]`,
`[2]` and `[4]`, with no accepted adaptive topology event to attribute across
documents. The next work must add model-level accepted-structure attribution
under frozen evaluation and run an adaptive validation block, rather than scale
the same fixed experiment.

Frozen-evaluation channel attribution has now been implemented behind
`record_channel_attribution`. When enabled, token evaluation reports each
address channel's counterfactual codelength credit sum, mean credit, positive
count and positive fraction without updating topology credit or training state.
It also records positive-document counts so R2 can test cross-document reuse
directly. The default remains disabled so normal training does not pay the
attribution cost.

## 2026-06-27 — R2 adaptive content attribution

The adaptive content-addressing R2 run used the same 10M FineWeb-Edu training
split with 1M validation and 1M independent test evaluation. Delta proposals
were disabled to force the proposal budget toward content semantics; the seed
program was lag 1, sparse-output capacity was 512 and token learning rates were
the repaired high-rate settings.

Accepted content structures survived held-out attribution on both splits.
`ContentFollow([2,3])` was the strongest: mean counterfactual credit was
0.16521 nats/token on validation and 0.16702 on test, with positive document
contribution in all 877 validation documents and all 994 test documents.
`ContentMatch([4])`, `ContentFollow([1,4])` and `ContentFollow([3,4])` also
had positive-document fractions above 0.98 on both splits.

This satisfies the R2 structural attribution requirement: accepted content
structures are not merely training-set artifacts. The result also exposes the
next blocker. The full adaptive model reached NLL 6.58731 on validation and
6.57489 on test, while the current-token controls were 6.49976 and 6.48778.
So the architecture now has real accepted structure but still fails to convert
that structure into better full-model held-out codelength. The next diagnosis
must isolate seed-only, accepted-channel and fused-model behavior before any
R3 scale-up.

## 2026-06-27 — plan redirected to neuronal address semantics

The immediate plan has been redirected away from fusion/rate/capacity sweeps.
The current evidence no longer supports treating P0-8 as a narrow channel
weighting problem: accepted content structures exist, but the implementation
still executes address programs as one-shot signature recipes without explicit
frames, lineage, dependency-aware attribution or recoverable accepted-structure
lifecycle.

The new canonical implementation plan is
`docs/superpowers/plans/2026-06-27-neuronal-address-semantics.md`. It requires
explicit address execution, persistent committed structures, dependency-aware
structural attribution and codelength/cost accounting before R3 scale-up or
large hyperparameter sweeps. Medium corpus runs remain validation gates for the
completed framework, not a substitute for implementing it.

## 2026-06-27 — address semantics smoke gate

Tasks 2 through 7 of the neuronal address semantics plan are implemented. The
machine now exposes address execution diagnostics, executes address programs as
frames, preserves accepted channels by default, supports quarantine rather than
physical deletion, reports program/dependency attribution and records
cost-penalized structural value for topology events.

A limited FineWeb-Edu smoke used 1,000,000 training examples and 100,000
validation examples from the 10M/1M manifest. It completed at 11,094.9
examples/s with eval NLL 6.9104, current-token control 7.8332 and interpolated
multiscale control 8.1617. The run emitted 6,528,320 address execution frames,
4,371,408 binding hits, 2,156,912 binding misses and 10 program-attribution
entries. Five topology programs were accepted and none were physically pruned
under the default preserve policy. The result is recorded in
`research_results/address_semantics_smoke_seed7.md`.

This smoke validates the completed semantic instrumentation and lifecycle path.
It is not the full 10M/1M gate and does not replace the required comparison
against the repaired fixed `[1,2,4]` control.

## 2026-06-27 — address semantics 10M/1M gate

The full FineWeb-Edu 10M/1M address-semantics gate completed with interpreted
address frames, accepted-channel preservation and frozen program attribution.
The model reached eval NLL 6.36637 after 9,989,918 training examples and
998,482 validation examples. This beats the current-token control at 6.49976,
the interpolated multiscale control at 6.76884 and the previous repaired fixed
`[1,2,4]` R1 result at 6.47603.

The run emitted 65,858,720 address execution frames, 43,990,050 binding hits
and 21,868,670 binding misses. Five topology programs were accepted and none
were physically pruned under the default preserve policy. Frozen evaluation
emitted 10 program-attribution entries. Active work remained bounded:
`avg_active = 5.99994` and `max_bucket_candidates_inspected = 8`.

The strongest content primitives were `ContentMatch([2])` and
`ContentFollow([1,2])`, both with positive full-document attribution. The
result is recorded in `research_results/address_semantics_10m_seed7.md`.

This is the first medium-scale positive result for the completed adaptive
content-addressing framework. It does not yet prove R3 readiness: exact
checkpoint/resume, multi-seed stability and shard transfer remain required.

## 2026-06-27 — model checkpoint foundation

The first R3 admission subtask now has an exact model-state checkpoint. A C++
regression trains two identical token models for 512 steps, saves one, reloads
it and verifies that the next 256 training steps match the uninterrupted model
on cross-entropy, target probability, predicted token, active nodes and live
nodes. The checkpoint stores the full `SparseBranchMachine` state rather than
only config/topology.

This is not yet complete long-run resume. The remaining runner-level work must
store corpus shard identity, cursor position, consumed-example counters and
output metadata so remote training can resume without replaying already
processed data.

## 2026-06-27 — resumable corpus training runner

The model checkpoint is now exposed through the stable C ABI and Python runtime:
callers can create a token machine, step it, save/load checkpoint state and read
diagnostics. `scripts/run_resumable_corpus_training.py` uses that API to save a
runner checkpoint with manifest path, train shard paths, current shard cursor,
consumed examples, accumulated train metrics, config overrides, output path and
the model checkpoint file.

`tests/test_resumable_runner.py` verifies the core property: a small mapped
corpus trained uninterrupted for 48 examples and the same corpus interrupted at
24 examples then resumed to 48 examples produce matching train cross-entropy
and target-probability metrics.

This removes the training-replay blocker for long remote runs. It is not yet a
drop-in replacement for canonical `run_corpus_training.py`, because eval shard
resume, strict-freeze transition reporting, baseline/control metrics and final
program-attribution JSON assembly still need to be integrated.

## 2026-06-27 — resumable canonical-style report

The resumable corpus runner now covers both train and eval phases. It persists
eval split identity, eval shard cursor, strict-freeze state, baseline tables,
channel/program attribution accumulators and final model summary. Its output
retains resumable-run metadata at the top level and embeds a canonical-style
`result` object with model eval metrics, current-token/interpolated baselines,
seed/active/content/tuple controls when available, learned address programs,
topology events and program attribution.

`tests/test_resumable_runner.py` now checks both interruption points: train is
interrupted at 24/48 examples and eval is interrupted at 5/12 examples. In both
cases the resumed run matches the uninterrupted run on final eval
cross-entropy. This closes the checkpoint/resume admission gate sufficiently to
start multi-seed 10M/1M validation without wasting completed remote training
after process interruption.

## 2026-06-27 — address semantics 10M/1M multi-seed gate

The completed address-semantics framework passed the three-seed 10M/1M gate on
FineWeb-Edu validation. Seeds 7, 11 and 19 reached eval NLL 6.366367,
6.366452 and 6.366331 respectively. The mean eval NLL is 6.366384 with
population standard deviation 0.000051. All three beat the current-token
control at 6.499756 and the interpolated multiscale control at 6.768842.

All three seeds converged to the same learned program set:
`[[1], [2], [1, 2], [2], [1, 2], [3]]` with operations
`[0, 0, 0, 2, 3, 0]`. Each run accepted five topology programs, physically
pruned zero accepted structures and emitted 10/10 positive-mean program
attribution entries.

This closes Task 9 Step 2. The next evidence gate is shard transfer: the same
framework must remain competitive on a different FineWeb-Edu shard sample.

## 2026-06-28 — parallel gate harness and short throughput probe

The R3 admission harness was updated before shard-transfer validation. The
corpus batch runner now supports parallel seed execution, split selection,
bounded train/eval windows, parameter overrides and skip-existing reuse, so
independent seeds do not need to run serially. This is infrastructure for Task 9
Step 3, not evidence that shard transfer has passed.

The throughput probe was changed from a long fixed workload to short C++
limited-window probes. The current local run on the 10M/1M manifest used
50k/100k/150k/200k train windows with 10k eval examples and stopped at the
configured 200k ceiling because the last three windows did not stabilize within
8%. Window speeds were 6.72k, 10.20k, 9.78k and 6.41k examples/s. This records
a real nonblocking performance concern and confirms that a fixed 1M probe is
not an appropriate default throughput test.

The first throughput repair removed a pure measurement cost from training:
sparse-token top-k ranking is now skipped during learning by default and
training top1/top5 are reported as unavailable. Evaluation ranking is unchanged,
as are cross-entropy, topology credit, parameter updates and attribution. On
the same 100k/10k window, throughput improved from about 6.43k to 15.42k
examples/s with identical eval NLL 7.34320677. On the 50k/100k/150k/200k
short-window probe, speeds were 26.04k, 16.28k, 15.00k and 19.25k examples/s.

## 2026-06-28 — address semantics test-transfer gate

Task 9 Step 3 passed on the held-out FineWeb-Edu test split. Seeds 7, 11 and
19 reached eval NLL 6.356485, 6.356029 and 6.356655 respectively. The mean
eval NLL is 6.356390 with population standard deviation 0.000264. All three
beat the test current-token control at 6.487780 and the interpolated multiscale
control at 6.748025.

All three seeds again converged to operations `[0, 0, 0, 2, 3, 0]` and
programs `[[1], [2], [1, 2], [2], [1, 2], [3]]`. Each run accepted five
topology programs, physically pruned zero accepted structures and emitted
10/10 positive-mean program-attribution entries. Bounded-work diagnostics
remained stable with `avg_active` about 6 and `max_bucket_candidates_inspected`
equal to 8. The result is recorded in
`research_results/address_semantics_10m_test_multiseed.md`.

The attempted three-worker local batch was stopped because it was memory-bound:
the workers left about 1 GB available memory while total CPU utilization stayed
below 30%. The completed run used two workers and reached mean throughput
10.28k examples/s across seeds.

## 2026-06-28 — structural-value admission calibration

Task 9 Step 4 passed for the default description-only structural-value gate.
Matched 10M/1M validation runs with
`topology_accept_uses_structural_value=true` learned the same operation/program
sequence as the raw-credit gate on seeds 7, 11 and 19:
`[0, 0, 0, 2, 3, 0]` over
`[[1], [2], [1, 2], [2], [1, 2], [3]]`. Each run accepted five programs,
physically pruned zero accepted structures and retained 10/10 positive-mean
frozen program-attribution entries.

Raw-gate NLLs were 6.366367, 6.366452 and 6.366331. Structural-gate NLLs were
6.366851, 6.366452 and 6.366292, for a mean delta of +0.000148 nats/token.
The smallest accepted structural event credit was 0.004321, still well above
the 0.0005 acceptance threshold. The result is recorded in
`research_results/structural_value_admission_10m_validation.md`.

This calibrates the current default description penalty. It does not justify
enabling nonzero `structural_execution_cost_weight`; execution-cost admission
remains a separate future experiment.

## 2026-06-28 — lineage-aware address attribution

The address-semantics implementation now carries channel-level lineage rather
than only binding-distance dependency values. Topology state, topology events,
execution frames, C/Python step stats, experiment JSON and C API summary JSON
include `parent_channel` and `dependency_channel`. A regression forces
`ContentMatch([2])` and `ContentFollow([1,2])` acceptance and verifies that the
follow channel depends on the accepted match channel, including its execution
frame metadata.

Frozen program attribution now reports `caller_removed_credit`,
`dependency_retained_credit` and `dependency_removed_credit`, so caller removal
with prerequisites retained is distinguishable from removing both caller and
dependency. The C ABI was incremented because `sbm_step_stats` changed.

`AddressExecutionFrame` now also carries a typed `AddressBindingState` with
current token, matched token, matched successor, matched distance, pattern span,
pattern-term count and match flag. The legacy `successor` and `dependency`
fields remain compatibility aliases. Frozen program attribution aggregates
binding match count, match fraction, mean binding distance and mean pattern
span. The C ABI is now v6 after adding the binding-state arrays.

Because topology state and topology events are raw-serialized in model
checkpoints, the model checkpoint magic was bumped to `SBMCKPT2`. The checkpoint
contract is exact same-format resume, not cross-version archive compatibility.

The accepted-channel lifecycle also now exposes `RecoverableRetire` as a real
policy. It masks routing, retains channel-owned state for audit and increments
`recoverable_retired_channels`; only `PhysicalErase` deletes channel-owned
nodes.

# GPAF Architecture Investigation — Why Unique Retrieval Is Negative

Date: 2026-07-02
Scope: root-cause analysis of the 10M FineWeb-Edu GPAF finding (unique gain
-0.0013, overlap gain +0.2288, all key net values negative) and architectural
proposals.

## 1. Executive summary

GPAF v1 cannot perform global retrieval because its role keys carry no
per-step context information: `gpaf_role_key_for_channel` is a pure function
of static channel metadata, so the entire "address field" degenerates into at
most a handful of per-channel recency caches (21 keys observed over 10M
tokens, ~4.4 bits of address entropy for the whole run). The +0.2288 overlap
gain is a measurement artifact — locally-earned node contributions are
re-attributed to GPAF because `push_candidate` overwrites the source of
already-present candidates — so GPAF's true causal marginal is the unique
group alone, which is slightly negative. Additionally, the costed admission
value (`gpaf_slot_costed_net_value_`) is never written by any code path, the
execution-cost model is dimensionally incoherent (resident counts subtracted
from nats), and GPAF candidates are scored by exactly the token-signature
Hamming similarity the spec says GPAF must move away from. The fix sequence
is: repair attribution and value plumbing first (days), then give keys
per-step context content — binding-conditioned keys and role-transition keys
— and give GPAF its own calibrated score channel (weeks).

## 2. Root cause analysis

### RC1 (primary): role keys are context-free constants

`gpaf_role_key_for_channel` (src/modules/machine_topology.cpp:268-281) hashes
only:

```
program.op, program.arity, channel_index,
dependency_edge_kind, parent_edge_kind, generation
```

Every one of these is a property of the *machine's topology*, not of the
current token window, bindings, history or prediction state. For a fixed
topology the key is a per-channel constant. The structural-call variant
(machine_topology.cpp:283-314) adds `address_program_key`, output state, a
reuse-count bucket and an execution bucket — still all slowly-varying
topology properties.

Consequences, all confirmed by the 10M diagnostics:

- **Address space collapse.** 21 unique keys over 10M tokens ≈ channels ×
  generation churn. An address field whose addresses have ~4.4 bits of total
  entropy cannot discriminate contexts. For comparison, each local channel's
  exact bucket carries `bucket_bits = 12` bits *per step*, plus Hamming
  ranking within the bucket. The information available to GPAF retrieval is
  orders of magnitude below what local addressing already uses; its retrieval
  value is upper-bounded by the mutual information between key and context,
  which is essentially zero per step.
- **Slots are recency caches, not roles.** The writer
  (`observe_gpaf_shadow_roles`, machine_topology.cpp:316-356) inserts every
  active node into its channel's slot, with rotation
  `residents[total_steps_ % residents.size()] = node.id`
  (machine_topology.cpp:341). A slot's residents are therefore "the last ≤4
  nodes that were active on channel c" — regardless of whether they helped.
- **Overlap is structurally guaranteed.** Retrieval queries are keyed by the
  channels of `previous_route_` nodes (machine_routing.cpp:113-134), i.e.
  "return nodes recently active on the channels I was just using". Recently
  active nodes are precisely the nodes that control edges
  (machine_routing.cpp:88-108) and exact buckets re-reach on the next step.
  The 89% overlap rate (55,582 / 62,471) is the predictable output of this
  design, not a tuning accident.
- **Key instability compounds the problem.** Because `generation` is hashed
  in, a re-proposed instance of the same program gets a fresh key and loses
  all accumulated slot evidence, while the key still carries no context. The
  design has the worst of both directions: unstable identity *and* zero
  per-step content.

The spec (2026-06-30 design, §Predictive role key) explicitly listed
context-bearing key ingredients — quantized loss/codelength buckets, binding
kind, output-tree region — none of which were implemented. V1 implemented
only the topology-metadata subset, which is the one subset that cannot
address anything.

### RC2: residency and promotion are popularity-gated, not value-gated

Probe→Active promotion requires only `gpaf_probe_min_observations`
observations and `gpaf_probe_min_residents` residents
(machine_topology.cpp:347-354). Observation count measures how often a
channel fires — popularity — not whether the slot's residents ever produced
held-out gain. Per-node counterfactual contribution is *already computed* in
the learn path (token_sparse_output.cpp:683-703, `node.contribution =
without_loss - cross_entropy`) immediately before `observe_gpaf_shadow_roles`
is called (token_sparse_output.cpp:719), and the writer ignores it. The
architecture computes exactly the evidence the spec's write path demands
("counterfactual/codelength contribution is computed → writer rules propose
or update predictive role slots") and then throws it away.

### RC3: the costed admission value is dead code

`gpaf_slot_costed_net_value_` is read at machine_topology.cpp:346 and
serialized (machine_checkpoint.cpp:440, 542) but **never written anywhere**.
Two concrete defects:

1. With `gpaf_active_requires_positive_net_value=true`, `operator[]`
   default-inserts 0.0, `0.0 > 0.0` is false, so **no slot can ever be
   promoted**. The flag silently disables promotion forever rather than
   gating it on evidence. (The 10M run confirms: gate blocks all promotions;
   NLL unchanged only because Probe and Active have identical routing
   eligibility — which also means the Probe/Active distinction currently has
   no behavioral meaning at all.)
2. There is no demotion path driven by measured value. `quarantine_gpaf_slots`
   etc. (machine_topology.cpp:358-397) are bulk manual operations; nothing
   in the training loop ever retires a harmful slot.

### RC4: GPAF candidates are scored by the signal GPAF exists to replace

`score()` (src/modules/machine_routing.cpp:208-221) applies
`0.42·exact + 0.72·hamming + 0.10·reliability + 0.02·novelty + edge_prior`
uniformly. For a genuinely global candidate — a node whose prototype
signature comes from a *different* context — Hamming similarity to the
current signature is low by construction, and `exact` is zero. So:

- Useful global candidates are systematically outscored by local ones and
  rarely enter the beam.
- When GPAF-unique candidates *do* activate, it is in low-evidence contexts
  (few/weak local candidates), where they inject sparse logits trained in
  unrelated contexts with no calibration. Mild harm (-0.0013) in exactly
  7,110 of 62,471 examples (11%) is the expected signature of this failure
  mode: uncontrolled activation precisely where the machine is most
  uncertain.
- The spec's candidate-merge-by-source-quota exists only on the candidate
  *generation* side (budget arithmetic, machine_routing.cpp:44-48). At beam
  *selection* (machine_routing.cpp:251-276) all sources compete in one
  score-ordered pool; there is no reserved or capped GPAF slot in the beam.

Minor but real waste: since keys are per-channel constants, multiple
previous-route nodes on the same channel probe the same slot repeatedly
(machine_routing.cpp:113-137 has no per-step key dedup), consuming query
budget (~4 probes/token for ≤ a couple of distinct keys) and inflating
measured execution cost.

### RC5: the frozen ablation misattributes local value to GPAF

Two attribution biases make the headline +0.2034 total gain invalid as a
causal claim:

1. **Source overwriting.** `push_candidate` (machine_routing.cpp:25-29): if a
   node already entered via ExactBucket or ControlEdge and GPAF pushes it
   again, its `source` is *rewritten* to `GpafRole` with
   `gpaf_overlap=true`. The frozen ablation
   (token_sparse_output.cpp:335-445) then removes that node's logits and
   books the loss increase as GPAF value. Without GPAF the candidate set,
   route and logits would be identical for these nodes; their true GPAF
   marginal is exactly zero. The +0.2288 overlap gain is local-retrieval
   value wearing a GPAF badge.
2. **Fixed-route ablation.** The ablation subtracts logits while holding the
   selected route and responsibilities fixed (token_sparse_output.cpp:
   361-409). The correct counterfactual for "remove GPAF" is to re-run
   candidate selection without GPAF injection — beam slots and responsibility
   mass would be reallocated to other candidates. Fixed-route removal
   overstates both harm and benefit.

Also note the unique/overlap boundary is order-dependent: neighbor-bucket
probing runs *after* GPAF injection (machine_routing.cpp:174-195), so a
neighbor-reachable node first pushed by GPAF is counted "unique" even though
it is locally reachable. "Unique" currently means "not reachable via exact
bucket or control edge", not "not locally reachable".

Net effect: GPAF's honest causal estimate from the 10M run is the unique
group only: **-0.0013 nats mean on 11% of examples, i.e. ≈ -0.00015
nats/example overall, before execution cost** — consistent with the observed
+0.0019 end-to-end NLL delta being pure noise plus a small drag.

### RC6: the cost model guarantees rejection regardless of key quality

`gpaf_ablation_key_net_value = gain − false_positive_cost − execution_cost`
where `execution_cost = resident_count` **per evaluation example**
(token_sparse_output.cpp:438-442), summed over ~56k examples → exec_cost
≈ 224k for key 692. Gain is in nats/example; execution cost is in
"residents × examples" with an implicit λ = 1 nat per resident per example.
No conceivable per-example gain (bounded by a few nats) can beat a cost of
~4 nats/example. The costed gate is therefore not a calibrated MDL
criterion; it is a hardcoded rejection. Until λ is set from an actual
resource-to-codelength exchange rate (or execution cost is amortized per
*probe* rather than per example × resident), net-value numbers cannot be
used for admission decisions in either direction.

### Summary of the causal chain

Context-free keys (RC1) → slots are per-channel recency caches → residents
overlap local retrieval (89%) and unique residents are stale/unrelated →
value-blind residency and promotion (RC2, RC3) keep harmful residents →
similarity-based scoring lets them activate only in low-evidence contexts,
uncalibrated (RC4) → measured "value" is an attribution artifact (RC5) and
measured "cost" is dimensionally meaningless (RC6). Every observed number in
the 10M run is the deterministic consequence of this chain; nothing about it
suggests the *concept* of predictive-role addressing was tested and failed.
What was tested is a degenerate instance with an unaddressable key space.

## 3. Proposal A — Honest attribution and live value plumbing (foundation)

Fix measurement and lifecycle before touching keys; otherwise no key redesign
can be evaluated.

1. **Stop overwriting candidate sources.** In `push_candidate`, keep the
   first source; record GPAF co-retrieval in a separate `gpaf_also` flag.
   Overlap nodes then contribute zero to GPAF ablation by construction, and
   the unique/overlap split becomes "GPAF-caused" vs "GPAF-redundant".
2. **Re-route counterfactual (diagnostic, frozen only).** Behind a flag, run
   `select_route` a second time per frozen step with GPAF injection disabled
   and report the NLL delta. This is the true causal ablation. Bounded: one
   extra route per eval token, eval-only.
3. **Implement the costed writer.** In `observe_gpaf_shadow_roles`, the
   caller already has per-node `contribution` (token_sparse_output.cpp:702);
   pass it in and update
   `gpaf_slot_costed_net_value_[key] ← EMA(contribution − λ·probe_cost)`,
   only for nodes whose candidate source was genuinely GPAF. This makes
   `gpaf_active_requires_positive_net_value` a real gate instead of a
   permanent block.
4. **Value-gated residency and demotion.** Insert a node into a slot's
   resident list only when its measured contribution is positive; evict the
   lowest-value resident, not `total_steps_ % size`. Demote Active →
   Quarantined when the slot's value EMA stays negative past a patience
   threshold (mirror the channel prune discipline,
   `topology_prune_patience`/`topology_prune_credit`).
5. **Calibrate λ.** Express execution cost per probe in nats via an explicit
   exchange rate derived from throughput impact (or simply make λ a config
   scalar, default small, and report both raw and costed values). Fix the
   per-example × resident-count unit error.
6. **Dedup probes per step** (keep a small fixed array of keys already
   probed this step).

Constraint check: no new unbounded work (EMA update is O(active); re-route is
eval-only, flag-gated); no leakage (contribution is computed after prediction
is fixed and only affects *future* slot state — same discipline as node logit
updates); frozen eval writes nothing.

## 4. Proposal B — Context-bearing keys: BindingReuseRole v1

The smallest change that gives the field an actual address argument. The
binding machinery already computes per-step, pre-target, content-conditioned
state: `AddressBindingState.binding_key`, matched distance, pattern span
(types.hpp:90-100), available in execution frames before the target is
observed.

Two key variants, both bounded and non-linguistic:

- **B1 (content identity):** `key = mix(program op/arity, lineage edge kinds,
  binding_key) mod gpaf_slots`. Two contexts that fired the *same long-range
  content binding* share a slot even when their positional window signatures
  differ completely. Globality comes from cross-document recurrence of the
  binding: the slot accumulates residents from every document where that
  binding fired, which exact buckets do not, because bucket signatures mix in
  positional detail.
- **B2 (content shape):** replace `binding_key` with quantized
  `(binding kind, distance bucket, pattern-span bucket)`. This encodes "a
  span-s recurrence at distance ~d fired" without the matched token identity
  — a structural role rather than content recurrence, further from token
  similarity, coarser but more transferable.

Read path: query keys computed from the *current step's* execution frames
(not from previous-route channels), up to `gpaf_query_keys_per_step` distinct
keys — strictly causal (frames derive from the history window). Write path:
value-gated insertion from Proposal A.

Expected effect on the diagnostic that currently fails: key cardinality moves
from ~21 to thousands (hashed into `gpaf_slots`), unique retrieval becomes
"nodes that helped under this binding elsewhere" instead of "nodes recently
active on this channel". Run B1 and B2 against the spec-mandated
shuffled-key and random-slot controls; B1's margin over B2 measures how much
of the value is content identity vs structural shape.

Risk: B1 partially overlaps what ContentMatch/ContentFollow buckets already
capture; the shuffled-key control and the (fixed) unique-marginal metric are
the arbiter. B2 has collision risk (few distinct shapes); widen buckets only
if slot phase statistics show saturation.

## 5. Proposal C — Role-transition addressing (CoPredictionRole as succession)

Control edges already learn node→node succession. Their weakness is zero
generalization: an edge from node A helps only when A itself was active.
GPAF can host the generalized version — **role→role transition** — which is
a genuinely global structure no local mechanism provides:

- **Write** (one step delayed, causal): at step t, after the target is
  observed, form `pending_key = mix(role(active node), region_prefix(target
  path, depth k))` where `region_prefix` is the first k binary decisions of
  the target's output-tree path — a non-linguistic, emergent coarse outcome
  class (the tree is hash-seeded; regions mean nothing a priori and acquire
  meaning only through reuse). At step t+1, insert the step-t+1 active nodes
  with positive contribution into the slots for the step-t pending keys.
- **Read**: at step t+1, the previous target *is* the current input token, so
  its path prefix is known pre-prediction. Query
  `mix(role(previous-route node), region_prefix(current input token, k))`.
  Retrieval means: "nodes that historically became active-and-useful right
  after this role produced this outcome class".

This is the spec's CoPredictionRole made concrete without leakage: the only
target information used is from *already-observed* tokens. Work is bounded
(≤ beam_width pending keys, one deferred insertion pass per step). It
subsumes and generalizes the control-edge prior, so the honest control is:
does role-transition GPAF beat the same budget spent on larger
`edge_scan_limit`? If it cannot beat node-level edges, role granularity is
too coarse at current scale and the proposal should be shelved.

## 6. Proposal D — GPAF-specific scoring and a reserved beam quota

Decouple GPAF candidate scoring from token-signature similarity
(spec §Scoring says exactly this; v1 did not implement it):

- Score GPAF candidates as
  `slot_prior + 0.10·reliability + 0.02·novelty`, where `slot_prior` is the
  calibrated per-slot value EMA from Proposal A (clamped, e.g. tanh-scaled),
  and *omit* the Hamming and exact terms entirely for GpafRole-sourced
  candidates.
- Reserve at most one beam slot for the best GPAF-unique candidate, and only
  when its slot prior is positive; never displace the guaranteed exact
  residents (machine_routing.cpp:251-262). Zero positive slots ⇒ zero GPAF
  beam presence ⇒ GPAF cannot hurt.

This converts GPAF's current failure mode (uncalibrated activation in
low-evidence contexts) into a controlled, evidence-gated injection. It is
deliberately asymmetric-conservative: GPAF can only add a candidate whose
slot has demonstrated positive costed value.

## 7. Proposal E — Epistemic-state addressing (research direction)

The deeper reading of the failure: v1 built addresses out of the machine's
*static structure*; local buckets build them out of the *surface context*.
Both answer "where am I?". A predictive address field should answer a
different question — **"what do I not yet know here?"** — and that question
has an observable, pre-target, non-linguistic answer:

- route confidence (max responsibility, already computed);
- channel disagreement (which channels found exact residents, whether their
  top logit directions on the first k output-tree levels agree);
- global-prior vs route disagreement pattern (sign pattern of
  `aggregate_sparse_logit` vs `global_output_logit` over the first k
  decisions).

Key = quantized epistemic signature (confidence bucket, k disagreement bits,
argmax region prefix). Residents = nodes whose corrections historically fixed
errors under that signature (value-gated writes from A). Properties worth
noting:

- **Structurally zero overlap with local retrieval**: the key lives in a
  space orthogonal to address signatures, so the overlap pathology cannot
  recur by construction.
- It targets exactly the contexts where GPAF-unique candidates currently
  activate by accident and hurt — but with residents *selected* for having
  helped in that epistemic state, rather than being whatever was cached.
- It is a sparse error-correcting associative memory: the machine learns to
  address by its own uncertainty pattern. No linguistic labels, no dense
  computation, O(k) key construction, standard slot bounds.

This is the highest-novelty, highest-risk direction; it should be attempted
only after A+D establish trustworthy measurement, because its value claim
depends entirely on honest unique-marginal accounting.

## 8. Recommendation

Order and gating:

1. **Proposal A (measurement + value plumbing) — do first, unconditionally.
   Effort: 1-2 days.** Nothing else is evaluable without it, and two of its
   items fix outright defects (source overwrite, dead net-value map /
   permanently-blocking gate, unit-incoherent cost). Success metric:
   corrected 10M diagnostics where overlap gain reads ≈ 0, GPAF NLL delta
   ≥ 0 vs baseline (self-pruning removes the drag), and every remaining
   Active slot has positive costed value.
2. **Proposal D (score + quota) — with A, same validation run. Effort: 1-2
   days.** Minimum experiment: 10M FineWeb-Edu, 2 seeds, upgrade-v1 vs
   A+D-retrieval. Success: no regression, and GPAF-unique frozen mean gain
   moves from -0.0013 to ≥ 0.
3. **Proposal B (binding-conditioned keys) — first real test of the GPAF
   hypothesis. Effort: ~1 week** including B1/B2 variants, shuffled-key and
   random-slot controls, 10M × 2 seeds. Success metric: observed seed spread
   is ~0.002 nats, so demand **≥ 0.010 nats NLL improvement on 10M (both
   seeds)** plus positive unique-group ablation net of calibrated cost, plus
   beating the shuffled-key control. A positive-unique but sub-0.01 result
   still justifies proceeding to R3 scale (the mechanism should grow with
   corpus-level recurrence).
4. **Proposal C (role transitions) — after B reports, regardless of B's
   sign. Effort: 1-2 weeks.** Its control (matched-budget larger
   edge_scan_limit) is cheap and decisive.
5. **Proposal E — multi-week research track**, gated on A+D evidence quality,
   pursued only if B or C shows the slot/lifecycle substrate can carry
   positive unique value.

What would falsify the GPAF concept rather than the v1 instance: if B, C
*and* E all fail to produce positive unique marginal after honest accounting,
then bounded global retrieval keyed on anything cheaper than content identity
adds nothing over exact buckets + control edges at this scale, and the
research focus should shift to enriching the local address programs instead.

All proposals preserve §4 constraints: no dense layers or global scans (all
keys are O(1) hashes; slot maps are bounded by `gpaf_slots` ×
`gpaf_residents_per_slot`); prediction is fixed before target in every read
path (B reads pre-target frames; C reads only already-observed tokens; E
reads pre-target internal state); frozen eval writes nothing (writers remain
inside `if (learn)`); no linguistic labels anywhere (binding keys, output-tree
regions and epistemic signatures are content/structure-derived, not
category-labeled). The token-signature system remains untouched as bootstrap
and control.

## 9. References

Code inspected (paths relative to repo root):

- `include/sbm/types.hpp:321-328` — GPAF Config fields; `types.hpp:396-418`
  — StepStats ablation fields; `types.hpp:466-488` — Diagnostics counters;
  `types.hpp:90-100` — AddressBindingState (binding_key, distance, span).
- `include/sbm/machine.hpp:92-109` — CandidateNode/ScoredNode gpaf fields;
  `machine.hpp:270-275` — GPAF maps, incl. `gpaf_slot_costed_net_value_`.
- `src/modules/machine_topology.cpp:268-281` — `gpaf_role_key_for_channel`
  (context-free key, RC1); `283-314` — structural-call key; `316-356` —
  `observe_gpaf_shadow_roles` (recency residency :341, popularity promotion
  :347-354, dead costed gate :344-346); `358-397` — bulk lifecycle ops.
- `src/modules/machine_routing.cpp:13-34` — `push_candidate` source
  overwrite (RC5); `44-48` — budget arithmetic; `88-108` — control edges;
  `110-172` — GPAF injection (previous-route channel keys, no probe dedup);
  `174-195` — neighbor probes after GPAF (unique-label caveat); `208-221` —
  uniform Hamming scoring (RC4); `251-276` — beam selection without source
  quota.
- `src/modules/token_sparse_output.cpp:311-445` — frozen GPAF ablation
  (fixed-route removal :361-409, exec-cost unit error :438-442); `640-703`
  — per-node counterfactual contribution in learn path; `:719` —
  `observe_gpaf_shadow_roles` call site (after contributions exist).
- `src/modules/machine_checkpoint.cpp:440,542` — only other references to
  `gpaf_slot_costed_net_value_` (serialization; never written elsewhere).
- `src/modules/experiment.cpp:440-485`, `src/modules/token_experiment.cpp`
  — GPAF diagnostic JSON emission.

Documents:

- `docs/superpowers/specs/2026-06-30-global-predictive-address-field-design.md`
  — key-family definitions (§Predictive role key), scoring intent (§Scoring),
  merge quotas (§Candidate merge), controls (§Controls and gates).
- `docs/superpowers/plans/2026-06-30-global-predictive-address-field.md` —
  implementation status (T1-T6).
- `DESIGN_NOTES.md:218-270` — §Next-generation global predictive addressing,
  10M validation summary.
- `RESEARCH_LOG.md:1478-1647` — 10M GPAF experiments (2026-07-01/02).
- `PROGRESS.md`, `ROADMAP_REAL_DATA.md`, `THEORY_ALIGNMENT.md` — current
  status references.

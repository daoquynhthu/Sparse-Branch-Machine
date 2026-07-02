# SPM Next-Generation Architecture Directions

## Status

Research design document. Not an implementation plan. Defines the structural
gap between SPM and dense baselines, and proposes four concrete architectural
directions for closing it while preserving the core §4 constraints.

## The structural gap

After the complete GPAF iteration (Proposals A-E, 5 key variants tested on
10M FineWeb-Edu), the honest result is:

| Model | 10M eval NLL | Gap to Transformer |
|---|---|---|
| Transformer small (10.5M params) | 5.527 | — |
| SPM upgrade-v1 (baseline) | 6.213 | 0.686 |
| SPM + gpaf-transition-v1 (best variant) | 6.212 | 0.685 |

Every GPAF variant (topology-only keys, binding-key B1, shape-key B2,
role-transition C, epistemic-state E) produces honest causal gain of
exactly 0.000000. Zero predictive value from global slot retrieval at
this scale.

The gap is not a tuning or measurement problem. It is structural:

1. **No depth**: single-pass routing → aggregation → prediction. Transformer
   has multiple layers of composition.
2. **Fixed routing**: Hamming similarity to hash-bucket prototypes is not
   learnable. Transformer attention weights are learned per-step.
3. **Independent nodes**: each node learns via 1/√visits with no mechanism
   for nodes to coordinate or specialize complementarily.
4. **Topology stagnation**: 2 accepted channels after 10M tokens. The
   adaptive topology lifecycle rarely accepts new programs.
5. **No gradient signal**: the only learning signal is per-node logit
   updates. There is no mechanism for error to propagate between nodes.

## Constraint framework

All proposals must preserve (§4 of Agent.md):

- No dense global layers or full-network backprop
- Active work bounded O(1) per token regardless of stored capacity
- No linguistic labels as supervision
- Prediction fixed before target (no leakage)
- Strict eval freeze (no state mutation during evaluation)
- Local learning only (node updates depend only on locally visible signal)

## Direction 1: Multi-pass routing

### Concept

Add 1-2 additional routing passes per token. Each pass uses the previous
pass's aggregate output as contextual signal for the next pass, creating
depth without dense layers.

```
token → signatures → pass 1 route → active set 1 → aggregate → state vector
                                                                     ↓
                             signatures + state vector → pass 2 route → active set 2
                                                                             ↓
                                   combine logits → prediction
```

### Key design decisions

- **State vector**: compact encoding of pass 1's output. Candidates include:
  top-K logit directions, responsibility distribution across channels,
  channel agreement bits (which channels agreed on the top-1 prediction).
- **Pass 2 score function**: extend the existing `score()` with an
  additional term `config_.multi_pass_weight * state_similarity(slot)`,
  where `state_similarity` measures alignment between the node's output
  vector and the pass 1 state.
- **Bounded work**: total active nodes ≤ 2 × beam_width. Each pass is an
  independent `select_route()` call. Total candidates examined bounded by
  2 × candidate budget.

### Implementation sketch

1. Add config: `multi_pass_count` (default 1), `multi_pass_weight` (default 0).
2. In `step_token_sparse` / `step_token_dense`: after the first
   `select_route` and `aggregate`, concatenate the aggregated vector to
   the per-channel signatures, then call `select_route` again.
3. The second pass's scoring includes similarity to the first pass's
   aggregate output.
4. Final prediction uses logits from both passes (weighted or gated).

### Why it might work

The first pass captures coarse address-matched content; the second pass
refines it with prediction-aware context. This mirrors the Transformer's
layer composition but with sparse routing instead of dense attention.

### Constraints check

- No dense layers: each routing pass is independent sparse selection.
- Bounded work: inactive nodes never fire. Total active nodes constant.
- No leakage: state vector computed from pre-target information only.
- Local learning: each node still updates independently.

### Expected effort: 1-2 weeks

---

## Direction 2: Learned embedding keys

### Concept

Replace Hamming similarity to hash-bucket prototypes with dot-product
matching between a learned context query and per-node learned key vectors.

```
execution frames → light query function → query vector (32-64 dims)
query · key₁, query · key₂, ... → softmax/routing weight → select top-K
```

### Key design decisions

- **Key vectors**: each node stores a small `std::vector<float> key`
  alongside its prototype/output. Dimension configurable (default 32).
- **Query function**: derive from execution frames (channel metadata,
  binding state, pattern span). A small learned linear transform or
  a fixed hash-to-embedding. Must stay bounded O(channels × key_dim).
- **Key learning**: local gradient update on the key vector. When a node's
  output is useful (contribution > 0), update key to be more similar to
  the current query. When harmful, update away. This is O(beam_width ×
  key_dim) per token — local, not global backprop.
- **Candidate merge**: keep existing source quotas (exact bucket, control
  edge, neighbor). Learned keys replace the "exact bucket" matching but
  not the full candidate set.

### Why it might work

The key weakness of Hamming similarity is that it measures surface-form
token overlap, not predictive relevance. A learned key can specialize on
"clusters of contexts where this node's output is useful" — exactly what
the node's learning signal provides. Two nodes that predict well in the
same type of context converge their keys; nodes that predict well in
different contexts diverge.

### Constraints check

- No dense layers: key comparison is dot product between two small vectors.
  The query function from execution frames is bounded.
- Bounded work: keys are per-node, matching cost is O(candidates × key_dim).
  Total candidates remain bounded by existing budget.
- No leakage: keys and queries use only pre-target information.
- Local learning: key updates are per-node, using locally visible gradient.

### Risk

The query function is critical. A fixed hash-derived query is a minor
improvement over Hamming. A learned query (small MLP) is more powerful
but must be kept small enough to not constitute a dense layer.

### Expected effort: 1-2 weeks (fixed query), 2-3 weeks (learned query)

---

## Direction 3: Channel composition

### Concept

Allow accepted address channels to form directed composition relationships.
A composed channel routes not just on the base signature but also on the
output state of its parent channel. This creates a program graph rather
than a flat channel list.

```
Channel A (Tuple, lag=1)  →  produces positional binding
Channel B (ContentMatch, lag=16)  →  produces content binding
Channel C (A's output + B's output)  →  combines both for routing
```

### Key design decisions

- **Composition rule**: a channel declares parent channels from its
  address program reference (already exists as `dependency_channel`).
  Extend to multi-parent: a channel may name up to 2 parent channels.
- **Routing signal**: when computing `candidate_ids` for a composed
  channel, the signature incorporates both the current window signature
  AND the parent channels' binding keys / execution frames.
- **Lifecycle**: composed channels inherit the probe/validate/admit/reject
  lifecycle. A composed channel is proposed after both its parents are
  accepted.
- **Bounded graph**: each channel has ≤ 2 parents, ≤ 4 children (config).
  Total channels bounded by `max_address_channels`.

### Why it might work

The current channel set is flat and independent: Tuple, ContentMatch,
ContentFollow each capture different aspects of the input but never
combine. A composed channel that says "this Tuple-1 node AND this
ContentMatch node together predict the next token" can represent
interactions that neither channel captures alone.

### Constraints check

- No dense layers: composition is sparse program-reference, not matrix
  multiplication.
- Bounded work: composed channels are just more channels. Each adds at
  most beam_width active nodes. Total bounded by max_address_channels ×
  beam_width.
- Local learning: each node in each channel still updates independently.

### Expected effort: 2-3 weeks

---

## Direction 4: Embedding-centric sparse retrieval

### Concept

Replace the entire fixed-hash addressing system with learned embedding
retrieval. Input tokens pass through a small embedding layer and context
encoder, producing a query vector. This query retrieves sparse nodes by
matching against stored key vectors.

```
[t0, t1, ..., tn] → embedding layer → context encoding (small RNN or
    positional blend) → query vector → sparse retrieval via key matching
    → top-K nodes → aggregate outputs → prediction
```

### Key design decisions

- **Embedding layer**: small learned embedding per token (vocab × 64 dims).
  This is a dense lookup table, not a computation — the same type as
  any sparse system's token ID mapping.
- **Context encoding**: blend recent embeddings via fixed positional
  weights (not learned attention). This is O(context_width × embed_dim),
  not O(all-pairs).
- **Retrieval**: per-node key vectors (Direction 2). Query × key dot
  product → top-K.
- **Memory**: each node stores (key_vector, output_vector). No prototypes.
  No bucket directory. Nodes are organized in a bounded approximate
  nearest-neighbor index (or simple hash table over key regions).
- **No address programs**: the full adaptive topology system is replaced
  by learned retrieval. The topology becomes implicit — "which nodes
  exist" rather than "which programs to run".

### Why it might work

The address-program system was the project's original contribution —
explicit, auditable, bounded address computation from token windows.
It works for small vocabularies and deterministic tasks. On real text
at 16K vocabulary, its signal is too weak. A learned embedding approach
is closer to how Transformers work (learned token representations,
learned matching) while preserving sparsity (bounded active nodes per
token, no self-attention over all positions).

### Constraints check

- **Borderline §4.1**: the embedding layer is a learned lookup table,
  acceptable. The context encoder could be a small learned transform—
  must stay small (≤ 2 layers, ≤ 128 dims). No dense global matrix.
- Bounded work: retrieval still selects K nodes. Total candidates bounded.
- No leakage: query computed from input tokens only.
- Local learning: node keys and outputs updated locally.

### Risk

This is the most radical departure from the existing architecture.
It replaces the adaptive topology system (thousands of lines of C++,
the full address interpreter, the proposal lifecycle) with learned
retrieval. Much of the existing codebase becomes dead code. The risk
is that learned retrieval at K candidates cannot match an attention
mechanism over all context positions.

### Expected effort: 4-6 weeks

---

## Comparison

| Criterion | D1: Multi-pass | D2: Learned keys | D3: Composition | D4: Embed retrieval |
|---|---|---|---|---|
| Depth | ✓ | — | ✓ (graph) | — |
| Learnable routing | — | ✓ | — | ✓ |
| Code reuse | high | high | medium | low |
| §4.1 risk | none | none | none | low |
| Novelty | low | medium | medium | high |
| Expected impact | small | medium | medium | high |
| Effort | 1-2 wk | 1-3 wk | 2-3 wk | 4-6 wk |

## Recommendation

Implement **Direction 1 (multi-pass routing) first**, then **Direction 2
(learned keys)**. These are complementary and can be developed
incrementally on top of the existing codebase:

1. Multi-pass creates depth with minimal code changes and immediately
   tests whether the model benefits from a second look at the input.
2. Learned keys address the fundamental limitation that routing is
   based on surface-form token hash rather than learned relevance.
   They can be added on top of multi-pass or independently.

Direction 3 (composition) is a natural extension of the existing
address-semantics framework and could be pursued in parallel if
multi-pass shows directional improvement.

Direction 4 (embedding-centric) should only be attempted after 1-3
have been tried and shown insufficient, because it requires rewriting
the core addressing system.

## References

- 10M GPAF results: `RESEARCH_LOG.md` (2026-07-02 entries)
- GPAF architecture investigation: `docs/superpowers/research/2026-07-02-gpaf-architecture-investigation.md`
- Existing architecture: `DESIGN_NOTES.md`
- Constraints: `Agent.md §4`

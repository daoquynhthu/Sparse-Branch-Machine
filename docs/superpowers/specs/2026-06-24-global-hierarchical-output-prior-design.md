# Global Hierarchical Output Prior Design

## Scope

This change resolves `ISSUES.md` items P0-1 and P0-4 without changing address
routing, topology lifecycle, corpus handling or per-node sparse capacity. It
adds one corpus-wide next-token base distribution and decouples the implicit
output tree from model stochasticity.

The falsifiable hypothesis is that a shared base distribution removes the
repeated uniform restart at every address. With local residual learning
disabled, frozen validation should match the corresponding hierarchical
unigram estimator. If the full fixed-topology model remains worse than unigram,
the global-prior hypothesis is insufficient and the result must be recorded as
negative rather than hidden with additional tuning.

## Output decomposition

For every implicit-tree decision `d`, prediction uses:

```text
combined_logit(d) = global_logit(d)
                    + sum_i responsibility_i * local_residual(i, d)
```

The global component is a count estimator. For each decision ID in
`[0, vocabulary - 2]`, store `total_count` and `right_count`. The fixed Jeffreys
pseudocount is `0.5` per branch:

```text
global_logit(d) = log((right_count[d] + 0.5) /
                      (total_count[d] - right_count[d] + 0.5))
```

Counts are read before the target is observed. When `learn=true`, the target
path increments counts only after loss, ranking and counterfactual credit are
fixed. When `learn=false`, counts and every local parameter remain unchanged.

The beam decoder and target NLL must use the same combined logit. Topology and
node counterfactuals remove only the selected local residual contribution; the
global prior remains in every counterfactual.

## Storage and complexity

The global prior owns two `uint64_t` count arrays and one derived `float` logit
cache of `vocabulary - 1` entries. The cache is refreshed only for decisions on
an observed target path and avoids repeated logarithms during beam decoding. It
is a single `O(V)` output bias, not per-node state. Per-token prediction and update
touch only the target or beam paths and remain `O(log V)` with bounded beam
width. No vocabulary scan is introduced.

`output_structure_bytes` continues to describe the constant-storage implicit
tree. New diagnostics report `global_output_prior_bytes` and
`global_output_prior_updates`; both are included in `estimated_bytes`.

## Output-tree seed

Add runtime parameter `output_tree_seed`, default `7`. `config.seed` continues
to control model stochasticity. `output_tree_seed` alone constructs
`ImplicitOutputTree`, is emitted in configuration/result JSON and must remain
fixed across multi-seed comparisons. This is additive to ABI v4 because the C
ABI passes configuration by opaque handle and registry name.

## Correctness gates

1. Parameter schema and C/Python configuration expose `output_tree_seed`.
2. Different model seeds with the same output-tree seed have identical
   prior-only token probabilities.
3. A power-of-two vocabulary starts from exactly uniform token probability.
4. Prior counts update after prediction and never during frozen evaluation.
5. A prior-only imbalanced-token fixture reaches its analytically computed
   hierarchical unigram NLL.
6. Target NLL and beam decoding both include the global prior.
7. `global_output_prior_bytes` is exactly bounded by two count arrays plus one
   derived float cache, and local sparse capacity remains unchanged.
8. Full C++, C and Python API tests pass before the implementation commit.

## Experiment gate

After correctness tests, run the existing fixed-topology compatibility smoke
with identical data, seeds and routing configuration. Then run the existing
approximately 1M/0.1M medium corpus only if the smoke is finite and faster than
the previous dense-baseline path. Acceptance requires the full model to close
the unigram gap materially; matching the prior-only control is the minimum
correctness gate, not evidence for contextual learning.

## Non-goals

- per-decision local learning statistics and evidence-based eviction (P0-2 and
  P0-3) remain the next isolated change;
- cross-channel mass normalization (P0-5) remains a separate adaptive-topology
  change;
- no tokenizer, dataset, checkpoint or content-address language work is part of
  this implementation.

# Theory alignment: validated sparse address programs

This branch began at the tokenizer-aligned baseline `b47713b`.  Version 9 made
address topology mutable, but its proposal space still consisted only of
single temporal lags.  The current version replaces that task-shaped object
with a minimal address-program language.

The intended learning loop is now:

```text
prediction error
    -> propose a sparse address program
    -> adapt only its local nodes
    -> freeze the candidate
    -> measure exact counterfactual loss
    -> accept, reject, or later retire it
```

## Minimal address-program language

An address program is an ordered, duplicate-free set of one or two positive
history offsets.  The current token is implicit.  Examples are:

```text
[1]       current token + token at lag 1
[1, 4]    current token + tokens at lags 1 and 4
[2, 4]    current token + tokens at lags 2 and 4
```

The language deliberately contains no arithmetic operators, semantic labels,
task-specific lag list, learned matrix, or nested program tree.  Its only
meta-rule is sparse selection of historical positions.

Programs are proposed in increasing temporal/structural complexity:

```text
[2], [1,2], [3], [1,3], [2,3], [4], [1,4], ...
```

The order does not encode the mathematical generator's known dependencies.
Every program must survive the same predictive-credit lifecycle.

## Coarse-to-fine addressing

For a two-offset program, the current token and earlier program operand form
the coarse address region; the remaining selected history and longer context
remain in the full prototype used for bucket-local discrimination.  This is a
bounded hierarchy rather than a flat joint table.

An attempted alternative hashed every operand directly into the top-level
bucket.  Across seeds 7, 11 and 19 it worsened mean frozen NLL from 3.3664 to
3.4277.  The joint address became too sparse for the available sample count.
That version was rejected rather than retained as a superficially cleaner
implementation.

## Structural lifecycle

Every proposed program passes through the same six stages:

1. **Proposal** — allocate a real channel, address namespace and local nodes.
2. **Adaptation** — update only candidate-local parameters.
3. **Frozen validation** — stop candidate learning during the validation tail.
4. **Exact channel ablation** — remove the complete channel contribution and
   recompute loss.
5. **Selection** — retain only positive mean validation credit.
6. **Mature audit** — retire an accepted non-seed program if sustained credit
   becomes sufficiently negative.

For token cross-entropy, credit is

\[
C_c=L(z-z_c,y)-L(z,y),
\]

where \(z_c\) is the complete active logit contribution of program \(c\).
For vector regression, the corresponding normalized-MSE difference is used.

Rejected or retired programs are physically reclaimed: their nodes are
removed, indexes rebuilt, stale route state cleared, and unrelated logical node
IDs remain stable.

## Strict freeze semantics

The training/evaluation boundary now explicitly freezes topology.  An unfinished
probe is rejected using training observations only.  During evaluation the
system no longer:

- proposes, accepts, rejects or retires programs;
- updates topology credit;
- computes node counterfactual credit used only for learning.

Thus frozen evaluation is both structurally and statistically read-only.

## Main result

Default task: 32-token mathematical next-token prediction, 64 independent
sequences of length 2,048, 80,000 training examples, then strict evaluation.

| model/control | mean frozen NLL, seeds 7/11/19 |
|---|---:|
| adaptive sparse programs, arity <= 2 | **3.36642** |
| fixed multiscale conditional-table baseline | 3.39199 |
| fixed singleton channels `[1,2,4]` | 3.41819 |
| adaptive singleton-only topology | 3.42692 |
| previous v9 lag-only adaptive topology | 3.41847 |
| unigram baseline | about 3.4658 |
| mathematical generator oracle | about 2.9143 |

The adaptive program model improves over the strong fixed multiscale table by
about 0.0256 nats/token and over the previous lag-only topology by about 0.0520
nats/token.

Learned final programs are not identical across seeds, but all three runs retain
interaction programs involving recent history.  Common retained structures are
`[1,2]`, `[1,3]` and `[1,4]`; additional accepted programs differ by seed and
remain subject to mature auditing.

This is the first experiment in the project where learned topology outperforms
a hand-constructed statistical control that knows the generator's three stated
time scales.

## Performance work that preserves semantics

The following optimizations were accepted only after result equality checks:

- reusable signature, channel-credit and zero-output buffers;
- contiguous fixed-capacity token history instead of allocating a window each
  step;
- direct log-sum-exp counterfactual loss without materializing a probability
  vector for every ablation;
- reuse of channel ablation when a channel has one active node;
- AVX2/FMA dense logit update in the softmax null-space;
- periodic rather than per-update logit recentering;
- no counterfactual work during frozen evaluation;
- thin CLI compiled at low optimization because it is not on the model path.

On seed 7, removing frozen-evaluation credit work raised observed throughput
from roughly 68k to roughly 89k steps/s in the direct before/after run, with
identical train/evaluation NLL, graph size and routes.  Repeated runs remain
noisy; the three-seed full-task mean for the adaptive-program model is about
71.8k steps/s because learned topology and graph size vary by seed.

## Vocabulary scaling

A short 32-sequence scaling probe produced:

| vocabulary | steps/s | estimated model bytes |
|---:|---:|---:|
| 32 | 151k | 5.9 MB |
| 128 | 89.9k | 6.2 MB |
| 512 | 29.4k | 18.8 MB |

The dense local-logit representation therefore remains the next major scaling
limit.  Replacing it requires a separate output-addressing design; sampled or
hierarchical normalization has not been added merely to improve this benchmark.

## Automated calibration

A 12-candidate, three-seed successive-halving search covered program arity,
probe duration, validation duration, acceptance threshold and retirement
threshold.  The unmodified default configuration was selected as best.  No
manually chosen parameter override was accepted.


## Retained vector-objective limitation

The topology lifecycle is currently calibrated on token cross-entropy.  On the
retained vector benchmark, adaptive sparse programs keep only `[1]` and reach
approximately `R2=0.528`, while the fixed singleton control `[1,2,4]` remains
approximately `R2=0.805`.  The token result therefore does not yet establish a
task-independent topology criterion.  No vector-specific acceptance threshold
or hand-written proposal order was introduced to conceal this gap.

## Remaining theoretical gap

The model now learns *which sparse history selections exist*, but not yet:

- transformations over selected values;
- variable binding or reusable operators;
- program calls and returns;
- learned proposal distributions;
- sparse large-vocabulary output normalization;
- a task-independent criterion for when address-program arity should exceed two.

The next step should not add a library of hand-written operators.  A defensible
extension would allow one additional generic operation only if it can be
proposed, validated and erased by the same lifecycle and if a simpler address
program cannot explain the gain.

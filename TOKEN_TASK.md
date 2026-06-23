# Token objectives and dataset contract

> **Document role:** This document defines the token-level learning contract,
> the mathematical regression fixture and the requirements for real tokenized
> corpora. It does not claim that the mathematical fixture is language data.

## 1. Common token-training contract

For every document or independent sequence:

\[
x_t\in\{0,\ldots,V-1\},\qquad
\hat p_t=M(x_{\le t}),\qquad
\mathcal L_t=-\log \hat p_t(x_{t+1}).
\]

The model receives:

- a flat token-ID stream;
- vocabulary size;
- explicit sequence/document offsets;
- a declared training boundary or streaming split.

At a document boundary the machine resets transient history, previous-route
state and delayed trace credit. Persistent learned nodes and edges remain.

The scored prediction is fixed before the target is used for learning.

## 2. Real natural-language token data

Real text is the primary next-stage dataset. The current workspace does not yet
contain it. Acquisition, tokenizer training, sharding and streaming are defined
in `ROADMAP_REAL_DATA.md`.

A real tokenized dataset must record:

- corpus source and pinned revision;
- document-level train/validation/test split;
- tokenizer artifact and checksum;
- vocabulary and special-token IDs;
- preprocessing version;
- document offsets;
- shard checksums;
- exact token counts.

Tokenizer training uses training documents only. Validation and test text must
not affect tokenizer learning, parameter search or topology calibration.

## 3. Mathematical token fixture

The mathematical generator remains a deterministic regression and mechanism
fixture. It uses the same token-ID and cross-entropy interface as real training,
but its source distribution is explicitly synthetic.

Let \(x_0,x_1,x_2,x_4\) denote the current token and tokens at delays 1, 2 and
4. Four circular centers are computed modulo vocabulary size \(V\):

\[
\begin{aligned}
 c_1 &= 5x_0+3x_1+11,\\
 c_2 &= 7x_0+9x_2+3(x_0\oplus x_2)+17,\\
 c_4 &= 13x_0+5x_4+19,\\
 c_I &= x_1(x_4+1)+3x_2+(x_0\oplus x_4)+23.
\end{aligned}
\]

Candidate-token logits are:

\[
\ell(y)=1.20\cos\frac{2\pi(y-c_1)}V
+0.92\cos\frac{2\pi(y-c_2)}V
+0.72\cos\frac{2\pi(y-c_4)}V
+\alpha\cos\frac{2\pi(y-c_I)}V
+0.18\cos\frac{4\pi(y-c_1)}V.
\]

The source distribution is:

\[
p^*(y\mid x_{\le t})=\operatorname{softmax}(\ell(y)/T),
\]

and the next token is sampled from it. The fixture records source NLL for the
realized target, providing an oracle reference.

There is no hidden stochastic state. All systematic dependency is a function of
visible token history; irreducible uncertainty is the categorical draw.

## 4. What the fixture may and may not establish

It may establish:

- token API correctness;
- no target leakage;
- exact/frozen cross-entropy behavior;
- deterministic save/reload;
- address-program and topology lifecycle mechanics;
- dense versus sparse output scaling;
- regression protection after code changes.

It may not establish:

- syntax, semantics or reference;
- content-conditioned variable binding;
- natural-language generalization;
- robust topology under heterogeneous text;
- competitive language-model quality.

Improving fixture NLL alone is not a theory milestone.

## 5. Metrics

For real and mathematical token data, report:

Primary:

- NLL in nats/token;
- bits/token;
- perplexity as a derived value;
- prequential codelength when block evaluation is available.

Secondary:

- top-k candidate recall;
- target probability;
- calibration;
- token-frequency and context-length bucket losses;
- structural and resource metrics.

Top-1 accuracy is not the primary criterion.

## 6. Output implementations

The repository contains two output paths:

1. a dense per-node vocabulary-logit control;
2. an experimental hierarchical sparse output storing observed binary decisions.

The sparse path is intended to reduce large-vocabulary storage and target-NLL
work. Its quality and decoding behavior must be re-evaluated on real tokenized
text. It is not accepted merely because it has better asymptotic storage.

## 7. External tokenizer API

The C/Python interface accepts tokenizer output as:

- unsigned token IDs;
- vocabulary size;
- document offsets.

No token strings or linguistic labels are required by the C++ model. The current
in-memory constructor copies arrays. Real-corpus training additionally requires
the streaming shard interface specified in `ROADMAP_REAL_DATA.md`.

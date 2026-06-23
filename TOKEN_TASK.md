# Mathematical token task

## Purpose

This task aligns the research machine with tokenized training without claiming
that synthetic mathematics is natural language.  The dataset and loss contract
are deliberately the same as next-token training, while the source distribution
is explicit, reproducible and auditable.

For every sequence and position:

\[
 x_t \in \{0,\ldots,V-1\},\qquad
 \hat p_t = M(x_{\le t}),\qquad
 \mathcal L_t=-\log \hat p_t(x_{t+1}).
\]

## Generator

Let \(x_0,x_1,x_2,x_4\) denote the current token and tokens at delays 1, 2 and
4.  Four circular centers are computed modulo the vocabulary size:

\[
\begin{aligned}
 c_1 &= 5x_0+3x_1+11,\\
 c_2 &= 7x_0+9x_2+3(x_0\oplus x_2)+17,\\
 c_4 &= 13x_0+5x_4+19,\\
 c_I &= x_1(x_4+1)+3x_2+(x_0\oplus x_4)+23.
\end{aligned}
\]

All expressions are reduced modulo \(V\).  Candidate-token logits are:

\[
\ell(y)=1.20\cos\frac{2\pi(y-c_1)}V
+0.92\cos\frac{2\pi(y-c_2)}V
+0.72\cos\frac{2\pi(y-c_4)}V
+\alpha\cos\frac{2\pi(y-c_I)}V
+0.18\cos\frac{4\pi(y-c_1)}V.
\]

The true distribution is

\[
 p^*(y\mid x_{\le t})=
 \operatorname{softmax}(\ell(y)/T),
\]

and \(x_{t+1}\) is sampled from \(p^*\).  The dataset records
\(-\log p^*(x_{t+1})\), providing an oracle lower-bound reference for the
realized samples.

There is no hidden stochastic state.  Every systematic component is a function
of visible token history; randomness is only the categorical draw itself.

## Sequence boundaries

A token dataset stores offsets

\[
0=o_0<o_1<\cdots<o_n=|X|.
\]

Each interval \([o_i,o_{i+1})\) is an independent sequence.  At the boundary the
machine resets transient history, previous-route state and delayed trace credit.
Learned long-term nodes and edges remain intact.

## Why accuracy is not the main metric

The distribution is intentionally broad.  Top-1 accuracy is therefore expected
to remain low even for a good model.  The primary metric is cross-entropy, with
bits/token and perplexity as monotonic representations.  Oracle cross-entropy
separates model error from irreducible sampling entropy.

## External tokenizer contract

Any tokenizer may provide:

- a flat array of unsigned token IDs;
- vocabulary size;
- sequence/document offsets.

No text, vocabulary strings or linguistic labels are required by the C++ model.
The current task generator can therefore be replaced later by real tokenized
shards without changing the learning call.

# Real-data transition roadmap

> **Document role:** This is the authoritative plan for the next research phase.
> It defines why the project is blocked, which data must be acquired, which
> engineering work is prerequisite, which experiments are admissible, and what
> evidence is required before another architecture claim is accepted.
>
> This document is normative for future work. Historical results remain in
> `RESEARCH_LOG.md`; theoretical goals remain in `THEORY_ALIGNMENT.md`.

## 1. Why the project is currently blocked

The present bottleneck is not that synthetic mathematical tasks have become
computationally impossible. The bottleneck is that the current cloud workspace
does not contain a real natural-language corpus suitable for training and
held-out evaluation.

The mathematical token generator has been useful for:

- validating token-ID ingestion and sequence boundaries;
- validating exact cross-entropy and frozen evaluation;
- testing sparse addressing, topology lifecycle and sparse output prototypes;
- exposing target leakage, task-sensitive topology thresholds and scaling
  failures;
- producing fully reproducible regression tests without external data.

It cannot establish that the machine learns:

- content-conditioned dependencies rather than positional regularities;
- entity or event reuse across variable distances;
- reference, agreement, scope or compositional relations;
- robust structures across heterogeneous documents and domains;
- a topology criterion that remains meaningful on natural-language entropy and
  frequency distributions.

Accordingly, further synthetic-benchmark optimization is no longer the primary
research path. The next phase must acquire, tokenize, shard and train on real
language data while retaining the mathematical task as a test fixture.

## 2. Research objective of the real-data phase

The immediate goal is **not** to claim competitive language-model quality. It is
to determine whether the current CPU-first sparse machine can learn nontrivial
regularities from genuine language without collapsing into one of four failure
modes:

1. a high-order positional lookup table;
2. a unigram or short-context frequency estimator;
3. uncontrolled structure growth proportional to token count;
4. a task-specific topology selected by fragile loss thresholds.

The primary question is:

\[
\text{Can bounded sparse execution discover reusable predictive structure in
real token streams?}
\]

The secondary questions are:

- Does model capacity grow while per-token active work remains bounded?
- Do accepted address programs persist across documents, seeds and corpus
  slices?
- Does the model improve held-out prequential codelength beyond simple
  statistical controls?
- Can the same structural value criterion compare common and rare phenomena?
- Which errors require content-conditioned binding or relation-following rather
  than additional positional selectors?

## 3. Data policy

### 3.1 Real-data definition

For this phase, a dataset counts as real natural-language data only when its
text was produced as ordinary human-facing language rather than generated
specifically as synthetic model-training examples.

A synthetic story corpus may be used to test ingestion or tokenizer plumbing,
but it does not satisfy any real-data research gate.

### 3.2 Dataset ladder

The project should advance through fixed stages rather than jump directly to a
large web corpus.

#### D0 — local corpus fixture

Purpose: parser, tokenizer, document-boundary and binary-shard tests.

Requirements:

- a few hundred short public-domain or repository-local text passages;
- at least dozens of independent documents;
- tiny enough for every CI run;
- not used for research claims.

The fixture may be generated from openly distributable text excerpts, but its
origin and license must be recorded in the manifest.

#### D1 — small real-language correctness corpus

Recommended initial source: a raw WikiText split, beginning with WikiText-2 and
then WikiText-103.

Purpose:

- first genuine next-token training run;
- tokenizer and shard reproducibility;
- full-epoch and streaming equivalence;
- overfit diagnostics on a small corpus;
- comparison against unigram, n-gram and fixed-context controls.

The validation and test splits supplied by the source must remain untouched.
No document from validation or test may enter tokenizer training, parameter
search or topology proposal calibration.

#### D2 — fixed heterogeneous web sample

Recommended source: a deterministic document-level sample from FineWeb or
FineWeb-Edu.

Initial scale ladder:

- 10 million training tokens;
- 100 million training tokens;
- 500 million training tokens, only after smaller gates pass.

Purpose:

- expose domain heterogeneity;
- test online topology stability;
- test memory growth and CPU throughput over longer streams;
- determine whether structures learned on one shard transfer to another;
- prevent results from depending on Wikipedia-specific regularity.

The sample must be created by a committed manifest containing source revision,
filter configuration, document hash policy and deterministic sampling seed. The
raw corpus itself must not be committed to Git.

#### D3 — domain transfer

Use at least two non-overlapping domain partitions from the real corpus.
Examples include encyclopedic prose versus general web prose, or technical text
versus narrative prose.

Train topology on domain A, then evaluate:

- frozen on A validation;
- frozen on B validation;
- continued local learning on B with topology frozen;
- continued topology learning on B under a separate run.

This stage tests whether a structural program is reusable or merely a compact
index into one distribution.

#### D4 — multilingual or second-language test

Only after D1–D3 are stable, repeat the pipeline on a second language using a
separate documented corpus shard. This is not intended to prove universal
linguistic structure; it tests whether the lifecycle and codelength criterion
are tied to English token statistics.

### 3.3 Dataset artifacts that may enter Git

Allowed:

- acquisition scripts;
- manifests and source revision identifiers;
- tokenizer JSON/model files when licensing permits;
- tiny CI fixtures;
- document split lists or content hashes;
- preprocessing configuration;
- aggregate statistics;
- checksums for generated shards.

Forbidden:

- full downloaded corpora;
- large tokenized shards;
- license-incompatible text excerpts;
- access credentials or cache paths tied to one machine.

## 4. Tokenizer plan

The tokenizer is part of the experimental condition and must not drift between
runs.

### 4.1 First tokenizer

Use one fixed byte-level subword tokenizer trained on the D1 training split only.
A vocabulary in the 4,096–8,192 range is preferred for the first real-data gate:
large enough to expose a genuine token distribution, but small enough that the
sparse output implementation can be diagnosed before 30k+ vocabulary scaling.

The tokenizer artifact must record:

- algorithm and library version;
- normalization and pre-tokenization rules;
- vocabulary size and special-token IDs;
- training-corpus manifest hash;
- deterministic seed where applicable;
- complete serialized tokenizer;
- checksum.

Do not reuse a pretrained tokenizer merely for convenience unless the experiment
explicitly studies compatibility with that tokenizer. A pretrained tokenizer
introduces information from an external corpus and complicates attribution.

### 4.2 Required special tokens

At minimum:

- beginning of document;
- end of document;
- unknown token only if required by the algorithm;
- optional padding token for tooling, never treated as ordinary training data.

Documents must remain explicit. Concatenating documents without a boundary token
and transient-state reset is forbidden.

### 4.3 Tokenizer controls

At least one later ablation should compare:

- the fixed subword tokenizer;
- a byte-level or character-level representation on the same text subset.

The purpose is to distinguish architecture behavior from tokenizer-induced
locality. This is not a first-gate requirement.

## 5. Corpus ingestion and shard format

The current `token_dataset_from_ids` API proves the in-memory contract but is
not sufficient for real training. The next engineering milestone is a streaming
corpus layer.

### 5.1 Required components

1. **Text source iterator**
   - yields one document at a time;
   - preserves source document ID and metadata needed for split verification;
   - does not concatenate documents.

2. **Tokenizer adapter**
   - converts one document at a time;
   - inserts explicit boundary tokens;
   - records token count and rejected/empty documents.

3. **Binary shard writer**
   - stores token IDs in a fixed endian format;
   - stores document offsets separately;
   - stores vocabulary size, tokenizer hash, corpus-manifest hash and checksum;
   - supports shards that can be memory-mapped independently.

4. **Streaming dataset handle**
   - iterates shards without loading the full corpus;
   - supports deterministic start position and epoch order;
   - exposes document boundaries to the machine;
   - supports a bounded read-ahead buffer;
   - never changes sample order silently.

5. **Checkpoint cursor**
   - records shard ID, document ID, token offset and RNG state;
   - allows exact resume without replaying or skipping data.

### 5.2 Split discipline

Splits are made at document level before tokenization. The same normalized
content hash must not appear in multiple splits. At minimum, exact normalized
hash deduplication must be performed across train, validation and test manifests.

Tokenizer training uses training documents only. Automatic hyperparameter search
uses validation data only. Test data is read only for final accepted runs.

### 5.3 Manifest example

Every experiment should be resolvable from a machine-readable manifest with
fields equivalent to:

```json
{
  "source": "Salesforce/wikitext",
  "source_revision": "pinned revision",
  "split_policy": "source train/validation/test",
  "tokenizer_sha256": "...",
  "preprocess_version": 1,
  "shards": [
    {"path": "train-00000.sbt", "sha256": "...", "tokens": 123456}
  ]
}
```

The concrete schema should be versioned before D1 is accepted.

## 6. Model work required before large real-data runs

### 6.1 Sparse output must be treated as an enabling component

The current hierarchical sparse output prototype changes storage from one dense
vocabulary vector per node to observed binary decisions. It is promising at
large vocabulary, but on the 32-token synthetic benchmark it loses quality and
its exact top-k traversal can negate the asymptotic benefit.

Before D1:

- retain exact target-token NLL;
- retain fixed-cost candidate decoding for diagnostics;
- report sparse decision count and actual bytes;
- compare dense and sparse output at the same tokenizer vocabulary on a small
  real corpus;
- do not select the output design solely from synthetic NLL.

For the first 4k–8k tokenizer, a dense control may be retained on a reduced node
budget if memory permits. The sparse output is accepted only if its quality,
capacity and CPU cost are reported together.

### 6.2 Checkpointing

Real-corpus runs require serialization of:

- configuration and parameter schema version;
- tokenizer and corpus manifest hashes;
- logical-node IDs and physical storage;
- address programs and lifecycle state;
- sparse edges;
- output parameters;
- optimizer/local update statistics;
- topology event history or a compact committed prefix;
- data cursor and RNG state.

A checkpoint must reload to an identical next prediction on a fixed continuation
fixture.

### 6.3 Bounded growth

The following curves are mandatory:

\[
N_{nodes}(t),\quad N_{edges}(t),\quad N_{active}(t),\quad
C_{candidates}(t),\quad bytes(t).
\]

The project goal requires persistent capacity to grow more rapidly than active
work. A run is invalid as evidence for the architecture if active nodes,
candidate scans or output work grow approximately linearly with stored nodes.

## 7. Baselines for real text

No result is meaningful without controls trained on the identical token stream.

Minimum controls:

1. unigram distribution;
2. bigram and trigram counts with documented smoothing;
3. a bounded-memory n-gram or Kneser–Ney-style control when available;
4. fixed positional address tables using the same memory budget;
5. the current machine with topology fixed;
6. the current machine with topology adaptive.

A small recurrent or Transformer baseline may be added to compare
quality-per-resource, but it is not required to establish the first structural
gate. If used, parameter count, training tokens, precision, hardware and wall
energy methodology must be reported.

## 8. Metrics

### 8.1 Predictive metrics

Primary:

- validation/test NLL in nats/token;
- bits/token;
- perplexity as a derived display value;
- prequential codelength over ordered blocks.

Secondary:

- top-k candidate recall;
- calibration by probability bin;
- rare-token and frequency-decile NLL;
- document-position NLL;
- context-length bucket NLL.

Top-1 accuracy is not a primary language metric.

### 8.2 Structural metrics

- accepted, rejected and retired program count;
- program survival time;
- direct and held-out codelength contribution;
- number of callers/contexts reusing a program;
- topology overlap across seeds;
- topology transfer across corpus shards;
- node and edge growth per million tokens;
- active nodes and candidate checks per token;
- bytes per learned token and bytes per retained program.

### 8.3 CPU metrics

- tokens/s separated into training and frozen evaluation;
- peak resident memory;
- model-state bytes excluding corpus cache;
- branch misses, cache misses and memory bandwidth when profiling is available;
- quality at fixed CPU time and fixed memory;
- quality gain per additional stored node.

A CPU advantage claim requires a matched baseline and cannot be inferred from
architecture shape alone.

## 9. Structural value criterion

The current raw loss-credit threshold is known to be task-sensitive. Real-data
work should migrate toward a common codelength account.

For a proposed structure \(P\), record:

\[
S(P)=\Delta C_{heldout}(P)-C_{describe}(P)-\lambda C_{execute}(P).
\]

Where:

- \(\Delta C_{heldout}\) is held-out prequential nats saved by retaining the
  structure;
- \(C_{describe}\) is the cost of encoding the program, bindings and persistent
  parameters under a declared code;
- \(C_{execute}\) is a declared resource penalty, initially measured rather
  than folded into acceptance;
- all terms are accumulated over an explicit validation window.

The first real-data implementation may log these terms without immediately using
them for acceptance. Changing the lifecycle decision rule is a separate theory
change and requires an ablation against the existing criterion.

Absolute task independence is not assumed. The goal is a comparable unit and a
transparent complexity tradeoff, not a universal magic threshold.

## 10. Content-conditioned program research

Real text is required to justify the next program-language redesign. Do not add
a large instruction set before observing where positional programs fail.

The intended minimal research direction is a relation-capable core such as:

- `Bind`: retain the identity/state of a selected item;
- `Match`: select candidates using current content and bound state;
- `Follow`: traverse a learned relation from one persistent state to another.

These are research placeholders, not approved instructions. Before one is added,
the proposal must specify:

1. its exact state and execution semantics;
2. how it is generated from a parent program by a local edit;
3. how dependencies and callers are tracked;
4. how it is ablated without attributing all downstream failure to one shared
   prerequisite;
5. how it is erased or rolled back;
6. which real-corpus diagnostic cannot be explained by simpler positional
   structure.

No linguistic labels such as subject, object, entity or modifier may be supplied
as supervision. If such concepts emerge, they must be inferred from prediction
and reuse.

## 11. Experimental sequence

### Phase R0 — acquisition and reproducibility

Deliverables:

- data acquisition script;
- pinned manifest;
- license record;
- tokenizer training script;
- tokenizer artifact and checksum;
- binary shard writer/reader;
- exact replay test.

Hard gate: two independent machines or clean directories produce identical
manifest hashes and token shards from the same pinned source.

### Phase R1 — small real-language baseline

Dataset: D1 small real corpus.

Runs:

- unigram/bigram/trigram;
- fixed-address dense output where feasible;
- fixed-address sparse output;
- adaptive topology sparse output.

Hard gate: model beats unigram and at least one short-context control on held-out
NLL without evaluation-time updates, target leakage or unbounded node growth.

### Phase R2 — structural diagnosis

Add analysis by context length, token frequency, document position and topology
reuse. Inspect examples only after aggregate metrics are frozen.

Hard gate: at least one accepted structure contributes held-out codelength across
multiple documents and survives an independently sampled validation block.

### Phase R3 — 100M-token heterogeneous stream

Use D2 fixed web sample.

Hard gate:

- exact checkpoint/resume;
- bounded active work;
- topology event rate does not grow without convergence;
- validation improvement persists across at least three corpus shards and seeds.

### Phase R4 — content-conditioned program prototype

Only begin after R2 identifies a reproducible failure that positional programs
cannot solve.

Hard gate: the new primitive improves held-out codelength or transfer at matched
persistent bytes and active work, and its complete lineage is auditable.

### Phase R5 — CPU scaling comparison

Compare against simple neural/statistical controls using matched corpus and
resource reporting.

Hard gate: no claim of CPU superiority without quality-matched time, memory and
energy evidence.

## 12. Work paused until real data is available

The following are not priority work:

- adding more arithmetic address operators;
- increasing positional-program arity merely to improve the mathematical task;
- manually tuning acceptance thresholds per objective;
- optimizing 32-token synthetic NLL beyond regression needs;
- interpreting synthetic topology as syntax;
- claiming language readiness from tokenizer-ID API compatibility;
- claiming CPU advantage from asymptotic design alone.

The mathematical task remains mandatory for fast regression tests, but a change
that improves only that task is not a theory milestone.

## 13. Definition of completion for the next milestone

The real-data transition milestone is complete only when all of the following
exist:

- a pinned real-language corpus manifest;
- a reproducible tokenizer trained without validation/test leakage;
- versioned memory-mappable token shards with document boundaries;
- streaming C/C++ and Python dataset APIs;
- exact checkpoint/resume;
- automated baseline and multi-seed experiment scripts;
- held-out NLL and prequential codelength reports;
- structural and CPU scaling reports;
- updated documentation with synthetic and real results clearly separated.

Until then, the repository should describe itself as a sparse learning research
platform with tokenizer-aligned synthetic validation, not as a trained language
model.

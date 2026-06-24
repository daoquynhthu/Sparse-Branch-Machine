# Long-Term Corpus Foundation Research

## Decision

Use a pinned document-level subset of **FineWeb-Edu `sample-10BT`** as the
primary corpus source. Build an owned, immutable corpus manifest with nested
1B/10B token views. Do not use `naime-corpus-v1` as the canonical source of
truth; retain it only as a compatibility and later bilingual-transfer corpus.

The canonical asset is normalized raw text plus document identity and split
membership. Tokenized shards are derived artifacts. This keeps future
tokenizers and model families comparable without changing the underlying
documents.

## Candidate assessment

| Candidate | Strength | Blocking weakness | Role |
|---|---|---|---|
| FineWeb-Edu `sample-10BT` | 10B-token official sample, document text and IDs, quality score, reproducible processing code, ODC-By | English-only, educational selection bias, no supplied validation/test split, deduplication is not global across crawls | Primary |
| FineWeb `sample-10BT` | Broader English web distribution and same strong metadata/tooling | Lower average quality; same split and cross-crawl duplication work remains | Domain-transfer control |
| DCLM-Baseline 1.0 | Strong 4T-token training corpus, documented model filtering and Bloom-filter deduplication, CC-BY-4.0 | Much larger operational surface and no equally convenient official 10B starter sample | Quality cross-check |
| Dolma v1.6 sample | Diverse sources, mature open tooling, documented 10B sample | Mixture composition and source-specific terms make attribution and licensing more complex | Later heterogeneity test |
| `naime-corpus-v1` | 29B-class pretokenized bilingual/domain mixture, locally accessible | No raw text or document ID, no held-out split, weak source identity, no tokenizer checksum/revision or deduplication evidence | Compatibility only |

## Audit of `naime-corpus-v1`

Repository revision inspected: `5f4b44b106af16a7ad020ee043d98fdc23438c48`.
The public card declares Qwen3-8B tokenization, 4096-token rows, 1,736 Parquet
files, about 28.1B tokens and 38,186,704 documents.

Five locally cached files were sampled across shard indices 0, 434, 868, 1302
and 1735:

- each row had exactly 4,096 tokens;
- four files had 4,096 rows and the final file had 3,765 rows;
- every sampled file contained only one `source_file`, such as
  `deduped_0028.jsonl`;
- `source_file` identifies an upstream batch, not a source document;
- `chunk_idx` is a batch-global chunk counter, not a recoverable document
  boundary;
- only a `train` split is published.

The card's document count cannot describe the published rows: 38.19M rows of
4,096 tokens would imply roughly 156B tokens. Conversely, a regular 1,736-file
layout near 4,096 rows per file implies roughly 29B tokens. The likely
explanation is that the card reports pre-chunk source documents while the
published schema no longer preserves their identity. That interpretation
cannot be verified from the artifact itself.

This prevents strict held-out splitting, cross-split deduplication, sequence
reset at true document boundaries and retokenization. The declared MIT license
also lacks a source-by-source licensing explanation for the aggregated web,
wiki, code and math content. These are provenance defects, not cosmetic card
omissions.

## FineWeb-Edu source facts

Pin the upstream dataset revision rather than following `main`. The revision
inspected here is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

The official `sample-10BT` consists of 14 Parquet files totaling about 28.52 GB
compressed. Remote Parquet metadata for the first file reports 726,000 rows and
the following document-level fields:

- `text`, `id`, `dump`, `url`, `file_path`;
- `language`, `language_score`;
- GPT-2 `token_count`;
- educational `score` and `int_score`.

FineWeb is filtered and deduplicated within each Common Crawl dump, not globally
across all dumps. FineWeb-Edu then applies an educational classifier and keeps
documents scoring at least 3. Therefore the owned subset must still perform
normalized exact-content deduplication across all selected files before split
assignment.

## Owned corpus contract

### Immutable source layer

For every retained document store:

- upstream repository, commit SHA, Parquet path and row index;
- upstream `id`, crawl dump, URL, date/file path when available;
- raw UTF-8 text hash and normalized-text SHA-256;
- language and educational scores;
- assigned split and deterministic selection rank.

Do not commit raw text to Git. Commit the source manifest, document hashes,
aggregate statistics, preprocessing code and generated shard checksums.

### Deterministic split and nested scales

1. Normalize only for deduplication; preserve original text for tokenization.
2. Compute `content_hash = SHA256(normalized_text)`.
3. Remove duplicate content hashes globally.
4. Assign split with a domain-separated hash of `content_hash` before
   tokenizer training: 98% train, 1% validation, 1% test.
5. Assign an independent deterministic rank hash within each split.
6. Materialize nested train budgets of 10M, 100M, 1B and 10B canonical-tokenizer
   tokens by rank. A smaller view must be a strict subset of every larger view.
7. Reserve validation and test documents permanently. Test is never used for
   tuning or topology acceptance.

The official `token_count` is GPT-2-specific and may estimate downloads, but
does not define the final 1B boundary. Final budgets are counted after the
project tokenizer is frozen.

### Tokenizer views

Train a byte-level BPE tokenizer only on the training split and version it
independently from the raw corpus. Start with 16,384 tokens: large enough for
realistic language training and small enough for controlled sparse-output
diagnosis. Preserve a second standard-tokenizer view later for external model
comparability. Both views must reference the same document manifest and split.

Every tokenizer artifact records library version, normalization,
pre-tokenization, special tokens, training-document hashes, vocabulary checksum
and serialized model checksum.

### First release target

`spm-corpus-v1` should contain at minimum:

- 1.00B training tokens under the frozen project tokenizer;
- at least 10M validation and 10M sealed test tokens;
- true document boundaries and BOS/EOS policy;
- exact normalized-content deduplication across all splits;
- source and tokenizer manifests with SHA-256 checksums;
- deterministic regeneration from the pinned FineWeb-Edu revision;
- shard-level token counts, document counts and checksums;
- a contamination report for any downstream benchmark later used.

The 1B release is the minimum durable gate. The same selection algorithm must
extend to 10B without redefining splits or replacing documents in the 1B view.

## Risks and limits

- FineWeb-Edu is not a neutral sample of language. Its educational classifier
  intentionally changes the distribution; use FineWeb or Dolma later as a
  transfer condition.
- ODC-By covers the database release, while underlying web content may carry
  separate rights. Preserve attribution and removal provenance.
- Dataset-level deduplication does not prove benchmark decontamination. Perform
  benchmark-specific n-gram or exact-match checks before downstream evaluation.
- A 1B corpus enables attributable training experiments but does not by itself
  make a 64M model converged or establish architecture superiority.

## Sources

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FineWeb dataset card and processing description](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FineWeb paper](https://openreview.net/forum?id=n6SCkn2QaG)
- [DataTrove FineWeb processing code](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py)
- [DCLM-Baseline dataset card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
- [DCLM paper](https://arxiv.org/abs/2406.11794)
- [Dolma dataset card](https://huggingface.co/datasets/allenai/dolma)
- [Dolma paper and datasheet](https://arxiv.org/abs/2402.00159)
- [`naime-corpus-v1` dataset card](https://huggingface.co/datasets/Leonharper/naime-corpus-v1)

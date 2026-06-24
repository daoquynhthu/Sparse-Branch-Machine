# Stable API contract

> **Document role:** This is the authoritative public C/Python ABI and ownership specification. It does not define research priorities or experiment interpretation.


## C ABI

The public ABI is declared in `include/sbm/api.h`.  It exposes opaque handles and
plain C types only; C++ classes and STL containers do not cross the boundary.
The current ABI major version is 4.

Ownership rules:

- `sbm_config_create` / `sbm_config_destroy` own configuration handles;
- dataset constructors or `sbm_dataset_load` / `sbm_dataset_destroy` own datasets;
- `sbm_run_experiment_json` and `sbm_config_get_json` return strings released by
  `sbm_string_free`;
- `sbm_parameter_schema_json` returns static read-only storage;
- `sbm_last_error` is thread-local until the next API call on that thread.

## Dataset kinds

A dataset handle can contain either:

```c
SBM_DATASET_VECTOR_REGRESSION
SBM_DATASET_TOKEN_CROSS_ENTROPY
```

Query it with:

```c
uint32_t sbm_dataset_kind_of(const sbm_dataset_handle* dataset);
```

Vector generation is retained:

```c
sbm_dataset_handle* sbm_dataset_generate(...);
```

Mathematical token generation uses:

```c
sbm_dataset_handle* sbm_token_dataset_generate_math(
    size_t sequence_count,
    size_t sequence_length,
    uint32_t vocab_size,
    uint64_t seed,
    float temperature,
    float interaction_strength);
```

Externally tokenized IDs use:

```c
sbm_dataset_handle* sbm_token_dataset_from_ids(
    const uint32_t* tokens,
    size_t token_count,
    uint32_t vocab_size,
    const uint64_t* sequence_offsets,
    size_t offset_count);
```

The constructor copies both arrays.  Callers may free their buffers immediately
after it returns.  Offsets must begin at zero, end at `token_count`, and describe
non-empty independent sequences.  Passing a null offset pointer with zero count
treats the entire stream as one sequence.

`sbm_run_experiment_json` dispatches by dataset kind.  A token dataset returns
cross-entropy/perplexity metrics; a vector dataset returns regression metrics.

## Runtime parameter registry

`sbm_parameter_schema_json()` is the authoritative discovery source.  Each entry
contains:

- name, type and default;
- tunability and automatic-search policy;
- optional range and sampling scale;
- dataset-bound status;
- applicable tasks (`token-ce`, `vector`, or both);
- semantic description.

Token-specific runtime parameters include:

- `classification_learning_rate`;
- `classification_mature_learning_rate`;
- `label_smoothing`;
- `softmax_temperature`;
- `logit_decay`.

## Python binding

```python
from sbm_runtime import Runtime

runtime = Runtime("build/libsbm_api.so")

with runtime.generate_math_token_dataset(
    sequence_count=64,
    sequence_length=2048,
    vocab_size=32,
    seed=7,
    temperature=0.8,
    interaction_strength=0.2,
) as dataset:
    with runtime.config() as config:
        result = dataset.run(config, warmup=80_000)
        print(result["eval_cross_entropy"])
```

External tokenizer output:

```python
with runtime.token_dataset_from_ids(
    tokens=token_ids,
    vocab_size=tokenizer_vocab_size,
    sequence_offsets=document_offsets,
) as dataset:
    with runtime.config() as config:
        result = dataset.run(config, warmup=train_examples)
```

The Python layer does not implement the model, softmax, metrics or task logic; it
only manages C handles and JSON conversion.

## Adaptive address-program results

Experiment JSON now includes `learned_address_programs`, represented as arrays
of history offsets, for example `[[1], [1,2], [1,4]]`. `topology_events` now
contains a `lags` array rather than one scalar lag. The compatibility field
`learned_address_lags` contains only surviving singleton programs.

The runtime parameter `topology_max_arity` is exposed through the same schema
and currently accepts 1 or 2. Changing it does not require recompilation.


## Memory-mapped token shards

ABI v4 now includes additive `sbm_token_shard_*` functions. A shard handle owns
a read-only file mapping and must be released with `sbm_token_shard_destroy`.
`sbm_token_shard_open` can verify the complete payload hash without copying the
payload. The format stores fixed little-endian token IDs, sequence offsets and a
versioned 128-byte header.

`sbm_token_shard_cursor` records shard index, sequence index and token offset.
`sbm_token_shard_next` returns only within-sequence next-token pairs: `1` means
an example was produced, `0` means end of shard and `-1` means an API error.
Copying the cursor and reopening the same shard reproduces the identical next
example.

Python exposes the same ownership through `Runtime.open_token_shard()` and the
`TokenShard` context manager. `Dataset.write_token_shard()` is intended for
small in-memory fixtures. Large corpora are written incrementally by the data
conversion utility rather than materialized as `TokenDataset`.

`sbm_run_token_shard_experiment_json` and `TokenShard.run()` execute the existing
C++ learner directly over mapped spans; Python does not iterate tokens. This is
a single-shard smoke/diagnostic interface: its `warmup` count splits examples
inside that shard and therefore is not a substitute for manifest-defined train
and validation splits.

Multi-shard training, manifest-owned iteration and model checkpoint state are
available through `sbm_token_corpus_*` and `Runtime.open_token_corpus()`. Corpus
handles own their mappings. All train shards are consumed first; the model then
freezes once and evaluates all validation shards. Sequence state resets at every
stored block and shard boundary. The Python manifest loader rejects paths that
escape the manifest directory.

Model checkpoint serialization is not implemented yet. A shard cursor proves
exact data replay but is not a full training checkpoint, and a corpus run cannot
yet resume in the middle of a shard without replaying model updates.

## Token baseline metrics

Synchronous corpus evaluation computes unigram and sparse conditional baseline
NLL by querying only the observed target probability. It does not scan or rank
the full vocabulary. Consequently, baseline cross-entropy, bits/token,
perplexity and mean target probability are defined, while baseline top-1 and
top-5 fields are JSON `null`.

`interpolated_multiscale_baseline_eval_*` is the arithmetic mean of the target
probabilities from the lag-1 pair, lag-2 and lag-4 conditional tables. Each
table backs off through the current-token and unigram distributions. The legacy
`multiscale_baseline_eval_*` key is currently an alias for compatibility; it no
longer denotes the old vocabulary-wide geometric product. Results separately
report `elapsed_seconds` for model work and `baseline_elapsed_seconds` for
baseline construction and evaluation.

The Python hardware scheduler is tooling above the stable C ABI. Resource
fractions apply to the current hardware snapshot, not theoretical installed
capacity. Worker processes remain isolated and do not share model or runtime
handles, so scheduling does not change a run's learning semantics.

## Global hierarchical output prior

Token result JSON reports `output_tree_seed`, `global_output_prior_bytes` and
`global_output_prior_updates`. Runtime parameter `output_tree_seed` controls the
fixed implicit class decomposition independently of model `seed`; comparison
runs must keep it constant.

The prior stores corpus-wide hierarchical branch counts. It is updated for
training examples only and is read-only during strict frozen evaluation. Its
storage is included in `estimated_bytes`, while `output_structure_bytes`
continues to describe only the constant-storage implicit tree and path scratch.

Sparse-output result diagnostics additionally report:

- `sparse_output_insertions` and `sparse_output_evictions`;
- `sparse_output_probable_reconstructions` from a bounded per-node fingerprint;
- `sparse_output_saturated_nodes` and `max_sparse_decision_visits`;
- `sparse_output_admission_rejections` and
  `sparse_output_admission_promotions` from the full-node evidence gate;
- `max_responsibility_mass_error` for cross-channel normalization.

Probable reconstructions are deliberately approximate because exact unbounded
eviction history would violate the model's storage contract.

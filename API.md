# Stable API contract

## C ABI

The public ABI is declared in `include/sbm/api.h`.  It exposes opaque handles and
plain C types only; C++ classes and STL containers do not cross the boundary.
The current feature version is 2.

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

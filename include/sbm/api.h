#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(SBM_API_BUILD)
    #define SBM_API __declspec(dllexport)
  #else
    #define SBM_API __declspec(dllimport)
  #endif
#else
  #define SBM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sbm_config_handle sbm_config_handle;
typedef struct sbm_dataset_handle sbm_dataset_handle;
typedef struct sbm_machine_handle sbm_machine_handle;
typedef struct sbm_token_shard_handle sbm_token_shard_handle;
typedef struct sbm_token_corpus_handle sbm_token_corpus_handle;

typedef struct sbm_token_shard_cursor {
    uint64_t shard_index;
    uint64_t sequence_index;
    uint64_t token_offset;
} sbm_token_shard_cursor;

typedef enum sbm_dataset_kind {
    SBM_DATASET_VECTOR_REGRESSION = 1,
    SBM_DATASET_TOKEN_CROSS_ENTROPY = 2
} sbm_dataset_kind;

typedef struct sbm_step_stats {
    float cross_entropy;
    float target_probability;
    uint32_t active_nodes;
    uint32_t candidates_examined;
    uint32_t live_nodes;
    uint32_t predicted_token;
    int top1_correct;
    int top5_correct;
    int ranking_available;
    uint8_t channel_credit_count;
    float channel_credit[8];
    float channel_responsibility_mass[8];
    uint32_t channel_dependency[8];
    uint8_t channel_parent_channel[8];
    uint8_t channel_dependency_channel[8];
    float channel_caller_removed_credit[8];
    float channel_dependency_retained_credit[8];
    float channel_dependency_removed_credit[8];
    uint8_t channel_binding_kind[8];
    uint8_t channel_binding_matched[8];
    uint32_t channel_binding_current_token[8];
    uint32_t channel_binding_matched_token[8];
    uint32_t channel_binding_successor[8];
    uint32_t channel_binding_distance[8];
    uint32_t channel_binding_pattern_span[8];
    uint64_t channel_binding_key[8];
    uint64_t channel_dependency_binding_key[8];
    uint64_t channel_call_key[8];
    float channel_description_cost[8];
    float channel_execution_cost[8];
    int channel_subset_available;
    float seed_only_cross_entropy;
    float active_only_cross_entropy;
    float content_only_cross_entropy;
    float tuple_only_cross_entropy;
} sbm_step_stats;

/* Stable C ABI version. Increment only for incompatible changes. */
SBM_API uint32_t sbm_api_version(void);
SBM_API const char* sbm_api_version_string(void);

/* Error text is thread-local and remains valid until the next API call. */
SBM_API const char* sbm_last_error(void);

/* Parameter discovery and runtime configuration. */
SBM_API const char* sbm_parameter_schema_json(void);
SBM_API sbm_config_handle* sbm_config_create(void);
SBM_API sbm_config_handle* sbm_config_clone(const sbm_config_handle* source);
SBM_API void sbm_config_destroy(sbm_config_handle* config);
SBM_API int sbm_config_set(sbm_config_handle* config, const char* name, const char* value);
SBM_API char* sbm_config_get_json(const sbm_config_handle* config);

/* Stateful token machine API used by resumable corpus runners. */
SBM_API sbm_machine_handle* sbm_machine_create(const sbm_config_handle* config);
SBM_API sbm_machine_handle* sbm_machine_load_checkpoint(const char* path);
SBM_API int sbm_machine_save_checkpoint(
    const sbm_machine_handle* machine,
    const char* path);
SBM_API void sbm_machine_destroy(sbm_machine_handle* machine);
SBM_API int sbm_machine_step_token(
    sbm_machine_handle* machine,
    uint32_t input_token,
    uint32_t target_token,
    int learn,
    sbm_step_stats* stats);
SBM_API void sbm_machine_reset_sequence(sbm_machine_handle* machine);
SBM_API void sbm_machine_freeze_topology(sbm_machine_handle* machine);
SBM_API char* sbm_machine_diagnostics_json(const sbm_machine_handle* machine);
SBM_API char* sbm_machine_summary_json(const sbm_machine_handle* machine);

/* Dataset ownership is explicit. Dataset objects are immutable after creation. */
SBM_API sbm_dataset_handle* sbm_dataset_generate(
    size_t length,
    uint32_t alphabet,
    uint32_t vector_dim,
    uint32_t hidden_dim,
    uint64_t seed,
    float noise_std);
SBM_API sbm_dataset_handle* sbm_token_dataset_generate_math(
    size_t sequence_count,
    size_t sequence_length,
    uint32_t vocab_size,
    uint64_t seed,
    float temperature,
    float interaction_strength);
SBM_API sbm_dataset_handle* sbm_token_dataset_from_ids(
    const uint32_t* tokens,
    size_t token_count,
    uint32_t vocab_size,
    const uint64_t* sequence_offsets,
    size_t offset_count);
SBM_API sbm_dataset_handle* sbm_dataset_load(const char* path);
SBM_API int sbm_dataset_save(const sbm_dataset_handle* dataset, const char* path);
SBM_API void sbm_dataset_destroy(sbm_dataset_handle* dataset);
SBM_API uint32_t sbm_dataset_kind_of(const sbm_dataset_handle* dataset);
SBM_API uint64_t sbm_dataset_hash(const sbm_dataset_handle* dataset);
SBM_API size_t sbm_dataset_length(const sbm_dataset_handle* dataset);
SBM_API uint32_t sbm_dataset_vector_dim(const sbm_dataset_handle* dataset);
SBM_API uint32_t sbm_dataset_alphabet(const sbm_dataset_handle* dataset);
SBM_API uint32_t sbm_dataset_vocab_size(const sbm_dataset_handle* dataset);
SBM_API size_t sbm_dataset_sequence_count(const sbm_dataset_handle* dataset);
SBM_API size_t sbm_dataset_example_count(const sbm_dataset_handle* dataset);

/* Versioned memory-mapped token shards. These functions are additive to ABI v4. */
SBM_API int sbm_token_shard_write(
    const sbm_dataset_handle* dataset,
    const char* path);
SBM_API sbm_token_shard_handle* sbm_token_shard_open(
    const char* path,
    uint64_t shard_index,
    int verify_payload);
SBM_API void sbm_token_shard_destroy(sbm_token_shard_handle* shard);
SBM_API uint32_t sbm_token_shard_vocab_size(const sbm_token_shard_handle* shard);
SBM_API uint64_t sbm_token_shard_token_count(const sbm_token_shard_handle* shard);
SBM_API uint64_t sbm_token_shard_sequence_count(const sbm_token_shard_handle* shard);
SBM_API uint64_t sbm_token_shard_dataset_hash(const sbm_token_shard_handle* shard);
/* Returns 1 for an example, 0 at end of shard and -1 on error. */
SBM_API int sbm_token_shard_next(
    const sbm_token_shard_handle* shard,
    sbm_token_shard_cursor* cursor,
    uint32_t* input,
    uint32_t* target);
SBM_API char* sbm_run_token_shard_experiment_json(
    const sbm_token_shard_handle* shard,
    size_t warmup,
    const sbm_config_handle* config,
    int strict_freeze,
    size_t prefill,
    size_t prune_interval,
    size_t merge_interval);
SBM_API sbm_token_corpus_handle* sbm_token_corpus_create(void);
SBM_API void sbm_token_corpus_destroy(sbm_token_corpus_handle* corpus);
/* split_kind: 0=train, 1=evaluation. */
SBM_API int sbm_token_corpus_add_shard(
    sbm_token_corpus_handle* corpus,
    const char* path,
    uint32_t split_kind,
    uint64_t shard_index,
    int verify_payload);
SBM_API char* sbm_run_token_corpus_experiment_json(
    const sbm_token_corpus_handle* corpus,
    const sbm_config_handle* config,
    int strict_freeze,
    size_t prefill,
    size_t prune_interval,
    size_t merge_interval);

SBM_API char* sbm_run_token_corpus_experiment_limited_json(
    const sbm_token_corpus_handle* corpus,
    size_t max_train_examples,
    size_t max_eval_examples,
    const sbm_config_handle* config,
    int strict_freeze,
    size_t prefill,
    size_t prune_interval,
    size_t merge_interval);

/* Returns a malloc-compatible UTF-8 JSON string, or NULL on failure. */
SBM_API char* sbm_run_experiment_json(
    const sbm_dataset_handle* dataset,
    size_t warmup,
    const sbm_config_handle* config,
    int strict_freeze,
    size_t prefill,
    size_t prune_interval,
    size_t merge_interval);

SBM_API void sbm_string_free(char* value);

#ifdef __cplusplus
}
#endif

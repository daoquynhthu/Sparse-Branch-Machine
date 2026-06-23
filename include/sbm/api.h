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

typedef enum sbm_dataset_kind {
    SBM_DATASET_VECTOR_REGRESSION = 1,
    SBM_DATASET_TOKEN_CROSS_ENTROPY = 2
} sbm_dataset_kind;

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

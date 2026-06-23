#include "sbm/api.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string_view>

int main() {
    assert(sbm_api_version() == 4U);
    assert(std::strlen(sbm_api_version_string()) > 0U);

    const std::string_view schema(sbm_parameter_schema_json());
    assert(schema.find("exact_region_mass") != std::string_view::npos);
    assert(schema.find("classification_learning_rate") != std::string_view::npos);
    assert(schema.find("max_sparse_decisions_per_node") != std::string_view::npos);
    assert(schema.find("search_default") != std::string_view::npos);

    sbm_config_handle* config = sbm_config_create();
    assert(config != nullptr);
    assert(sbm_config_set(config, "exact_region_mass", "0.91") == 0);
    assert(sbm_config_set(config, "address_lags", "1,2,4") == 0);
    assert(sbm_config_set(config, "label_smoothing", "0.02") == 0);
    assert(sbm_config_set(config, "does_not_exist", "1") != 0);
    assert(std::strlen(sbm_last_error()) > 0U);

    char* config_json = sbm_config_get_json(config);
    assert(config_json != nullptr);
    assert(std::string_view(config_json).find("classification_learning_rate") !=
           std::string_view::npos);
    sbm_string_free(config_json);

    sbm_dataset_handle* vector_dataset = sbm_dataset_generate(8000, 32, 16, 20, 9, 0.035F);
    assert(vector_dataset != nullptr);
    assert(sbm_dataset_kind_of(vector_dataset) == SBM_DATASET_VECTOR_REGRESSION);
    assert(sbm_dataset_length(vector_dataset) == 8000U);
    assert(sbm_dataset_vector_dim(vector_dataset) == 16U);
    assert(sbm_dataset_alphabet(vector_dataset) == 32U);

    char* vector_result = sbm_run_experiment_json(vector_dataset, 3000, config, 1, 0, 0, 0);
    assert(vector_result != nullptr);
    assert(std::string_view(vector_result).find("\"eval_r2\"") != std::string_view::npos);
    sbm_string_free(vector_result);
    sbm_dataset_destroy(vector_dataset);

    sbm_dataset_handle* token_dataset = sbm_token_dataset_generate_math(
        8, 512, 16, 13, 0.8F, 0.2F);
    assert(token_dataset != nullptr);
    assert(sbm_dataset_kind_of(token_dataset) == SBM_DATASET_TOKEN_CROSS_ENTROPY);
    assert(sbm_dataset_vocab_size(token_dataset) == 16U);
    assert(sbm_dataset_sequence_count(token_dataset) == 8U);
    assert(sbm_dataset_example_count(token_dataset) == 4088U);

    char* token_result = sbm_run_experiment_json(token_dataset, 2500, config, 1, 0, 0, 0);
    assert(token_result != nullptr);
    const std::string_view token_view(token_result);
    assert(token_view.find("\"eval_cross_entropy\"") != std::string_view::npos);
    assert(token_view.find("\"objective\": \"token_cross_entropy\"") !=
           std::string_view::npos);
    sbm_string_free(token_result);
    sbm_dataset_destroy(token_dataset);

    const uint32_t ids[]{1, 2, 3, 1, 2, 4};
    const uint64_t offsets[]{0, 3, 6};
    sbm_dataset_handle* external = sbm_token_dataset_from_ids(ids, 6, 8, offsets, 3);
    assert(external != nullptr);
    assert(sbm_dataset_sequence_count(external) == 2U);
    assert(sbm_dataset_example_count(external) == 4U);
    const char* shard_path = "sbm_c_api_shard.sbt";
    std::remove(shard_path);
    assert(sbm_token_shard_write(external, shard_path) == 0);
    sbm_token_shard_handle* shard = sbm_token_shard_open(shard_path, 3U, 1);
    assert(shard != nullptr);
    assert(sbm_token_shard_vocab_size(shard) == 8U);
    assert(sbm_token_shard_token_count(shard) == 6U);
    assert(sbm_token_shard_sequence_count(shard) == 2U);
    sbm_token_shard_cursor cursor{3U, 0U, 0U};
    uint32_t input = 0U;
    uint32_t target = 0U;
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 1U && target == 2U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 2U && target == 3U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 1U && target == 2U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 2U && target == 4U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 0);
    char* shard_result = sbm_run_token_shard_experiment_json(
        shard, 2U, config, 1, 0U, 0U, 0U);
    assert(shard_result != nullptr);
    assert(std::string_view(shard_result).find(
        "\"task\": \"real_corpus_next_token_cross_entropy\"") !=
        std::string_view::npos);
    sbm_string_free(shard_result);
    sbm_token_shard_destroy(shard);

    sbm_token_corpus_handle* corpus = sbm_token_corpus_create();
    assert(corpus != nullptr);
    assert(sbm_token_corpus_add_shard(corpus, shard_path, 0U, 0U, 1) == 0);
    assert(sbm_token_corpus_add_shard(corpus, shard_path, 1U, 1U, 1) == 0);
    char* corpus_result = sbm_run_token_corpus_experiment_json(
        corpus, config, 1, 0U, 0U, 0U);
    assert(corpus_result != nullptr);
    assert(std::string_view(corpus_result).find("\"train_examples\": 4") !=
           std::string_view::npos);
    assert(std::string_view(corpus_result).find("\"eval_examples\": 4") !=
           std::string_view::npos);
    sbm_string_free(corpus_result);
    sbm_token_corpus_destroy(corpus);
    std::remove(shard_path);
    sbm_dataset_destroy(external);

    sbm_config_destroy(config);
    std::cout << "C API tests passed\n";
}

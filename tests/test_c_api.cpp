#include "sbm/api.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <string_view>

int main() {
    assert(sbm_api_version() == 1U);
    assert(std::strlen(sbm_api_version_string()) > 0U);

    const std::string_view schema(sbm_parameter_schema_json());
    assert(schema.find("exact_region_mass") != std::string_view::npos);
    assert(schema.find("search_default") != std::string_view::npos);

    sbm_config_handle* config = sbm_config_create();
    assert(config != nullptr);
    assert(sbm_config_set(config, "exact_region_mass", "0.91") == 0);
    assert(sbm_config_set(config, "address_lags", "1,2,4") == 0);
    assert(sbm_config_set(config, "does_not_exist", "1") != 0);
    assert(std::strlen(sbm_last_error()) > 0U);

    char* config_json = sbm_config_get_json(config);
    assert(config_json != nullptr);
    assert(std::string_view(config_json).find("0.910") != std::string_view::npos);
    sbm_string_free(config_json);

    sbm_dataset_handle* dataset = sbm_dataset_generate(8000, 32, 16, 20, 9, 0.035F);
    assert(dataset != nullptr);
    assert(sbm_dataset_length(dataset) == 8000U);
    assert(sbm_dataset_vector_dim(dataset) == 16U);
    assert(sbm_dataset_alphabet(dataset) == 32U);
    assert(sbm_dataset_hash(dataset) != 0U);

    char* result = sbm_run_experiment_json(dataset, 3000, config, 1, 0, 0, 0);
    assert(result != nullptr);
    const std::string_view result_view(result);
    assert(result_view.find("\"eval_r2\"") != std::string_view::npos);
    assert(result_view.find("\"strict_freeze\": true") != std::string_view::npos);
    sbm_string_free(result);

    sbm_dataset_destroy(dataset);
    sbm_config_destroy(config);
    std::cout << "C API tests passed\n";
}

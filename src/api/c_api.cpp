#include "sbm/api.h"

#include "sbm/dataset.hpp"
#include "sbm/experiment.hpp"
#include "sbm/types.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

struct sbm_config_handle {
    sbm::Config value;
};

struct sbm_dataset_handle {
    sbm::VectorDataset value;
};

namespace {
thread_local std::string g_last_error;

void clear_error() noexcept { g_last_error.clear(); }

void set_error(std::string message) noexcept {
    try {
        g_last_error = std::move(message);
    } catch (...) {
        g_last_error = "unknown API error";
    }
}

template <class Function>
auto guarded(Function&& function) noexcept -> decltype(function()) {
    using Return = decltype(function());
    clear_error();
    try {
        return function();
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("unknown C++ exception");
    }
    if constexpr (std::is_pointer_v<Return>) return nullptr;
    else return Return{};
}

char* duplicate_string(const std::string& value) {
    auto* result = static_cast<char*>(std::malloc(value.size() + 1U));
    if (result == nullptr) throw std::bad_alloc();
    std::memcpy(result, value.c_str(), value.size() + 1U);
    return result;
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(ch) << std::dec;
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

template <class T>
T parse_integer(std::string_view text, const char* name) {
    T value{};
    const auto* begin = text.data();
    const auto* end = text.data() + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end) {
        throw std::invalid_argument(std::string("invalid integer for ") + name);
    }
    return value;
}

float parse_float(std::string_view text, const char* name) {
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(owned.c_str(), &end);
    if (errno != 0 || end != owned.c_str() + owned.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid float for ") + name);
    }
    return value;
}

double parse_double(std::string_view text, const char* name) {
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(owned.c_str(), &end);
    if (errno != 0 || end != owned.c_str() + owned.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid double for ") + name);
    }
    return value;
}

bool parse_bool(std::string_view text, const char* name) {
    if (text == "1" || text == "true" || text == "TRUE" || text == "yes") return true;
    if (text == "0" || text == "false" || text == "FALSE" || text == "no") return false;
    throw std::invalid_argument(std::string("invalid boolean for ") + name);
}

std::array<std::uint32_t, sbm::kAddressChannelCount> parse_lags(std::string_view text) {
    std::array<std::uint32_t, sbm::kAddressChannelCount> result{};
    std::size_t offset = 0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto comma = text.find(',', offset);
        const auto part = text.substr(offset, comma == std::string_view::npos
                                              ? std::string_view::npos
                                              : comma - offset);
        if (part.empty()) throw std::invalid_argument("address_lags contains an empty item");
        result[i] = parse_integer<std::uint32_t>(part, "address_lags");
        if (comma == std::string_view::npos) {
            if (i + 1U != result.size()) {
                throw std::invalid_argument("address_lags must contain exactly three values");
            }
            offset = text.size();
        } else {
            offset = comma + 1U;
        }
    }
    if (offset != text.size()) {
        throw std::invalid_argument("address_lags must contain exactly three values");
    }
    return result;
}

struct ParameterDescriptor {
    const char* name;
    const char* type;
    const char* default_value;
    const char* minimum;
    const char* maximum;
    const char* scale;
    bool tunable;
    bool search_default;
    bool dataset_bound;
    const char* description;
};

// This table is the single discovery source for the CLI and Python tuner.
constexpr ParameterDescriptor kParameters[] = {
    {"token_alphabet", "uint32", "64", "1", "1048576", "linear", false, false, true, "Dataset token alphabet; overwritten by the dataset at run time."},
    {"vector_dim", "uint32", "16", "1", "4096", "linear", false, false, true, "Target vector dimension; overwritten by the dataset at run time."},
    {"context_width", "uint32", "12", "2", "256", "linear", true, false, false, "Maximum retained token history."},
    {"bucket_bits", "uint32", "12", "6", "18", "linear", true, false, false, "Bits used by each address-channel bucket index."},
    {"beam_width", "uint32", "6", "3", "16", "linear", true, false, false, "Maximum active nodes per step."},
    {"bucket_scan_limit", "uint32", "32", "4", "128", "log", true, false, false, "Maximum nodes examined from an address bucket."},
    {"edge_scan_limit", "uint32", "8", "1", "32", "linear", true, false, false, "Maximum outgoing edges examined per source node."},
    {"max_edges_per_node", "uint32", "32", "4", "128", "log", true, false, false, "Maximum stored sparse control edges per node."},
    {"edge_reinforce_width", "uint32", "2", "1", "8", "linear", true, false, false, "Number of route positions eligible for edge reinforcement."},
    {"max_specializations_per_bucket", "uint32", "4", "1", "16", "linear", true, false, false, "Maximum conflict-driven specializations per address bucket."},
    {"split_min_visits", "uint32", "48", "8", "256", "log", true, false, false, "Minimum address-local visits before specialization."},
    {"split_cooldown", "uint32", "128", "8", "2048", "log", true, false, false, "Minimum steps between bucket specializations."},
    {"split_context_similarity", "double", "0.78", "0.40", "0.99", "linear", true, false, false, "Maximum context mismatch tolerated by specialization."},
    {"split_loss_threshold", "float", "0.72", "0.10", "2.00", "log", true, false, false, "Address-local normalized-loss threshold for specialization."},
    {"allow_growth_when_frozen", "bool", "false", "", "", "categorical", false, false, false, "Allow structural growth during evaluation."},
    {"trace_horizon", "uint32", "8", "1", "64", "log", true, false, false, "Number of recent routes retained for delayed credit."},
    {"trace_decay", "float", "0.80", "0.30", "0.999", "linear", true, false, false, "Temporal decay for delayed node credit."},
    {"edge_decay", "float", "0.999", "0.95", "0.99999", "linear", true, true, false, "Multiplicative sparse-edge decay."},
    {"edge_learning_rate", "float", "0.08", "0.002", "0.30", "log", true, true, false, "Positive sparse-edge update rate."},
    {"edge_score_weight", "float", "0.32", "0.0", "0.80", "linear", true, true, false, "Routing score weight assigned to learned control edges."},
    {"edge_min_contribution", "float", "0.004", "0.0", "0.05", "linear", true, true, false, "Minimum counterfactual contribution for edge reinforcement."},
    {"responsibility_temperature", "float", "3.0", "0.25", "12.0", "log", true, true, false, "Soft responsibility temperature within a route."},
    {"exact_region_mass", "float", "0.88", "0.50", "1.0", "linear", true, true, false, "Prediction mass reserved for exact address regions."},
    {"min_update_responsibility", "float", "0.01", "0.0", "0.20", "linear", true, true, false, "Minimum local responsibility required for a parameter update."},
    {"address_lags", "uint32[3]", "1,2,4", "", "", "categorical", false, false, false, "Three positive, unique temporal address lags."},
    {"residual_channel_gain", "float", "1.0", "0.25", "2.0", "linear", true, true, false, "Gain applied to additive residual channels."},
    {"residual_learning_rate", "float", "0.10", "0.005", "0.50", "log", true, false, false, "Legacy residual update cap used by non-mean paths."},
    {"residual_mature_learning_rate", "float", "0.03", "0.001", "0.20", "log", true, false, false, "Legacy mature residual update cap."},
    {"residual_recency_pseudocount", "float", "0.75", "0.0", "8.0", "linear", true, true, false, "Recency bias for sequential residual means."},
    {"warm_visits", "uint32", "12", "2", "128", "log", true, false, false, "Visits needed to leave the cold lifecycle phase."},
    {"mature_visits", "uint32", "96", "16", "1024", "log", true, false, false, "Visits needed to enter the mature lifecycle phase."},
    {"dormant_utility", "float", "-0.30", "-2.0", "0.0", "linear", true, false, false, "Utility threshold for the dormant lifecycle phase."},
    {"node_learning_rate", "float", "0.12", "0.005", "0.50", "log", true, false, false, "Legacy anchor update cap used by non-mean paths."},
    {"mature_learning_rate", "float", "0.035", "0.001", "0.20", "log", true, false, false, "Legacy mature anchor update cap."},
    {"merge_similarity", "float", "0.95", "0.70", "0.999", "linear", true, false, false, "Minimum address similarity for node merge."},
    {"merge_vector_distance", "float", "0.08", "0.005", "0.50", "log", true, false, false, "Maximum output-vector distance for node merge."},
    {"seed", "uint64", "7", "0", "18446744073709551615", "linear", false, false, false, "Model random seed."},
};

bool set_parameter(sbm::Config& config, std::string_view name, std::string_view value) {
#define SBM_SET_UINT(field) if (name == #field) { config.field = parse_integer<std::uint32_t>(value, #field); return true; }
#define SBM_SET_U64(field) if (name == #field) { config.field = parse_integer<std::uint64_t>(value, #field); return true; }
#define SBM_SET_FLOAT(field) if (name == #field) { config.field = parse_float(value, #field); return true; }
#define SBM_SET_DOUBLE(field) if (name == #field) { config.field = parse_double(value, #field); return true; }
#define SBM_SET_BOOL(field) if (name == #field) { config.field = parse_bool(value, #field); return true; }
    SBM_SET_UINT(token_alphabet)
    SBM_SET_UINT(vector_dim)
    SBM_SET_UINT(context_width)
    SBM_SET_UINT(bucket_bits)
    SBM_SET_UINT(beam_width)
    SBM_SET_UINT(bucket_scan_limit)
    SBM_SET_UINT(edge_scan_limit)
    SBM_SET_UINT(max_edges_per_node)
    SBM_SET_UINT(edge_reinforce_width)
    SBM_SET_UINT(max_specializations_per_bucket)
    SBM_SET_UINT(split_min_visits)
    SBM_SET_UINT(split_cooldown)
    SBM_SET_DOUBLE(split_context_similarity)
    SBM_SET_FLOAT(split_loss_threshold)
    SBM_SET_BOOL(allow_growth_when_frozen)
    SBM_SET_UINT(trace_horizon)
    SBM_SET_FLOAT(trace_decay)
    SBM_SET_FLOAT(edge_decay)
    SBM_SET_FLOAT(edge_learning_rate)
    SBM_SET_FLOAT(edge_score_weight)
    SBM_SET_FLOAT(edge_min_contribution)
    SBM_SET_FLOAT(responsibility_temperature)
    SBM_SET_FLOAT(exact_region_mass)
    SBM_SET_FLOAT(min_update_responsibility)
    if (name == "address_lags") { config.address_lags = parse_lags(value); return true; }
    SBM_SET_FLOAT(residual_channel_gain)
    SBM_SET_FLOAT(residual_learning_rate)
    SBM_SET_FLOAT(residual_mature_learning_rate)
    SBM_SET_FLOAT(residual_recency_pseudocount)
    SBM_SET_UINT(warm_visits)
    SBM_SET_UINT(mature_visits)
    SBM_SET_FLOAT(dormant_utility)
    SBM_SET_FLOAT(node_learning_rate)
    SBM_SET_FLOAT(mature_learning_rate)
    SBM_SET_FLOAT(merge_similarity)
    SBM_SET_FLOAT(merge_vector_distance)
    SBM_SET_U64(seed)
#undef SBM_SET_UINT
#undef SBM_SET_U64
#undef SBM_SET_FLOAT
#undef SBM_SET_DOUBLE
#undef SBM_SET_BOOL
    return false;
}

std::string config_json(const sbm::Config& c) {
    std::ostringstream out;
    out << std::setprecision(9) << std::boolalpha << "{\n"
        << "  \"token_alphabet\": " << c.token_alphabet << ",\n"
        << "  \"vector_dim\": " << c.vector_dim << ",\n"
        << "  \"context_width\": " << c.context_width << ",\n"
        << "  \"bucket_bits\": " << c.bucket_bits << ",\n"
        << "  \"beam_width\": " << c.beam_width << ",\n"
        << "  \"bucket_scan_limit\": " << c.bucket_scan_limit << ",\n"
        << "  \"edge_scan_limit\": " << c.edge_scan_limit << ",\n"
        << "  \"max_edges_per_node\": " << c.max_edges_per_node << ",\n"
        << "  \"edge_reinforce_width\": " << c.edge_reinforce_width << ",\n"
        << "  \"max_specializations_per_bucket\": " << c.max_specializations_per_bucket << ",\n"
        << "  \"split_min_visits\": " << c.split_min_visits << ",\n"
        << "  \"split_cooldown\": " << c.split_cooldown << ",\n"
        << "  \"split_context_similarity\": " << c.split_context_similarity << ",\n"
        << "  \"split_loss_threshold\": " << c.split_loss_threshold << ",\n"
        << "  \"allow_growth_when_frozen\": " << c.allow_growth_when_frozen << ",\n"
        << "  \"trace_horizon\": " << c.trace_horizon << ",\n"
        << "  \"trace_decay\": " << c.trace_decay << ",\n"
        << "  \"edge_decay\": " << c.edge_decay << ",\n"
        << "  \"edge_learning_rate\": " << c.edge_learning_rate << ",\n"
        << "  \"edge_score_weight\": " << c.edge_score_weight << ",\n"
        << "  \"edge_min_contribution\": " << c.edge_min_contribution << ",\n"
        << "  \"responsibility_temperature\": " << c.responsibility_temperature << ",\n"
        << "  \"exact_region_mass\": " << c.exact_region_mass << ",\n"
        << "  \"min_update_responsibility\": " << c.min_update_responsibility << ",\n"
        << "  \"address_lags\": [" << c.address_lags[0] << ", " << c.address_lags[1]
        << ", " << c.address_lags[2] << "],\n"
        << "  \"residual_channel_gain\": " << c.residual_channel_gain << ",\n"
        << "  \"residual_learning_rate\": " << c.residual_learning_rate << ",\n"
        << "  \"residual_mature_learning_rate\": " << c.residual_mature_learning_rate << ",\n"
        << "  \"residual_recency_pseudocount\": " << c.residual_recency_pseudocount << ",\n"
        << "  \"warm_visits\": " << c.warm_visits << ",\n"
        << "  \"mature_visits\": " << c.mature_visits << ",\n"
        << "  \"dormant_utility\": " << c.dormant_utility << ",\n"
        << "  \"node_learning_rate\": " << c.node_learning_rate << ",\n"
        << "  \"mature_learning_rate\": " << c.mature_learning_rate << ",\n"
        << "  \"merge_similarity\": " << c.merge_similarity << ",\n"
        << "  \"merge_vector_distance\": " << c.merge_vector_distance << ",\n"
        << "  \"seed\": " << c.seed << "\n"
        << "}\n";
    return out.str();
}

std::string schema_json() {
    std::ostringstream out;
    out << "{\n  \"api_version\": 1,\n  \"parameters\": [\n";
    for (std::size_t i = 0; i < std::size(kParameters); ++i) {
        const auto& p = kParameters[i];
        out << "    {\"name\": \"" << json_escape(p.name)
            << "\", \"type\": \"" << p.type
            << "\", \"default\": \"" << p.default_value
            << "\", \"tunable\": " << (p.tunable ? "true" : "false")
            << ", \"search_default\": " << (p.search_default ? "true" : "false")
            << ", \"dataset_bound\": " << (p.dataset_bound ? "true" : "false")
            << ", \"scale\": \"" << p.scale << "\"";
        if (p.minimum[0] != '\0') out << ", \"minimum\": " << p.minimum;
        if (p.maximum[0] != '\0') out << ", \"maximum\": " << p.maximum;
        out << ", \"description\": \"" << json_escape(p.description) << "\"}";
        if (i + 1U != std::size(kParameters)) out << ',';
        out << '\n';
    }
    out << "  ]\n}\n";
    return out.str();
}

const std::string& static_schema() {
    static const std::string value = schema_json();
    return value;
}

} // namespace

extern "C" {

uint32_t sbm_api_version(void) { return 1U; }
const char* sbm_api_version_string(void) { return "1.0.0"; }
const char* sbm_last_error(void) { return g_last_error.c_str(); }
const char* sbm_parameter_schema_json(void) { return static_schema().c_str(); }

sbm_config_handle* sbm_config_create(void) {
    return guarded([] { return new sbm_config_handle{}; });
}

sbm_config_handle* sbm_config_clone(const sbm_config_handle* source) {
    return guarded([&] {
        if (source == nullptr) throw std::invalid_argument("config is null");
        return new sbm_config_handle{source->value};
    });
}

void sbm_config_destroy(sbm_config_handle* config) { delete config; }

int sbm_config_set(sbm_config_handle* config, const char* name, const char* value) {
    clear_error();
    try {
        if (config == nullptr || name == nullptr || value == nullptr) {
            throw std::invalid_argument("config, name and value must be non-null");
        }
        if (!set_parameter(config->value, name, value)) {
            throw std::invalid_argument(std::string("unknown parameter: ") + name);
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

char* sbm_config_get_json(const sbm_config_handle* config) {
    return guarded([&] {
        if (config == nullptr) throw std::invalid_argument("config is null");
        return duplicate_string(config_json(config->value));
    });
}

sbm_dataset_handle* sbm_dataset_generate(size_t length, uint32_t alphabet,
                                         uint32_t vector_dim, uint32_t hidden_dim,
                                         uint64_t seed, float noise_std) {
    return guarded([&] {
        return new sbm_dataset_handle{
            sbm::generate_vector_process(length, alphabet, vector_dim,
                                         hidden_dim, seed, noise_std)};
    });
}

sbm_dataset_handle* sbm_dataset_load(const char* path) {
    return guarded([&] {
        if (path == nullptr) throw std::invalid_argument("path is null");
        return new sbm_dataset_handle{sbm::load_dataset(path)};
    });
}

int sbm_dataset_save(const sbm_dataset_handle* dataset, const char* path) {
    clear_error();
    try {
        if (dataset == nullptr || path == nullptr) {
            throw std::invalid_argument("dataset and path must be non-null");
        }
        sbm::save_dataset(dataset->value, path);
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

void sbm_dataset_destroy(sbm_dataset_handle* dataset) { delete dataset; }

uint64_t sbm_dataset_hash(const sbm_dataset_handle* dataset) {
    return dataset == nullptr ? 0U : sbm::hash_dataset(dataset->value);
}

size_t sbm_dataset_length(const sbm_dataset_handle* dataset) {
    return dataset == nullptr ? 0U : dataset->value.tokens.size();
}

uint32_t sbm_dataset_vector_dim(const sbm_dataset_handle* dataset) {
    return dataset == nullptr ? 0U : dataset->value.vector_dim;
}

uint32_t sbm_dataset_alphabet(const sbm_dataset_handle* dataset) {
    return dataset == nullptr ? 0U : dataset->value.token_alphabet;
}

char* sbm_run_experiment_json(const sbm_dataset_handle* dataset, size_t warmup,
                              const sbm_config_handle* config, int strict_freeze,
                              size_t prefill, size_t prune_interval,
                              size_t merge_interval) {
    return guarded([&] {
        if (dataset == nullptr || config == nullptr) {
            throw std::invalid_argument("dataset and config must be non-null");
        }
        const auto result = sbm::run_experiment(
            dataset->value, warmup, config->value, strict_freeze != 0,
            prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

void sbm_string_free(char* value) { std::free(value); }

} // extern "C"

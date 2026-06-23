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
#include <variant>
#include <vector>

struct sbm_config_handle {
    sbm::Config value;
};

struct sbm_dataset_handle {
    std::variant<sbm::VectorDataset, sbm::TokenDataset> value;
};

struct sbm_token_shard_handle {
    sbm::MappedTokenShard value;
};

struct sbm_token_corpus_handle {
    std::vector<std::unique_ptr<sbm::MappedTokenShard>> train;
    std::vector<std::unique_ptr<sbm::MappedTokenShard>> eval;
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

std::vector<std::uint32_t> parse_lags(std::string_view text) {
    std::vector<std::uint32_t> result;
    std::size_t offset = 0;
    while (offset <= text.size()) {
        const auto comma = text.find(',', offset);
        const auto part = text.substr(offset, comma == std::string_view::npos
                                              ? std::string_view::npos
                                              : comma - offset);
        if (part.empty()) throw std::invalid_argument("address_lags contains an empty item");
        result.push_back(parse_integer<std::uint32_t>(part, "address_lags"));
        if (result.size() > sbm::kMaxAddressChannels) {
            throw std::invalid_argument("too many address_lags");
        }
        if (comma == std::string_view::npos) break;
        offset = comma + 1U;
    }
    if (result.empty()) throw std::invalid_argument("address_lags must not be empty");
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
    {"address_lags", "uint32[]", "1", "", "", "categorical", false, false, false, "Seed temporal address lags; adaptive topology may add more."},
    {"adaptive_topology", "bool", "true", "", "", "categorical", false, false, false, "Enable predictive-credit topology proposals."},
    {"max_address_channels", "uint32", "6", "1", "8", "linear", true, false, false, "Maximum simultaneous address channels."},
    {"topology_max_lag", "uint32", "16", "2", "256", "log", true, false, false, "Largest temporal lag eligible for proposal."},
    {"topology_max_arity", "uint32", "2", "1", "2", "linear", true, false, false, "Maximum number of history offsets in an address program."},
    {"topology_enable_delta", "bool", "true", "", "", "categorical", false, false, false, "Allow the generic modular-difference address operator to be proposed."},
    {"topology_probe_interval", "uint32", "2048", "128", "16384", "log", true, false, false, "Delay between topology proposals."},
    {"topology_probe_warmup", "uint32", "512", "0", "4096", "linear", true, false, false, "Probe steps ignored before credit collection."},
    {"topology_probe_steps", "uint32", "4096", "512", "32768", "log", true, false, false, "Lifetime of a candidate address channel."},
    {"topology_validation_steps", "uint32", "1024", "128", "8192", "log", true, false, false, "Frozen tail used for out-of-sample topology selection."},
    {"topology_min_observations", "uint32", "512", "64", "8192", "log", true, false, false, "Minimum probe observations before a decision."},
    {"topology_accept_credit", "float", "0.0005", "0.0", "0.05", "linear", true, false, false, "Mean counterfactual NLL gain required to retain a channel."},
    {"topology_credit_decay", "float", "0.995", "0.90", "0.9999", "linear", true, false, false, "EMA decay for address-channel credit."},
    {"topology_prune_patience", "uint32", "32768", "512", "65536", "log", true, false, false, "Mature observations required before channel retirement."},
    {"topology_prune_credit", "float", "-0.01", "-0.05", "0.0", "linear", true, false, false, "Credit threshold for retiring an accepted channel."},
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
    {"merge_vector_distance", "float", "0.08", "0.005", "0.50", "log", true, false, false, "Maximum output-vector/logit distance for node merge."},
    {"classification_learning_rate", "float", "0.35", "0.01", "1.0", "log", true, true, false, "Initial local logit learning rate for token cross-entropy."},
    {"classification_mature_learning_rate", "float", "0.08", "0.002", "0.40", "log", true, true, false, "Local logit learning rate for mature token nodes."},
    {"label_smoothing", "float", "0.01", "0.0", "0.20", "linear", true, true, false, "Label smoothing for mathematical next-token cross-entropy."},
    {"softmax_temperature", "float", "1.0", "0.25", "3.0", "log", true, true, false, "Temperature applied to aggregated token logits."},
    {"logit_decay", "float", "0.0001", "0.0", "0.02", "linear", true, true, false, "Per-update decay applied to local token logits."},
    {"sparse_token_output", "bool", "true", "", "", "categorical", false, false, false, "Use exact hierarchical softmax with sparse per-node binary decisions."},
    {"sparse_output_topk", "uint32", "5", "1", "32", "linear", true, false, false, "Number of hierarchical candidates reported by token decoding."},
    {"sparse_output_beam_width", "uint32", "16", "5", "128", "log", true, false, false, "Fixed candidate beam for O(B log V) hierarchical decoding."},
    {"max_sparse_decisions_per_node", "uint32", "64", "8", "4096", "log", true, false, false, "Hard bound on local hierarchical decisions stored by one address node."},
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
    SBM_SET_BOOL(adaptive_topology)
    SBM_SET_UINT(max_address_channels)
    SBM_SET_UINT(topology_max_lag)
    SBM_SET_UINT(topology_max_arity)
    SBM_SET_BOOL(topology_enable_delta)
    SBM_SET_UINT(topology_probe_interval)
    SBM_SET_UINT(topology_probe_warmup)
    SBM_SET_UINT(topology_probe_steps)
    SBM_SET_UINT(topology_validation_steps)
    SBM_SET_UINT(topology_min_observations)
    SBM_SET_FLOAT(topology_accept_credit)
    SBM_SET_FLOAT(topology_credit_decay)
    SBM_SET_UINT(topology_prune_patience)
    SBM_SET_FLOAT(topology_prune_credit)
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
    SBM_SET_FLOAT(classification_learning_rate)
    SBM_SET_FLOAT(classification_mature_learning_rate)
    SBM_SET_FLOAT(label_smoothing)
    SBM_SET_FLOAT(softmax_temperature)
    SBM_SET_FLOAT(logit_decay)
    SBM_SET_BOOL(sparse_token_output)
    SBM_SET_UINT(sparse_output_topk)
    SBM_SET_UINT(sparse_output_beam_width)
    SBM_SET_UINT(max_sparse_decisions_per_node)
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
        << "  \"address_lags\": [";
    for (std::size_t i = 0; i < c.address_lags.size(); ++i) {
        if (i != 0U) out << ", ";
        out << c.address_lags[i];
    }
    out << "],\n"
        << "  \"adaptive_topology\": " << c.adaptive_topology << ",\n"
        << "  \"max_address_channels\": " << c.max_address_channels << ",\n"
        << "  \"topology_max_lag\": " << c.topology_max_lag << ",\n"
        << "  \"topology_max_arity\": " << c.topology_max_arity << ",\n"
        << "  \"topology_enable_delta\": " << c.topology_enable_delta << ",\n"
        << "  \"topology_probe_interval\": " << c.topology_probe_interval << ",\n"
        << "  \"topology_probe_warmup\": " << c.topology_probe_warmup << ",\n"
        << "  \"topology_probe_steps\": " << c.topology_probe_steps << ",\n"
        << "  \"topology_validation_steps\": " << c.topology_validation_steps << ",\n"
        << "  \"topology_min_observations\": " << c.topology_min_observations << ",\n"
        << "  \"topology_accept_credit\": " << c.topology_accept_credit << ",\n"
        << "  \"topology_credit_decay\": " << c.topology_credit_decay << ",\n"
        << "  \"topology_prune_patience\": " << c.topology_prune_patience << ",\n"
        << "  \"topology_prune_credit\": " << c.topology_prune_credit << ",\n"
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
        << "  \"classification_learning_rate\": " << c.classification_learning_rate << ",\n"
        << "  \"classification_mature_learning_rate\": " << c.classification_mature_learning_rate << ",\n"
        << "  \"label_smoothing\": " << c.label_smoothing << ",\n"
        << "  \"softmax_temperature\": " << c.softmax_temperature << ",\n"
        << "  \"logit_decay\": " << c.logit_decay << ",\n"
        << "  \"sparse_token_output\": " << c.sparse_token_output << ",\n"
        << "  \"sparse_output_topk\": " << c.sparse_output_topk << ",\n"
        << "  \"sparse_output_beam_width\": " << c.sparse_output_beam_width << ",\n"
        << "  \"max_sparse_decisions_per_node\": "
        << c.max_sparse_decisions_per_node << ",\n"
        << "  \"seed\": " << c.seed << "\n"
        << "}\n";
    return out.str();
}

std::string_view parameter_tasks(std::string_view name) {
    if (name == "classification_learning_rate" ||
        name == "classification_mature_learning_rate" ||
        name == "label_smoothing" ||
        name == "softmax_temperature" ||
        name == "logit_decay" ||
        name == "sparse_output_topk" ||
        name == "sparse_output_beam_width" ||
        name == "max_sparse_decisions_per_node") {
        return "token-ce";
    }
    if (name == "residual_learning_rate" ||
        name == "residual_mature_learning_rate" ||
        name == "residual_recency_pseudocount" ||
        name == "node_learning_rate" ||
        name == "mature_learning_rate") {
        return "vector";
    }
    return "both";
}

std::string schema_json() {
    std::ostringstream out;
    out << "{\n  \"api_version\": 4,\n  \"parameters\": [\n";
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
        const auto tasks = parameter_tasks(p.name);
        out << ", \"tasks\": [";
        if (tasks == "both") out << "\"token-ce\", \"vector\"";
        else out << "\"" << tasks << "\"";
        out << "], \"description\": \"" << json_escape(p.description) << "\"}";
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

uint32_t sbm_api_version(void) { return 4U; }
const char* sbm_api_version_string(void) { return "4.0.0"; }
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
        return new sbm_dataset_handle{sbm::generate_vector_process(
            length, alphabet, vector_dim, hidden_dim, seed, noise_std)};
    });
}

sbm_dataset_handle* sbm_token_dataset_generate_math(size_t sequence_count,
                                                     size_t sequence_length,
                                                     uint32_t vocab_size,
                                                     uint64_t seed,
                                                     float temperature,
                                                     float interaction_strength) {
    return guarded([&] {
        return new sbm_dataset_handle{sbm::generate_math_token_process(
            sequence_count, sequence_length, vocab_size, seed,
            temperature, interaction_strength)};
    });
}

sbm_dataset_handle* sbm_token_dataset_from_ids(const uint32_t* tokens,
                                                size_t token_count,
                                                uint32_t vocab_size,
                                                const uint64_t* sequence_offsets,
                                                size_t offset_count) {
    return guarded([&] {
        if (tokens == nullptr || token_count == 0U) {
            throw std::invalid_argument("tokens must be non-null and non-empty");
        }
        if ((sequence_offsets == nullptr) != (offset_count == 0U)) {
            throw std::invalid_argument(
                "sequence_offsets must be null iff offset_count is zero");
        }
        const std::span<const std::uint32_t> token_span(tokens, token_count);
        const std::span<const std::uint64_t> offset_span = sequence_offsets == nullptr
            ? std::span<const std::uint64_t>{}
            : std::span<const std::uint64_t>(sequence_offsets, offset_count);
        return new sbm_dataset_handle{
            sbm::make_token_dataset(token_span, vocab_size, offset_span)};
    });
}

sbm_dataset_handle* sbm_dataset_load(const char* path) {
    return guarded([&] {
        if (path == nullptr) throw std::invalid_argument("path is null");
        if (sbm::inspect_dataset_kind(path) == sbm::DatasetKind::TokenCrossEntropy) {
            return new sbm_dataset_handle{sbm::load_token_dataset(path)};
        }
        return new sbm_dataset_handle{sbm::load_dataset(path)};
    });
}

int sbm_dataset_save(const sbm_dataset_handle* dataset, const char* path) {
    clear_error();
    try {
        if (dataset == nullptr || path == nullptr) {
            throw std::invalid_argument("dataset and path must be non-null");
        }
        std::visit([&](const auto& value) {
            using Dataset = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Dataset, sbm::VectorDataset>) {
                sbm::save_dataset(value, path);
            } else {
                sbm::save_token_dataset(value, path);
            }
        }, dataset->value);
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

uint32_t sbm_dataset_kind_of(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    return std::holds_alternative<sbm::VectorDataset>(dataset->value)
        ? static_cast<uint32_t>(SBM_DATASET_VECTOR_REGRESSION)
        : static_cast<uint32_t>(SBM_DATASET_TOKEN_CROSS_ENTROPY);
}

uint64_t sbm_dataset_hash(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    return std::visit([](const auto& value) { return sbm::hash_dataset(value); },
                      dataset->value);
}

size_t sbm_dataset_length(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    return std::visit([](const auto& value) { return value.tokens.size(); },
                      dataset->value);
}

uint32_t sbm_dataset_vector_dim(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* vector = std::get_if<sbm::VectorDataset>(&dataset->value)) {
        return vector->vector_dim;
    }
    return 0U;
}

uint32_t sbm_dataset_alphabet(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* vector = std::get_if<sbm::VectorDataset>(&dataset->value)) {
        return vector->token_alphabet;
    }
    return std::get<sbm::TokenDataset>(dataset->value).vocab_size;
}

uint32_t sbm_dataset_vocab_size(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value)) {
        return token->vocab_size;
    }
    return 0U;
}

size_t sbm_dataset_sequence_count(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value)) {
        return token->sequence_count();
    }
    return 1U;
}

size_t sbm_dataset_example_count(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value)) {
        return token->example_count();
    }
    return std::get<sbm::VectorDataset>(dataset->value).tokens.size();
}

int sbm_token_shard_write(const sbm_dataset_handle* dataset, const char* path) {
    clear_error();
    try {
        if (dataset == nullptr || path == nullptr) {
            throw std::invalid_argument("dataset and path must be non-null");
        }
        const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value);
        if (token == nullptr) {
            throw std::invalid_argument("token shard requires a token dataset");
        }
        sbm::write_token_shard(*token, path);
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

sbm_token_shard_handle* sbm_token_shard_open(const char* path,
                                              uint64_t shard_index,
                                              int verify_payload) {
    return guarded([&] {
        if (path == nullptr) throw std::invalid_argument("path is null");
        return new sbm_token_shard_handle{
            sbm::MappedTokenShard(path, shard_index, verify_payload != 0)};
    });
}

void sbm_token_shard_destroy(sbm_token_shard_handle* shard) { delete shard; }

uint32_t sbm_token_shard_vocab_size(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.vocab_size();
}

uint64_t sbm_token_shard_token_count(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.token_count();
}

uint64_t sbm_token_shard_sequence_count(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.sequence_count();
}

uint64_t sbm_token_shard_dataset_hash(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.dataset_hash();
}

int sbm_token_shard_next(const sbm_token_shard_handle* shard,
                         sbm_token_shard_cursor* cursor,
                         uint32_t* input,
                         uint32_t* target) {
    clear_error();
    try {
        if (shard == nullptr || cursor == nullptr || input == nullptr || target == nullptr) {
            throw std::invalid_argument("shard, cursor, input and target must be non-null");
        }
        sbm::TokenShardCursor cpp_cursor{
            cursor->shard_index, cursor->sequence_index, cursor->token_offset};
        sbm::TokenExample example{};
        if (!shard->value.next(cpp_cursor, example)) return 0;
        cursor->shard_index = cpp_cursor.shard_index;
        cursor->sequence_index = cpp_cursor.sequence_index;
        cursor->token_offset = cpp_cursor.token_offset;
        *input = example.input;
        *target = example.target;
        return 1;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

char* sbm_run_token_shard_experiment_json(const sbm_token_shard_handle* shard,
                                          size_t warmup,
                                          const sbm_config_handle* config,
                                          int strict_freeze,
                                          size_t prefill,
                                          size_t prune_interval,
                                          size_t merge_interval) {
    return guarded([&] {
        if (shard == nullptr || config == nullptr) {
            throw std::invalid_argument("shard and config must be non-null");
        }
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        const auto result = sbm::run_token_experiment(
            shard->value, warmup, resolved, strict_freeze != 0,
            prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

sbm_token_corpus_handle* sbm_token_corpus_create(void) {
    return guarded([] { return new sbm_token_corpus_handle{}; });
}

void sbm_token_corpus_destroy(sbm_token_corpus_handle* corpus) { delete corpus; }

int sbm_token_corpus_add_shard(sbm_token_corpus_handle* corpus,
                               const char* path,
                               uint32_t split_kind,
                               uint64_t shard_index,
                               int verify_payload) {
    clear_error();
    try {
        if (corpus == nullptr || path == nullptr) {
            throw std::invalid_argument("corpus and path must be non-null");
        }
        if (split_kind > 1U) throw std::invalid_argument("invalid corpus split kind");
        auto shard = std::make_unique<sbm::MappedTokenShard>(
            path, shard_index, verify_payload != 0);
        (split_kind == 0U ? corpus->train : corpus->eval).push_back(std::move(shard));
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

char* sbm_run_token_corpus_experiment_json(const sbm_token_corpus_handle* corpus,
                                           const sbm_config_handle* config,
                                           int strict_freeze,
                                           size_t prefill,
                                           size_t prune_interval,
                                           size_t merge_interval) {
    return guarded([&] {
        if (corpus == nullptr || config == nullptr) {
            throw std::invalid_argument("corpus and config must be non-null");
        }
        std::vector<const sbm::MappedTokenShard*> train;
        std::vector<const sbm::MappedTokenShard*> eval;
        train.reserve(corpus->train.size());
        eval.reserve(corpus->eval.size());
        for (const auto& shard : corpus->train) train.push_back(shard.get());
        for (const auto& shard : corpus->eval) eval.push_back(shard.get());
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        const auto result = sbm::run_token_corpus_experiment(
            train, eval, resolved, strict_freeze != 0,
            prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

char* sbm_run_experiment_json(const sbm_dataset_handle* dataset, size_t warmup,
                              const sbm_config_handle* config, int strict_freeze,
                              size_t prefill, size_t prune_interval,
                              size_t merge_interval) {
    return guarded([&] {
        if (dataset == nullptr || config == nullptr) {
            throw std::invalid_argument("dataset and config must be non-null");
        }
        if (const auto* vector = std::get_if<sbm::VectorDataset>(&dataset->value)) {
            auto resolved = config->value;
            resolved.objective = sbm::ObjectiveKind::VectorRegression;
            const auto result = sbm::run_experiment(
                *vector, warmup, resolved, strict_freeze != 0,
                prefill, prune_interval, merge_interval);
            return duplicate_string(sbm::to_json(result));
        }
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        const auto result = sbm::run_token_experiment(
            std::get<sbm::TokenDataset>(dataset->value), warmup, resolved,
            strict_freeze != 0, prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

void sbm_string_free(char* value) { std::free(value); }

} // extern "C"

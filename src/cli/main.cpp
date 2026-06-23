#include "sbm/api.h"

#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

template <class T>
T number(std::string_view text, const char* name) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return value;
}

float real_number(std::string_view text, const char* name) {
    std::string owned(text);
    char* end = nullptr;
    const float value = std::strtof(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return value;
}

[[noreturn]] void api_failure(const char* action) {
    const char* detail = sbm_last_error();
    throw std::runtime_error(std::string(action) + ": " +
                             (detail == nullptr ? "unknown error" : detail));
}

struct ConfigOwner {
    ConfigOwner() : value(sbm_config_create()) {}
    sbm_config_handle* value{};
    ~ConfigOwner() { sbm_config_destroy(value); }
    ConfigOwner(const ConfigOwner&) = delete;
    ConfigOwner& operator=(const ConfigOwner&) = delete;
};

struct DatasetOwner {
    DatasetOwner() = default;
    sbm_dataset_handle* value{};
    ~DatasetOwner() { sbm_dataset_destroy(value); }
    DatasetOwner(const DatasetOwner&) = delete;
    DatasetOwner& operator=(const DatasetOwner&) = delete;
};

struct StringOwner {
    explicit StringOwner(char* pointer = nullptr) : value(pointer) {}
    char* value{};
    ~StringOwner() { sbm_string_free(value); }
    StringOwner(const StringOwner&) = delete;
    StringOwner& operator=(const StringOwner&) = delete;
};

void usage() {
    std::cout
        << "sbm_cli [dataset options] [experiment options] [--set name=value ...]\n\n"
        << "Task selection:\n"
        << "  --task token-ce|vector     Generated task (default token-ce)\n"
        << "  --dataset FILE             Load either supported binary dataset\n"
        << "  --write-dataset FILE       Save generated/loaded dataset\n\n"
        << "Mathematical token task:\n"
        << "  --sequences N              Independent token sequences (default 64)\n"
        << "  --sequence-length N        Tokens per sequence (default 2048)\n"
        << "  --vocab-size N             Token vocabulary (default 32)\n"
        << "  --math-temperature X       Generator distribution temperature (default 0.8)\n"
        << "  --interaction-strength X   Non-additive mathematical term (default 0.2)\n\n"
        << "Legacy vector task:\n"
        << "  --length N                 Vector dataset length (default 120000)\n"
        << "  --alphabet N               Input alphabet (default 64)\n"
        << "  --vector-dim N             Target dimension (default 16)\n"
        << "  --hidden-dim N             Generator feature dimension (default 24)\n"
        << "  --noise-std X              Observation noise (default 0.035)\n\n"
        << "Experiment options:\n"
        << "  --seed N                   Dataset/model seed (default 7)\n"
        << "  --warmup N                 Training examples (default 80000)\n"
        << "  --prefill N                Cold distractor nodes\n"
        << "  --prune-interval N         Structural prune interval\n"
        << "  --merge-interval N         Structural merge interval\n"
        << "  --allow-eval-growth        Disable strict structural freeze\n"
        << "  --set NAME=VALUE           Set any registered model parameter\n"
        << "  --output FILE              Result JSON path\n"
        << "  --describe-parameters      Print runtime parameter schema\n"
        << "  --dump-config              Print resolved configuration and exit\n";
}

std::pair<std::string, std::string> split_assignment(std::string_view text) {
    const auto equals = text.find('=');
    if (equals == std::string_view::npos || equals == 0U || equals + 1U >= text.size()) {
        throw std::invalid_argument("--set requires NAME=VALUE");
    }
    return {std::string(text.substr(0, equals)), std::string(text.substr(equals + 1U))};
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string task = "token-ce";
        std::size_t sequence_count = 64;
        std::size_t sequence_length = 2048;
        std::uint32_t vocab_size = 32;
        float math_temperature = 0.8F;
        float interaction_strength = 0.2F;

        std::size_t length = 120000;
        std::uint32_t alphabet = 64;
        std::uint32_t vector_dim = 16;
        std::uint32_t hidden_dim = 24;
        float noise_std = 0.035F;

        std::size_t warmup = 80000;
        std::size_t prefill = 0;
        std::size_t prune_interval = 0;
        std::size_t merge_interval = 0;
        std::uint64_t seed = 7;
        bool strict_freeze = true;
        bool describe_parameters = false;
        bool dump_config = false;
        std::string dataset_path;
        std::string write_dataset_path;
        std::string output_path = "results.json";
        std::vector<std::pair<std::string, std::string>> overrides;

        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--help" || argument == "-h") {
                usage();
                return 0;
            }
            if (argument == "--allow-eval-growth") {
                strict_freeze = false;
                continue;
            }
            if (argument == "--describe-parameters") {
                describe_parameters = true;
                continue;
            }
            if (argument == "--dump-config") {
                dump_config = true;
                continue;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing option value");
            const std::string_view value(argv[++i]);
            if (argument == "--task") task = value;
            else if (argument == "--sequences") sequence_count = number<std::size_t>(value, "sequences");
            else if (argument == "--sequence-length") sequence_length = number<std::size_t>(value, "sequence-length");
            else if (argument == "--vocab-size") vocab_size = number<std::uint32_t>(value, "vocab-size");
            else if (argument == "--math-temperature") math_temperature = real_number(value, "math-temperature");
            else if (argument == "--interaction-strength") interaction_strength = real_number(value, "interaction-strength");
            else if (argument == "--length") length = number<std::size_t>(value, "length");
            else if (argument == "--warmup") warmup = number<std::size_t>(value, "warmup");
            else if (argument == "--seed") seed = number<std::uint64_t>(value, "seed");
            else if (argument == "--alphabet") alphabet = number<std::uint32_t>(value, "alphabet");
            else if (argument == "--vector-dim") vector_dim = number<std::uint32_t>(value, "vector-dim");
            else if (argument == "--hidden-dim") hidden_dim = number<std::uint32_t>(value, "hidden-dim");
            else if (argument == "--noise-std") noise_std = real_number(value, "noise-std");
            else if (argument == "--prefill") prefill = number<std::size_t>(value, "prefill");
            else if (argument == "--prune-interval") prune_interval = number<std::size_t>(value, "prune-interval");
            else if (argument == "--merge-interval") merge_interval = number<std::size_t>(value, "merge-interval");
            else if (argument == "--dataset") dataset_path = value;
            else if (argument == "--write-dataset") write_dataset_path = value;
            else if (argument == "--output") output_path = value;
            else if (argument == "--set") overrides.push_back(split_assignment(value));
            else throw std::invalid_argument(std::string("unknown argument: ") + std::string(argument));
        }

        if (describe_parameters) {
            std::cout << sbm_parameter_schema_json();
            return 0;
        }
        if (task != "token-ce" && task != "vector") {
            throw std::invalid_argument("--task must be token-ce or vector");
        }

        ConfigOwner config;
        if (config.value == nullptr) api_failure("create config");
        if (sbm_config_set(config.value, "seed", std::to_string(seed).c_str()) != 0) {
            api_failure("set seed");
        }
        for (const auto& [name, value] : overrides) {
            if (sbm_config_set(config.value, name.c_str(), value.c_str()) != 0) {
                api_failure(("set " + name).c_str());
            }
        }

        if (dump_config) {
            StringOwner json{sbm_config_get_json(config.value)};
            if (json.value == nullptr) api_failure("dump config");
            std::cout << json.value;
            return 0;
        }

        DatasetOwner dataset;
        if (!dataset_path.empty()) {
            dataset.value = sbm_dataset_load(dataset_path.c_str());
        } else if (task == "token-ce") {
            dataset.value = sbm_token_dataset_generate_math(
                sequence_count, sequence_length, vocab_size, seed,
                math_temperature, interaction_strength);
        } else {
            dataset.value = sbm_dataset_generate(
                length, alphabet, vector_dim, hidden_dim, seed, noise_std);
        }
        if (dataset.value == nullptr) api_failure("create dataset");
        if (!write_dataset_path.empty() &&
            sbm_dataset_save(dataset.value, write_dataset_path.c_str()) != 0) {
            api_failure("save dataset");
        }

        StringOwner result{sbm_run_experiment_json(
            dataset.value, warmup, config.value, strict_freeze ? 1 : 0,
            prefill, prune_interval, merge_interval)};
        if (result.value == nullptr) api_failure("run experiment");

        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot open output file");
        output << result.value;
        std::cout << result.value;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

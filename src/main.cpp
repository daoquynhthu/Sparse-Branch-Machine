#include "sparse_branch_machine.hpp"

#include <charconv>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

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

void usage() {
    std::cout
        << "sbm_benchmark [--length N] [--warmup N] [--seed N] "
           "[--alphabet N] [--vector-dim N] [--hidden-dim N] "
           "[--dataset FILE] [--write-dataset FILE] [--prefill N] "
           "[--prune-interval N] [--merge-interval N] "
           "[--allow-eval-growth] [--output FILE]\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t length = 120000;
        std::size_t warmup = 40000;
        std::size_t prefill = 0;
        std::size_t prune_interval = 0;
        std::size_t merge_interval = 0;
        std::uint64_t seed = 7;
        std::uint32_t alphabet = 64;
        std::uint32_t vector_dim = 16;
        std::uint32_t hidden_dim = 24;
        bool strict_freeze = true;
        std::string dataset_path;
        std::string write_dataset_path;
        std::string output_path = "results_vector_v5.json";

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
            if (i + 1 >= argc) throw std::invalid_argument("missing value");
            const std::string_view value(argv[++i]);
            if (argument == "--length") length = number<std::size_t>(value, "length");
            else if (argument == "--warmup") warmup = number<std::size_t>(value, "warmup");
            else if (argument == "--seed") seed = number<std::uint64_t>(value, "seed");
            else if (argument == "--alphabet") alphabet = number<std::uint32_t>(value, "alphabet");
            else if (argument == "--vector-dim") vector_dim = number<std::uint32_t>(value, "vector-dim");
            else if (argument == "--hidden-dim") hidden_dim = number<std::uint32_t>(value, "hidden-dim");
            else if (argument == "--prefill") prefill = number<std::size_t>(value, "prefill");
            else if (argument == "--prune-interval") prune_interval = number<std::size_t>(value, "prune-interval");
            else if (argument == "--merge-interval") merge_interval = number<std::size_t>(value, "merge-interval");
            else if (argument == "--dataset") dataset_path = value;
            else if (argument == "--write-dataset") write_dataset_path = value;
            else if (argument == "--output") output_path = value;
            else throw std::invalid_argument("unknown argument");
        }

        auto dataset = dataset_path.empty()
            ? sbm::generate_vector_process(length, alphabet, vector_dim, hidden_dim, seed)
            : sbm::load_dataset(dataset_path);
        if (!write_dataset_path.empty()) sbm::save_dataset(dataset, write_dataset_path);

        const auto result = sbm::run_experiment(dataset, warmup, seed, strict_freeze,
                                                prefill, prune_interval, merge_interval);
        const auto json = sbm::to_json(result);
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot open output");
        output << json;
        std::cout << json;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

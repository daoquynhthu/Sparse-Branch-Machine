#include "sparse_branch_machine.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

sbm::Config sparse_config(std::uint32_t bucket_bits, std::uint32_t vocabulary) {
    sbm::Config config;
    config.objective = sbm::ObjectiveKind::TokenCrossEntropy;
    config.token_alphabet = vocabulary;
    config.vector_dim = vocabulary;
    config.bucket_bits = bucket_bits;
    config.address_lags = {1U};
    config.adaptive_topology = false;
    config.max_address_channels = 1U;
    config.sparse_token_output = true;
    config.seed = 17U;
    return config;
}

std::uint64_t address_bytes(std::uint32_t bucket_bits) {
    sbm::SparseBranchMachine machine(sparse_config(bucket_bits, 4096U));
    return machine.diagnostics().address_index_bytes;
}

std::uint64_t output_bytes(std::uint32_t vocabulary) {
    sbm::SparseBranchMachine machine(sparse_config(4U, vocabulary));
    return machine.diagnostics().output_structure_bytes;
}

} // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "all";
    const auto address_10 = address_bytes(10U);
    const auto address_16 = address_bytes(16U);
    const auto address_20 = address_bytes(20U);
    const auto output_4k = output_bytes(4096U);
    const auto output_50k = output_bytes(50257U);
    const auto output_250k = output_bytes(250000U);
    std::cerr << "address_bytes=" << address_10 << ',' << address_16 << ','
              << address_20 << " output_bytes=" << output_4k << ',' << output_50k
              << ',' << output_250k << '\n';
    auto collision_config = sparse_config(1U, 128U);
    collision_config.bucket_scan_limit = 16U;
    sbm::SparseBranchMachine collision_machine(collision_config);
    collision_machine.prefill_distractors(20000U);
    collision_machine.reset_sequence();
    (void)collision_machine.step_token(7U, 11U, true);
    const auto collision = collision_machine.diagnostics();
    std::cerr << "max_bucket_scan="
              << collision.max_bucket_candidates_inspected << '\n';

    if (mode == "address" || mode == "all") {
        assert(address_16 <= address_10 * 105U / 100U + 4096U);
        assert(address_20 <= address_10 * 105U / 100U + 4096U);
        assert(collision.max_bucket_candidates_inspected <=
               collision_config.bucket_scan_limit);
    }
    if (mode == "output" || mode == "all") {
        constexpr std::uint64_t one_mebibyte = 1024ULL * 1024ULL;
        assert(output_50k <= output_4k + one_mebibyte);
        assert(output_250k <= output_4k + one_mebibyte);
    }

    std::cout << "scaling " << mode << " completed\n";
}

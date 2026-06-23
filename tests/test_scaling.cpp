#include "sparse_branch_machine.hpp"
#include "sbm/detail/implicit_output.hpp"

#include <cassert>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

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

double address_p95_microseconds(std::size_t nodes) {
    auto config = sparse_config(10U, 128U);
    config.bucket_scan_limit = 16U;
    config.max_edges_per_node = 1U;
    config.edge_scan_limit = 1U;
    sbm::SparseBranchMachine machine(config);
    machine.prefill_distractors(nodes);
    machine.reset_sequence();
    for (std::uint32_t index = 0U; index < 64U; ++index) {
        (void)machine.step_token(index % 128U, (index + 1U) % 128U, false);
    }
    constexpr std::uint32_t batch_steps = 256U;
    std::vector<double> samples;
    samples.reserve(64U);
    std::uint32_t token_index = 0U;
    for (std::uint32_t batch = 0U; batch < 64U; ++batch) {
        const auto start = std::chrono::steady_clock::now();
        for (std::uint32_t index = 0U; index < batch_steps; ++index, ++token_index) {
            (void)machine.step_token(
                token_index % 128U, (token_index + 1U) % 128U, false);
        }
        const auto elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() /
            static_cast<double>(batch_steps);
        samples.push_back(elapsed);
    }
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<std::size_t>(samples.size() * 95U / 100U)];
}

void verify_implicit_output(std::uint32_t vocabulary) {
    sbm::detail::ImplicitOutputTree tree(vocabulary, 29U);
    std::vector<sbm::detail::ImplicitDecision> path;
    const std::uint32_t stride = std::max(1U, vocabulary / 997U);
    for (std::uint32_t token = 0U; token < vocabulary; token += stride) {
        const auto rank = tree.rank_from_token(token);
        assert(rank < vocabulary);
        assert(tree.token_from_rank(rank) == token);
        tree.target_path(token, path);
        std::uint32_t lo = 0U;
        std::uint32_t hi = vocabulary;
        for (const auto& decision : path) {
            const auto split = tree.split(lo, hi);
            assert(decision.id == split.decision_id);
            assert(decision.id < vocabulary - 1U);
            if (decision.right) lo = split.middle;
            else hi = split.middle;
        }
        assert(hi - lo == 1U);
        assert(lo == rank);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "all";
    if (mode == "latency") {
        const auto p95_100k = address_p95_microseconds(100000U);
        const auto p95_1m = address_p95_microseconds(1000000U);
        std::cout << "address_p95_us_100k=" << p95_100k
                  << " address_p95_us_1m=" << p95_1m
                  << " ratio=" << p95_1m / p95_100k << '\n';
        return 0;
    }
    if (mode == "implicit") {
        for (const auto vocabulary : {2U, 3U, 4096U, 50257U, 250000U}) {
            verify_implicit_output(vocabulary);
        }
        std::cout << "implicit output tests passed\n";
        return 0;
    }
    if (mode == "capacity") {
        auto config = sparse_config(4U, 257U);
        config.max_sparse_decisions_per_node = 8U;
        config.max_specializations_per_bucket = 1U;
        config.split_min_visits = UINT32_MAX;
        sbm::SparseBranchMachine machine(config);
        for (std::uint32_t step = 0U; step < 4096U; ++step) {
            (void)machine.step_token(3U, step % 257U, true);
        }
        assert(machine.diagnostics().max_sparse_entries_per_node <= 8U);
        std::cout << "bounded decision capacity passed\n";
        return 0;
    }
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
    sbm::SparseBranchMachine collision_control(collision_config);
    collision_machine.prefill_distractors(20000U);
    collision_control.prefill_distractors(20000U);
    collision_machine.reset_sequence();
    collision_control.reset_sequence();
    const auto collision_step = collision_machine.step_token(7U, 11U, true);
    const auto control_step = collision_control.step_token(7U, 11U, true);
    assert(collision_step.route == control_step.route);
    const auto collision = collision_machine.diagnostics();
    std::cerr << "max_bucket_scan="
              << collision.max_bucket_candidates_inspected << '\n';

    if (mode == "address" || mode == "all") {
        assert(address_16 <= address_10 * 105U / 100U + 4096U);
        assert(address_20 <= address_10 * 105U / 100U + 4096U);
        assert(collision.max_bucket_candidates_inspected <=
               collision_config.bucket_scan_limit);
        const auto live_before_prune = collision_machine.live_nodes();
        const auto pruned = collision_machine.prune(UINT32_MAX, 1.0F);
        assert(pruned == live_before_prune);
        assert(collision_machine.live_nodes() == 0U);
        collision_machine.prefill_distractors(1024U);
        (void)collision_machine.step_token(7U, 11U, false);
        assert(collision_machine.diagnostics().max_bucket_candidates_inspected <=
               collision_config.bucket_scan_limit);
    }
    if (mode == "output" || mode == "all") {
        constexpr std::uint64_t one_mebibyte = 1024ULL * 1024ULL;
        assert(output_50k <= output_4k + one_mebibyte);
        assert(output_250k <= output_4k + one_mebibyte);
    }

    std::cout << "scaling " << mode << " completed\n";
}

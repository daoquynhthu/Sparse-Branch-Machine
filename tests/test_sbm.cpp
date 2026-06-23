#include "sparse_branch_machine.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

namespace {
bool close(float a, float b, float tolerance = 1e-6F) {
    return std::abs(a - b) <= tolerance;
}
}

int main() {
    const std::uint32_t fixed_window[]{1, 2, 3, 4};
    assert(sbm::rolling_signature(fixed_window) == sbm::rolling_signature(fixed_window));
    assert(sbm::hamming_similarity(42, 42) == 1.0);

    // Compact token packing must use both symbols for a 64-token alphabet.
    const std::array<std::uint32_t, 2> context_a{1, 7};
    const std::array<std::uint32_t, 2> context_b{33, 7};
    const auto signature_a = sbm::token_context_signature(context_a, 64);
    const auto signature_b = sbm::token_context_signature(context_b, 64);
    assert((signature_a >> 52U) != (signature_b >> 52U));

    auto dataset = sbm::generate_vector_process(8000, 32, 16, 20, 9);
    assert(dataset.tokens.size() == 8000);
    assert(dataset.targets.size() == 8000 * 16);

    const auto expected_hash = sbm::hash_dataset(dataset);
    const char* path = "sbm_vector_test.bin";
    sbm::save_dataset(dataset, path);
    auto loaded = sbm::load_dataset(path);
    std::remove(path);
    assert(sbm::hash_dataset(loaded) == expected_hash);

    // A target may update the model only after the prediction has been fixed.
    sbm::Config leakage_config;
    leakage_config.token_alphabet = 8;
    leakage_config.vector_dim = 8;
    leakage_config.bucket_bits = 6;
    leakage_config.seed = 11;
    sbm::SparseBranchMachine positive_model(leakage_config);
    sbm::SparseBranchMachine negative_model(leakage_config);
    const std::vector<float> positive_target(8, 1.0F);
    const std::vector<float> negative_target(8, -1.0F);
    (void)positive_model.step(3, positive_target, true);
    (void)negative_model.step(3, negative_target, true);
    const auto positive_prediction = positive_model.last_prediction();
    const auto negative_prediction = negative_model.last_prediction();
    for (std::size_t i = 0; i < positive_prediction.size(); ++i) {
        assert(close(positive_prediction[i], 0.0F));
        assert(close(positive_prediction[i], negative_prediction[i]));
    }

    sbm::Config config;
    config.token_alphabet = 32;
    config.vector_dim = 16;
    config.bucket_bits = 10;
    config.seed = 9;
    sbm::SparseBranchMachine machine(config);
    for (std::size_t i = 0; i < 3000; ++i) {
        const auto stats = machine.step(dataset.tokens[i], dataset.target(i), true);
        assert(std::isfinite(stats.normalized_mse));
        assert(stats.active_nodes <= config.beam_width);
    }
    const auto frozen_nodes = machine.live_nodes();
    for (std::size_t i = 3000; i < 4000; ++i) {
        (void)machine.step(dataset.tokens[i], dataset.target(i), false);
    }
    assert(machine.live_nodes() == frozen_nodes);

    const auto result = sbm::run_experiment(dataset, 3000, 9, true, 1000, 0, 0);
    assert(std::isfinite(result.eval.normalized_rmse));
    assert(std::isfinite(result.token_baseline_eval.r2));
    assert(std::isfinite(result.pair_baseline_eval.r2));
    assert(std::abs(result.mean_baseline_eval.r2) < 1e-5);
    assert(result.token_baseline_eval.r2 < 0.85);
    assert(result.pair_baseline_eval.r2 < 0.90);
    assert(result.seen_context_vectors + result.unseen_context_vectors == 5000);
    assert(result.diagnostics.steps == 8000);

    std::cout << "all tests passed; SIMD=" << sbm::simd_available() << '\n';
}

#pragma once

#include "sbm/dataset.hpp"
#include "sbm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace sbm {

struct VectorMetrics {
    double normalized_rmse{};
    double cosine{};
    double r2{};
};

struct ExperimentResult {
    Diagnostics diagnostics;
    VectorMetrics train;
    VectorMetrics eval;
    VectorMetrics mean_baseline_eval;
    VectorMetrics token_baseline_eval;
    VectorMetrics pair_baseline_eval;
    VectorMetrics seen_context_eval;
    VectorMetrics unseen_context_eval;
    std::uint64_t seen_context_vectors{};
    std::uint64_t unseen_context_vectors{};
    double steps_per_second{};
    double elapsed_seconds{};
    bool strict_freeze{};
    std::uint64_t dataset_hash{};
    std::uint32_t vector_dim{};
    std::uint32_t token_alphabet{};
};

[[nodiscard]] ExperimentResult run_experiment(
    const VectorDataset& dataset,
    std::size_t warmup,
    std::uint64_t seed = 7,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] std::string to_json(const ExperimentResult& result);

} // namespace sbm

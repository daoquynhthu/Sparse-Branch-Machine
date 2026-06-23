#pragma once

#include "sbm/dataset.hpp"
#include "sbm/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace sbm {

struct VectorMetrics {
    double normalized_rmse{};
    double cosine{};
    double r2{};
};

struct TokenMetrics {
    double cross_entropy{};
    double bits_per_token{};
    double perplexity{};
    double top1_accuracy{};
    double top5_accuracy{};
    double mean_target_probability{};
};

struct ExperimentResult {
    Diagnostics diagnostics;
    VectorMetrics train;
    VectorMetrics eval;
    VectorMetrics mean_baseline_eval;
    VectorMetrics token_baseline_eval;
    VectorMetrics pair_baseline_eval;
    VectorMetrics pair_lag2_baseline_eval;
    VectorMetrics multiscale_baseline_eval;
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
    std::array<std::uint32_t, kAddressChannelCount> address_lags{};
    float exact_region_mass{};
    float residual_channel_gain{};
    float residual_recency_pseudocount{};
    float edge_score_weight{};
};

struct TokenExperimentResult {
    Diagnostics diagnostics;
    TokenMetrics train;
    TokenMetrics eval;
    TokenMetrics unigram_baseline_eval;
    TokenMetrics current_token_baseline_eval;
    TokenMetrics pair_context_baseline_eval;
    TokenMetrics multiscale_baseline_eval;
    double oracle_cross_entropy{};
    double excess_cross_entropy{};
    double steps_per_second{};
    double elapsed_seconds{};
    bool strict_freeze{};
    std::uint64_t dataset_hash{};
    std::uint32_t vocab_size{};
    std::uint64_t train_examples{};
    std::uint64_t eval_examples{};
    std::uint64_t sequence_count{};
    std::array<std::uint32_t, kAddressChannelCount> address_lags{};
    float exact_region_mass{};
    float edge_score_weight{};
    float softmax_temperature{};
    float label_smoothing{};
};

[[nodiscard]] ExperimentResult run_experiment(
    const VectorDataset& dataset,
    std::size_t warmup,
    std::uint64_t seed = 7,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] ExperimentResult run_experiment(
    const VectorDataset& dataset,
    std::size_t warmup,
    Config config,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] TokenExperimentResult run_token_experiment(
    const TokenDataset& dataset,
    std::size_t warmup_examples,
    Config config,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] std::string to_json(const ExperimentResult& result);
[[nodiscard]] std::string to_json(const TokenExperimentResult& result);

} // namespace sbm

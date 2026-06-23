#pragma once

#include <cstdint>
#include <vector>

namespace sbm {

using NodeId = std::uint32_t;
inline constexpr NodeId kInvalidNode = UINT32_MAX;

enum class NodePhase : std::uint8_t { Cold, Warm, Mature, Dormant };

struct Edge {
    NodeId dst{kInvalidNode};
    float weight{};
    float eligibility{};
};

struct Config {
    std::uint32_t token_alphabet{64};
    std::uint32_t vector_dim{16};
    std::uint32_t context_width{12};
    std::uint32_t bucket_bits{12};
    std::uint32_t beam_width{6};
    std::uint32_t bucket_scan_limit{32};
    std::uint32_t edge_scan_limit{8};
    std::uint32_t max_edges_per_node{32};
    std::uint32_t edge_reinforce_width{2};
    std::uint32_t max_specializations_per_bucket{4};
    std::uint32_t split_min_visits{48};
    std::uint32_t split_cooldown{128};

    double split_context_similarity{0.78};
    float split_loss_threshold{0.72F};
    bool allow_growth_when_frozen{false};

    std::uint32_t trace_horizon{8};
    float trace_decay{0.80F};
    float edge_decay{0.999F};
    float edge_learning_rate{0.08F};
    float edge_score_weight{0.16F};
    float edge_min_contribution{0.004F};
    float responsibility_temperature{3.0F};
    float exact_region_mass{0.86F};
    float min_update_responsibility{0.01F};
    std::uint32_t warm_visits{12};
    std::uint32_t mature_visits{96};
    float dormant_utility{-0.30F};
    float node_learning_rate{0.12F};
    float mature_learning_rate{0.035F};
    float merge_similarity{0.95F};
    float merge_vector_distance{0.08F};

    std::uint64_t seed{7};
};

struct StepStats {
    float normalized_mse{};
    float cosine{};
    std::uint32_t active_nodes{};
    std::uint32_t candidates_examined{};
    std::uint32_t live_nodes{};
    std::uint32_t created{};
    std::vector<NodeId> route;
};

struct Diagnostics {
    std::uint64_t steps{};
    std::uint64_t live_nodes{};
    std::uint64_t logical_ids_issued{};
    std::uint64_t edges{};
    double avg_active{};
    double avg_candidates{};
    std::uint64_t created_total{};
    std::uint64_t merged_total{};
    std::uint64_t pruned_total{};
    std::uint64_t cold_nodes{};
    std::uint64_t warm_nodes{};
    std::uint64_t mature_nodes{};
    std::uint64_t dormant_nodes{};
    std::uint64_t stale_bucket_refs_skipped{};
    std::uint64_t stale_edge_refs_skipped{};
    std::uint64_t estimated_bytes{};
    bool simd_enabled{};
};

} // namespace sbm

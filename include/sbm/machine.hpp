#pragma once

#include "sbm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <utility>
#include <vector>

namespace sbm {

class SparseBranchMachine {
public:
    explicit SparseBranchMachine(Config config);

    [[nodiscard]] StepStats step(std::uint32_t token, std::span<const float> target,
                                 bool learn = true);
    [[nodiscard]] std::size_t prune(std::uint32_t min_visits = 8,
                                    float utility_threshold = -0.05F);
    [[nodiscard]] std::size_t merge_redundant(std::size_t max_merges = 64);
    void rebuild_indexes();
    void prefill_distractors(std::size_t count);

    [[nodiscard]] Diagnostics diagnostics() const noexcept;
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] bool contains(NodeId id) const noexcept;
    [[nodiscard]] std::size_t live_nodes() const noexcept { return ids_.size(); }
    [[nodiscard]] NodePhase phase(NodeId id) const noexcept;
    [[nodiscard]] std::span<const float> last_prediction() const noexcept {
        return prediction_buffer_;
    }

private:
    struct CandidateNode {
        NodeId id{kInvalidNode};
        float edge_prior{};
    };
    struct ScoredNode {
        double score{};
        NodeId id{kInvalidNode};
        float responsibility{};
        float contribution{};
        bool exact_region{};
        std::uint8_t channel{};
    };
    struct TraceFrame {
        std::vector<NodeId> route;
        std::vector<float> contribution;
        float loss{};
    };

    [[nodiscard]] std::uint32_t bucket(std::uint64_t signature) const noexcept;
    [[nodiscard]] std::size_t bucket_index(std::uint8_t channel,
                                           std::uint64_t signature) const noexcept;
    [[nodiscard]] NodeId new_node(std::uint64_t signature, std::span<const float> initial,
                                  NodeId parent = kInvalidNode,
                                  std::uint8_t channel = 0U);
    [[nodiscard]] std::size_t slot_of(NodeId id) const noexcept;
    [[nodiscard]] std::vector<CandidateNode> candidate_ids(
        std::span<const std::uint64_t> signatures);
    [[nodiscard]] double score(std::size_t slot,
                               std::span<const std::uint64_t> signatures,
                               float edge_prior) const noexcept;
    [[nodiscard]] std::pair<std::vector<ScoredNode>, std::uint32_t>
        select_route(std::span<const std::uint64_t> signatures);
    void assign_responsibilities(std::vector<ScoredNode>& active) const;
    void aggregate(const std::vector<ScoredNode>& active, std::span<float> output) const;
    void compute_counterfactual_contributions(std::vector<ScoredNode>& active,
                                              std::span<const float> target,
                                              float target_energy,
                                              float full_loss) const;
    [[nodiscard]] double output_distance(std::size_t a, std::size_t b) const noexcept;
    [[nodiscard]] NodePhase phase_of_slot(std::size_t slot) const noexcept;

    void reinforce_edge(NodeId source, NodeId destination, float delta);
    void decay_edges(NodeId source);
    void apply_trace_credit(float normalized_loss);
    void update_phase(std::size_t slot);
    void absorb_node(std::size_t survivor_slot, std::size_t victim_slot);
    void erase_slot(std::size_t slot);
    [[nodiscard]] bool push_candidate(std::vector<CandidateNode>& values, NodeId id,
                                      float edge_prior) const noexcept;

    Config config_;
    std::uint64_t rng_state_{};

    std::vector<NodeId> ids_;
    std::vector<std::uint64_t> prototypes_;
    std::vector<std::uint32_t> visits_;
    std::vector<std::uint32_t> address_visits_;
    std::vector<float> utility_ema_;
    std::vector<float> loss_ema_;
    std::vector<float> address_loss_ema_;
    std::vector<std::uint8_t> phases_;
    std::vector<std::uint8_t> channels_;
    std::vector<std::uint8_t> hot_indexed_;
    std::vector<NodeId> parents_;
    std::vector<float> output_vectors_;
    std::vector<std::vector<Edge>> edges_;

    std::vector<std::uint32_t> id_to_slot_;
    NodeId next_id_{};

    std::vector<std::vector<NodeId>> buckets_;
    std::vector<std::vector<NodeId>> hot_buckets_;
    std::vector<std::uint64_t> bucket_last_split_step_;
    std::vector<NodeId> previous_route_;
    std::vector<float> previous_responsibilities_;
    std::deque<TraceFrame> trace_;
    std::deque<std::uint32_t> history_;
    std::vector<float> prediction_buffer_;

    std::uint64_t total_steps_{};
    std::uint64_t total_candidates_{};
    std::uint64_t total_active_{};
    std::uint64_t total_created_{};
    std::uint64_t total_merged_{};
    std::uint64_t total_pruned_{};
    mutable std::uint64_t stale_bucket_refs_skipped_{};
    mutable std::uint64_t stale_edge_refs_skipped_{};
};

} // namespace sbm

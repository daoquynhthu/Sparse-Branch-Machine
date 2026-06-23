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
    [[nodiscard]] StepStats step_token(std::uint32_t token,
                                       std::uint32_t target_token,
                                       bool learn = true);
    void reset_sequence();
    void freeze_topology();
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
    [[nodiscard]] std::vector<std::uint32_t> learned_address_lags() const;
    [[nodiscard]] std::vector<AddressProgram> learned_address_programs() const;
    [[nodiscard]] std::vector<float> learned_channel_credit() const;
    [[nodiscard]] std::vector<std::uint8_t> learned_channel_phase() const;
    [[nodiscard]] const std::vector<TopologyEvent>& topology_events() const noexcept {
        return topology_events_;
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
    struct AddressChannelState {
        AddressProgram program{};
        ChannelPhase phase{ChannelPhase::Retired};
        float credit_ema{};
        double credit_sum{};
        std::uint64_t observations{};
        std::uint64_t born_step{};
    };
    struct SparseOutputEntry {
        std::uint32_t decision{};
        float logit{};
    };
    struct OutputTreeNode {
        std::uint32_t left_ref{};
        std::uint32_t right_ref{};
    };
    struct TokenPathStep {
        std::uint32_t decision{};
        bool right{};
    };
    struct TraceFrame {
        std::vector<NodeId> route;
        std::vector<float> contribution;
        float loss{};
    };

    [[nodiscard]] std::uint32_t bucket(std::uint64_t signature) const noexcept;
    [[nodiscard]] bool uses_sparse_token_output() const noexcept;
    void build_output_tree();
    [[nodiscard]] float sparse_logit(std::size_t slot, std::uint32_t decision) const noexcept;
    float& mutable_sparse_logit(std::size_t slot, std::uint32_t decision);
    [[nodiscard]] float aggregate_sparse_logit(std::span<const ScoredNode> active,
                                               std::uint32_t decision) const noexcept;
    [[nodiscard]] StepStats step_token_dense(std::uint32_t token,
                                             std::uint32_t target_token,
                                             bool learn);
    [[nodiscard]] StepStats step_token_sparse(std::uint32_t token,
                                              std::uint32_t target_token,
                                              bool learn);
    [[nodiscard]] bool channel_enabled(std::size_t channel) const noexcept;
    [[nodiscard]] bool channel_learning_enabled(std::size_t channel) const noexcept;
    [[nodiscard]] std::span<const std::uint64_t> make_signatures(
        std::span<const std::uint32_t> window);
    void maybe_begin_topology_probe(bool learn);
    void maybe_finalize_topology_probe();
    void observe_topology_credit(std::span<const float> channel_credit);
    void retire_channel(std::size_t channel);
    [[nodiscard]] std::size_t bucket_index(std::uint8_t channel,
                                           std::uint64_t signature) const noexcept;
    [[nodiscard]] NodeId new_node(std::uint64_t signature, std::span<const float> initial,
                                  NodeId parent = kInvalidNode,
                                  std::uint8_t channel = 0U);
    [[nodiscard]] std::size_t slot_of(NodeId id) const noexcept;
    [[nodiscard]] std::span<const CandidateNode> candidate_ids(
        std::span<const std::uint64_t> signatures);
    [[nodiscard]] double score(std::size_t slot,
                               std::span<const std::uint64_t> signatures,
                               float edge_prior) const noexcept;
    [[nodiscard]] std::pair<std::span<ScoredNode>, std::uint32_t>
        select_route(std::span<const std::uint64_t> signatures);
    void assign_responsibilities(std::span<ScoredNode> active) const;
    void aggregate(std::span<const ScoredNode> active, std::span<float> output) const;
    void compute_counterfactual_contributions(std::span<ScoredNode> active,
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
    std::vector<AddressChannelState> topology_;
    std::uint64_t proposal_cursor_{};
    std::vector<std::uint64_t> proposed_program_keys_;
    std::uint64_t next_probe_step_{};
    std::uint64_t topology_proposals_{};
    std::uint64_t topology_accepted_{};
    std::uint64_t topology_rejected_{};
    std::uint64_t topology_pruned_{};
    std::vector<TopologyEvent> topology_events_;
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
    std::vector<std::vector<SparseOutputEntry>> sparse_outputs_;
    std::vector<OutputTreeNode> output_tree_;
    std::vector<std::uint32_t> token_path_offsets_;
    std::vector<TokenPathStep> token_path_steps_;
    std::uint32_t output_root_ref_{};
    std::vector<std::vector<Edge>> edges_;

    std::vector<std::uint32_t> id_to_slot_;
    NodeId next_id_{};

    std::vector<std::vector<NodeId>> buckets_;
    std::vector<std::vector<NodeId>> hot_buckets_;
    std::vector<std::uint64_t> bucket_last_split_step_;
    std::vector<NodeId> previous_route_;
    std::vector<float> previous_responsibilities_;
    std::deque<TraceFrame> trace_;
    std::vector<std::uint32_t> history_;
    std::vector<float> prediction_buffer_;
    std::vector<float> logit_buffer_;
    std::vector<float> zero_output_buffer_;
    std::vector<float> channel_output_buffer_;
    std::vector<float> channel_credit_buffer_;
    std::vector<std::uint64_t> signature_buffer_;
    std::vector<CandidateNode> candidate_scratch_;
    std::vector<ScoredNode> scored_scratch_;
    std::vector<ScoredNode> selected_scratch_;
    std::vector<std::uint8_t> chosen_scratch_;
    std::vector<std::size_t> order_scratch_;

    std::uint64_t total_steps_{};
    std::uint64_t total_candidates_{};
    std::uint64_t total_active_{};
    std::uint64_t total_created_{};
    std::uint64_t total_merged_{};
    std::uint64_t total_pruned_{};
    mutable std::uint64_t stale_bucket_refs_skipped_{};
    mutable std::uint64_t stale_edge_refs_skipped_{};
    std::uint64_t max_bucket_candidates_inspected_{};
};

} // namespace sbm

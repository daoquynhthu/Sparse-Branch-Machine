#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <string>
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

struct VectorDataset {
    std::uint32_t token_alphabet{};
    std::uint32_t vector_dim{};
    std::vector<std::uint32_t> tokens;
    std::vector<float> targets; // row-major [tokens.size(), vector_dim]

    [[nodiscard]] std::span<const float> target(std::size_t index) const noexcept {
        return {targets.data() + index * vector_dim, vector_dim};
    }
};

std::uint64_t mix64(std::uint64_t x) noexcept;
std::uint64_t rolling_signature(std::span<const std::uint32_t> window,
                                std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept;
std::uint64_t token_context_signature(std::span<const std::uint32_t> window,
                                      std::uint32_t alphabet,
                                      std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept;
double hamming_similarity(std::uint64_t a, std::uint64_t b) noexcept;
std::uint64_t hash_dataset(const VectorDataset& dataset) noexcept;
bool simd_available() noexcept;

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
    [[nodiscard]] std::span<const float> last_prediction() const noexcept { return prediction_buffer_; }

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
    };
    struct TraceFrame {
        std::vector<NodeId> route;
        std::vector<float> contribution;
        float loss{};
    };

    [[nodiscard]] std::uint32_t bucket(std::uint64_t signature) const noexcept;
    [[nodiscard]] NodeId new_node(std::uint64_t signature, std::span<const float> initial,
                                  NodeId parent = kInvalidNode);
    [[nodiscard]] std::size_t slot_of(NodeId id) const noexcept;
    [[nodiscard]] std::vector<CandidateNode> candidate_ids(std::uint64_t signature);
    [[nodiscard]] double score(std::size_t slot, std::uint64_t signature,
                               float edge_prior) const noexcept;
    [[nodiscard]] std::pair<std::vector<ScoredNode>, std::uint32_t>
        select_route(std::uint64_t signature);
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

[[nodiscard]] VectorDataset generate_vector_process(std::size_t length,
                                                    std::uint32_t alphabet = 64,
                                                    std::uint32_t vector_dim = 16,
                                                    std::uint32_t hidden_dim = 24,
                                                    std::uint64_t seed = 7,
                                                    float noise_std = 0.035F);
void save_dataset(const VectorDataset& dataset, const std::string& path);
[[nodiscard]] VectorDataset load_dataset(const std::string& path);

[[nodiscard]] ExperimentResult run_experiment(const VectorDataset& dataset,
                                              std::size_t warmup,
                                              std::uint64_t seed = 7,
                                              bool strict_freeze = true,
                                              std::size_t prefill = 0,
                                              std::size_t prune_interval = 0,
                                              std::size_t merge_interval = 0);

[[nodiscard]] std::string to_json(const ExperimentResult& result);

} // namespace sbm

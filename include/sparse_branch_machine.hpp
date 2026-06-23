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
    std::uint32_t alphabet_size{8};
    std::uint32_t context_width{10};
    std::uint32_t bucket_bits{11};
    std::uint32_t beam_width{4};
    std::uint32_t bucket_scan_limit{24};
    std::uint32_t edge_scan_limit{8};
    std::uint32_t max_edges_per_node{32};

    double create_similarity{0.60};
    bool create_on_error{true};
    bool allow_growth_when_frozen{false};

    // Architecture/learning controls.
    std::uint32_t trace_horizon{8};
    float trace_decay{0.78F};
    float positive_credit{1.0F};
    float negative_credit{0.55F};
    float edge_decay{0.999F};
    std::uint32_t split_conflict_threshold{1};
    float split_error_threshold{0.0F};
    std::uint32_t warm_visits{8};
    std::uint32_t mature_visits{48};
    float dormant_utility{-0.35F};
    float merge_similarity{0.94F};
    float merge_output_distance{0.12F};

    std::uint64_t seed{7};
};

struct StepStats {
    std::uint32_t prediction{};
    bool correct{};
    std::uint32_t active_nodes{};
    std::uint32_t candidates_examined{};
    std::uint32_t live_nodes{};
    std::uint32_t created{};
    std::uint32_t split{};
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
    std::uint64_t split_total{};
    std::uint64_t merged_total{};
    std::uint64_t pruned_total{};
    std::uint64_t cold_nodes{};
    std::uint64_t warm_nodes{};
    std::uint64_t mature_nodes{};
    std::uint64_t dormant_nodes{};
    std::uint64_t stale_bucket_refs_skipped{};
    std::uint64_t stale_edge_refs_skipped{};
    std::uint64_t estimated_bytes{};
};

struct ExperimentResult {
    Diagnostics diagnostics;
    double train_accuracy{};
    double eval_accuracy{};
    double steps_per_second{};
    double elapsed_seconds{};
    bool strict_freeze{};
    std::uint64_t dataset_hash{};
};

struct Dataset {
    std::uint32_t alphabet_size{};
    std::vector<std::uint32_t> sequence;
};

std::uint64_t mix64(std::uint64_t x) noexcept;
std::uint64_t rolling_signature(std::span<const std::uint32_t> window,
                                std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept;
double hamming_similarity(std::uint64_t a, std::uint64_t b) noexcept;
std::uint64_t hash_dataset(const Dataset& dataset) noexcept;

class SparseBranchMachine {
public:
    explicit SparseBranchMachine(Config config);

    [[nodiscard]] StepStats step(std::uint32_t token, std::uint32_t target, bool learn = true);
    [[nodiscard]] std::size_t prune(std::uint32_t min_visits = 4,
                                    float utility_threshold = 0.0F);
    [[nodiscard]] std::size_t merge_redundant(std::size_t max_merges = 64);
    void rebuild_indexes();
    void prefill_distractors(std::size_t count);

    [[nodiscard]] Diagnostics diagnostics() const noexcept;
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] bool contains(NodeId id) const noexcept;
    [[nodiscard]] std::size_t live_nodes() const noexcept { return ids_.size(); }
    [[nodiscard]] NodePhase phase(NodeId id) const noexcept;

private:
    struct ScoredNode { double score{}; NodeId id{kInvalidNode}; };
    struct TraceFrame { std::vector<NodeId> route; };

    [[nodiscard]] std::uint32_t bucket(std::uint64_t signature) const noexcept;
    [[nodiscard]] NodeId new_node(std::uint64_t signature, NodeId parent = kInvalidNode);
    [[nodiscard]] std::size_t slot_of(NodeId id) const noexcept;
    [[nodiscard]] std::vector<NodeId> candidate_ids(std::uint64_t signature);
    [[nodiscard]] double score(std::size_t slot, std::uint64_t signature) const noexcept;
    [[nodiscard]] std::pair<std::vector<ScoredNode>, std::uint32_t> select_route(std::uint64_t signature);
    [[nodiscard]] std::uint32_t aggregate(const std::vector<ScoredNode>& active) const;
    [[nodiscard]] std::uint32_t node_prediction(std::size_t slot) const noexcept;
    [[nodiscard]] double output_distance(std::size_t a, std::size_t b) const noexcept;
    [[nodiscard]] NodePhase phase_of_slot(std::size_t slot) const noexcept;

    void reinforce_edge(NodeId source, NodeId destination, float delta);
    void apply_trace_credit(bool correct);
    void update_phase(std::size_t slot);
    void absorb_node(std::size_t survivor_slot, std::size_t victim_slot);
    void erase_slot(std::size_t slot);
    [[nodiscard]] bool push_unique(std::vector<NodeId>& values, NodeId id) const noexcept;

    Config config_;
    std::uint64_t rng_state_{};

    std::vector<NodeId> ids_;
    std::vector<std::uint64_t> prototypes_;
    std::vector<std::uint32_t> visits_;
    std::vector<std::uint32_t> correct_;
    std::vector<std::uint32_t> conflicts_;
    std::vector<float> utility_ema_;
    std::vector<float> error_ema_;
    std::vector<std::uint8_t> phases_;
    std::vector<std::uint8_t> hot_indexed_;
    std::vector<NodeId> parents_;
    std::vector<float> output_counts_;
    std::vector<std::vector<Edge>> edges_;

    std::vector<std::uint32_t> id_to_slot_;
    NodeId next_id_{};

    std::vector<std::vector<NodeId>> buckets_;
    std::vector<std::vector<NodeId>> hot_buckets_;
    std::vector<NodeId> previous_route_;
    std::deque<TraceFrame> trace_;
    std::deque<std::uint32_t> history_;

    std::uint64_t total_steps_{};
    std::uint64_t total_candidates_{};
    std::uint64_t total_active_{};
    std::uint64_t total_created_{};
    std::uint64_t total_split_{};
    std::uint64_t total_merged_{};
    std::uint64_t total_pruned_{};
    mutable std::uint64_t stale_bucket_refs_skipped_{};
    mutable std::uint64_t stale_edge_refs_skipped_{};
};

[[nodiscard]] Dataset generate_hidden_fsm(std::size_t length,
                                          std::uint32_t alphabet = 8,
                                          std::uint32_t states = 64,
                                          std::uint64_t seed = 7);
void save_dataset(const Dataset& dataset, const std::string& path);
[[nodiscard]] Dataset load_dataset(const std::string& path);

[[nodiscard]] ExperimentResult run_experiment(const Dataset& dataset,
                                              std::size_t warmup,
                                              std::uint64_t seed = 7,
                                              bool strict_freeze = true,
                                              std::size_t prefill = 0,
                                              std::size_t prune_interval = 0,
                                              std::size_t merge_interval = 0);

[[nodiscard]] std::string to_json(const ExperimentResult& result);

} // namespace sbm

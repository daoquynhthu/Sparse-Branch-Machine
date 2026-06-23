#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <string>
#include <vector>

namespace sbm {

using NodeId = std::uint32_t;
inline constexpr NodeId kInvalidNode = UINT32_MAX;

struct Edge {
    NodeId dst{kInvalidNode};
    float weight{};
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
    std::uint64_t seed{7};
};

struct StepStats {
    std::uint32_t prediction{};
    bool correct{};
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
    std::uint64_t pruned_total{};
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
    void rebuild_indexes();
    void prefill_distractors(std::size_t count);

    [[nodiscard]] Diagnostics diagnostics() const noexcept;
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] bool contains(NodeId id) const noexcept;
    [[nodiscard]] std::size_t live_nodes() const noexcept { return ids_.size(); }

private:
    struct ScoredNode {
        double score{};
        NodeId id{kInvalidNode};
    };

    [[nodiscard]] std::uint32_t bucket(std::uint64_t signature) const noexcept;
    [[nodiscard]] NodeId new_node(std::uint64_t signature);
    [[nodiscard]] std::size_t slot_of(NodeId id) const noexcept;
    [[nodiscard]] std::vector<NodeId> candidate_ids(std::uint64_t signature);
    [[nodiscard]] double score(std::size_t slot, std::uint64_t signature) const noexcept;
    [[nodiscard]] std::pair<std::vector<ScoredNode>, std::uint32_t>
    select_route(std::uint64_t signature);
    [[nodiscard]] std::uint32_t aggregate(const std::vector<ScoredNode>& active) const;
    void reinforce_edge(NodeId source, NodeId destination, float delta);
    void erase_slot(std::size_t slot);
    [[nodiscard]] bool push_unique(std::vector<NodeId>& values, NodeId id) const noexcept;

    Config config_;
    std::uint64_t rng_state_{};

    // Structure-of-arrays node store. Edges remain compact per-node vectors.
    std::vector<NodeId> ids_;
    std::vector<std::uint64_t> prototypes_;
    std::vector<std::uint32_t> visits_;
    std::vector<std::uint32_t> correct_;
    std::vector<float> utility_ema_;
    std::vector<std::uint8_t> hot_indexed_;
    std::vector<float> output_counts_; // flattened [slot * alphabet + symbol]
    std::vector<std::vector<Edge>> edges_;

    // Stable logical ID -> movable physical slot.
    std::vector<std::uint32_t> id_to_slot_;
    NodeId next_id_{};

    std::vector<std::vector<NodeId>> buckets_;
    std::vector<std::vector<NodeId>> hot_buckets_;
    std::vector<NodeId> previous_route_;
    std::deque<std::uint32_t> history_;

    std::uint64_t total_steps_{};
    std::uint64_t total_candidates_{};
    std::uint64_t total_active_{};
    std::uint64_t total_created_{};
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
                                              std::size_t prune_interval = 0);

[[nodiscard]] std::string to_json(const ExperimentResult& result);

} // namespace sbm

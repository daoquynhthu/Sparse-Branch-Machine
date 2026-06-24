#include "sbm/machine.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace sbm {

bool SparseBranchMachine::uses_sparse_token_output() const noexcept {
    return config_.objective == ObjectiveKind::TokenCrossEntropy &&
           config_.sparse_token_output;
}

SparseBranchMachine::SparseBranchMachine(Config config)
    : config_(config),
      rng_state_(config.seed),
      prediction_buffer_(config.objective == ObjectiveKind::TokenCrossEntropy &&
                                 config.sparse_token_output ? 0U : config.vector_dim,
                             0.0F),
      logit_buffer_(config.objective == ObjectiveKind::TokenCrossEntropy &&
                          config.sparse_token_output ? 0U : config.vector_dim,
                    0.0F),
      zero_output_buffer_(config.objective == ObjectiveKind::TokenCrossEntropy &&
                                config.sparse_token_output ? 0U : config.vector_dim,
                          0.0F),
      channel_output_buffer_(config.objective == ObjectiveKind::TokenCrossEntropy &&
                                   config.sparse_token_output ? 0U :
                             static_cast<std::size_t>(config.max_address_channels) *
                                 config.vector_dim,
                             0.0F),
      channel_credit_buffer_(config.max_address_channels, 0.0F),
      signature_buffer_(config.max_address_channels, 0U) {
    if (config.token_alphabet == 0 || config.vector_dim == 0) {
        throw std::invalid_argument("token_alphabet and vector_dim must be positive");
    }
    if (config.bucket_bits == 0 || config.bucket_bits > 20) {
        throw std::invalid_argument("bucket_bits must be in [1,20]");
    }
    if (config.address_lags.empty() || config.max_address_channels == 0U ||
        config.max_address_channels > kMaxAddressChannels ||
        config.address_lags.size() > config.max_address_channels ||
        config.beam_width < (config.adaptive_topology
            ? config.max_address_channels
            : config.address_lags.size()) ||
        config.bucket_scan_limit == 0 || config.max_edges_per_node == 0 ||
        config.max_specializations_per_bucket == 0) {
        throw std::invalid_argument("invalid routing budget");
    }
    if (config.address_lags.front() == 0U || config.topology_max_lag == 0U ||
        config.topology_max_arity == 0U ||
        config.topology_max_arity > kMaxAddressProgramArity) {
        throw std::invalid_argument("invalid address-program configuration");
    }
    if (!std::all_of(config.address_lags.begin(), config.address_lags.end(),
                     [](std::uint32_t lag) { return lag > 0U; })) {
        throw std::invalid_argument("address lags must be positive");
    }
    for (std::size_t left = 0; left < config.address_lags.size(); ++left) {
        for (std::size_t right = left + 1U; right < config.address_lags.size(); ++right) {
            if (config.address_lags[left] == config.address_lags[right]) {
                throw std::invalid_argument("address lags must be unique");
            }
        }
    }
    if (config.topology_probe_steps == 0U ||
        config.topology_validation_steps == 0U ||
        config.topology_validation_steps > config.topology_probe_steps ||
        config.topology_min_observations == 0U ||
        config.topology_min_observations > config.topology_validation_steps ||
        config.topology_credit_decay < 0.0F || config.topology_credit_decay >= 1.0F ||
        config.exact_region_mass < 0.0F || config.exact_region_mass > 1.0F ||
        config.residual_channel_gain <= 0.0F ||
        config.residual_recency_pseudocount < 0.0F) {
        throw std::invalid_argument("invalid mixture or residual configuration");
    }
    if (config.softmax_temperature <= 0.0F ||
        config.label_smoothing < 0.0F || config.label_smoothing >= 1.0F ||
        config.classification_learning_rate <= 0.0F ||
        config.classification_mature_learning_rate <= 0.0F ||
        config.logit_decay < 0.0F || config.logit_decay >= 1.0F) {
        throw std::invalid_argument("invalid token-objective configuration");
    }
    if (config.max_sparse_decisions_per_node < 8U ||
        config.max_sparse_decisions_per_node > 4096U) {
        throw std::invalid_argument("max_sparse_decisions_per_node must be in [8,4096]");
    }
    history_.reserve(config.context_width);
    const std::size_t candidate_capacity =
        static_cast<std::size_t>(config.max_address_channels) * config.bucket_scan_limit +
        static_cast<std::size_t>(config.beam_width) * config.edge_scan_limit;
    candidate_scratch_.reserve(candidate_capacity);
    scored_scratch_.reserve(candidate_capacity);
    selected_scratch_.reserve(config.beam_width);
    chosen_scratch_.reserve(candidate_capacity);
    order_scratch_.reserve(candidate_capacity);
    bucket_node_scratch_.reserve(config.bucket_scan_limit);
    topology_.reserve(config.max_address_channels);
    proposed_program_keys_.reserve(config.max_address_channels * 4U);
    for (std::size_t index = 0; index < config.address_lags.size(); ++index) {
        const auto program = singleton_address_program(config.address_lags[index]);
        topology_.push_back({program,
                             index == 0U ? ChannelPhase::Seed : ChannelPhase::Active,
                             0.0F, 0.0, 0U, 0U});
        proposed_program_keys_.push_back(address_program_key(program));
    }
    next_probe_step_ = config.topology_probe_interval;
    if (uses_sparse_token_output()) {
        if (config_.vector_dim < 2U) {
            throw std::invalid_argument("hierarchical token output requires vocabulary >= 2");
        }
        implicit_output_.emplace(config_.vector_dim, config_.output_tree_seed);
        global_output_total_.assign(config_.vector_dim - 1U, 0U);
        global_output_right_.assign(config_.vector_dim - 1U, 0U);
        global_output_logit_cache_.assign(config_.vector_dim - 1U, 0.0F);
        token_path_scratch_.reserve(
            static_cast<std::size_t>(std::bit_width(config_.vector_dim - 1U)) + 1U);
    }
}

void SparseBranchMachine::reset_sequence() {
    history_.clear();
    previous_route_.clear();
    previous_responsibilities_.clear();
    trace_.clear();
}

std::uint32_t SparseBranchMachine::bucket(std::uint64_t signature) const noexcept {
    return static_cast<std::uint32_t>(signature >> (64U - config_.bucket_bits));
}

std::size_t SparseBranchMachine::bucket_index(std::uint8_t channel,
                                              std::uint64_t signature) const noexcept {
    const std::size_t bucket_count = std::size_t{1} << config_.bucket_bits;
    return static_cast<std::size_t>(channel) * bucket_count + bucket(signature);
}

const SparseBranchMachine::BucketState*
SparseBranchMachine::find_bucket(std::size_t index) const noexcept {
    const auto found = bucket_directory_.find(index);
    return found == bucket_directory_.end() ? nullptr : &found->second;
}

SparseBranchMachine::BucketState& SparseBranchMachine::ensure_bucket(std::size_t index) {
    return bucket_directory_.try_emplace(index).first->second;
}

std::span<const NodeId> SparseBranchMachine::bounded_bucket_nodes(
    std::size_t index, std::size_t limit) {
    bucket_node_scratch_.clear();
    limit = std::min<std::size_t>(limit, config_.bucket_scan_limit);
    auto found = bucket_directory_.find(index);
    if (found == bucket_directory_.end() || limit == 0U) return {};
    auto& state = found->second;
    std::size_t inspected = 0U;
    const auto append = [&](NodeId id) {
        if (!contains(id)) {
            ++stale_bucket_refs_skipped_;
            return;
        }
        if (std::find(bucket_node_scratch_.begin(), bucket_node_scratch_.end(), id) !=
                bucket_node_scratch_.end()) {
            return;
        }
        bucket_node_scratch_.push_back(id);
    };
    for (const auto id : state.hot) {
        if (inspected >= limit) break;
        ++inspected;
        append(id);
    }
    if (inspected < limit) {
        const auto attempts = std::min<std::size_t>(
            state.cold.size(), limit - inspected);
        for (std::size_t offset = 0U; offset < attempts; ++offset) {
            append(state.cold[offset]);
        }
        inspected += attempts;
    }
    max_bucket_candidates_inspected_ = std::max<std::uint64_t>(
        max_bucket_candidates_inspected_, inspected);
    return {bucket_node_scratch_.data(), bucket_node_scratch_.size()};
}

void SparseBranchMachine::mark_hot(NodeId id) {
    const auto slot = slot_of(id);
    if (slot == SIZE_MAX) return;
    auto& state = ensure_bucket(bucket_index(channels_[slot], prototypes_[slot]));
    if (std::find(state.hot.begin(), state.hot.end(), id) != state.hot.end()) {
        hot_indexed_[slot] = 1U;
        return;
    }
    const auto limit = static_cast<std::size_t>(config_.bucket_scan_limit);
    if (state.hot.size() < limit) {
        state.hot.push_back(id);
        hot_indexed_[slot] = 1U;
        return;
    }
    const auto weaker = [&](NodeId left, NodeId right) {
        const auto left_slot = slot_of(left);
        const auto right_slot = slot_of(right);
        if (left_slot == SIZE_MAX) return true;
        if (right_slot == SIZE_MAX) return false;
        if (utility_ema_[left_slot] != utility_ema_[right_slot]) {
            return utility_ema_[left_slot] < utility_ema_[right_slot];
        }
        if (visits_[left_slot] != visits_[right_slot]) {
            return visits_[left_slot] < visits_[right_slot];
        }
        return left > right;
    };
    const auto victim = std::min_element(state.hot.begin(), state.hot.end(), weaker);
    if (victim == state.hot.end() || !weaker(*victim, id)) return;
    const auto victim_slot = slot_of(*victim);
    if (victim_slot != SIZE_MAX) hot_indexed_[victim_slot] = 0U;
    *victim = id;
    hot_indexed_[slot] = 1U;
}

std::size_t SparseBranchMachine::slot_of(NodeId id) const noexcept {
    if (id >= id_to_slot_.size()) return SIZE_MAX;
    const auto slot = id_to_slot_[id];
    return (slot == UINT32_MAX || slot >= ids_.size() || ids_[slot] != id) ? SIZE_MAX : slot;
}

bool SparseBranchMachine::contains(NodeId id) const noexcept {
    return slot_of(id) != SIZE_MAX;
}

NodeId SparseBranchMachine::new_node(std::uint64_t signature,
                                     std::span<const float> initial,
                                     NodeId parent,
                                     std::uint8_t channel) {
    if (next_id_ == kInvalidNode) throw std::overflow_error("node ids exhausted");
    if (channel >= topology_.size() || !channel_enabled(channel)) {
        throw std::out_of_range("invalid address channel");
    }
    const NodeId id = next_id_++;
    const auto slot = static_cast<std::uint32_t>(ids_.size());
    if (id_to_slot_.size() <= id) {
        id_to_slot_.resize(static_cast<std::size_t>(id) + 1, UINT32_MAX);
    }
    id_to_slot_[id] = slot;
    ids_.push_back(id);
    prototypes_.push_back(signature);
    visits_.push_back(0);
    address_visits_.push_back(0);
    utility_ema_.push_back(0);
    loss_ema_.push_back(1);
    address_loss_ema_.push_back(1);
    phases_.push_back(static_cast<std::uint8_t>(NodePhase::Cold));
    channels_.push_back(channel);
    hot_indexed_.push_back(0);
    parents_.push_back(parent);
    if (!uses_sparse_token_output()) {
        output_vectors_.insert(output_vectors_.end(), initial.begin(), initial.end());
    }
    sparse_outputs_.emplace_back();
    sparse_output_evicted_masks_.push_back(0U);
    edges_.emplace_back();
    edges_.back().reserve(config_.max_edges_per_node);
    auto& bucket_state = ensure_bucket(bucket_index(channel, signature));
    bucket_state.residents.push_back(id);
    if (bucket_state.cold.size() < config_.bucket_scan_limit) {
        bucket_state.cold.push_back(id);
    }
    ++total_created_;
    return id;
}

} // namespace sbm

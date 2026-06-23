#include "sbm/machine.hpp"

#include <algorithm>
#include <stdexcept>

namespace sbm {

SparseBranchMachine::SparseBranchMachine(Config config)
    : config_(config),
      rng_state_(config.seed),
      buckets_(static_cast<std::size_t>(config.max_address_channels) * (std::size_t{1} << config.bucket_bits)),
      hot_buckets_(static_cast<std::size_t>(config.max_address_channels) * (std::size_t{1} << config.bucket_bits)),
      bucket_last_split_step_(static_cast<std::size_t>(config.max_address_channels) *
                              (std::size_t{1} << config.bucket_bits), 0),
      prediction_buffer_(config.vector_dim, 0.0F),
      logit_buffer_(config.vector_dim, 0.0F) {
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
    if (config.address_lags.front() == 0U || config.topology_max_lag == 0U) {
        throw std::invalid_argument("address lags must be positive");
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
    topology_.reserve(config.max_address_channels);
    for (std::size_t index = 0; index < config.address_lags.size(); ++index) {
        topology_.push_back({config.address_lags[index],
                             index == 0U ? ChannelPhase::Seed : ChannelPhase::Active,
                             0.0F, 0.0, 0U, 0U});
    }
    next_probe_step_ = config.topology_probe_interval;
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
    output_vectors_.insert(output_vectors_.end(), initial.begin(), initial.end());
    edges_.emplace_back();
    edges_.back().reserve(config_.max_edges_per_node);
    buckets_[bucket_index(channel, signature)].push_back(id);
    ++total_created_;
    return id;
}

} // namespace sbm

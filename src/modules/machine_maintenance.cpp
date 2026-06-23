#include "sbm/machine.hpp"

#include "sbm/detail/random.hpp"
#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <cmath>

namespace sbm {

void SparseBranchMachine::erase_slot(std::size_t slot) {
    const NodeId victim = ids_[slot];
    const std::size_t last = ids_.size() - 1U;
    if (slot != last) {
        ids_[slot] = ids_[last];
        prototypes_[slot] = prototypes_[last];
        visits_[slot] = visits_[last];
        address_visits_[slot] = address_visits_[last];
        utility_ema_[slot] = utility_ema_[last];
        loss_ema_[slot] = loss_ema_[last];
        address_loss_ema_[slot] = address_loss_ema_[last];
        phases_[slot] = phases_[last];
        channels_[slot] = channels_[last];
        hot_indexed_[slot] = hot_indexed_[last];
        parents_[slot] = parents_[last];
        if (!uses_sparse_token_output()) {
            std::copy_n(output_vectors_.data() + last * config_.vector_dim,
                        config_.vector_dim,
                        output_vectors_.data() + slot * config_.vector_dim);
        }
        sparse_outputs_[slot] = std::move(sparse_outputs_[last]);
        edges_[slot] = std::move(edges_[last]);
        id_to_slot_[ids_[slot]] = static_cast<std::uint32_t>(slot);
    }
    ids_.pop_back();
    prototypes_.pop_back();
    visits_.pop_back();
    address_visits_.pop_back();
    utility_ema_.pop_back();
    loss_ema_.pop_back();
    address_loss_ema_.pop_back();
    phases_.pop_back();
    channels_.pop_back();
    hot_indexed_.pop_back();
    parents_.pop_back();
    if (!uses_sparse_token_output()) {
        output_vectors_.resize(ids_.size() * config_.vector_dim);
    }
    sparse_outputs_.pop_back();
    edges_.pop_back();
    id_to_slot_[victim] = UINT32_MAX;
}

std::size_t SparseBranchMachine::prune(std::uint32_t min_visits,
                                       float utility_threshold) {
    std::size_t pruned = 0;
    for (std::size_t slot = ids_.size(); slot-- > 0;) {
        if (visits_[slot] < min_visits && utility_ema_[slot] < utility_threshold) {
            erase_slot(slot);
            ++pruned;
        }
    }
    total_pruned_ += pruned;
    if (pruned != 0) rebuild_indexes();
    return pruned;
}

double SparseBranchMachine::output_distance(std::size_t a,
                                             std::size_t b) const noexcept {
    if (!uses_sparse_token_output()) {
        return std::sqrt(detail::squared_distance(
            output_vectors_.data() + a * config_.vector_dim,
            output_vectors_.data() + b * config_.vector_dim,
            config_.vector_dim) / static_cast<float>(config_.vector_dim));
    }
    const auto& left = sparse_outputs_[a];
    const auto& right = sparse_outputs_[b];
    std::size_t i = 0U;
    std::size_t j = 0U;
    double squared = 0.0;
    std::size_t count = 0U;
    while (i < left.size() || j < right.size()) {
        if (j >= right.size() || (i < left.size() && left[i].decision < right[j].decision)) {
            squared += static_cast<double>(left[i].logit) * left[i].logit;
            ++i;
        } else if (i >= left.size() || right[j].decision < left[i].decision) {
            squared += static_cast<double>(right[j].logit) * right[j].logit;
            ++j;
        } else {
            const double delta = static_cast<double>(left[i].logit) - right[j].logit;
            squared += delta * delta;
            ++i;
            ++j;
        }
        ++count;
    }
    return count == 0U ? 0.0 : std::sqrt(squared / static_cast<double>(count));
}

void SparseBranchMachine::absorb_node(std::size_t survivor_slot,
                                      std::size_t victim_slot) {
    const float survivor_weight = static_cast<float>(std::max(1U, visits_[survivor_slot]));
    const float victim_weight = static_cast<float>(std::max(1U, visits_[victim_slot]));
    const float inverse = 1.0F / (survivor_weight + victim_weight);
    if (!uses_sparse_token_output()) {
        float* survivor = output_vectors_.data() + survivor_slot * config_.vector_dim;
        const float* victim = output_vectors_.data() + victim_slot * config_.vector_dim;
        for (std::uint32_t component = 0; component < config_.vector_dim; ++component) {
            survivor[component] = (survivor_weight * survivor[component] +
                                   victim_weight * victim[component]) * inverse;
        }
    } else {
        for (const auto& entry : sparse_outputs_[victim_slot]) {
            float& destination = mutable_sparse_logit(survivor_slot, entry.decision);
            destination = (survivor_weight * destination +
                           victim_weight * entry.logit) * inverse;
        }
    }
    visits_[survivor_slot] += visits_[victim_slot];
    address_visits_[survivor_slot] += address_visits_[victim_slot];
    utility_ema_[survivor_slot] = (survivor_weight * utility_ema_[survivor_slot] +
        victim_weight * utility_ema_[victim_slot]) * inverse;
    loss_ema_[survivor_slot] = (survivor_weight * loss_ema_[survivor_slot] +
        victim_weight * loss_ema_[victim_slot]) * inverse;
    address_loss_ema_[survivor_slot] =
        (survivor_weight * address_loss_ema_[survivor_slot] +
         victim_weight * address_loss_ema_[victim_slot]) * inverse;
    for (const auto& edge : edges_[victim_slot]) {
        reinforce_edge(ids_[survivor_slot], edge.dst, edge.weight);
    }
    const NodeId victim_id = ids_[victim_slot];
    const NodeId survivor_id = ids_[survivor_slot];
    for (auto& list : edges_) {
        for (auto& edge : list) {
            if (edge.dst == victim_id) edge.dst = survivor_id;
        }
    }
    for (auto& id : previous_route_) if (id == victim_id) id = survivor_id;
    for (auto& frame : trace_) {
        for (auto& id : frame.route) if (id == victim_id) id = survivor_id;
    }
    erase_slot(victim_slot);
}

std::size_t SparseBranchMachine::merge_redundant(std::size_t max_merges) {
    std::size_t merged = 0;
    for (std::size_t a = 0; a < ids_.size() && merged < max_merges; ++a) {
        for (std::size_t b = a + 1U; b < ids_.size() && merged < max_merges; ++b) {
            if (channels_[a] != channels_[b]) continue;
            if (hamming_similarity(prototypes_[a], prototypes_[b]) <
                config_.merge_similarity) continue;
            if (output_distance(a, b) > config_.merge_vector_distance) continue;
            if (visits_[b] > visits_[a]) absorb_node(b, a);
            else absorb_node(a, b);
            ++merged;
            break;
        }
    }
    total_merged_ += merged;
    if (merged != 0) rebuild_indexes();
    return merged;
}

void SparseBranchMachine::rebuild_indexes() {
    for (auto& entries : buckets_) entries.clear();
    for (auto& entries : hot_buckets_) entries.clear();
    for (std::size_t slot = 0; slot < ids_.size(); ++slot) {
        const auto index = bucket_index(channels_[slot], prototypes_[slot]);
        buckets_[index].push_back(ids_[slot]);
        if (hot_indexed_[slot]) hot_buckets_[index].push_back(ids_[slot]);
    }
    for (auto& list : edges_) {
        list.erase(std::remove_if(list.begin(), list.end(), [&](const Edge& edge) {
            return !contains(edge.dst);
        }), list.end());
    }
}

void SparseBranchMachine::prefill_distractors(std::size_t count) {
    std::vector<float> value(uses_sparse_token_output() ? 0U : config_.vector_dim);
    for (std::size_t index = 0; index < count; ++index) {
        for (auto& component : value) {
            const auto sample = static_cast<int>(detail::next_random(rng_state_) % 2001U) - 1000;
            component = static_cast<float>(sample) / 1000.0F;
        }
        std::vector<std::uint8_t> enabled_channels;
        for (std::uint8_t channel = 0; channel < topology_.size(); ++channel) {
            if (channel_enabled(channel)) enabled_channels.push_back(channel);
        }
        if (enabled_channels.empty()) break;
        const auto channel = enabled_channels[static_cast<std::size_t>(
            detail::next_random(rng_state_) % enabled_channels.size())];
        (void)new_node(detail::next_random(rng_state_), value, kInvalidNode, channel);
    }
}

Diagnostics SparseBranchMachine::diagnostics() const noexcept {
    std::uint64_t edge_count = 0;
    std::uint64_t edge_capacity = 0;
    std::uint64_t bucket_capacity = 0;
    std::uint64_t cold = 0;
    std::uint64_t warm = 0;
    std::uint64_t mature = 0;
    std::uint64_t dormant = 0;
    std::uint64_t anchor = 0;
    std::uint64_t residual = 0;
    for (const auto& list : edges_) {
        edge_count += list.size();
        edge_capacity += list.capacity();
    }
    for (const auto& entries : buckets_) bucket_capacity += entries.capacity();
    for (const auto& entries : hot_buckets_) bucket_capacity += entries.capacity();
    for (std::size_t slot = 0; slot < ids_.size(); ++slot) {
        switch (phase_of_slot(slot)) {
            case NodePhase::Cold: ++cold; break;
            case NodePhase::Warm: ++warm; break;
            case NodePhase::Mature: ++mature; break;
            case NodePhase::Dormant: ++dormant; break;
        }
        if (channels_[slot] == 0U) ++anchor;
        else ++residual;
    }
    std::uint64_t sparse_entries = 0U;
    std::uint64_t sparse_capacity = 0U;
    std::uint64_t max_sparse_entries = 0U;
    for (const auto& entries : sparse_outputs_) {
        sparse_entries += entries.size();
        sparse_capacity += entries.capacity();
        max_sparse_entries = std::max<std::uint64_t>(max_sparse_entries, entries.size());
    }
    const std::uint64_t address_index_bytes =
        buckets_.capacity() * sizeof(std::vector<NodeId>) +
        hot_buckets_.capacity() * sizeof(std::vector<NodeId>) +
        bucket_last_split_step_.capacity() * sizeof(std::uint64_t) +
        bucket_capacity * sizeof(NodeId);
    const std::uint64_t output_structure_bytes =
        output_tree_.capacity() * sizeof(OutputTreeNode) +
        token_path_offsets_.capacity() * sizeof(std::uint32_t) +
        token_path_steps_.capacity() * sizeof(TokenPathStep);
    const auto denominator = std::max<std::uint64_t>(1, total_steps_);
    const std::uint64_t bytes =
        ids_.capacity() * sizeof(NodeId) +
        prototypes_.capacity() * sizeof(std::uint64_t) +
        visits_.capacity() * sizeof(std::uint32_t) +
        address_visits_.capacity() * sizeof(std::uint32_t) +
        utility_ema_.capacity() * sizeof(float) +
        loss_ema_.capacity() * sizeof(float) +
        address_loss_ema_.capacity() * sizeof(float) +
        phases_.capacity() * sizeof(std::uint8_t) +
        channels_.capacity() * sizeof(std::uint8_t) +
        hot_indexed_.capacity() * sizeof(std::uint8_t) +
        parents_.capacity() * sizeof(NodeId) +
        output_vectors_.capacity() * sizeof(float) +
        sparse_capacity * sizeof(SparseOutputEntry) +
        id_to_slot_.capacity() * sizeof(std::uint32_t) +
        edge_capacity * sizeof(Edge) +
        address_index_bytes + output_structure_bytes;
    std::uint64_t seed_channels = 0U;
    std::uint64_t probe_channels = 0U;
    std::uint64_t active_channels = 0U;
    std::uint64_t retired_channels = 0U;
    for (const auto& state : topology_) {
        switch (state.phase) {
            case ChannelPhase::Seed: ++seed_channels; break;
            case ChannelPhase::Probe: ++probe_channels; break;
            case ChannelPhase::Active: ++active_channels; break;
            case ChannelPhase::Retired: ++retired_channels; break;
        }
    }
    Diagnostics result{total_steps_, ids_.size(), next_id_, edge_count,
            static_cast<double>(total_active_) / static_cast<double>(denominator),
            static_cast<double>(total_candidates_) / static_cast<double>(denominator),
            total_created_, total_merged_, total_pruned_, cold, warm, mature, dormant,
            anchor, residual, stale_bucket_refs_skipped_, stale_edge_refs_skipped_,
            bytes, sparse_entries, topology_proposals_, topology_accepted_, topology_rejected_,
            topology_pruned_, seed_channels, probe_channels, active_channels, retired_channels,
            simd_available()};
    result.address_index_bytes = address_index_bytes;
    result.output_structure_bytes = output_structure_bytes;
    result.max_bucket_candidates_inspected = max_bucket_candidates_inspected_;
    result.max_sparse_entries_per_node = max_sparse_entries;
    return result;
}

} // namespace sbm

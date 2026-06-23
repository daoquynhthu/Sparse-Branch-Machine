#include "sbm/machine.hpp"

#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sbm {

bool SparseBranchMachine::push_candidate(std::vector<CandidateNode>& values,
                                         NodeId id,
                                         float edge_prior) const noexcept {
    if (!contains(id)) return false;
    const auto found = std::find_if(values.begin(), values.end(),
                                    [&](const CandidateNode& candidate) {
                                        return candidate.id == id;
                                    });
    if (found != values.end()) {
        if (edge_prior > found->edge_prior) found->edge_prior = edge_prior;
        return false;
    }
    values.push_back({id, edge_prior});
    return true;
}

std::span<const SparseBranchMachine::CandidateNode>
SparseBranchMachine::candidate_ids(std::span<const std::uint64_t> signatures) {
    const std::size_t channel_count = topology_.size();
    const std::size_t exact_budget = channel_count * config_.bucket_scan_limit;
    const std::size_t hard_limit = exact_budget +
        static_cast<std::size_t>(config_.beam_width) * config_.edge_scan_limit;
    auto& output = candidate_scratch_;
    output.clear();

    auto append_bucket = [&](std::size_t bucket_id, std::size_t target_size) {
        if (output.size() >= target_size) return;
        const auto candidates = bounded_bucket_nodes(
            bucket_id, target_size - output.size());
        for (const NodeId id : candidates) {
            (void)push_candidate(output, id, 0.0F);
        }
    };

    // Each temporal view receives its own exact-address budget.  A large or
    // noisy view therefore cannot displace the other views.
    for (std::uint8_t channel = 0; channel < channel_count; ++channel) {
        if (!channel_enabled(channel)) continue;
        const std::size_t target_size = output.size() + config_.bucket_scan_limit;
        const std::size_t index = bucket_index(channel, signatures[channel]);
        append_bucket(index, target_size);
    }

    // Learned control-flow edges add alternatives after all exact views have
    // been represented.
    for (const NodeId source : previous_route_) {
        const auto slot = slot_of(source);
        if (slot == SIZE_MAX) continue;
        const auto limit = std::min<std::size_t>(edges_[slot].size(),
                                                config_.edge_scan_limit);
        for (std::size_t index = 0;
             index < limit && output.size() < hard_limit;
             ++index) {
            const auto& edge = edges_[slot][index];
            if (!contains(edge.dst)) {
                ++stale_edge_refs_skipped_;
                continue;
            }
            const float prior = edge.weight /
                (1.0F + std::max(0.0F, edge.weight));
            (void)push_candidate(output, edge.dst, prior);
        }
    }

    // Neighboring address regions are exploration fallback only.  Probe each
    // temporal view independently to retain channel identity.
    const auto raw_bucket_count = static_cast<std::int64_t>(
        std::size_t{1} << config_.bucket_bits);
    for (std::int64_t radius = 1;
         output.size() < hard_limit && radius <= 2;
         ++radius) {
        for (std::uint8_t channel = 0; channel < channel_count; ++channel) {
            if (!channel_enabled(channel)) continue;
            const auto base = static_cast<std::int64_t>(bucket(signatures[channel]));
            for (const auto probe : {base - radius, base + radius}) {
                if (probe < 0 || probe >= raw_bucket_count) continue;
                const std::size_t flattened =
                    static_cast<std::size_t>(channel) *
                    static_cast<std::size_t>(raw_bucket_count) +
                    static_cast<std::size_t>(probe);
                append_bucket(flattened, hard_limit);
                if (output.size() >= hard_limit) break;
            }
        }
    }
    return std::span<const CandidateNode>(output.data(), output.size());
}

NodePhase SparseBranchMachine::phase_of_slot(std::size_t slot) const noexcept {
    return static_cast<NodePhase>(phases_[slot]);
}

NodePhase SparseBranchMachine::phase(NodeId id) const noexcept {
    const auto slot = slot_of(id);
    return slot == SIZE_MAX ? NodePhase::Dormant : phase_of_slot(slot);
}

double SparseBranchMachine::score(std::size_t slot,
                                  std::span<const std::uint64_t> signatures,
                                  float edge_prior) const noexcept {
    const auto channel = channels_[slot];
    const auto signature = signatures[channel];
    const double similarity = hamming_similarity(prototypes_[slot], signature);
    const double reliability = 1.0 / (1.0 + loss_ema_[slot]);
    const double novelty = 1.0 /
        std::sqrt(static_cast<double>(visits_[slot]) + 1.0);
    const double exact = bucket(prototypes_[slot]) == bucket(signature) ? 0.42 : 0.0;
    const double phase_bias = phase_of_slot(slot) == NodePhase::Dormant ? -0.18 : 0.0;
    return exact + 0.72 * similarity + 0.10 * reliability + 0.02 * novelty +
           static_cast<double>(config_.edge_score_weight * edge_prior) + phase_bias;
}

std::pair<std::span<SparseBranchMachine::ScoredNode>, std::uint32_t>
SparseBranchMachine::select_route(std::span<const std::uint64_t> signatures) {
    const auto candidates = candidate_ids(signatures);
    auto& scored = scored_scratch_;
    scored.clear();
    for (const auto& candidate : candidates) {
        const auto slot = slot_of(candidate.id);
        if (slot == SIZE_MAX) continue;
        const auto channel = channels_[slot];
        const bool exact = bucket(prototypes_[slot]) == bucket(signatures[channel]);
        scored.push_back({score(slot, signatures, candidate.edge_prior), candidate.id,
                          0.0F, 0.0F, exact, channel});
    }

    auto& selected = selected_scratch_;
    selected.clear();
    chosen_scratch_.assign(scored.size(), 0U);
    auto& chosen = chosen_scratch_;

    // Guarantee one exact resident from every available temporal view.
    for (std::uint8_t channel = 0; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        std::size_t best = SIZE_MAX;
        for (std::size_t index = 0; index < scored.size(); ++index) {
            if (!scored[index].exact_region || scored[index].channel != channel) continue;
            if (best == SIZE_MAX || scored[index].score > scored[best].score) best = index;
        }
        if (best != SIZE_MAX && selected.size() < config_.beam_width) {
            selected.push_back(scored[best]);
            chosen[best] = 1U;
        }
    }

    auto& order = order_scratch_;
    order.clear();
    for (std::size_t index = 0; index < scored.size(); ++index) {
        if (!chosen[index]) order.push_back(index);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (scored[a].score == scored[b].score) return scored[a].id > scored[b].id;
        return scored[a].score > scored[b].score;
    });
    for (const auto index : order) {
        if (selected.size() >= config_.beam_width) break;
        selected.push_back(scored[index]);
    }

    assign_responsibilities(std::span<ScoredNode>(selected.data(), selected.size()));
    return {std::span<ScoredNode>(selected.data(), selected.size()),
            static_cast<std::uint32_t>(candidates.size())};
}

void SparseBranchMachine::assign_responsibilities(std::span<ScoredNode> active) const {
    for (auto& node : active) node.responsibility = 0.0F;
    for (std::uint8_t channel = 0; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        const float channel_mass = channel == 0U ? 1.0F : config_.residual_channel_gain;
        bool has_exact = false;
        bool has_non_exact = false;
        for (const auto& node : active) {
            if (node.channel != channel) continue;
            has_exact = has_exact || node.exact_region;
            has_non_exact = has_non_exact || !node.exact_region;
        }
        if (!has_exact && !has_non_exact) continue;

        const auto normalize_group = [&](bool exact, float mass) {
            if (mass <= 0.0F) return;
            double max_score = -std::numeric_limits<double>::infinity();
            for (const auto& node : active) {
                if (node.channel == channel && node.exact_region == exact) {
                    max_score = std::max(max_score, node.score);
                }
            }
            if (!std::isfinite(max_score)) return;
            double sum = 0.0;
            for (auto& node : active) {
                if (node.channel != channel || node.exact_region != exact) continue;
                node.responsibility = static_cast<float>(std::exp(
                    static_cast<double>(config_.responsibility_temperature) *
                    (node.score - max_score)));
                sum += node.responsibility;
            }
            if (sum <= 0.0) return;
            for (auto& node : active) {
                if (node.channel == channel && node.exact_region == exact) {
                    node.responsibility *= mass / static_cast<float>(sum);
                }
            }
        };

        if (has_exact && has_non_exact) {
            const float exact_fraction = std::clamp(config_.exact_region_mass, 0.0F, 1.0F);
            normalize_group(true, channel_mass * exact_fraction);
            normalize_group(false, channel_mass * (1.0F - exact_fraction));
        } else if (has_exact) {
            normalize_group(true, channel_mass);
        } else {
            normalize_group(false, channel_mass);
        }
    }
}

void SparseBranchMachine::aggregate(std::span<const ScoredNode> active,
                                    std::span<float> output) const {
    std::fill(output.begin(), output.end(), 0.0F);
    for (const auto& node : active) {
        const auto slot = slot_of(node.id);
        if (slot == SIZE_MAX || node.responsibility == 0.0F) continue;
        detail::axpy(output.data(),
                     output_vectors_.data() + slot * config_.vector_dim,
                     node.responsibility,
                     config_.vector_dim);
    }
}

void SparseBranchMachine::compute_counterfactual_contributions(
    std::span<ScoredNode> active,
    std::span<const float> target,
    float target_energy,
    float full_loss) const {
    std::vector<float> without(prediction_buffer_.begin(), prediction_buffer_.end());
    for (auto& node : active) {
        const auto slot = slot_of(node.id);
        if (slot == SIZE_MAX || std::abs(node.responsibility) < 1e-6F) {
            node.contribution = 0.0F;
            continue;
        }
        std::copy(prediction_buffer_.begin(), prediction_buffer_.end(), without.begin());
        detail::axpy(without.data(),
                     output_vectors_.data() + slot * config_.vector_dim,
                     -node.responsibility,
                     config_.vector_dim);
        const float mse_without = detail::squared_distance(
            without.data(), target.data(), config_.vector_dim) /
            static_cast<float>(config_.vector_dim);
        node.contribution = mse_without / std::max(target_energy, 1e-5F) - full_loss;
    }
}

} // namespace sbm

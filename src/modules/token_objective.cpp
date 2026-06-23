#include "sbm/machine.hpp"

#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sbm {
namespace {

void softmax(std::span<const float> logits, std::span<float> probabilities,
             float temperature) {
    const float inverse_temperature = 1.0F / temperature;
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float value : logits) maximum = std::max(maximum, value * inverse_temperature);
    double normalizer = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] * inverse_temperature - maximum);
        normalizer += probabilities[index];
    }
    const float inverse = static_cast<float>(1.0 / std::max(normalizer, 1e-30));
    for (auto& value : probabilities) value *= inverse;
}

float centered_logit_mean(std::span<const float> logits) {
    double total = 0.0;
    for (const float value : logits) total += value;
    return static_cast<float>(total / static_cast<double>(logits.size()));
}

} // namespace

StepStats SparseBranchMachine::step_token(std::uint32_t token,
                                          std::uint32_t target_token,
                                          bool learn) {
    if (config_.objective != ObjectiveKind::TokenCrossEntropy) {
        throw std::logic_error("step_token requires the token cross-entropy objective");
    }
    if (token >= config_.token_alphabet || target_token >= config_.vector_dim) {
        throw std::out_of_range("invalid input or target token");
    }

    history_.push_back(token);
    if (history_.size() > config_.context_width) history_.pop_front();
    const std::vector<std::uint32_t> window(history_.begin(), history_.end());
    std::array<std::uint64_t, kAddressChannelCount> signatures{};
    for (std::size_t channel = 0; channel < signatures.size(); ++channel) {
        signatures[channel] = lagged_token_signature(
            window, config_.token_alphabet, config_.address_lags[channel]);
    }

    const bool can_grow = learn || config_.allow_growth_when_frozen;
    std::uint32_t created = 0U;
    std::vector<float> zero_logits(config_.vector_dim, 0.0F);

    // Address allocation depends only on the observed context, never on the
    // target token.  New residents therefore make a uniform prediction on
    // their first use instead of leaking the label into the current metric.
    for (std::uint8_t channel = 0; channel < signatures.size(); ++channel) {
        const auto signature = signatures[channel];
        const auto exact_bucket = bucket_index(channel, signature);
        std::size_t exact_count = 0U;
        NodeId nearest_exact = kInvalidNode;
        double nearest_similarity = -1.0;
        float nearest_persistent_loss = std::numeric_limits<float>::infinity();
        std::uint32_t nearest_visits = 0U;
        for (const NodeId id : buckets_[exact_bucket]) {
            const auto slot = slot_of(id);
            if (slot == SIZE_MAX || channels_[slot] != channel) continue;
            ++exact_count;
            const double similarity = hamming_similarity(prototypes_[slot], signature);
            if (similarity > nearest_similarity) {
                nearest_similarity = similarity;
                nearest_exact = id;
                nearest_visits = address_visits_[slot];
                nearest_persistent_loss = address_loss_ema_[slot];
            }
        }
        const bool cooldown_ready = total_steps_ >=
            bucket_last_split_step_[exact_bucket] + config_.split_cooldown;
        const bool persistent_conflict = nearest_exact != kInvalidNode &&
            nearest_visits >= config_.split_min_visits &&
            exact_count < config_.max_specializations_per_bucket &&
            cooldown_ready &&
            nearest_similarity < config_.split_context_similarity &&
            nearest_persistent_loss > config_.split_loss_threshold;

        if (can_grow && (exact_count == 0U || persistent_conflict)) {
            const NodeId id = new_node(signature, zero_logits, nearest_exact, channel);
            const auto slot = slot_of(id);
            if (slot != SIZE_MAX) {
                loss_ema_[slot] = 1.0F;
                address_loss_ema_[slot] = 1.0F;
                utility_ema_[slot] = 0.0F;
                hot_buckets_[exact_bucket].push_back(id);
                hot_indexed_[slot] = 1U;
            }
            bucket_last_split_step_[exact_bucket] = total_steps_;
            ++created;
        }
    }

    auto [active, examined] = select_route(signatures);
    aggregate(active, logit_buffer_);
    softmax(logit_buffer_, prediction_buffer_, config_.softmax_temperature);

    const float target_probability = std::max(prediction_buffer_[target_token], 1e-12F);
    const float cross_entropy = -std::log(target_probability);
    const float normalized_loss = cross_entropy /
        std::max(std::log(static_cast<float>(config_.vector_dim)), 1e-5F);
    const auto predicted = static_cast<std::uint32_t>(std::distance(
        prediction_buffer_.begin(),
        std::max_element(prediction_buffer_.begin(), prediction_buffer_.end())));

    // Counterfactual contribution is the increase in cross-entropy when a
    // node's logit contribution is removed from the active route.
    std::vector<float> without_logits(logit_buffer_.size(), 0.0F);
    std::vector<float> without_probabilities(prediction_buffer_.size(), 0.0F);
    for (auto& node : active) {
        const auto slot = slot_of(node.id);
        if (slot == SIZE_MAX || std::abs(node.responsibility) < 1e-7F) {
            node.contribution = 0.0F;
            continue;
        }
        std::copy(logit_buffer_.begin(), logit_buffer_.end(), without_logits.begin());
        detail::axpy(without_logits.data(),
                     output_vectors_.data() + slot * config_.vector_dim,
                     -node.responsibility,
                     config_.vector_dim);
        softmax(without_logits, without_probabilities, config_.softmax_temperature);
        const float without_loss = -std::log(
            std::max(without_probabilities[target_token], 1e-12F));
        node.contribution = without_loss - cross_entropy;
    }

    std::vector<NodeId> route;
    std::vector<float> contributions;
    route.reserve(active.size());
    contributions.reserve(active.size());
    for (const auto& node : active) {
        route.push_back(node.id);
        contributions.push_back(node.contribution);
    }

    if (learn) {
        apply_trace_credit(normalized_loss);
        const float smoothing = config_.label_smoothing;
        const float off_target = config_.vector_dim > 1U
            ? smoothing / static_cast<float>(config_.vector_dim - 1U)
            : 0.0F;

        for (const auto& node : active) {
            if (!node.exact_region ||
                node.responsibility < config_.min_update_responsibility) {
                continue;
            }
            const auto slot = slot_of(node.id);
            if (slot == SIZE_MAX) continue;
            if (visits_[slot] == 0U && !hot_indexed_[slot]) {
                hot_buckets_[bucket_index(channels_[slot], prototypes_[slot])]
                    .push_back(ids_[slot]);
                hot_indexed_[slot] = 1U;
            }
            ++visits_[slot];
            ++address_visits_[slot];
            const bool mature = phase_of_slot(slot) == NodePhase::Mature;
            const float base_rate = mature
                ? config_.classification_mature_learning_rate
                : config_.classification_learning_rate;
            const float schedule = 1.0F /
                std::sqrt(static_cast<float>(std::max(1U, address_visits_[slot])));
            const float rate = base_rate * schedule * node.responsibility /
                std::max(config_.softmax_temperature, 1e-5F);
            float* logits = output_vectors_.data() + slot * config_.vector_dim;
            const float retain = 1.0F - config_.logit_decay;
            for (std::uint32_t candidate = 0; candidate < config_.vector_dim; ++candidate) {
                const float target_value = candidate == target_token
                    ? 1.0F - smoothing
                    : off_target;
                logits[candidate] = retain * logits[candidate] +
                    rate * (target_value - prediction_buffer_[candidate]);
            }
            // Softmax is invariant to a shared offset.  Re-centering prevents
            // numerically irrelevant drift from accumulating in long runs.
            const float mean = centered_logit_mean(
                std::span<const float>(logits, config_.vector_dim));
            for (std::uint32_t candidate = 0; candidate < config_.vector_dim; ++candidate) {
                logits[candidate] -= mean;
            }
            loss_ema_[slot] = 0.96F * loss_ema_[slot] + 0.04F * normalized_loss;
            address_loss_ema_[slot] = 0.96F * address_loss_ema_[slot] +
                                      0.04F * normalized_loss;
            utility_ema_[slot] = 0.985F * utility_ema_[slot] +
                0.015F * std::clamp(node.contribution, -1.0F, 1.0F);
            update_phase(slot);
        }

        const std::size_t source_limit = std::min<std::size_t>(
            previous_route_.size(), config_.edge_reinforce_width);
        const std::size_t destination_limit = std::min<std::size_t>(
            active.size(), config_.edge_reinforce_width);
        for (std::size_t source_index = 0; source_index < source_limit; ++source_index) {
            decay_edges(previous_route_[source_index]);
            const float source_responsibility = source_index < previous_responsibilities_.size()
                ? previous_responsibilities_[source_index]
                : 1.0F;
            for (std::size_t destination_index = 0;
                 destination_index < destination_limit;
                 ++destination_index) {
                const float helpful = std::max(0.0F, active[destination_index].contribution);
                if (helpful < config_.edge_min_contribution) continue;
                reinforce_edge(previous_route_[source_index],
                               active[destination_index].id,
                               source_responsibility *
                                   active[destination_index].responsibility * helpful);
            }
        }
        trace_.push_back({route, contributions, normalized_loss});
        while (trace_.size() > config_.trace_horizon) trace_.pop_front();
    }

    previous_route_ = route;
    previous_responsibilities_.clear();
    previous_responsibilities_.reserve(active.size());
    for (const auto& node : active) {
        previous_responsibilities_.push_back(node.responsibility);
    }
    ++total_steps_;
    total_candidates_ += examined;
    total_active_ += route.size();

    StepStats stats;
    stats.normalized_mse = normalized_loss;
    stats.cosine = 0.0F;
    stats.active_nodes = static_cast<std::uint32_t>(route.size());
    stats.candidates_examined = examined;
    stats.live_nodes = static_cast<std::uint32_t>(ids_.size());
    stats.created = created;
    stats.route = std::move(route);
    stats.cross_entropy = cross_entropy;
    stats.target_probability = target_probability;
    stats.top1_correct = predicted == target_token;
    return stats;
}

} // namespace sbm

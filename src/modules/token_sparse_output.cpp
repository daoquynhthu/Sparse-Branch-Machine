#include "sbm/machine.hpp"
#include "sbm/math.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbm {
namespace {

[[nodiscard]] float softplus(float value) noexcept {
    if (value > 20.0F) return value;
    if (value < -20.0F) return std::exp(value);
    return std::log1p(std::exp(value));
}

[[nodiscard]] float branch_loss(float normalized_logit, bool right) noexcept {
    return right ? softplus(-normalized_logit) : softplus(normalized_logit);
}

[[nodiscard]] float log_branch_probability(float normalized_logit,
                                           bool right) noexcept {
    return -branch_loss(normalized_logit, right);
}

[[nodiscard]] float sigmoid(float value) noexcept {
    if (value >= 0.0F) {
        const float inverse = std::exp(-value);
        return 1.0F / (1.0F + inverse);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

} // namespace

float SparseBranchMachine::sparse_logit(std::size_t slot,
                                        std::uint32_t decision) const noexcept {
    if (slot >= sparse_outputs_.size()) return 0.0F;
    const auto& entries = sparse_outputs_[slot];
    const auto found = std::lower_bound(
        entries.begin(), entries.end(), decision,
        [](const SparseOutputEntry& entry, std::uint32_t value) {
            return entry.decision < value;
        });
    return found != entries.end() && found->decision == decision
        ? found->logit : 0.0F;
}

float& SparseBranchMachine::mutable_sparse_logit(std::size_t slot,
                                                  std::uint32_t decision) {
    auto& entries = sparse_outputs_.at(slot);
    auto found = std::lower_bound(
        entries.begin(), entries.end(), decision,
        [](const SparseOutputEntry& entry, std::uint32_t value) {
            return entry.decision < value;
        });
    if (found != entries.end() && found->decision == decision) {
        return found->logit;
    }
    if (entries.size() >= config_.max_sparse_decisions_per_node) {
        const auto victim = std::min_element(
            entries.begin(), entries.end(),
            [](const SparseOutputEntry& left, const SparseOutputEntry& right) {
                const float left_magnitude = std::abs(left.logit);
                const float right_magnitude = std::abs(right.logit);
                return left_magnitude != right_magnitude
                    ? left_magnitude < right_magnitude
                    : left.decision > right.decision;
            });
        entries.erase(victim);
        found = std::lower_bound(
            entries.begin(), entries.end(), decision,
            [](const SparseOutputEntry& entry, std::uint32_t value) {
                return entry.decision < value;
            });
    }
    return entries.insert(found, {decision, 0.0F})->logit;
}

float SparseBranchMachine::aggregate_sparse_logit(
    std::span<const ScoredNode> active,
    std::uint32_t decision) const noexcept {
    float value = 0.0F;
    for (const auto& node : active) {
        if (std::abs(node.responsibility) < 1e-8F) continue;
        const auto slot = slot_of(node.id);
        if (slot == SIZE_MAX) continue;
        value += node.responsibility * sparse_logit(slot, decision);
    }
    return value;
}

float SparseBranchMachine::global_output_logit(
    std::uint32_t decision) const noexcept {
    return decision < global_output_logit_cache_.size()
        ? global_output_logit_cache_[decision] : 0.0F;
}

void SparseBranchMachine::observe_global_output_path(
    std::span<const detail::ImplicitDecision> path) {
    if (global_output_prior_updates_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("global output prior update count overflow");
    }
    for (const auto& step : path) {
        auto& total = global_output_total_.at(step.id);
        auto& right = global_output_right_.at(step.id);
        if (total == std::numeric_limits<std::uint64_t>::max() ||
            (step.right && right == std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error("global output prior branch count overflow");
        }
        ++total;
        if (step.right) ++right;
        constexpr double pseudocount = 0.5;
        const double left = static_cast<double>(total - right);
        global_output_logit_cache_[step.id] = static_cast<float>(std::log(
            (static_cast<double>(right) + pseudocount) / (left + pseudocount)));
    }
    ++global_output_prior_updates_;
}

StepStats SparseBranchMachine::step_token(std::uint32_t token,
                                          std::uint32_t target_token,
                                          bool learn) {
    return uses_sparse_token_output()
        ? step_token_sparse(token, target_token, learn)
        : step_token_dense(token, target_token, learn);
}

StepStats SparseBranchMachine::step_token_sparse(std::uint32_t token,
                                                 std::uint32_t target_token,
                                                 bool learn) {
    if (config_.objective != ObjectiveKind::TokenCrossEntropy) {
        throw std::logic_error("step_token requires the token cross-entropy objective");
    }
    if (token >= config_.token_alphabet || target_token >= config_.vector_dim) {
        throw std::out_of_range("invalid input or target token");
    }

    if (history_.size() == config_.context_width) {
        std::move(history_.begin() + 1, history_.end(), history_.begin());
        history_.back() = token;
    } else {
        history_.push_back(token);
    }
    maybe_begin_topology_probe(learn);
    const auto signatures = make_signatures(history_);

    const bool can_grow = learn || config_.allow_growth_when_frozen;
    std::uint32_t created = 0U;
    const std::span<const float> empty_initial;
    for (std::uint8_t channel = 0; channel < signatures.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        const auto signature = signatures[channel];
        const auto exact_bucket = bucket_index(channel, signature);
        const auto* bucket_state = find_bucket(exact_bucket);
        const std::size_t exact_count = bucket_state == nullptr
            ? 0U : bucket_state->residents.size();
        NodeId nearest_exact = kInvalidNode;
        double nearest_similarity = -1.0;
        float nearest_persistent_loss = std::numeric_limits<float>::infinity();
        std::uint32_t nearest_visits = 0U;
        for (const NodeId id : bounded_bucket_nodes(
                 exact_bucket, config_.bucket_scan_limit)) {
            const auto slot = slot_of(id);
            if (slot == SIZE_MAX || channels_[slot] != channel) continue;
            const double similarity = hamming_similarity(prototypes_[slot], signature);
            if (similarity > nearest_similarity) {
                nearest_similarity = similarity;
                nearest_exact = id;
                nearest_visits = address_visits_[slot];
                nearest_persistent_loss = address_loss_ema_[slot];
            }
        }
        const bool cooldown_ready = total_steps_ >=
            (bucket_state == nullptr ? 0U : bucket_state->last_split_step) +
                config_.split_cooldown;
        const bool persistent_conflict = nearest_exact != kInvalidNode &&
            nearest_visits >= config_.split_min_visits &&
            exact_count < config_.max_specializations_per_bucket &&
            cooldown_ready && nearest_similarity < config_.split_context_similarity &&
            nearest_persistent_loss > config_.split_loss_threshold;
        if (can_grow && channel_learning_enabled(channel) &&
            (exact_count == 0U || persistent_conflict)) {
            const NodeId id = new_node(signature, empty_initial, nearest_exact, channel);
            const auto slot = slot_of(id);
            if (slot != SIZE_MAX) {
                loss_ema_[slot] = 1.0F;
                address_loss_ema_[slot] = 1.0F;
                utility_ema_[slot] = 0.0F;
                mark_hot(id);
            }
            ensure_bucket(exact_bucket).last_split_step = total_steps_;
            ++created;
        }
    }

    auto [active, examined] = select_route(signatures);
    auto& output_tree = *implicit_output_;
    output_tree.target_path(target_token, token_path_scratch_);
    const float temperature = std::max(config_.softmax_temperature, 1e-5F);
    std::vector<float> path_logits;
    path_logits.reserve(token_path_scratch_.size());
    float cross_entropy = 0.0F;
    for (const auto& step : token_path_scratch_) {
        const float logit = global_output_logit(step.id) +
            aggregate_sparse_logit(active, step.id);
        path_logits.push_back(logit);
        cross_entropy += branch_loss(logit / temperature, step.right);
    }
    const float target_probability = std::exp(-std::min(cross_entropy, 80.0F));
    const float normalized_loss = cross_entropy /
        std::max(std::log(static_cast<float>(config_.vector_dim)), 1e-5F);

    struct SearchItem {
        float log_probability{};
        std::uint32_t lo{};
        std::uint32_t hi{};
    };
    std::vector<SearchItem> frontier{{0.0F, 0U, config_.vector_dim}};
    const auto decoder_beam = std::max(config_.sparse_output_topk,
                                       config_.sparse_output_beam_width);
    const auto maximum_depth = static_cast<std::uint32_t>(
        std::bit_width(config_.vector_dim - 1U) + 2U);
    for (std::uint32_t depth = 0U; depth < maximum_depth; ++depth) {
        bool all_leaves = true;
        std::vector<SearchItem> expanded;
        expanded.reserve(frontier.size() * 2U);
        for (const auto& item : frontier) {
            if (item.hi - item.lo == 1U) {
                expanded.push_back(item);
                continue;
            }
            all_leaves = false;
            const auto decision = output_tree.split(item.lo, item.hi);
            const float logit = (global_output_logit(decision.decision_id) +
                aggregate_sparse_logit(active, decision.decision_id)) / temperature;
            expanded.push_back({item.log_probability +
                                    log_branch_probability(logit, false),
                                item.lo, decision.middle});
            expanded.push_back({item.log_probability +
                                    log_branch_probability(logit, true),
                                decision.middle, item.hi});
        }
        if (all_leaves) break;
        const auto keep = std::min<std::size_t>(decoder_beam, expanded.size());
        std::partial_sort(expanded.begin(), expanded.begin() + keep, expanded.end(),
                          [](const SearchItem& left, const SearchItem& right) {
                              return left.log_probability > right.log_probability;
                          });
        expanded.resize(keep);
        frontier = std::move(expanded);
    }
    std::sort(frontier.begin(), frontier.end(),
              [](const SearchItem& left, const SearchItem& right) {
                  return left.log_probability > right.log_probability;
              });
    std::vector<std::uint32_t> top_tokens;
    top_tokens.reserve(config_.sparse_output_topk);
    for (const auto& item : frontier) {
        if (item.hi - item.lo != 1U) continue;
        top_tokens.push_back(output_tree.token_from_rank(item.lo));
        if (top_tokens.size() >= config_.sparse_output_topk) break;
    }

    const auto predicted = top_tokens.empty() ? 0U : top_tokens.front();
    const bool top5 = std::find(top_tokens.begin(), top_tokens.end(), target_token) !=
                      top_tokens.end();

    if (learn) {
        std::fill(channel_credit_buffer_.begin(), channel_credit_buffer_.end(), 0.0F);
        std::array<std::uint8_t, kMaxAddressChannels> channel_members{};
        for (const auto& node : active) {
            if (node.channel < channel_members.size() &&
                std::abs(node.responsibility) >= 1e-7F) {
                ++channel_members[node.channel];
            }
        }

        for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
            if (!channel_enabled(channel) || channel_members[channel] == 0U) continue;
            float without_loss = 0.0F;
            std::size_t path_position = 0U;
            for (const auto& step : token_path_scratch_) {
                float removed = 0.0F;
                for (const auto& node : active) {
                    if (node.channel != channel) continue;
                    const auto slot = slot_of(node.id);
                    if (slot == SIZE_MAX) continue;
                    removed += node.responsibility * sparse_logit(slot, step.id);
                }
                without_loss += branch_loss(
                    (path_logits[path_position] - removed) / temperature, step.right);
                ++path_position;
            }
            channel_credit_buffer_[channel] = without_loss - cross_entropy;
        }

        for (auto& node : active) {
            const auto slot = slot_of(node.id);
            if (slot == SIZE_MAX || std::abs(node.responsibility) < 1e-7F) {
                node.contribution = 0.0F;
                continue;
            }
            if (channel_members[node.channel] == 1U) {
                node.contribution = channel_credit_buffer_[node.channel];
                continue;
            }
            float without_loss = 0.0F;
            std::size_t path_position = 0U;
            for (const auto& step : token_path_scratch_) {
                const float removed = node.responsibility *
                    sparse_logit(slot, step.id);
                without_loss += branch_loss(
                    (path_logits[path_position] - removed) / temperature, step.right);
                ++path_position;
            }
            node.contribution = without_loss - cross_entropy;
        }
        observe_topology_credit(std::span<const float>(channel_credit_buffer_.data(),
                                                        topology_.size()));
    }

    std::vector<NodeId> route;
    std::vector<float> contributions;
    route.reserve(active.size());
    contributions.reserve(active.size());
    for (const auto& node : active) {
        route.push_back(node.id);
        contributions.push_back(node.contribution);
    }

    if (learn) observe_global_output_path(token_path_scratch_);

    if (learn) {
        apply_trace_credit(normalized_loss);
        for (const auto& node : active) {
            if (!node.exact_region ||
                node.responsibility < config_.min_update_responsibility) continue;
            const auto slot = slot_of(node.id);
            if (slot == SIZE_MAX || !channel_learning_enabled(node.channel)) continue;
            if (visits_[slot] == 0U && !hot_indexed_[slot]) {
                mark_hot(ids_[slot]);
            }
            ++visits_[slot];
            ++address_visits_[slot];
            const bool mature = phase_of_slot(slot) == NodePhase::Mature;
            const float base_rate = mature
                ? config_.classification_mature_learning_rate
                : config_.classification_learning_rate;
            const float schedule = 1.0F /
                std::sqrt(static_cast<float>(std::max(1U, address_visits_[slot])));
            const float rate = base_rate * schedule * node.responsibility / temperature;
            std::size_t path_position = 0U;
            for (const auto& step : token_path_scratch_) {
                const float probability_right = sigmoid(path_logits[path_position] / temperature);
                const float target_right = step.right
                    ? 1.0F - 0.5F * config_.label_smoothing
                    : 0.5F * config_.label_smoothing;
                float& local = mutable_sparse_logit(slot, step.id);
                local = (1.0F - config_.logit_decay) * local +
                        rate * (target_right - probability_right);
                ++path_position;
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
                ? previous_responsibilities_[source_index] : 1.0F;
            for (std::size_t destination_index = 0;
                 destination_index < destination_limit; ++destination_index) {
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
    stats.active_nodes = static_cast<std::uint32_t>(route.size());
    stats.candidates_examined = examined;
    stats.live_nodes = static_cast<std::uint32_t>(ids_.size());
    stats.created = created;
    stats.route = std::move(route);
    stats.cross_entropy = cross_entropy;
    stats.target_probability = target_probability;
    stats.top1_correct = predicted == target_token;
    stats.top5_correct = top5;
    stats.predicted_token = predicted;
    return stats;
}

} // namespace sbm

#include "sbm/machine.hpp"

#include "sbm/detail/address_interpreter.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace sbm {
namespace {

std::optional<AddressProgram> proposal_at(std::uint64_t ordinal,
                                          std::uint32_t max_lag,
                                          std::uint32_t max_arity,
                                          bool enable_delta,
                                          bool enable_content_match) {
    for (std::uint32_t outer = 2U; outer <= max_lag; ++outer) {
        const auto visit = [&](AddressOp op) -> std::optional<AddressProgram> {
            AddressProgram singleton = singleton_address_program(outer);
            singleton.op = op;
            if (ordinal == 0U) return singleton;
            --ordinal;
            if (max_arity < 2U) return std::nullopt;
            for (std::uint32_t inner = 1U; inner < outer; ++inner) {
                AddressProgram program;
                program.lags[0] = inner;
                program.lags[1] = outer;
                program.arity = 2U;
                program.op = op;
                if (ordinal == 0U) return program;
                --ordinal;
            }
            return std::nullopt;
        };
        if (auto value = visit(AddressOp::Tuple); value.has_value()) return value;
        if (enable_delta) {
            if (auto value = visit(AddressOp::DeltaMod); value.has_value()) return value;
        }
        if (enable_content_match) {
            AddressProgram content = singleton_address_program(outer);
            content.op = AddressOp::ContentMatch;
            if (ordinal == 0U) return content;
            --ordinal;
            if (max_arity >= 2U) {
                for (std::uint32_t inner = 1U; inner < outer; ++inner) {
                    AddressProgram follow;
                    follow.lags[0] = inner;
                    follow.lags[1] = outer;
                    follow.arity = 2U;
                    follow.op = AddressOp::ContentFollow;
                    if (ordinal == 0U) return follow;
                    --ordinal;
                }
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] double program_description_cost(const AddressProgram& program) noexcept {
    return 1.0 + static_cast<double>(program.arity);
}

[[nodiscard]] double program_execution_cost(const AddressProgram& program) noexcept {
    if (program.op == AddressOp::ContentMatch || program.op == AddressOp::ContentFollow) {
        return program.arity == 0U ? 1.0 :
            1.0 + static_cast<double>(program.lags[program.arity - 1U]);
    }
    return 1.0 + static_cast<double>(program.arity);
}

} // namespace

bool SparseBranchMachine::channel_enabled(std::size_t channel) const noexcept {
    if (channel >= topology_.size()) return false;
    const auto phase = topology_[channel].phase;
    return phase == ChannelPhase::Seed || phase == ChannelPhase::Probe ||
           phase == ChannelPhase::Active;
}

bool SparseBranchMachine::channel_learning_enabled(std::size_t channel) const noexcept {
    if (!channel_enabled(channel)) return false;
    const auto& state = topology_[channel];
    if (state.phase != ChannelPhase::Probe) return true;
    const auto age = total_steps_ - state.born_step;
    const auto validation = std::min(config_.topology_validation_steps,
                                     config_.topology_probe_steps);
    return age + validation < config_.topology_probe_steps;
}

std::span<const AddressExecutionFrame> SparseBranchMachine::execute_address_programs(
    std::span<const std::uint32_t> window) {
    execution_frames_.clear();
    std::fill(signature_buffer_.begin(), signature_buffer_.end(), 0U);
    for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        const auto& program = topology_[channel].program;
        AddressExecutionFrame frame{};
        const auto seed = config_.seed ^ mix64(address_program_key(program) +
                                              0x9E3779B97F4A7C15ULL);
        const bool matched = execute_address_program(
            window, config_.token_alphabet, program, seed, frame);
        signature_buffer_[channel] = frame.signature;
        execution_frames_.push_back(frame);
        ++address_execution_frames_;
        if (matched) ++address_binding_hits_;
        else ++address_binding_misses_;
        structural_description_cost_ += static_cast<double>(frame.description_cost);
        structural_execution_cost_ += static_cast<double>(frame.execution_cost);
    }
    return std::span<const AddressExecutionFrame>(
        execution_frames_.data(), execution_frames_.size());
}

std::span<const std::uint64_t> SparseBranchMachine::make_signatures(
    std::span<const std::uint32_t> window) {
    if (config_.address_execution_mode == AddressExecutionMode::LegacySignature) {
        std::fill(signature_buffer_.begin(), signature_buffer_.end(), 0U);
        for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
            if (!channel_enabled(channel)) continue;
            const auto& program = topology_[channel].program;
            signature_buffer_[channel] = address_program_signature(
                window, config_.token_alphabet,
                std::span<const std::uint32_t>(program.lags.data(), program.arity),
                program.op,
                config_.seed ^ mix64(address_program_key(program) +
                                     0x9E3779B97F4A7C15ULL));
        }
        return std::span<const std::uint64_t>(
            signature_buffer_.data(), topology_.size());
    }
    (void)execute_address_programs(window);
    return std::span<const std::uint64_t>(signature_buffer_.data(), topology_.size());
}

void SparseBranchMachine::quarantine_channel(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return;
    topology_[channel].phase = ChannelPhase::Quarantined;
}

void SparseBranchMachine::recoverably_retire_channel(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return;
    topology_[channel].phase = ChannelPhase::RecoverableRetired;
}

void SparseBranchMachine::physically_erase_channel(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return;
    topology_[channel].phase = ChannelPhase::Retired;

    for (std::size_t slot = ids_.size(); slot-- > 0;) {
        if (channels_[slot] == channel) erase_slot(slot);
    }
    std::vector<NodeId> filtered_route;
    std::vector<float> filtered_responsibility;
    filtered_route.reserve(previous_route_.size());
    filtered_responsibility.reserve(previous_responsibilities_.size());
    for (std::size_t index = 0; index < previous_route_.size(); ++index) {
        const auto slot = slot_of(previous_route_[index]);
        if (slot == SIZE_MAX || channels_[slot] == channel) continue;
        filtered_route.push_back(previous_route_[index]);
        filtered_responsibility.push_back(index < previous_responsibilities_.size()
            ? previous_responsibilities_[index] : 1.0F);
    }
    previous_route_ = std::move(filtered_route);
    previous_responsibilities_ = std::move(filtered_responsibility);
    trace_.clear();
    rebuild_indexes();
}

void SparseBranchMachine::maybe_finalize_topology_probe() {
    if (!config_.adaptive_topology) return;
    for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (state.phase != ChannelPhase::Probe) continue;
        const auto age = total_steps_ - state.born_step;
        if (age < config_.topology_probe_steps ||
            state.observations < config_.topology_min_observations) {
            return;
        }
        const double mean_credit = state.credit_sum /
            static_cast<double>(std::max<std::uint64_t>(1U, state.observations));
        const double description_penalty =
            static_cast<double>(config_.structural_description_cost_weight) *
            program_description_cost(state.program) /
            static_cast<double>(std::max<std::uint64_t>(1U, state.observations));
        const double execution_penalty =
            static_cast<double>(config_.structural_execution_cost_weight) *
            program_execution_cost(state.program);
        const double structural_value =
            mean_credit - description_penalty - execution_penalty;
        const double decision_value = config_.topology_accept_uses_structural_value
            ? structural_value : mean_credit;
        if (decision_value >= static_cast<double>(config_.topology_accept_credit)) {
            topology_events_.push_back({total_steps_, state.program,
                                        TopologyDecision::Accepted,
                                        static_cast<float>(structural_value)});
            state.phase = ChannelPhase::Active;
            state.born_step = total_steps_;
            state.observations = 0U;
            state.credit_sum = 0.0;
            ++topology_accepted_;
        } else {
            topology_events_.push_back({total_steps_, state.program,
                                        TopologyDecision::Rejected,
                                        static_cast<float>(structural_value)});
            physically_erase_channel(channel);
            ++topology_rejected_;
        }
        next_probe_step_ = total_steps_ + config_.topology_probe_interval;
        return;
    }
    for (std::size_t channel = 1U; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (state.phase != ChannelPhase::Active) continue;
        const auto age = total_steps_ - state.born_step;
        if (age < config_.topology_prune_patience ||
            state.observations < config_.topology_prune_patience ||
            state.credit_ema >= config_.topology_prune_credit) {
            continue;
        }
        if (config_.accepted_channel_retirement ==
            AcceptedChannelRetirement::Preserve) {
            continue;
        }
        topology_events_.push_back({total_steps_, state.program,
                                    TopologyDecision::Pruned,
                                    state.credit_ema});
        if (config_.accepted_channel_retirement ==
            AcceptedChannelRetirement::Quarantine) {
            quarantine_channel(channel);
        } else {
            physically_erase_channel(channel);
        }
        ++topology_pruned_;
        next_probe_step_ = total_steps_ + config_.topology_probe_interval;
        return;
    }
}


void SparseBranchMachine::freeze_topology() {
    for (std::size_t channel = 1U; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (state.phase != ChannelPhase::Probe) continue;
        const double mean_credit = state.observations == 0U
            ? 0.0
            : state.credit_sum / static_cast<double>(state.observations);
        topology_events_.push_back({total_steps_, state.program,
                                    TopologyDecision::Rejected,
                                    static_cast<float>(mean_credit)});
        physically_erase_channel(channel);
        ++topology_rejected_;
    }
}

void SparseBranchMachine::maybe_begin_topology_probe(bool learn) {
    if (!learn) return;
    maybe_finalize_topology_probe();
    if (!config_.adaptive_topology || total_steps_ < next_probe_step_) return;

    for (const auto& state : topology_) {
        if (state.phase == ChannelPhase::Probe) return;
    }

    std::size_t enabled = 0U;
    for (const auto& state : topology_) {
        if (state.phase == ChannelPhase::Seed || state.phase == ChannelPhase::Active ||
            state.phase == ChannelPhase::Probe) {
            ++enabled;
        }
    }
    if (enabled >= config_.max_address_channels) return;

    std::optional<AddressProgram> proposal;
    while (true) {
        const auto candidate = proposal_at(proposal_cursor_++, config_.topology_max_lag,
                                           config_.topology_max_arity,
                                           config_.topology_enable_delta,
                                           config_.topology_enable_content_match);
        if (!candidate.has_value()) break;
        const auto key = address_program_key(*candidate);
        if (std::find(proposed_program_keys_.begin(), proposed_program_keys_.end(), key) ==
            proposed_program_keys_.end()) {
            proposal = candidate;
            proposed_program_keys_.push_back(key);
            break;
        }
    }
    if (!proposal.has_value()) return;

    std::size_t slot = topology_.size();
    for (std::size_t index = 1U; index < topology_.size(); ++index) {
        if (topology_[index].phase == ChannelPhase::Retired) {
            slot = index;
            break;
        }
    }
    if (slot == topology_.size()) {
        if (topology_.size() >= config_.max_address_channels) return;
        topology_.push_back({});
    }

    topology_[slot] = {*proposal, ChannelPhase::Probe, 0.0F, 0.0, 0U, total_steps_};
    topology_events_.push_back({total_steps_, *proposal,
                                TopologyDecision::Proposed, 0.0F});
    ++topology_proposals_;
    next_probe_step_ = total_steps_ + config_.topology_probe_steps;
}

void SparseBranchMachine::observe_topology_credit(
    std::span<const float> channel_credit) {
    if (!config_.adaptive_topology) return;
    for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (!channel_enabled(channel) || channel >= channel_credit.size()) continue;
        const float credit = channel_credit[channel];
        state.credit_ema = config_.topology_credit_decay * state.credit_ema +
            (1.0F - config_.topology_credit_decay) * credit;
        if (state.phase == ChannelPhase::Active) ++state.observations;
        if (state.phase == ChannelPhase::Probe) {
            const auto age = total_steps_ - state.born_step;
            const auto validation = std::min(config_.topology_validation_steps,
                                             config_.topology_probe_steps);
            const auto validation_start = config_.topology_probe_steps - validation;
            if (age >= validation_start && age >= config_.topology_probe_warmup) {
                state.credit_sum += static_cast<double>(credit);
                ++state.observations;
            }
        }
    }
}

std::vector<std::uint32_t> SparseBranchMachine::learned_address_lags() const {
    std::vector<std::uint32_t> result;
    for (const auto& state : topology_) {
        if ((state.phase == ChannelPhase::Seed || state.phase == ChannelPhase::Probe ||
             state.phase == ChannelPhase::Active) && state.program.arity == 1U) {
            result.push_back(state.program.lags[0]);
        }
    }
    return result;
}

std::vector<AddressProgram> SparseBranchMachine::learned_address_programs() const {
    std::vector<AddressProgram> result;
    for (const auto& state : topology_) {
        if (state.phase == ChannelPhase::Seed || state.phase == ChannelPhase::Probe ||
            state.phase == ChannelPhase::Active) {
            result.push_back(state.program);
        }
    }
    return result;
}

std::vector<float> SparseBranchMachine::learned_channel_credit() const {
    std::vector<float> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) result.push_back(state.credit_ema);
    return result;
}

std::vector<std::uint8_t> SparseBranchMachine::learned_channel_phase() const {
    std::vector<std::uint8_t> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) {
        result.push_back(static_cast<std::uint8_t>(state.phase));
    }
    return result;
}

} // namespace sbm

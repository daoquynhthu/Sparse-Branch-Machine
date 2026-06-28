#include "sbm/detail/address_interpreter.hpp"

#include "sbm/math.hpp"

#include <algorithm>

namespace sbm {
namespace {

[[nodiscard]] std::uint32_t safe_history_index(
    std::span<const std::uint32_t> window,
    std::uint32_t lag) noexcept {
    if (window.empty()) return 0U;
    if (window.size() > lag) {
        return static_cast<std::uint32_t>(window.size() - 1U - lag);
    }
    return 0U;
}

[[nodiscard]] AddressBindingKind binding_kind(AddressOp op) noexcept {
    switch (op) {
        case AddressOp::Tuple:
        case AddressOp::DeltaMod:
            return AddressBindingKind::Positional;
        case AddressOp::ContentMatch:
            return AddressBindingKind::ContentMatch;
        case AddressOp::ContentFollow:
            return AddressBindingKind::ContentFollow;
    }
    return AddressBindingKind::None;
}

} // namespace

bool execute_address_program(std::span<const std::uint32_t> window,
                             std::uint32_t alphabet,
                             const AddressProgram& program,
                             std::uint64_t seed,
                             AddressExecutionFrame& out) noexcept {
    out = {};
    out.program = program;
    out.binding = binding_kind(program.op);
    out.binding_state.pattern_terms = program.op == AddressOp::ContentFollow &&
        program.arity > 1U
        ? static_cast<std::uint8_t>(program.arity - 1U)
        : 0U;
    if (!window.empty()) {
        out.binding_state.current_token = window.back();
    }
    out.signature = address_program_signature(
        window, alphabet,
        std::span<const std::uint32_t>(program.lags.data(), program.arity),
        program.op, seed);
    out.description_cost = 1.0F + static_cast<float>(program.arity);
    out.execution_cost = 1.0F;
    if (window.empty() || program.arity == 0U) return false;

    if (program.op == AddressOp::Tuple || program.op == AddressOp::DeltaMod) {
        out.matched = true;
        out.binding_state.matched = true;
        out.source_index = safe_history_index(window, program.lags[program.arity - 1U]);
        out.matched_index = out.source_index;
        out.binding_state.matched_token = window[out.matched_index];
        out.binding_state.matched_successor = out.binding_state.matched_token;
        out.binding_state.matched_distance = program.lags[program.arity - 1U];
        out.binding_state.pattern_span = program.lags[program.arity - 1U];
        out.successor = out.binding_state.matched_successor;
        out.dependency = program.lags[program.arity - 1U];
        out.execution_cost += static_cast<float>(program.arity);
        return true;
    }

    const auto current_index = window.size() - 1U;
    const auto current = window[current_index];
    const std::uint32_t max_lag = program.lags[program.arity - 1U];
    const auto bounded_lag = std::min<std::size_t>(max_lag, current_index);
    const std::size_t pattern_count =
        program.op == AddressOp::ContentFollow && program.arity > 1U
            ? program.arity - 1U
            : 0U;

    for (std::size_t distance = 1U; distance <= bounded_lag; ++distance) {
        out.execution_cost += 1.0F;
        const auto candidate_index = current_index - distance;
        if (window[candidate_index] != current) continue;
        bool pattern_matches = true;
        for (std::size_t index = 0; index < pattern_count; ++index) {
            const auto lag = static_cast<std::size_t>(program.lags[index]);
            if (lag > current_index || lag > candidate_index ||
                window[current_index - lag] != window[candidate_index - lag]) {
                pattern_matches = false;
                break;
            }
        }
        if (!pattern_matches) continue;
        out.matched = true;
        out.binding_state.matched = true;
        out.source_index = static_cast<std::uint32_t>(candidate_index);
        out.matched_index = static_cast<std::uint32_t>(candidate_index);
        out.binding_state.matched_token = window[candidate_index];
        out.binding_state.matched_successor =
            window[std::min(candidate_index + 1U, current_index)];
        out.binding_state.matched_distance = static_cast<std::uint32_t>(distance);
        out.binding_state.pattern_span = max_lag;
        out.successor = out.binding_state.matched_successor;
        out.dependency = out.binding_state.matched_distance;
        return true;
    }
    return false;
}

} // namespace sbm

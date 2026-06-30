#include "sbm/detail/address_interpreter.hpp"

#include "sbm/math.hpp"

#include <algorithm>

namespace sbm {
namespace {

[[nodiscard]] AddressBindingKind binding_kind(AddressOp op) noexcept {
    switch (op) {
        case AddressOp::Tuple:
        case AddressOp::DeltaMod:
            return AddressBindingKind::Positional;
        case AddressOp::ContentMatch:
            return AddressBindingKind::ContentMatch;
        case AddressOp::ContentFollow:
        case AddressOp::ContentFollowMulti2:
        case AddressOp::ContentFollowMulti3:
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
    out.input_state = address_program_input_state(program);
    out.output_state = address_program_output_state(program);
    out.binding = binding_kind(program.op);
    out.required_dependency_binding =
        address_program_required_dependency_binding(program);
    out.binding_state = resolve_address_binding_state(window, program);
    out.signature = address_program_signature(
        window, alphabet,
        std::span<const std::uint32_t>(program.lags.data(), program.arity),
        program.op, seed);
    out.description_cost = 1.0F + static_cast<float>(program.arity);
    out.execution_cost = 1.0F;
    if (window.empty() || program.arity == 0U) return false;

    if (program.op == AddressOp::Tuple || program.op == AddressOp::DeltaMod) {
        out.matched = out.binding_state.matched;
        out.source_index = out.binding_state.matched_index;
        out.matched_index = out.source_index;
        out.successor = out.binding_state.matched_successor;
        out.dependency = out.binding_state.matched_distance;
        out.execution_cost += static_cast<float>(program.arity);
        return out.matched;
    }

    const std::uint8_t hops = content_follow_hop_count(program.op);
    const std::uint32_t max_lag = program.lags[program.arity - 1U];
    const auto bounded_lag = std::min<std::size_t>(max_lag, window.size() - 1U);
    out.execution_cost += static_cast<float>(hops * bounded_lag);
    out.matched = out.binding_state.matched;
    out.source_index = out.binding_state.matched_index;
    out.matched_index = out.binding_state.matched_index;
    out.successor = out.binding_state.matched_successor;
    out.dependency = out.binding_state.matched_distance;
    return out.matched;
}

} // namespace sbm

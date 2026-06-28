#pragma once

#include "sbm/types.hpp"

#include <cstdint>
#include <span>

namespace sbm {

[[nodiscard]] AddressBindingState resolve_address_binding_state(
    std::span<const std::uint32_t> window,
    const AddressProgram& program) noexcept;

[[nodiscard]] bool execute_address_program(
    std::span<const std::uint32_t> window,
    std::uint32_t alphabet,
    const AddressProgram& program,
    std::uint64_t seed,
    AddressExecutionFrame& out) noexcept;

} // namespace sbm

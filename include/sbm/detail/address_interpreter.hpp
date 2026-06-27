#pragma once

#include "sbm/types.hpp"

#include <cstdint>
#include <span>

namespace sbm {

[[nodiscard]] bool execute_address_program(
    std::span<const std::uint32_t> window,
    std::uint32_t alphabet,
    const AddressProgram& program,
    std::uint64_t seed,
    AddressExecutionFrame& out) noexcept;

} // namespace sbm

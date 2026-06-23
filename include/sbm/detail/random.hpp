#pragma once

#include "sbm/math.hpp"

#include <cstdint>

namespace sbm::detail {

inline std::uint64_t next_random(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    return mix64(state);
}

} // namespace sbm::detail

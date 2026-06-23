#pragma once

#include <cstdint>
#include <span>

namespace sbm {

[[nodiscard]] std::uint64_t mix64(std::uint64_t x) noexcept;
[[nodiscard]] std::uint64_t rolling_signature(
    std::span<const std::uint32_t> window,
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept;
[[nodiscard]] std::uint64_t token_context_signature(
    std::span<const std::uint32_t> window,
    std::uint32_t alphabet,
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept;
[[nodiscard]] double hamming_similarity(std::uint64_t a, std::uint64_t b) noexcept;
[[nodiscard]] bool simd_available() noexcept;

} // namespace sbm

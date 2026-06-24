#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace sbm::detail {

struct SparseOutputEntry {
    std::uint32_t decision{};
    float logit{};
    std::uint32_t visits{};
    float gain_ema{};
    std::uint64_t last_update_step{};
};

[[nodiscard]] float sparse_decision_learning_rate(
    const SparseOutputEntry& entry,
    float initial_rate,
    float mature_rate,
    std::uint32_t mature_visits) noexcept;

[[nodiscard]] std::size_t select_sparse_output_victim(
    std::span<const SparseOutputEntry> entries,
    std::uint64_t current_step);

} // namespace sbm::detail

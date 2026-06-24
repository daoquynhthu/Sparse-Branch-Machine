#include "sbm/detail/sparse_output.hpp"

#include <cmath>
#include <stdexcept>

namespace sbm::detail {
namespace {

constexpr std::uint64_t kProbationSteps = 32U;

[[nodiscard]] bool in_probation(const SparseOutputEntry& entry,
                                std::uint64_t current_step) noexcept {
    return current_step >= entry.last_update_step &&
           current_step - entry.last_update_step < kProbationSteps;
}

[[nodiscard]] double retention_score(const SparseOutputEntry& entry) noexcept {
    const double visits = static_cast<double>(entry.visits);
    return static_cast<double>(std::max(entry.gain_ema, 0.0F)) *
               std::sqrt(visits + 1.0) +
           1e-4 * std::log1p(visits);
}

} // namespace

float sparse_decision_learning_rate(const SparseOutputEntry& entry,
                                    float initial_rate,
                                    float mature_rate,
                                    std::uint32_t mature_visits) noexcept {
    const float base = entry.visits >= mature_visits ? mature_rate : initial_rate;
    return base / std::sqrt(static_cast<float>(entry.visits) + 1.0F);
}

std::size_t select_sparse_output_victim(
    std::span<const SparseOutputEntry> entries,
    std::uint64_t current_step) {
    if (entries.empty()) throw std::invalid_argument("cannot evict from empty entries");
    bool has_non_probation = false;
    for (const auto& entry : entries) {
        has_non_probation = has_non_probation || !in_probation(entry, current_step);
    }
    std::size_t victim = 0U;
    bool selected = false;
    for (std::size_t index = 0U; index < entries.size(); ++index) {
        const auto& candidate = entries[index];
        if (has_non_probation && in_probation(candidate, current_step)) continue;
        if (!selected) {
            victim = index;
            selected = true;
            continue;
        }
        const auto& current = entries[victim];
        const double candidate_score = retention_score(candidate);
        const double current_score = retention_score(current);
        if (candidate_score < current_score ||
            (candidate_score == current_score &&
             (candidate.last_update_step < current.last_update_step ||
              (candidate.last_update_step == current.last_update_step &&
               candidate.decision > current.decision)))) {
            victim = index;
        }
    }
    return victim;
}

} // namespace sbm::detail

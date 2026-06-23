#pragma once

#include <cstdint>
#include <vector>

namespace sbm::detail {

struct ImplicitDecision {
    std::uint32_t id{};
    bool right{};
};

struct ImplicitSplit {
    std::uint32_t middle{};
    std::uint32_t decision_id{};
};

class ImplicitOutputTree {
public:
    ImplicitOutputTree(std::uint32_t vocabulary, std::uint64_t seed);

    [[nodiscard]] std::uint32_t vocabulary() const noexcept { return vocabulary_; }
    [[nodiscard]] std::uint32_t rank_from_token(std::uint32_t token) const;
    [[nodiscard]] std::uint32_t token_from_rank(std::uint32_t rank) const;
    [[nodiscard]] ImplicitSplit split(std::uint32_t lo, std::uint32_t hi) const;
    void target_path(std::uint32_t token,
                     std::vector<ImplicitDecision>& output) const;

private:
    std::uint32_t vocabulary_{};
    std::uint32_t multiplier_{};
    std::uint32_t offset_{};
    std::uint32_t inverse_{};
};

} // namespace sbm::detail

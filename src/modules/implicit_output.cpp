#include "sbm/detail/implicit_output.hpp"

#include "sbm/math.hpp"

#include <numeric>
#include <stdexcept>

namespace sbm::detail {
namespace {

std::uint32_t modular_inverse(std::uint32_t value, std::uint32_t modulus) {
    std::int64_t previous = 0;
    std::int64_t current = 1;
    std::int64_t remainder = modulus;
    std::int64_t next_remainder = value;
    while (next_remainder != 0) {
        const std::int64_t quotient = remainder / next_remainder;
        const std::int64_t next = previous - quotient * current;
        previous = current;
        current = next;
        const std::int64_t reduced = remainder - quotient * next_remainder;
        remainder = next_remainder;
        next_remainder = reduced;
    }
    if (remainder != 1) throw std::invalid_argument("output permutation is not invertible");
    previous %= static_cast<std::int64_t>(modulus);
    if (previous < 0) previous += modulus;
    return static_cast<std::uint32_t>(previous);
}

} // namespace

ImplicitOutputTree::ImplicitOutputTree(std::uint32_t vocabulary,
                                       std::uint64_t seed)
    : vocabulary_(vocabulary) {
    if (vocabulary < 2U) {
        throw std::invalid_argument("implicit output requires vocabulary >= 2");
    }
    multiplier_ = static_cast<std::uint32_t>(
        mix64(seed ^ 0xA0761D6478BD642FULL) % vocabulary_);
    if (multiplier_ == 0U) multiplier_ = 1U;
    while (std::gcd(multiplier_, vocabulary_) != 1U) {
        ++multiplier_;
        if (multiplier_ == vocabulary_) multiplier_ = 1U;
    }
    offset_ = static_cast<std::uint32_t>(
        mix64(seed ^ 0xE7037ED1A0B428DBULL) % vocabulary_);
    inverse_ = modular_inverse(multiplier_, vocabulary_);
}

std::uint32_t ImplicitOutputTree::rank_from_token(std::uint32_t token) const {
    if (token >= vocabulary_) throw std::out_of_range("token is outside vocabulary");
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(multiplier_) * token + offset_) % vocabulary_);
}

std::uint32_t ImplicitOutputTree::token_from_rank(std::uint32_t rank) const {
    if (rank >= vocabulary_) throw std::out_of_range("rank is outside vocabulary");
    const auto shifted = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(rank) + vocabulary_ - offset_) % vocabulary_);
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(inverse_) * shifted % vocabulary_);
}

ImplicitSplit ImplicitOutputTree::split(std::uint32_t lo, std::uint32_t hi) const {
    if (lo >= hi || hi > vocabulary_ || hi - lo < 2U) {
        throw std::out_of_range("invalid implicit output interval");
    }
    const auto middle = lo + (hi - lo) / 2U;
    return {middle, middle - 1U};
}

std::uint64_t ImplicitOutputTree::region_prefix(
    std::uint32_t token, std::uint32_t depth) const {
    const auto rank = rank_from_token(token);
    std::uint64_t prefix = 0U;
    std::uint32_t lo = 0U;
    std::uint32_t hi = vocabulary_;
    for (std::uint32_t i = 0; i < depth && hi - lo > 1U; ++i) {
        const auto branch = split(lo, hi);
        const bool right = rank >= branch.middle;
        if (right) { prefix |= (1ULL << i); lo = branch.middle; }
        else { hi = branch.middle; }
    }
    return prefix;
}

void ImplicitOutputTree::target_path(
    std::uint32_t token,
    std::vector<ImplicitDecision>& output) const {
    output.clear();
    const auto rank = rank_from_token(token);
    std::uint32_t lo = 0U;
    std::uint32_t hi = vocabulary_;
    while (hi - lo > 1U) {
        const auto branch = split(lo, hi);
        const bool right = rank >= branch.middle;
        output.push_back({branch.decision_id, right});
        if (right) lo = branch.middle;
        else hi = branch.middle;
    }
}

} // namespace sbm::detail

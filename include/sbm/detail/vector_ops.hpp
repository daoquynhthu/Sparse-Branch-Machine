#pragma once

#include <cstddef>
#include <span>

namespace sbm::detail {

[[nodiscard]] float dot(const float* a, const float* b, std::size_t n) noexcept;
[[nodiscard]] float squared_distance(const float* a, const float* b, std::size_t n) noexcept;
void axpy(float* destination, const float* source, float alpha, std::size_t n) noexcept;
void lerp(float* destination, const float* target, float learning_rate, std::size_t n) noexcept;
[[nodiscard]] float cosine(std::span<const float> a, std::span<const float> b) noexcept;

} // namespace sbm::detail

#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sbm {
namespace {

float dot_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float squared_distance_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        const float delta = a[i] - b[i];
        sum += delta * delta;
    }
    return sum;
}

void axpy_scalar(float* destination, const float* source, float alpha, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) destination[i] += alpha * source[i];
}

void scale_axpy_scalar(float* destination, const float* source, float scale,
                       float alpha, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        destination[i] = scale * destination[i] + alpha * source[i];
    }
}

void lerp_scalar(float* destination, const float* target, float rate, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) destination[i] += rate * (target[i] - destination[i]);
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float dot_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    }
    alignas(32) float values[8];
    _mm256_store_ps(values, sum);
    float result = 0.0F;
    for (const float value : values) result += value;
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float squared_distance_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum = _mm256_fmadd_ps(delta, delta, sum);
    }
    alignas(32) float values[8];
    _mm256_store_ps(values, sum);
    float result = 0.0F;
    for (const float value : values) result += value;
    for (; i < n; ++i) {
        const float delta = a[i] - b[i];
        result += delta * delta;
    }
    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void axpy_avx2(float* destination, const float* source, float alpha, std::size_t n) noexcept {
    const __m256 scale = _mm256_set1_ps(alpha);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 current = _mm256_loadu_ps(destination + i);
        const __m256 value = _mm256_loadu_ps(source + i);
        _mm256_storeu_ps(destination + i, _mm256_fmadd_ps(scale, value, current));
    }
    for (; i < n; ++i) destination[i] += alpha * source[i];
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void scale_axpy_avx2(float* destination, const float* source, float scale_value,
                     float alpha, std::size_t n) noexcept {
    const __m256 scale = _mm256_set1_ps(scale_value);
    const __m256 source_scale = _mm256_set1_ps(alpha);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 current = _mm256_loadu_ps(destination + i);
        const __m256 value = _mm256_loadu_ps(source + i);
        _mm256_storeu_ps(destination + i,
                         _mm256_fmadd_ps(source_scale, value,
                                         _mm256_mul_ps(scale, current)));
    }
    for (; i < n; ++i) {
        destination[i] = scale_value * destination[i] + alpha * source[i];
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void lerp_avx2(float* destination, const float* target, float rate, std::size_t n) noexcept {
    const __m256 scale = _mm256_set1_ps(rate);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 current = _mm256_loadu_ps(destination + i);
        const __m256 desired = _mm256_loadu_ps(target + i);
        _mm256_storeu_ps(destination + i,
                         _mm256_fmadd_ps(scale, _mm256_sub_ps(desired, current), current));
    }
    for (; i < n; ++i) destination[i] += rate * (target[i] - destination[i]);
}
#endif

bool has_avx2_fma() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

} // namespace

bool simd_available() noexcept { return has_avx2_fma(); }

namespace detail {

float dot(const float* a, const float* b, std::size_t n) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx2_fma()) return dot_avx2(a, b, n);
#endif
    return dot_scalar(a, b, n);
}

float squared_distance(const float* a, const float* b, std::size_t n) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx2_fma()) return squared_distance_avx2(a, b, n);
#endif
    return squared_distance_scalar(a, b, n);
}

void axpy(float* destination, const float* source, float alpha, std::size_t n) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx2_fma()) {
        axpy_avx2(destination, source, alpha, n);
        return;
    }
#endif
    axpy_scalar(destination, source, alpha, n);
}


void scale_axpy(float* destination, const float* source, float scale, float alpha,
                std::size_t n) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx2_fma()) {
        scale_axpy_avx2(destination, source, scale, alpha, n);
        return;
    }
#endif
    scale_axpy_scalar(destination, source, scale, alpha, n);
}

void lerp(float* destination, const float* target, float rate, std::size_t n) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx2_fma()) {
        lerp_avx2(destination, target, rate, n);
        return;
    }
#endif
    lerp_scalar(destination, target, rate, n);
}

float cosine(std::span<const float> a, std::span<const float> b) noexcept {
    const float ab = dot(a.data(), b.data(), a.size());
    const float aa = dot(a.data(), a.data(), a.size());
    const float bb = dot(b.data(), b.data(), b.size());
    return ab / std::sqrt(std::max(aa * bb, 1e-12F));
}

} // namespace detail
} // namespace sbm

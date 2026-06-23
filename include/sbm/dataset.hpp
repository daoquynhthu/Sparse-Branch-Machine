#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sbm {

struct VectorDataset {
    std::uint32_t token_alphabet{};
    std::uint32_t vector_dim{};
    std::vector<std::uint32_t> tokens;
    std::vector<float> targets;

    [[nodiscard]] std::span<const float> target(std::size_t index) const noexcept {
        return {targets.data() + index * vector_dim, vector_dim};
    }
};

[[nodiscard]] std::uint64_t hash_dataset(const VectorDataset& dataset) noexcept;
[[nodiscard]] VectorDataset generate_vector_process(
    std::size_t length,
    std::uint32_t alphabet = 64,
    std::uint32_t vector_dim = 16,
    std::uint32_t hidden_dim = 24,
    std::uint64_t seed = 7,
    float noise_std = 0.035F);
void save_dataset(const VectorDataset& dataset, const std::string& path);
[[nodiscard]] VectorDataset load_dataset(const std::string& path);

} // namespace sbm

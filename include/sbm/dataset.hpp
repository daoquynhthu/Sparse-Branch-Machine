#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sbm {

enum class DatasetKind : std::uint32_t {
    VectorRegression = 1U,
    TokenCrossEntropy = 2U,
};

struct VectorDataset {
    std::uint32_t token_alphabet{};
    std::uint32_t vector_dim{};
    std::vector<std::uint32_t> tokens;
    std::vector<float> targets;

    [[nodiscard]] std::span<const float> target(std::size_t index) const noexcept {
        return {targets.data() + index * vector_dim, vector_dim};
    }
};

// A tokenizer-aligned dataset: every sequence is a stream of token IDs and the
// training target at position t is the token at t+1.  Sequence offsets contain
// a leading zero and a final tokens.size() sentinel.  oracle_nll is optional;
// when present it stores the generator's exact -log p(x_t | x_<t) for each
// token position (sequence starts are zero and are not scored).
struct TokenDataset {
    std::uint32_t vocab_size{};
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint64_t> sequence_offsets;
    std::vector<float> oracle_nll;

    [[nodiscard]] std::size_t sequence_count() const noexcept {
        return sequence_offsets.empty() ? 0U : sequence_offsets.size() - 1U;
    }
    [[nodiscard]] std::size_t example_count() const noexcept {
        return tokens.size() >= sequence_count() ? tokens.size() - sequence_count() : 0U;
    }
};

struct TokenShardCursor {
    std::uint64_t shard_index{};
    std::uint64_t sequence_index{};
    std::uint64_t token_offset{};
};

struct TokenExample {
    std::uint32_t input{};
    std::uint32_t target{};
};

// A read-only, independently memory-mapped real-corpus shard.  The cursor is
// the position of the next within-sequence (input, target) pair.
class MappedTokenShard {
public:
    explicit MappedTokenShard(const std::string& path,
                              std::uint64_t shard_index = 0U,
                              bool verify_payload = true);
    ~MappedTokenShard();
    MappedTokenShard(MappedTokenShard&&) noexcept;
    MappedTokenShard& operator=(MappedTokenShard&&) noexcept;
    MappedTokenShard(const MappedTokenShard&) = delete;
    MappedTokenShard& operator=(const MappedTokenShard&) = delete;

    [[nodiscard]] std::uint32_t vocab_size() const noexcept;
    [[nodiscard]] std::uint64_t token_count() const noexcept;
    [[nodiscard]] std::uint64_t sequence_count() const noexcept;
    [[nodiscard]] std::uint64_t dataset_hash() const noexcept;
    [[nodiscard]] bool next(TokenShardCursor& cursor, TokenExample& example) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::uint64_t hash_dataset(const VectorDataset& dataset) noexcept;
[[nodiscard]] std::uint64_t hash_dataset(const TokenDataset& dataset) noexcept;

[[nodiscard]] VectorDataset generate_vector_process(
    std::size_t length,
    std::uint32_t alphabet = 64,
    std::uint32_t vector_dim = 16,
    std::uint32_t hidden_dim = 24,
    std::uint64_t seed = 7,
    float noise_std = 0.035F);

// Generates a mathematically defined autoregressive token process.  It uses
// only observable token history; uncertainty comes from sampling the explicit
// categorical distribution rather than from hidden, unobservable state.
[[nodiscard]] TokenDataset generate_math_token_process(
    std::size_t sequence_count = 64,
    std::size_t sequence_length = 2048,
    std::uint32_t vocab_size = 128,
    std::uint64_t seed = 7,
    float temperature = 1.0F,
    float interaction_strength = 0.35F);

// Creates a training dataset from externally tokenized IDs.  Offsets may be
// empty, in which case the whole stream is treated as one sequence.
[[nodiscard]] TokenDataset make_token_dataset(
    std::span<const std::uint32_t> tokens,
    std::uint32_t vocab_size,
    std::span<const std::uint64_t> sequence_offsets = {});

void save_dataset(const VectorDataset& dataset, const std::string& path);
[[nodiscard]] VectorDataset load_dataset(const std::string& path);
void save_token_dataset(const TokenDataset& dataset, const std::string& path);
[[nodiscard]] TokenDataset load_token_dataset(const std::string& path);
void write_token_shard(const TokenDataset& dataset, const std::string& path);
[[nodiscard]] DatasetKind inspect_dataset_kind(const std::string& path);

} // namespace sbm

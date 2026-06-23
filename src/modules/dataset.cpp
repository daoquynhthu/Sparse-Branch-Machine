#include "sbm/dataset.hpp"

#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace sbm {
namespace {
constexpr std::uint32_t kVectorDatasetVersion = 2;
constexpr std::uint32_t kTokenDatasetVersion = 1;
constexpr std::array<char, 8> kVectorMagic{'S','B','M','V','E','C','0','2'};
constexpr std::array<char, 8> kTokenMagic{'S','B','M','T','O','K','0','1'};

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("dataset write failed");
}

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("dataset read failed");
    return value;
}

void validate_token_dataset(const TokenDataset& dataset) {
    if (dataset.vocab_size == 0U) {
        throw std::invalid_argument("token dataset vocabulary must be positive");
    }
    if (dataset.sequence_offsets.size() < 2U ||
        dataset.sequence_offsets.front() != 0U ||
        dataset.sequence_offsets.back() != dataset.tokens.size()) {
        throw std::invalid_argument("invalid token dataset sequence offsets");
    }
    for (std::size_t index = 1; index < dataset.sequence_offsets.size(); ++index) {
        if (dataset.sequence_offsets[index] <= dataset.sequence_offsets[index - 1U]) {
            throw std::invalid_argument("token sequences must be non-empty and ordered");
        }
    }
    for (const auto token : dataset.tokens) {
        if (token >= dataset.vocab_size) {
            throw std::invalid_argument("token id is outside the declared vocabulary");
        }
    }
    if (!dataset.oracle_nll.empty() && dataset.oracle_nll.size() != dataset.tokens.size()) {
        throw std::invalid_argument("oracle_nll must be empty or match token count");
    }
}

float circular_cosine(std::uint32_t value, std::uint32_t center,
                      std::uint32_t vocabulary, float harmonic = 1.0F) {
    const float phase = static_cast<float>(2.0 * std::numbers::pi) * harmonic *
        static_cast<float>((value + vocabulary - center) % vocabulary) /
        static_cast<float>(vocabulary);
    return std::cos(phase);
}

std::uint32_t history_at(const std::vector<std::uint32_t>& sequence,
                         std::size_t lag) {
    if (sequence.empty()) return 0U;
    return lag < sequence.size() ? sequence[sequence.size() - 1U - lag]
                                 : sequence.front();
}

std::uint32_t sample_categorical(std::mt19937_64& rng,
                                 std::span<const float> probabilities) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double draw = uniform(rng);
    double cumulative = 0.0;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        cumulative += probabilities[index];
        if (draw <= cumulative || index + 1U == probabilities.size()) {
            return static_cast<std::uint32_t>(index);
        }
    }
    return static_cast<std::uint32_t>(probabilities.size() - 1U);
}

} // namespace

std::uint64_t hash_dataset(const VectorDataset& dataset) noexcept {
    std::uint64_t hash = mix64(dataset.token_alphabet) ^
        std::rotl(mix64(dataset.vector_dim), 13);
    for (std::size_t index = 0; index < dataset.tokens.size(); ++index) {
        hash ^= mix64(static_cast<std::uint64_t>(dataset.tokens[index]) +
                      index * 0x9E37ULL);
        hash = std::rotl(hash, 7);
    }
    for (std::size_t index = 0; index < dataset.targets.size(); ++index) {
        std::uint32_t bits{};
        std::memcpy(&bits, &dataset.targets[index], sizeof(bits));
        hash ^= mix64(static_cast<std::uint64_t>(bits) + index * 0x85EBULL);
        hash = std::rotl(hash, 9);
    }
    return mix64(hash);
}

std::uint64_t hash_dataset(const TokenDataset& dataset) noexcept {
    std::uint64_t hash = mix64(dataset.vocab_size) ^ 0x544F4B454EULL;
    for (std::size_t index = 0; index < dataset.tokens.size(); ++index) {
        hash ^= mix64(static_cast<std::uint64_t>(dataset.tokens[index]) +
                      index * 0x9E3779B97F4A7C15ULL);
        hash = std::rotl(hash, 11);
    }
    for (std::size_t index = 0; index < dataset.sequence_offsets.size(); ++index) {
        hash ^= mix64(dataset.sequence_offsets[index] + index * 0xC2B2AE3D27D4EB4FULL);
        hash = std::rotl(hash, 5);
    }
    for (std::size_t index = 0; index < dataset.oracle_nll.size(); ++index) {
        std::uint32_t bits{};
        std::memcpy(&bits, &dataset.oracle_nll[index], sizeof(bits));
        hash ^= mix64(static_cast<std::uint64_t>(bits) + index * 0x165667B1ULL);
        hash = std::rotl(hash, 13);
    }
    return mix64(hash);
}

VectorDataset generate_vector_process(std::size_t length, std::uint32_t alphabet,
                                      std::uint32_t dim, std::uint32_t hidden_dim,
                                      std::uint64_t seed, float noise_std) {
    VectorDataset dataset{alphabet, dim, {}, {}};
    if (!length || !alphabet || !dim || !hidden_dim) return dataset;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::normal_distribution<float> noise(0.0F, noise_std);
    std::uniform_int_distribution<std::uint32_t> token_draw(0, alphabet - 1U);

    std::vector<float> token_embedding(static_cast<std::size_t>(alphabet) * hidden_dim);
    constexpr std::uint32_t regime_count = 8U;
    std::vector<float> regime_embedding(regime_count * hidden_dim);
    std::vector<float> projection(static_cast<std::size_t>(dim) * hidden_dim);
    for (auto& value : token_embedding) value = 0.55F * normal(rng);
    for (auto& value : regime_embedding) value = 0.22F * normal(rng);
    for (auto& value : projection) {
        value = normal(rng) / std::sqrt(static_cast<float>(hidden_dim));
    }

    dataset.tokens.reserve(length);
    dataset.targets.resize(length * dim);
    std::array<std::uint32_t, 6> history{};
    for (auto& token : history) token = token_draw(rng);

    std::vector<float> features(hidden_dim);
    for (std::size_t step = 0; step < length; ++step) {
        const std::uint32_t innovation = token_draw(rng);
        const std::uint32_t token = static_cast<std::uint32_t>(
            (5U * history[5] + 3U * history[3] + innovation) % alphabet);
        for (std::size_t index = 0; index + 1U < history.size(); ++index) {
            history[index] = history[index + 1U];
        }
        history.back() = token;
        dataset.tokens.push_back(token);

        const std::uint32_t regime = static_cast<std::uint32_t>(
            (history[5] + 3U * history[3] + 5U * history[1]) % regime_count);
        for (std::uint32_t index = 0; index < hidden_dim; ++index) {
            const auto offset = [&](std::uint32_t token_id) {
                return static_cast<std::size_t>(token_id) * hidden_dim + index;
            };
            const float e0 = token_embedding[offset(history[5])];
            const float e1 = token_embedding[offset(history[4])];
            const float e2 = token_embedding[offset(history[3])];
            const float e4 = token_embedding[offset(history[1])];
            const float regime_value = regime_embedding[
                static_cast<std::size_t>(regime) * hidden_dim + index];
            const float cross = e0 * e2 - 0.55F * e1 * e4;
            const float gate = ((history[5] ^ history[2] ^ index) & 1U) ? 1.0F : -1.0F;
            features[index] = std::tanh(0.30F * e0 + 0.28F * e1 + 0.25F * e2 +
                                        0.20F * e4 + 0.24F * regime_value +
                                        0.16F * gate * cross);
        }
        for (std::uint32_t component = 0; component < dim; ++component) {
            float target = detail::dot(
                projection.data() + static_cast<std::size_t>(component) * hidden_dim,
                features.data(), hidden_dim);
            target += 0.16F * features[(component * 5U + regime) % hidden_dim] *
                      features[(component * 11U + 3U) % hidden_dim];
            dataset.targets[step * dim + component] = std::tanh(target) + noise(rng);
        }
    }
    return dataset;
}

TokenDataset generate_math_token_process(std::size_t sequence_count,
                                         std::size_t sequence_length,
                                         std::uint32_t vocab_size,
                                         std::uint64_t seed,
                                         float temperature,
                                         float interaction_strength) {
    if (sequence_count == 0U || sequence_length < 2U || vocab_size < 8U) {
        throw std::invalid_argument(
            "math token process requires sequences>0, length>=2 and vocab>=8");
    }
    if (!(temperature > 0.0F) || interaction_strength < 0.0F) {
        throw std::invalid_argument("invalid mathematical token process parameters");
    }

    TokenDataset dataset;
    dataset.vocab_size = vocab_size;
    dataset.tokens.reserve(sequence_count * sequence_length);
    dataset.oracle_nll.reserve(sequence_count * sequence_length);
    dataset.sequence_offsets.reserve(sequence_count + 1U);
    dataset.sequence_offsets.push_back(0U);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> initial_draw(0U, vocab_size - 1U);
    std::vector<float> logits(vocab_size, 0.0F);
    std::vector<float> probabilities(vocab_size, 0.0F);

    for (std::size_t sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
        std::vector<std::uint32_t> sequence;
        sequence.reserve(sequence_length);
        sequence.push_back(initial_draw(rng));
        dataset.tokens.push_back(sequence.front());
        dataset.oracle_nll.push_back(0.0F);

        for (std::size_t position = 1; position < sequence_length; ++position) {
            const std::uint32_t x0 = history_at(sequence, 0U);
            const std::uint32_t x1 = history_at(sequence, 1U);
            const std::uint32_t x2 = history_at(sequence, 2U);
            const std::uint32_t x4 = history_at(sequence, 4U);

            const std::uint32_t center1 = static_cast<std::uint32_t>(
                (5ULL * x0 + 3ULL * x1 + 11ULL) % vocab_size);
            const std::uint32_t center2 = static_cast<std::uint32_t>(
                (7ULL * x0 + 9ULL * x2 + 3ULL * (x0 ^ x2) + 17ULL) % vocab_size);
            const std::uint32_t center4 = static_cast<std::uint32_t>(
                (13ULL * x0 + 5ULL * x4 + 19ULL) % vocab_size);
            const std::uint32_t interaction_center = static_cast<std::uint32_t>(
                ((static_cast<std::uint64_t>(x1) * (x4 + 1ULL)) +
                 3ULL * x2 + (x0 ^ x4) + 23ULL) % vocab_size);

            float maximum = -std::numeric_limits<float>::infinity();
            for (std::uint32_t candidate = 0; candidate < vocab_size; ++candidate) {
                float value = 1.20F * circular_cosine(candidate, center1, vocab_size) +
                              0.92F * circular_cosine(candidate, center2, vocab_size) +
                              0.72F * circular_cosine(candidate, center4, vocab_size) +
                              interaction_strength *
                                  circular_cosine(candidate, interaction_center, vocab_size) +
                              0.18F * circular_cosine(candidate, center1, vocab_size, 2.0F);
                value /= temperature;
                logits[candidate] = value;
                maximum = std::max(maximum, value);
            }
            double normalizer = 0.0;
            for (std::uint32_t candidate = 0; candidate < vocab_size; ++candidate) {
                probabilities[candidate] = std::exp(logits[candidate] - maximum);
                normalizer += probabilities[candidate];
            }
            for (auto& probability : probabilities) {
                probability = static_cast<float>(probability / normalizer);
            }

            const std::uint32_t next = sample_categorical(rng, probabilities);
            sequence.push_back(next);
            dataset.tokens.push_back(next);
            dataset.oracle_nll.push_back(-std::log(std::max(probabilities[next], 1e-12F)));
        }
        dataset.sequence_offsets.push_back(dataset.tokens.size());
    }
    validate_token_dataset(dataset);
    return dataset;
}

TokenDataset make_token_dataset(std::span<const std::uint32_t> tokens,
                                std::uint32_t vocab_size,
                                std::span<const std::uint64_t> sequence_offsets) {
    TokenDataset dataset;
    dataset.vocab_size = vocab_size;
    dataset.tokens.assign(tokens.begin(), tokens.end());
    if (sequence_offsets.empty()) {
        dataset.sequence_offsets = {0U, static_cast<std::uint64_t>(tokens.size())};
    } else {
        dataset.sequence_offsets.assign(sequence_offsets.begin(), sequence_offsets.end());
    }
    validate_token_dataset(dataset);
    return dataset;
}

void save_dataset(const VectorDataset& dataset, const std::string& path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open dataset for writing");
    output.write(kVectorMagic.data(), static_cast<std::streamsize>(kVectorMagic.size()));
    write_pod(output, kVectorDatasetVersion);
    write_pod(output, dataset.token_alphabet);
    write_pod(output, dataset.vector_dim);
    const auto count = static_cast<std::uint64_t>(dataset.tokens.size());
    write_pod(output, count);
    write_pod(output, hash_dataset(dataset));
    output.write(reinterpret_cast<const char*>(dataset.tokens.data()),
                 static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    output.write(reinterpret_cast<const char*>(dataset.targets.data()),
                 static_cast<std::streamsize>(count * dataset.vector_dim * sizeof(float)));
    if (!output) throw std::runtime_error("dataset write failed");
}

VectorDataset load_dataset(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open dataset");
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != kVectorMagic) throw std::runtime_error("invalid vector dataset magic");
    if (read_pod<std::uint32_t>(input) != kVectorDatasetVersion) {
        throw std::runtime_error("unsupported vector dataset version");
    }
    VectorDataset dataset;
    dataset.token_alphabet = read_pod<std::uint32_t>(input);
    dataset.vector_dim = read_pod<std::uint32_t>(input);
    const auto count = read_pod<std::uint64_t>(input);
    const auto expected_hash = read_pod<std::uint64_t>(input);
    if (count > (1ULL << 34U)) throw std::runtime_error("dataset too large");
    dataset.tokens.resize(static_cast<std::size_t>(count));
    dataset.targets.resize(static_cast<std::size_t>(count) * dataset.vector_dim);
    input.read(reinterpret_cast<char*>(dataset.tokens.data()),
               static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    input.read(reinterpret_cast<char*>(dataset.targets.data()),
               static_cast<std::streamsize>(count * dataset.vector_dim * sizeof(float)));
    if (!input || hash_dataset(dataset) != expected_hash) {
        throw std::runtime_error("vector dataset corrupt");
    }
    return dataset;
}

void save_token_dataset(const TokenDataset& dataset, const std::string& path) {
    validate_token_dataset(dataset);
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open token dataset for writing");
    output.write(kTokenMagic.data(), static_cast<std::streamsize>(kTokenMagic.size()));
    write_pod(output, kTokenDatasetVersion);
    write_pod(output, dataset.vocab_size);
    const auto token_count = static_cast<std::uint64_t>(dataset.tokens.size());
    const auto offset_count = static_cast<std::uint64_t>(dataset.sequence_offsets.size());
    const std::uint8_t has_oracle = dataset.oracle_nll.empty() ? 0U : 1U;
    write_pod(output, token_count);
    write_pod(output, offset_count);
    write_pod(output, has_oracle);
    write_pod(output, hash_dataset(dataset));
    output.write(reinterpret_cast<const char*>(dataset.sequence_offsets.data()),
                 static_cast<std::streamsize>(offset_count * sizeof(std::uint64_t)));
    output.write(reinterpret_cast<const char*>(dataset.tokens.data()),
                 static_cast<std::streamsize>(token_count * sizeof(std::uint32_t)));
    if (has_oracle != 0U) {
        output.write(reinterpret_cast<const char*>(dataset.oracle_nll.data()),
                     static_cast<std::streamsize>(token_count * sizeof(float)));
    }
    if (!output) throw std::runtime_error("token dataset write failed");
}

TokenDataset load_token_dataset(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open token dataset");
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != kTokenMagic) throw std::runtime_error("invalid token dataset magic");
    if (read_pod<std::uint32_t>(input) != kTokenDatasetVersion) {
        throw std::runtime_error("unsupported token dataset version");
    }
    TokenDataset dataset;
    dataset.vocab_size = read_pod<std::uint32_t>(input);
    const auto token_count = read_pod<std::uint64_t>(input);
    const auto offset_count = read_pod<std::uint64_t>(input);
    const auto has_oracle = read_pod<std::uint8_t>(input);
    const auto expected_hash = read_pod<std::uint64_t>(input);
    if (token_count > (1ULL << 34U) || offset_count > token_count + 1U) {
        throw std::runtime_error("token dataset too large");
    }
    dataset.sequence_offsets.resize(static_cast<std::size_t>(offset_count));
    dataset.tokens.resize(static_cast<std::size_t>(token_count));
    input.read(reinterpret_cast<char*>(dataset.sequence_offsets.data()),
               static_cast<std::streamsize>(offset_count * sizeof(std::uint64_t)));
    input.read(reinterpret_cast<char*>(dataset.tokens.data()),
               static_cast<std::streamsize>(token_count * sizeof(std::uint32_t)));
    if (has_oracle != 0U) {
        dataset.oracle_nll.resize(static_cast<std::size_t>(token_count));
        input.read(reinterpret_cast<char*>(dataset.oracle_nll.data()),
                   static_cast<std::streamsize>(token_count * sizeof(float)));
    }
    validate_token_dataset(dataset);
    if (!input || hash_dataset(dataset) != expected_hash) {
        throw std::runtime_error("token dataset corrupt");
    }
    return dataset;
}

DatasetKind inspect_dataset_kind(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open dataset");
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input) throw std::runtime_error("cannot read dataset header");
    if (magic == kVectorMagic) return DatasetKind::VectorRegression;
    if (magic == kTokenMagic) return DatasetKind::TokenCrossEntropy;
    throw std::runtime_error("unknown dataset magic");
}

} // namespace sbm

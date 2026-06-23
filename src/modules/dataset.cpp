#include "sbm/dataset.hpp"
#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

namespace sbm {
namespace {
constexpr std::uint32_t kDatasetVersion = 2;
constexpr std::array<char, 8> kMagic{'S','B','M','V','E','C','0','2'};

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
} // namespace

std::uint64_t hash_dataset(const VectorDataset& d) noexcept {
    std::uint64_t h=mix64(d.token_alphabet)^std::rotl(mix64(d.vector_dim),13);
    for(std::size_t i=0;i<d.tokens.size();++i){h^=mix64(static_cast<std::uint64_t>(d.tokens[i])+i*0x9E37ULL);h=std::rotl(h,7);} 
    for(std::size_t i=0;i<d.targets.size();++i){std::uint32_t bits;std::memcpy(&bits,&d.targets[i],4);h^=mix64(static_cast<std::uint64_t>(bits)+i*0x85EBULL);h=std::rotl(h,9);} return mix64(h);
}
VectorDataset generate_vector_process(std::size_t length, std::uint32_t alphabet,
                                             std::uint32_t dim, std::uint32_t hidden_dim,
                                             std::uint64_t seed, float noise_std) {
    VectorDataset d{alphabet, dim, {}, {}};
    if (!length || !alphabet || !dim || !hidden_dim) return d;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::normal_distribution<float> noise(0.0F, noise_std);
    std::uniform_int_distribution<std::uint32_t> token_draw(0, alphabet - 1U);

    // The observed token history fully determines the noise-free target.  The
    // hidden dimension is a feature bank, not an unobservable stochastic state.
    std::vector<float> token_embedding(static_cast<std::size_t>(alphabet) * hidden_dim);
    constexpr std::uint32_t regime_count = 8U;
    std::vector<float> regime_embedding(regime_count * hidden_dim);
    std::vector<float> projection(static_cast<std::size_t>(dim) * hidden_dim);
    for (auto& v : token_embedding) v = 0.55F * normal(rng);
    for (auto& v : regime_embedding) v = 0.22F * normal(rng);
    for (auto& v : projection) v = normal(rng) / std::sqrt(static_cast<float>(hidden_dim));

    d.tokens.reserve(length);
    d.targets.resize(length * dim);
    std::array<std::uint32_t, 6> history{};
    for (auto& x : history) x = token_draw(rng);

    std::vector<float> features(hidden_dim);
    for (std::size_t t = 0; t < length; ++t) {
        // The token stream has broad support instead of nearly deterministic
        // recurrence.  Every target driver remains observable in token history.
        const std::uint32_t innovation = token_draw(rng);
        const std::uint32_t token = static_cast<std::uint32_t>(
            (5U * history[5] + 3U * history[3] + innovation) % alphabet);
        for (std::size_t i = 0; i + 1U < history.size(); ++i) history[i] = history[i + 1U];
        history.back() = token;
        d.tokens.push_back(token);

        const std::uint32_t regime = static_cast<std::uint32_t>(
            (history[5] + 3U * history[3] + 5U * history[1]) % regime_count);
        for (std::uint32_t j = 0; j < hidden_dim; ++j) {
            const float e0 = token_embedding[static_cast<std::size_t>(history[5]) * hidden_dim + j];
            const float e1 = token_embedding[static_cast<std::size_t>(history[4]) * hidden_dim + j];
            const float e2 = token_embedding[static_cast<std::size_t>(history[3]) * hidden_dim + j];
            const float e4 = token_embedding[static_cast<std::size_t>(history[1]) * hidden_dim + j];
            const float rg = regime_embedding[static_cast<std::size_t>(regime) * hidden_dim + j];
            const float cross = e0 * e2 - 0.55F * e1 * e4;
            const float gate = ((history[5] ^ history[2] ^ j) & 1U) ? 1.0F : -1.0F;
            features[j] = std::tanh(0.30F * e0 + 0.28F * e1 + 0.25F * e2 +
                                    0.20F * e4 + 0.24F * rg +
                                    0.16F * gate * cross);
        }
        for (std::uint32_t k = 0; k < dim; ++k) {
            float y = detail::dot(projection.data() + static_cast<std::size_t>(k) * hidden_dim,
                           features.data(), hidden_dim);
            y += 0.16F * features[(k * 5U + regime) % hidden_dim] *
                 features[(k * 11U + 3U) % hidden_dim];
            d.targets[t * dim + k] = std::tanh(y) + noise(rng);
        }
    }
    return d;
}
void save_dataset(const VectorDataset& d,const std::string&p){std::ofstream o(p,std::ios::binary);if(!o)throw std::runtime_error("cannot open dataset for writing");o.write(kMagic.data(),8);write_pod(o,kDatasetVersion);write_pod(o,d.token_alphabet);write_pod(o,d.vector_dim);auto n=(std::uint64_t)d.tokens.size();write_pod(o,n);auto h=hash_dataset(d);write_pod(o,h);o.write((char*)d.tokens.data(),(std::streamsize)(n*4));o.write((char*)d.targets.data(),(std::streamsize)(n*d.vector_dim*4));if(!o)throw std::runtime_error("dataset write failed");}
VectorDataset load_dataset(const std::string&p){std::ifstream i(p,std::ios::binary);if(!i)throw std::runtime_error("cannot open dataset");std::array<char,8>m{};i.read(m.data(),8);if(m!=kMagic)throw std::runtime_error("invalid dataset magic");if(read_pod<std::uint32_t>(i)!=kDatasetVersion)throw std::runtime_error("unsupported dataset version");VectorDataset d;d.token_alphabet=read_pod<std::uint32_t>(i);d.vector_dim=read_pod<std::uint32_t>(i);auto n=read_pod<std::uint64_t>(i);auto expected=read_pod<std::uint64_t>(i);if(n>(1ULL<<34))throw std::runtime_error("dataset too large");d.tokens.resize((std::size_t)n);d.targets.resize((std::size_t)n*d.vector_dim);i.read((char*)d.tokens.data(),(std::streamsize)(n*4));i.read((char*)d.targets.data(),(std::streamsize)(n*d.vector_dim*4));if(!i||hash_dataset(d)!=expected)throw std::runtime_error("dataset corrupt");return d;}


} // namespace sbm

#include "sbm/math.hpp"
#include <bit>
#include <cstring>

namespace sbm {
std::uint64_t mix64(std::uint64_t x) noexcept { x^=x>>30U;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27U;x*=0x94d049bb133111ebULL;x^=x>>31U;return x; }
std::uint64_t rolling_signature(std::span<const std::uint32_t> w,std::uint64_t seed) noexcept {
    // Preserve recent-token locality in the high bits used by the bucket index.
    // The lower bits remain mixed to separate longer histories inside a bucket.
    std::uint64_t h = seed;
    for (std::size_t i = 0; i < w.size(); ++i) {
        h ^= std::rotl(mix64(static_cast<std::uint64_t>(w[i]) + 0x10001ULL * (i + 1U)),
                       static_cast<int>((i * 9U) & 63U));
    }
    h = mix64(h);
    std::uint64_t recent = 0;
    const std::size_t take = std::min<std::size_t>(4, w.size());
    // Most recent token occupies the most significant byte.
    for (std::size_t k = 0; k < take; ++k) {
        recent |= static_cast<std::uint64_t>(w[w.size() - 1U - k] & 0xFFU)
                  << (56U - static_cast<unsigned>(8U * k));
    }
    const unsigned bits = static_cast<unsigned>(take * 8U);
    if (bits != 0U) {
        const std::uint64_t low_mask = bits == 64U ? 0U : ((std::uint64_t{1} << (64U - bits)) - 1U);
        h = recent | (h & low_mask);
    }
    return h;
}
std::uint64_t token_context_signature(std::span<const std::uint32_t> window,
                                      std::uint32_t alphabet,
                                      std::uint64_t seed) noexcept {
    if(window.empty()) return mix64(seed);
    const unsigned symbol_bits=std::max(1U,static_cast<unsigned>(std::bit_width(alphabet>0?alphabet-1U:0U)));
    std::uint64_t mixed=seed;
    for(std::size_t i=0;i<window.size();++i){
        mixed^=std::rotl(mix64(static_cast<std::uint64_t>(window[i])+0x10001ULL*(i+1U)),
                         static_cast<int>((i*9U)&63U));
    }
    mixed=mix64(mixed);

    std::uint64_t packed=0;
    unsigned used=0;
    for(std::size_t offset=0;offset<window.size()&&used+symbol_bits<=64U;++offset){
        const std::uint64_t symbol=window[window.size()-1U-offset];
        const unsigned shift=64U-used-symbol_bits;
        packed|=(symbol&((std::uint64_t{1}<<symbol_bits)-1U))<<shift;
        used+=symbol_bits;
    }
    if(used==64U) return packed;
    const std::uint64_t low_mask=(std::uint64_t{1}<<(64U-used))-1U;
    return packed|(mixed&low_mask);
}
std::uint64_t lagged_token_signature(std::span<const std::uint32_t> window,
                                     std::uint32_t alphabet,
                                     std::uint32_t lag,
                                     std::uint64_t seed) noexcept {
    if (window.empty()) return mix64(seed ^ lag);
    const auto current = window.back();
    const auto previous = window.size() > lag
        ? window[window.size() - 1U - static_cast<std::size_t>(lag)]
        : current;
    const std::uint32_t pair[]{previous, current};
    const auto pair_signature = token_context_signature(pair, alphabet,
        seed ^ mix64(static_cast<std::uint64_t>(lag) + 0xA0761D6478BD642FULL));

    // Preserve the lagged pair in the high address bits while letting the
    // lower bits distinguish longer contexts for future specialization.
    const unsigned symbol_bits = std::max(
        1U, static_cast<unsigned>(std::bit_width(alphabet > 0 ? alphabet - 1U : 0U)));
    const unsigned address_bits = std::min(64U, 2U * symbol_bits);
    if (address_bits == 64U) return pair_signature;
    const std::uint64_t high_mask = ~((std::uint64_t{1} << (64U - address_bits)) - 1U);
    const auto context_mix = token_context_signature(window, alphabet,
        seed ^ mix64(static_cast<std::uint64_t>(lag) * 0xE7037ED1A0B428DBULL));
    return (pair_signature & high_mask) | (context_mix & ~high_mask);
}
double hamming_similarity(std::uint64_t a,std::uint64_t b) noexcept { return 1.0-static_cast<double>(std::popcount(a^b))/64.0; }

} // namespace sbm

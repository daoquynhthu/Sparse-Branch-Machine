#include "sbm/detail/address_interpreter.hpp"

#include "sbm/math.hpp"
#include "sbm/types.hpp"
#include <bit>
#include <cstring>
#include <optional>

namespace sbm {

std::uint64_t mix64(std::uint64_t x) noexcept;

namespace {

[[nodiscard]] std::uint32_t safe_history_index(
    std::span<const std::uint32_t> window,
    std::uint32_t lag) noexcept {
    if (window.empty()) return 0U;
    if (window.size() > lag) {
        return static_cast<std::uint32_t>(window.size() - 1U - lag);
    }
    return 0U;
}

[[nodiscard]] std::uint64_t binding_state_key(
    const AddressBindingState& state,
    const AddressProgram& program) noexcept {
    if (!state.matched) return 0U;
    std::uint64_t key = mix64(
        static_cast<std::uint64_t>(program.op) + 0x8CB92BA72F3D8DD7ULL);
    key ^= std::rotl(mix64(address_program_key(program)), 7);
    key ^= std::rotl(mix64(state.current_token), 13);
    key ^= std::rotl(mix64(state.matched_token), 23);
    key ^= std::rotl(mix64(state.matched_successor), 31);
    key ^= std::rotl(mix64(state.pattern_span), 43);
    key ^= std::rotl(mix64(state.pattern_terms), 53);
    key = mix64(key);
    return key == 0U ? 1U : key;
}

} // namespace

namespace {

struct ContentMatchResult {
    std::uint32_t index{};
    std::uint32_t distance{};
};

[[nodiscard]] std::optional<ContentMatchResult> find_content_match(
    std::span<const std::uint32_t> window,
    std::size_t current_index,
    std::uint32_t max_lag,
    const std::array<std::uint32_t, kMaxAddressProgramArity>& lags,
    std::size_t pattern_count) noexcept {
    if (current_index == 0U || window.empty()) return std::nullopt;
    const auto bounded_lag = std::min<std::size_t>(max_lag, current_index);
    const auto current_token = window[current_index];
    for (std::size_t distance = 1U; distance <= bounded_lag; ++distance) {
        const auto candidate_index = current_index - distance;
        if (window[candidate_index] != current_token) continue;
        bool pattern_matches = true;
        for (std::size_t index = 0; index < pattern_count; ++index) {
            const auto lag = static_cast<std::size_t>(lags[index]);
            if (lag > current_index || lag > candidate_index ||
                window[current_index - lag] != window[candidate_index - lag]) {
                pattern_matches = false;
                break;
            }
        }
        if (!pattern_matches) continue;
        return ContentMatchResult{
            static_cast<std::uint32_t>(candidate_index),
            static_cast<std::uint32_t>(distance)};
    }
    return std::nullopt;
}

} // namespace

AddressBindingState resolve_address_binding_state(
    std::span<const std::uint32_t> window,
    const AddressProgram& program) noexcept {
    AddressBindingState state;
    state.pattern_terms = is_content_follow_op(program.op) &&
        program.arity > 1U
        ? static_cast<std::uint8_t>(program.arity - 1U)
        : 0U;
    if (window.empty() || program.arity == 0U) return state;
    state.current_token = window.back();

    if (program.op == AddressOp::Tuple || program.op == AddressOp::DeltaMod) {
        state.matched = true;
        state.matched_index =
            safe_history_index(window, program.lags[program.arity - 1U]);
        state.matched_token = window[state.matched_index];
        state.matched_successor = state.matched_token;
        state.matched_distance = program.lags[program.arity - 1U];
        state.pattern_span = program.lags[program.arity - 1U];
        state.binding_key = binding_state_key(state, program);
        return state;
    }

    const auto original_current_index = window.size() - 1U;
    const std::uint32_t max_lag = program.lags[program.arity - 1U];
    const std::size_t pattern_count = state.pattern_terms;
    const std::uint8_t hops = content_follow_hop_count(program.op);

    std::uint32_t final_index = 0U;
    std::uint32_t final_distance = 0U;
    auto current_index = original_current_index;

    for (std::uint8_t hop = 0U; hop < hops; ++hop) {
        const auto match = find_content_match(
            window, current_index, max_lag, program.lags, pattern_count);
        if (!match.has_value()) {
            state.matched = false;
            return state;
        }
        final_index = match->index;
        final_distance = match->distance;
        current_index = static_cast<std::size_t>(final_index) + 1U;
    }

    state.matched = true;
    state.matched_index = final_index;
    state.matched_token = window[final_index];
    state.matched_successor =
        window[std::min(static_cast<std::size_t>(final_index) + 1U,
                        original_current_index)];
    state.matched_distance = final_distance;
    state.pattern_span = max_lag;
    state.binding_key = binding_state_key(state, program);
    return state;
}

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

std::uint64_t address_program_signature(std::span<const std::uint32_t> window,
                                        std::uint32_t alphabet,
                                        std::span<const std::uint32_t> lags,
                                        AddressOp op,
                                        std::uint64_t seed) noexcept {
    if (window.empty()) return mix64(seed);
    const unsigned symbol_bits = std::max(
        1U, static_cast<unsigned>(std::bit_width(alphabet > 0 ? alphabet - 1U : 0U)));

    std::uint32_t selected[kMaxAddressProgramArity + 1U]{};
    std::size_t count = 0U;
    const auto current = window.back();
    selected[count++] = current;
    std::uint32_t matched_distance = 0U;
    if (op == AddressOp::ContentMatch || is_content_follow_op(op)) {
        AddressProgram program;
        program.arity = static_cast<std::uint8_t>(
            std::min<std::size_t>(lags.size(), kMaxAddressProgramArity));
        program.op = op;
        for (std::size_t index = 0U; index < program.arity; ++index) {
            program.lags[index] = lags[index];
        }
        const auto binding = resolve_address_binding_state(window, program);
        matched_distance = binding.matched_distance;
        selected[count++] = binding.matched ? 1U : 0U;
        selected[count++] = binding.matched ? binding.matched_successor : 0U;
    } else {
        for (const auto lag : lags) {
            if (count >= std::size(selected)) break;
            const auto historical = window.size() > lag
                ? window[window.size() - 1U - static_cast<std::size_t>(lag)]
                : window.front();
            if (op == AddressOp::DeltaMod) {
                const auto modulus = std::max(1U, alphabet);
                selected[count++] = (current + modulus - (historical % modulus)) % modulus;
            } else {
                selected[count++] = historical;
            }
        }
    }

    std::uint64_t mixed = seed ^ mix64(static_cast<std::uint64_t>(count)) ^
        mix64(static_cast<std::uint64_t>(op) + 0xD1B54A32D192ED03ULL);
    if (op == AddressOp::ContentMatch || is_content_follow_op(op)) {
        mixed ^= mix64(static_cast<std::uint64_t>(matched_distance) +
                       0xA24BAED4963EE407ULL);
        for (std::size_t index = 0; index < lags.size(); ++index) {
            const auto lag = static_cast<std::size_t>(lags[index]);
            const auto pattern = window.size() > lag ? window[window.size() - 1U - lag]
                                                     : window.front();
            mixed ^= std::rotl(mix64(static_cast<std::uint64_t>(pattern) +
                                     0xD6E8ULL * (index + 1U)),
                               static_cast<int>(((index + 3U) * 11U) & 63U));
        }
    }
    for (std::size_t index = 0; index < count; ++index) {
        mixed ^= std::rotl(mix64(static_cast<std::uint64_t>(selected[index]) +
                                 0x9E37ULL * (index + 1U)),
                           static_cast<int>((index * 13U) & 63U));
    }
    mixed = mix64(mixed ^ token_context_signature(window, alphabet, seed));

    std::uint64_t packed = 0U;
    unsigned used = 0U;
    for (std::size_t index = 0; index < count && used + symbol_bits <= 64U; ++index) {
        const unsigned shift = 64U - used - symbol_bits;
        const std::uint64_t mask = symbol_bits == 64U
            ? ~std::uint64_t{0}
            : ((std::uint64_t{1} << symbol_bits) - 1U);
        packed |= (static_cast<std::uint64_t>(selected[index]) & mask) << shift;
        used += symbol_bits;
    }
    if (used == 64U) return packed;
    const std::uint64_t low_mask = (std::uint64_t{1} << (64U - used)) - 1U;
    return packed | (mixed & low_mask);
}

double hamming_similarity(std::uint64_t a,std::uint64_t b) noexcept { return 1.0-static_cast<double>(std::popcount(a^b))/64.0; }

} // namespace sbm

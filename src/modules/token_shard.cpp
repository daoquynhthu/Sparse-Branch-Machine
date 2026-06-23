#include "sbm/dataset.hpp"

#include "sbm/math.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace sbm {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'S','B','M','S','H','R','0','1'};
constexpr std::uint32_t kVersion = 1U;
constexpr std::uint32_t kHeaderSize = 128U;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint64_t kAlignment = 64U;

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U)) {
        throw std::overflow_error("token shard layout overflow");
    }
    return (value + alignment - 1U) & ~(alignment - 1U);
}

void put_u32(std::array<std::uint8_t, kHeaderSize>& header,
             std::size_t offset, std::uint32_t value) {
    for (std::size_t index = 0; index < 4U; ++index) {
        header[offset + index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
}

void put_u64(std::array<std::uint8_t, kHeaderSize>& header,
             std::size_t offset, std::uint64_t value) {
    for (std::size_t index = 0; index < 8U; ++index) {
        header[offset + index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
}

std::uint32_t get_u32(const std::uint8_t* bytes) {
    std::uint32_t value = 0U;
    for (std::size_t index = 0; index < 4U; ++index) {
        value |= static_cast<std::uint32_t>(bytes[index]) << (index * 8U);
    }
    return value;
}

std::uint64_t get_u64(const std::uint8_t* bytes) {
    std::uint64_t value = 0U;
    for (std::size_t index = 0; index < 8U; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
}

void validate_for_write(const TokenDataset& dataset) {
    if (dataset.vocab_size == 0U || dataset.sequence_offsets.size() < 2U ||
        dataset.sequence_offsets.front() != 0U ||
        dataset.sequence_offsets.back() != dataset.tokens.size()) {
        throw std::invalid_argument("invalid token shard dataset");
    }
    for (std::size_t index = 1; index < dataset.sequence_offsets.size(); ++index) {
        if (dataset.sequence_offsets[index] <= dataset.sequence_offsets[index - 1U]) {
            throw std::invalid_argument("token shard sequences must be non-empty");
        }
    }
    for (const auto token : dataset.tokens) {
        if (token >= dataset.vocab_size) {
            throw std::invalid_argument("token shard ID exceeds vocabulary");
        }
    }
    if (!dataset.oracle_nll.empty()) {
        throw std::invalid_argument("real-corpus shards do not store oracle NLL");
    }
}

std::uint64_t hash_mapped_dataset(std::uint32_t vocab_size,
                                  const std::uint32_t* tokens,
                                  std::uint64_t token_count,
                                  const std::uint64_t* offsets,
                                  std::uint64_t offset_count) {
    std::uint64_t hash = mix64(vocab_size) ^ 0x544F4B454EULL;
    for (std::uint64_t index = 0U; index < token_count; ++index) {
        if (tokens[index] >= vocab_size) {
            throw std::runtime_error("token shard ID exceeds vocabulary");
        }
        hash ^= mix64(static_cast<std::uint64_t>(tokens[index]) +
                      index * 0x9E3779B97F4A7C15ULL);
        hash = std::rotl(hash, 11);
    }
    for (std::uint64_t index = 0U; index < offset_count; ++index) {
        hash ^= mix64(offsets[index] + index * 0xC2B2AE3D27D4EB4FULL);
        hash = std::rotl(hash, 5);
    }
    return mix64(hash);
}

void write_zeros(std::ofstream& output, std::uint64_t count) {
    constexpr std::array<char, 64> zeros{};
    while (count > 0U) {
        const auto chunk = static_cast<std::streamsize>(
            std::min<std::uint64_t>(count, zeros.size()));
        output.write(zeros.data(), chunk);
        count -= static_cast<std::uint64_t>(chunk);
    }
}

} // namespace

struct MappedTokenShard::Impl {
    const std::uint8_t* data{};
    std::size_t size{};
    std::uint64_t shard_index{};
    std::uint32_t vocab_size{};
    std::uint64_t token_count{};
    std::uint64_t sequence_count{};
    std::uint64_t dataset_hash{};
    const std::uint64_t* offsets{};
    const std::uint32_t* tokens{};
#ifdef _WIN32
    HANDLE file{INVALID_HANDLE_VALUE};
    HANDLE mapping{};
#else
    int file{-1};
#endif

    ~Impl() {
#ifdef _WIN32
        if (data != nullptr) UnmapViewOfFile(data);
        if (mapping != nullptr) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
        if (data != nullptr) munmap(const_cast<std::uint8_t*>(data), size);
        if (file >= 0) close(file);
#endif
    }
};

void write_token_shard(const TokenDataset& dataset, const std::string& path) {
    validate_for_write(dataset);
    const auto offset_count = static_cast<std::uint64_t>(dataset.sequence_offsets.size());
    const auto token_count = static_cast<std::uint64_t>(dataset.tokens.size());
    const auto offsets_bytes = offset_count * sizeof(std::uint64_t);
    const auto tokens_bytes = token_count * sizeof(std::uint32_t);
    const auto offsets_offset = static_cast<std::uint64_t>(kHeaderSize);
    const auto tokens_offset = align_up(offsets_offset + offsets_bytes, kAlignment);
    if (tokens_bytes > std::numeric_limits<std::uint64_t>::max() - tokens_offset) {
        throw std::overflow_error("token shard layout overflow");
    }
    const auto file_size = tokens_offset + tokens_bytes;

    std::array<std::uint8_t, kHeaderSize> header{};
    std::copy(kMagic.begin(), kMagic.end(), header.begin());
    put_u32(header, 8U, kVersion);
    put_u32(header, 12U, kHeaderSize);
    put_u32(header, 16U, kEndianMarker);
    put_u32(header, 20U, dataset.vocab_size);
    put_u64(header, 24U, token_count);
    put_u64(header, 32U, offset_count - 1U);
    put_u64(header, 40U, offsets_offset);
    put_u64(header, 48U, tokens_offset);
    put_u64(header, 56U, hash_dataset(dataset));
    put_u64(header, 64U, file_size);

    const auto destination = std::filesystem::path(path);
    if (std::filesystem::exists(destination)) {
        throw std::runtime_error("refusing to overwrite existing token shard");
    }
    auto temporary = destination;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open token shard for writing");
    output.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(dataset.sequence_offsets.data()),
                 static_cast<std::streamsize>(offsets_bytes));
    write_zeros(output, tokens_offset - offsets_offset - offsets_bytes);
    output.write(reinterpret_cast<const char*>(dataset.tokens.data()),
                 static_cast<std::streamsize>(tokens_bytes));
    output.flush();
    if (!output) throw std::runtime_error("token shard write failed");
    output.close();
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot commit token shard: " + error.message());
    }
}

MappedTokenShard::MappedTokenShard(const std::string& path,
                                   std::uint64_t shard_index,
                                   bool verify_payload)
    : impl_(std::make_unique<Impl>()) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("token shard mapping requires a little-endian host");
    }
    impl_->shard_index = shard_index;
#ifdef _WIN32
    impl_->file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open token shard");
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(impl_->file, &size) || size.QuadPart <= 0) {
        throw std::runtime_error("cannot inspect token shard size");
    }
    impl_->size = static_cast<std::size_t>(size.QuadPart);
    impl_->mapping = CreateFileMappingA(impl_->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (impl_->mapping == nullptr) throw std::runtime_error("cannot map token shard");
    impl_->data = static_cast<const std::uint8_t*>(
        MapViewOfFile(impl_->mapping, FILE_MAP_READ, 0, 0, 0));
    if (impl_->data == nullptr) throw std::runtime_error("cannot view token shard");
#else
    impl_->file = open(path.c_str(), O_RDONLY);
    if (impl_->file < 0) throw std::runtime_error("cannot open token shard");
    struct stat status {};
    if (fstat(impl_->file, &status) != 0 || status.st_size <= 0) {
        throw std::runtime_error("cannot inspect token shard size");
    }
    impl_->size = static_cast<std::size_t>(status.st_size);
    void* view = mmap(nullptr, impl_->size, PROT_READ, MAP_PRIVATE, impl_->file, 0);
    if (view == MAP_FAILED) throw std::runtime_error("cannot map token shard");
    impl_->data = static_cast<const std::uint8_t*>(view);
#endif
    if (impl_->size < kHeaderSize ||
        !std::equal(kMagic.begin(), kMagic.end(), impl_->data) ||
        get_u32(impl_->data + 8U) != kVersion ||
        get_u32(impl_->data + 12U) != kHeaderSize ||
        get_u32(impl_->data + 16U) != kEndianMarker) {
        throw std::runtime_error("invalid token shard header");
    }
    impl_->vocab_size = get_u32(impl_->data + 20U);
    impl_->token_count = get_u64(impl_->data + 24U);
    impl_->sequence_count = get_u64(impl_->data + 32U);
    const auto offsets_offset = get_u64(impl_->data + 40U);
    const auto tokens_offset = get_u64(impl_->data + 48U);
    impl_->dataset_hash = get_u64(impl_->data + 56U);
    const auto declared_size = get_u64(impl_->data + 64U);
    if (impl_->sequence_count == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("invalid token shard sequence count");
    }
    const auto offset_count = impl_->sequence_count + 1U;
    if (offset_count > (std::numeric_limits<std::uint64_t>::max() - kHeaderSize) /
                           sizeof(std::uint64_t) ||
        impl_->token_count > std::numeric_limits<std::uint64_t>::max() /
                                 sizeof(std::uint32_t)) {
        throw std::runtime_error("token shard layout overflow");
    }
    const auto expected_tokens_offset = align_up(
        kHeaderSize + offset_count * sizeof(std::uint64_t), kAlignment);
    const auto token_bytes = impl_->token_count * sizeof(std::uint32_t);
    if (token_bytes > std::numeric_limits<std::uint64_t>::max() - expected_tokens_offset) {
        throw std::runtime_error("token shard layout overflow");
    }
    const auto expected_size = expected_tokens_offset +
        token_bytes;
    if (impl_->vocab_size == 0U || impl_->sequence_count == 0U ||
        impl_->token_count == 0U || declared_size != impl_->size ||
        offsets_offset != kHeaderSize || tokens_offset != expected_tokens_offset ||
        declared_size != expected_size ||
        offset_count > (impl_->size - offsets_offset) / sizeof(std::uint64_t) ||
        tokens_offset > impl_->size ||
        impl_->token_count > (impl_->size - tokens_offset) / sizeof(std::uint32_t)) {
        throw std::runtime_error("invalid token shard layout");
    }
    impl_->offsets = reinterpret_cast<const std::uint64_t*>(impl_->data + offsets_offset);
    impl_->tokens = reinterpret_cast<const std::uint32_t*>(impl_->data + tokens_offset);
    if (impl_->offsets[0] != 0U ||
        impl_->offsets[impl_->sequence_count] != impl_->token_count) {
        throw std::runtime_error("invalid token shard offsets");
    }
    for (std::uint64_t index = 1U; index <= impl_->sequence_count; ++index) {
        if (impl_->offsets[index] <= impl_->offsets[index - 1U]) {
            throw std::runtime_error("token shard contains an empty sequence");
        }
    }
    if (verify_payload) {
        if (hash_mapped_dataset(impl_->vocab_size, impl_->tokens, impl_->token_count,
                                impl_->offsets, offset_count) != impl_->dataset_hash) {
            throw std::runtime_error("token shard checksum mismatch");
        }
    }
}

MappedTokenShard::~MappedTokenShard() = default;
MappedTokenShard::MappedTokenShard(MappedTokenShard&&) noexcept = default;
MappedTokenShard& MappedTokenShard::operator=(MappedTokenShard&&) noexcept = default;

std::uint32_t MappedTokenShard::vocab_size() const noexcept { return impl_->vocab_size; }
std::uint64_t MappedTokenShard::token_count() const noexcept { return impl_->token_count; }
std::uint64_t MappedTokenShard::sequence_count() const noexcept { return impl_->sequence_count; }
std::uint64_t MappedTokenShard::dataset_hash() const noexcept { return impl_->dataset_hash; }
std::span<const std::uint32_t> MappedTokenShard::tokens() const noexcept {
    return {impl_->tokens, static_cast<std::size_t>(impl_->token_count)};
}
std::span<const std::uint64_t> MappedTokenShard::sequence_offsets() const noexcept {
    return {impl_->offsets, static_cast<std::size_t>(impl_->sequence_count + 1U)};
}

bool MappedTokenShard::next(TokenShardCursor& cursor, TokenExample& example) const {
    if (cursor.shard_index != impl_->shard_index) {
        throw std::invalid_argument("token shard cursor belongs to another shard");
    }
    while (cursor.sequence_index < impl_->sequence_count) {
        const auto start = impl_->offsets[cursor.sequence_index];
        const auto end = impl_->offsets[cursor.sequence_index + 1U];
        const auto length = end - start;
        if (cursor.token_offset < length - 1U) {
            const auto position = start + cursor.token_offset;
            example.input = impl_->tokens[position];
            example.target = impl_->tokens[position + 1U];
            ++cursor.token_offset;
            return true;
        }
        ++cursor.sequence_index;
        cursor.token_offset = 0U;
    }
    return false;
}

} // namespace sbm

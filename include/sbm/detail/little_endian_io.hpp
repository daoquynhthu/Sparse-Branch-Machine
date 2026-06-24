#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <type_traits>

namespace sbm::detail {

template <typename Value>
void write_little_endian(std::ostream& output, std::span<const Value> values) {
    static_assert(std::is_same_v<Value, std::uint32_t> ||
                  std::is_same_v<Value, std::uint64_t>);
    constexpr std::size_t buffer_bytes = 16U * 1024U;
    constexpr std::size_t values_per_buffer = buffer_bytes / sizeof(Value);
    std::array<char, buffer_bytes> buffer{};
    std::size_t position = 0U;
    while (position < values.size()) {
        const auto count = std::min(values_per_buffer, values.size() - position);
        for (std::size_t index = 0U; index < count; ++index) {
            const Value value = values[position + index];
            for (std::size_t byte = 0U; byte < sizeof(Value); ++byte) {
                buffer[index * sizeof(Value) + byte] = static_cast<char>(
                    value >> (byte * 8U));
            }
        }
        output.write(buffer.data(),
                     static_cast<std::streamsize>(count * sizeof(Value)));
        position += count;
    }
}

} // namespace sbm::detail

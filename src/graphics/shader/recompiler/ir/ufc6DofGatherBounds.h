#pragma once

#include <bit>
#include <cstdint>

namespace KytyUfc6DofGatherBounds {
inline constexpr uint64_t Shader = 0x1e180cead1bdcd6cull;
inline constexpr uint32_t Slot = 1;
inline constexpr uint32_t Disabled = 0x7f800000u;
constexpr uint32_t LimitDword(uint32_t logical_width, uint32_t native_width) {
    // Gather covers floor(U * width - 0.5) and the next texel. The midpoint
    // between the last two valid centers leaves room for floating-point rounding.
    return logical_width >= 2u && logical_width < native_width
        ? std::bit_cast<uint32_t>(static_cast<float>(logical_width - 1u) /
                                  static_cast<float>(native_width))
        : Disabled;
}
} // namespace KytyUfc6DofGatherBounds

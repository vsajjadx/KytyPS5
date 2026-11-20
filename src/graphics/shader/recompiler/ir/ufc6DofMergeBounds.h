#pragma once

#include <bit>
#include <cstdint>

namespace KytyUfc6DofMergeBounds {
inline constexpr uint64_t Shader = 0xed4b36c85001b428ull;
inline constexpr uint32_t Slot = 2;
inline constexpr uint32_t DepthSlot = 1;
inline constexpr uint32_t Disabled = 0x7f800000u;

constexpr uint32_t LimitDword(uint32_t viewport_width, uint32_t depth_width,
                              uint32_t requested_width, uint32_t native_width) {
    // Only trust the viewport when the corresponding full-resolution depth view agrees.
    // Clamp a linear sample to the center of the final valid texel, leaving V alone.
    return viewport_width != 0 && viewport_width == depth_width &&
                   viewport_width < native_width &&
                   (requested_width == viewport_width || requested_width == native_width)
               ? std::bit_cast<uint32_t>((static_cast<float>(viewport_width) - 0.5f) /
                                         static_cast<float>(native_width))
               : Disabled;
}
} // namespace KytyUfc6DofMergeBounds

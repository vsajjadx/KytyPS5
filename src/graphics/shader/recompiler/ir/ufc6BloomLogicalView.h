#pragma once

#include <cstdint>

namespace KytyUfc6BloomLogicalView {
inline constexpr uint64_t Shader = 0xc455169aed362db1ull;
inline constexpr uint32_t Slot = 1;
inline constexpr uint32_t ResourceSource = 3;

// The bloom upsampler reads the previous, smaller render target with normalized UVs.
// Keep its bilinear footprint within the guest's view when the native allocation is wider.
constexpr bool HasPaddedWidth(uint32_t width, uint32_t height, uint32_t native_width,
                              uint32_t native_height) {
	return width != 0 && height != 0 && width < native_width && height == native_height;
}
constexpr float Scale(uint32_t width, uint32_t native_width) {
	return static_cast<float>(width) / static_cast<float>(native_width);
}
constexpr float MaxU(uint32_t width, uint32_t native_width) {
	return (static_cast<float>(width) - 0.5f) / static_cast<float>(native_width);
}
} // namespace KytyUfc6BloomLogicalView

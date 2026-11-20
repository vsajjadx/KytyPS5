#pragma once
#include <bit>
#include <cstdint>

namespace KytyUfc6WhiteRecovery {
inline constexpr uint64_t DofA = 0x12df451a109b4f17ull;
inline constexpr uint64_t DofB = 0x305d4f388615a943ull;
inline constexpr uint64_t FirstMip = 0x82a58b0a6aaec8bfull;
inline constexpr uint64_t HorizontalWeight = 0x175b582e55e6a51full;
inline constexpr uint64_t Upsample = 0xc455169aed362db1ull;
inline constexpr uint64_t Compose = 0x1227e0bc1235069bull;
inline constexpr uint32_t Disabled = 0x7f800000u;
enum class Mode { None, Dof, FirstMip, UpsampleMip, UpsampleHalf, Compose };
constexpr Mode ForSlot(uint64_t h, uint32_t s) {
    if ((h == DofA && (s == 1 || s == 2)) ||
        (h == DofB && s >= 1 && s <= 3)) return Mode::Dof;
    if (h == FirstMip && s == 0) return Mode::FirstMip;
    if (h == Upsample && s <= 1) return s == 0 ? Mode::UpsampleMip : Mode::UpsampleHalf;
    if (h == Compose && s == 3) return Mode::Compose;
    return Mode::None;
}
constexpr uint32_t Dwords(uint64_t h) {
    return h == DofA || h == Upsample ? 4u : h == DofB ? 6u :
           h == FirstMip || h == Compose ? 2u : 0u;
}
constexpr uint32_t Index(uint64_t h, uint32_t slot) {
    return (h == DofA || h == DofB) ? slot - 1u : h == Upsample ? slot : 0u;
}
struct Bounds {
    uint32_t scale, limit;
};
constexpr Bounds ForWidths(Mode mode, uint32_t valid, uint32_t native) {
    if (mode == Mode::None || valid == 0 || valid >= native)
        return {0x3f800000u, Disabled};
    const auto n = static_cast<float>(native);
    const auto w = static_cast<float>(valid);
    return {mode == Mode::Dof ? 0x3f800000u : std::bit_cast<uint32_t>(w / n),
            std::bit_cast<uint32_t>((w - (mode == Mode::Dof ? .51f : .5f)) / n)};
}
} // namespace KytyUfc6WhiteRecovery

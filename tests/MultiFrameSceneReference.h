#pragma once
// Intended synchronized pinned-SDK SCD math for synthetic encoded RGBA8 pairs.
// Test-only: no SDK/GPU state, counters, policy or measured output is consumed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>

namespace MultiFgScdReference {
struct Result {
    bool valid = false;
    float score = 0;
    std::array<uint32_t, 3> sums {};
};
using Histogram = std::array<uint32_t, 256>;
using Distribution = std::array<float, 256>;

inline float SumTree(Distribution values) {
    for (uint32_t stop : {128u, 64u, 32u, 16u, 8u, 4u, 2u})
        for (uint32_t i = 0; i < stop; ++i) values[i] += values[i + stop];
    return values[0] + values[1];
}
inline uint32_t Luma(uint32_t rgba) {
    const float r = float(rgba & 255u) / 255.0f;
    const float g = float((rgba >> 8u) & 255u) / 255.0f;
    const float b = float((rgba >> 16u) & 255u) / 255.0f;
    const float y = (.2126f * r + .7152f * g + .0722f * b) * 255.0f;
    return static_cast<uint32_t>(std::clamp(y, 0.0f, 255.0f));
}
inline Distribution Filter(const Histogram& histogram, uint32_t shift) {
    constexpr std::array kernel {.0088122291f, .027143577f, .065114059f, .12164907f,
        .17699835f, .20056541f, .17699835f, .12164907f, .065114059f, .027143577f, .0088122291f};
    Distribution smooth {}, output {};
    for (int i = 0; i < 256; ++i) {
        float value = 0;
        for (int tap = 0; tap < 11; ++tap)
            value += kernel[tap] * float(histogram[std::clamp(i - 5 + tap, 0, 255)]);
        smooth[i] = value + 1.0f;
    }
    for (int i = 0; i < 256; ++i)
        output[i] = shift == 0 ? (i == 255 ? 1.0f : smooth[i + 1]) :
                    shift == 2 ? (i == 0 ? 1.0f : smooth[i - 1]) : smooth[i];
    const float sum = SumTree(output);
    for (float& value : output) value /= sum;
    return output;
}
inline Result Evaluate(std::span<const uint32_t> previous, std::span<const uint32_t> current,
                       uint32_t width, uint32_t height) {
    Result result;
    // The original histogram shader samples four pixels at a time. Restrict its
    // CPU reference to exact 4-pixel-wide strata, avoiding any edge/OOB assumption.
    if (width < 12 || width > 4096 || width % 12 != 0 || height < 3 || height > 4096 ||
        uint64_t(width) * height != previous.size() || previous.size() != current.size()) return result;
    const uint32_t dx = width / 3, dy = height / 3;
    for (uint32_t hy = 0; hy < 3; ++hy)
        for (uint32_t hx = 0; hx < 3; ++hx) {
            Histogram old_hist {}, new_hist {};
            for (uint32_t y = hy * dy; y < (hy + 1) * dy; ++y)
                for (uint32_t x = hx * dx; x < (hx + 1) * dx; ++x) {
                    ++old_hist[Luma(previous[std::size_t(y) * width + x])];
                    ++new_hist[Luma(current[std::size_t(y) * width + x])];
                }
            const auto old = Filter(old_hist, 1); // Previous histogram stores the unshifted distribution.
            for (uint32_t shift = 0; shift < 3; ++shift) {
                const auto now = Filter(new_hist, shift);
                Distribution forward {}, backward {};
                for (uint32_t i = 0; i < 256; ++i) {
                    forward[i] = now[i] * std::log(now[i] / old[i]);
                    backward[i] = old[i] * std::log(old[i] / now[i]);
                }
                const float distance = 1.0f - std::exp(-(std::abs(SumTree(forward)) + std::abs(SumTree(backward))));
                if (!std::isfinite(distance) || distance < 0 || distance > 1) return {};
                const auto contribution = static_cast<uint32_t>((distance / 9.0f) * 1'000'000.0f);
                result.sums[shift] += contribution; // Per-group truncation BEFORE integer addition.
            }
        }
    result.score = float(std::min({result.sums[0], result.sums[1], result.sums[2]})) / 1'000'000.0f;
    result.valid = true;
    return result;
}
} // namespace MultiFgScdReference

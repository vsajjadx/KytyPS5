#pragma once

// Independent test-only image oracle. No emulator/GPU/presentation code is used.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace MultiFgOracle {
using Pixel = uint32_t;
constexpr Pixel Canary = 0x53af0db2u;

struct View {
    std::span<const Pixel> words;
    uint32_t width = 0, height = 0, stride = 0;
};
struct Scene {
    uint32_t width = 256, height = 128, variant = 0;
    double dx = 12, dy = 6;
    uint32_t previous_frame = 11; // The native driver warms at least eleven pairs first.
};
struct Result {
    bool okay = false;
    std::string reason;
    double mae_bytes = 0, rmse_bytes = 0;
    double alternate_mse = 0, blend_mse = 0;
};

inline uint32_t Byte(Pixel p, uint32_t channel) { return (p >> (channel * 8u)) & 255u; }
inline Pixel Pack(double r, double g, double b) {
    const auto q = [](double x) { return uint32_t(std::clamp(std::lround(x), 0l, 255l)); };
    return q(r) | (q(g) << 8u) | (q(b) << 16u) | 0xff000000u;
}

// Rich two-dimensional, nonperiodic-in-window texture: multiple incommensurate
// phases, a checker and position-dependent modulation. Not an aperture-only edge.
inline Pixel Grid(int64_t x, int64_t y, uint32_t variant) {
    const double phase = variant ? 1.13 : 0.0;
    const double checker = ((int64_t(std::floor(double(x) / 7)) +
                              int64_t(std::floor(double(y) / 5))) & 1) ? 1.0 : -1.0;
    return Pack(125 + 45 * std::sin(.237 * x + .071 * y + phase) + 24 * std::cos(.113 * y),
                125 + 43 * std::cos(.311 * x - .193 * y + phase) + 21 * std::sin(.137 * x),
                125 + 28 * checker + 25 * std::sin(.053 * x + .097 * y + phase));
}

// Quantized texels + independent bilinear translation match a correctly warped
// RGBA8 image; exact known flow needs at most ~1-2 byte rounding error.
inline Pixel Sample(double x, double y, uint32_t variant) {
    const int64_t ix = int64_t(std::floor(x)), iy = int64_t(std::floor(y));
    const double fx = x - ix, fy = y - iy;
    const std::array<Pixel, 4> p {Grid(ix, iy, variant), Grid(ix + 1, iy, variant),
                                 Grid(ix, iy + 1, variant), Grid(ix + 1, iy + 1, variant)};
    std::array<double, 3> value {};
    for (uint32_t c = 0; c < 3; ++c) {
        value[c] = ((1 - fx) * Byte(p[0], c) + fx * Byte(p[1], c)) * (1 - fy) +
                   ((1 - fx) * Byte(p[2], c) + fx * Byte(p[3], c)) * fy;
    }
    return Pack(value[0], value[1], value[2]);
}

inline std::vector<Pixel> Frame(const Scene& scene, double frame) {
    std::vector<Pixel> pixels(size_t(scene.width) * scene.height);
    for (uint32_t y = 0; y < scene.height; ++y)
        for (uint32_t x = 0; x < scene.width; ++x)
            pixels[size_t(y) * scene.width + x] =
                Sample(x - frame * scene.dx, y - frame * scene.dy, scene.variant);
    return pixels;
}
inline std::vector<Pixel> Blend(const Scene& scene, double t) {
    const auto a = Frame(scene, scene.previous_frame);
    const auto b = Frame(scene, scene.previous_frame + 1);
    std::vector<Pixel> pixels(a.size());
    for (size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = Pack((1 - t) * Byte(a[i], 0) + t * Byte(b[i], 0),
                         (1 - t) * Byte(a[i], 1) + t * Byte(b[i], 1),
                         (1 - t) * Byte(a[i], 2) + t * Byte(b[i], 2));
    return pixels;
}
inline bool Valid(View image) {
    return image.width != 0 && image.height != 0 && image.stride >= image.width &&
           uint64_t(image.stride) * (image.height - 1u) + image.width <= image.words.size();
}
inline View Packed(const std::vector<Pixel>& pixels, const Scene& scene) {
    return {pixels, scene.width, scene.height, scene.width};
}

inline double Mse(View image, View reference, uint32_t margin = 32) {
    if (!Valid(image) || !Valid(reference) || image.width != reference.width ||
        image.height != reference.height || image.width <= margin * 2 || image.height <= margin * 2)
        return INFINITY;
    double error = 0; uint64_t channels = 0;
    for (uint32_t y = margin; y < image.height - margin; ++y)
        for (uint32_t x = margin; x < image.width - margin; ++x)
            for (uint32_t c = 0; c < 3; ++c) {
                const double d = double(Byte(image.words[size_t(y) * image.stride + x], c)) -
                                 Byte(reference.words[size_t(y) * reference.stride + x], c);
                error += d * d; ++channels;
            }
    return error / channels;
}

inline Result Position(View image, const Scene& scene, uint32_t factor, uint32_t ordinal) {
    Result result;
    if (factor < 2 || factor > 6 || ordinal == 0 || ordinal >= factor || !Valid(image) ||
        image.width != scene.width || image.height != scene.height ||
        scene.width <= 64 || scene.height <= 64) {
        result.reason = "invalid factor/ordinal/view"; return result;
    }
    const double t = double(ordinal) / factor;
    const auto reference = Frame(scene, scene.previous_frame + t);
    const auto blend = Blend(scene, t);
    const double mse = Mse(image, Packed(reference, scene));
    result.rmse_bytes = std::sqrt(mse);
    result.blend_mse = Mse(image, Packed(blend, scene));
    result.alternate_mse = INFINITY;
    for (uint32_t other = 0; other <= factor; ++other) {
        if (other == ordinal) continue;
        const auto alternative = Frame(scene, scene.previous_frame + double(other) / factor);
        result.alternate_mse = std::min(result.alternate_mse, Mse(image, Packed(alternative, scene)));
    }
    double absolute = 0; uint64_t channels = 0;
    for (uint32_t y = 32; y < scene.height - 32; ++y)
        for (uint32_t x = 32; x < scene.width - 32; ++x) {
            const auto actual = image.words[size_t(y) * image.stride + x];
            if (Byte(actual, 3) != 255) { result.reason = "unexpected alpha"; return result; }
            for (uint32_t c = 0; c < 3; ++c) {
                absolute += std::abs(int(Byte(actual, c)) - int(Byte(reference[size_t(y) * scene.width + x], c)));
                ++channels;
            }
        }
    result.mae_bytes = absolute / channels;
    // Predeclared easy-motion envelope, not tuned from a GPU run. Approximate
    // flow may vary, but every image must match its own t better than ALL other
    // endpoints/subpositions and better than the endpoint cross-fade mutant.
    result.okay = std::isfinite(mse) && result.mae_bytes <= 8 && result.rmse_bytes <= 16 &&
                  mse < .60 * result.alternate_mse && mse + 4 < .80 * result.blend_mse;
    result.reason = result.okay ? "okay" : "temporal position/texture error or cross-fade";
    return result;
}

inline bool OutsideLogicalUntouched(View physical, uint32_t logical_width, uint32_t logical_height) {
    if (!Valid(physical) || logical_width > physical.width || logical_height > physical.height) return false;
    for (uint32_t y = 0; y < physical.height; ++y)
        for (uint32_t x = 0; x < physical.width; ++x)
            if ((x >= logical_width || y >= logical_height) &&
                physical.words[size_t(y) * physical.stride + x] != Canary) return false;
    return true;
}
} // namespace MultiFgOracle

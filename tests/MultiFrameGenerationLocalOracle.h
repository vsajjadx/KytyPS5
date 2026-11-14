#pragma once
#include "MultiFrameGenerationOracle.h"

namespace MultiFgLocalOracle {
using namespace MultiFgOracle;
enum class Kind { OppositePanels, Occlusion };
enum class Region { Left, Right, Foreground, Background };
struct Scene {
    Kind kind;
    uint32_t Width() const { return kind == Kind::OppositePanels ? 512 : 384; }
    uint32_t Height() const { return 256; }
};
struct Result {
    Region region;
    uint64_t pixels = 0;
    double mae = 0, rmse = 0, alternate_mse = INFINITY, blend_mse = INFINITY;
    bool okay = false;
};
inline const char* Name(Region region) {
    switch (region) {
    case Region::Left: return "left";
    case Region::Right: return "right";
    case Region::Foreground: return "foreground";
    case Region::Background: return "background";
    }
    return "invalid";
}
inline bool Rectangle(double x, double y, double frame, double inset = 0) {
    const double u = frame - 11;
    return x >= 112 + u * 12 + inset && x < 256 + u * 12 - inset &&
           y >= 80 - u * 6 + inset && y < 176 - u * 6 - inset;
}
inline Pixel Texel(const Scene& scene, uint32_t x, uint32_t y, double frame) {
    if (scene.kind == Kind::OppositePanels) {
        const bool left = x < 256;
        const double dx = left ? 12 : -12, dy = left ? -6 : 6;
        return Sample(x - frame * dx, y - frame * dy, left ? 0 : 1);
    }
    const double u = frame - 11;
    return Rectangle(x + .5, y + .5, frame) ? Sample(x - u * 12, y + u * 6, 0) : Grid(x, y, 1);
}
inline std::vector<Pixel> Frame(const Scene& scene, double frame) {
    std::vector<Pixel> result(size_t(scene.Width()) * scene.Height());
    for (uint32_t y = 0; y < scene.Height(); ++y)
        for (uint32_t x = 0; x < scene.Width(); ++x)
            result[size_t(y) * scene.Width() + x] = Texel(scene, x, y, frame);
    return result;
}
inline View Packed(const std::vector<Pixel>& pixels, const Scene& scene) {
    return {pixels, scene.Width(), scene.Height(), scene.Width()};
}
// Immutable test references are shared by the positive and mutant evaluations;
// they depend only on this mathematical scene, never on measured GPU pixels.
inline const std::vector<Pixel>& Reference(const Scene& scene, uint32_t factor, uint32_t ordinal) {
    static std::array<std::array<std::array<std::vector<Pixel>, 7>, 7>, 2> images;
    auto& image = images[static_cast<unsigned>(scene.kind)][factor][ordinal];
    if (image.empty()) image = Frame(scene, 11 + double(ordinal) / factor);
    return image;
}
inline bool InMask(const Scene& scene, Region region, uint32_t x, uint32_t y, double t) {
    if (x < 32 || y < 32 || x >= scene.Width() - 32 || y >= scene.Height() - 32) return false;
    if (scene.kind == Kind::OppositePanels)
        return region == Region::Left ? x < 224 : region == Region::Right && x >= 288;
    const double px = x + .5, py = y + .5;
    if (region == Region::Foreground)
        // The same rigid-surface sample must be safely inside the object at BOTH endpoints.
        return Rectangle(px, py, 11 + t, 3) && Rectangle(px - t * 12, py + t * 6, 11, 3) &&
               Rectangle(px + (1 - t) * 12, py - (1 - t) * 6, 12, 3);
    if (region == Region::Background)
        // Never grade a reveal sample hidden by the foreground in either real input.
        return !Rectangle(px, py, 11, -3) && !Rectangle(px, py, 12, -3) &&
               !Rectangle(px, py, 11 + t, -3);
    return false;
}
inline std::array<Region, 2> Regions(const Scene& scene) {
    return scene.kind == Kind::OppositePanels ? std::array {Region::Left, Region::Right} :
        std::array {Region::Foreground, Region::Background};
}
inline double MaskMse(View image, const std::vector<Pixel>& reference, const Scene& scene,
                      Region region, double t) {
    double error = 0; uint64_t count = 0;
    for (uint32_t y = 32; y < scene.Height() - 32; ++y)
        for (uint32_t x = 32; x < scene.Width() - 32; ++x) if (InMask(scene, region, x, y, t))
            for (uint32_t c = 0; c < 3; ++c) {
                const double d = double(Byte(image.words[size_t(y) * image.stride + x], c)) -
                    Byte(reference[size_t(y) * scene.Width() + x], c);
                error += d * d; ++count;
            }
    return count ? error / count : INFINITY;
}
inline std::vector<Pixel> Blend(const Scene& scene, double t) {
    const auto& a = Reference(scene, 2, 0);
    const auto& b = Reference(scene, 2, 2);
    std::vector<Pixel> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = Pack((1 - t) * Byte(a[i], 0) + t * Byte(b[i], 0),
                         (1 - t) * Byte(a[i], 1) + t * Byte(b[i], 1),
                         (1 - t) * Byte(a[i], 2) + t * Byte(b[i], 2));
    return result;
}
inline std::array<Result, 2> Position(View image, const Scene& scene, uint32_t factor, uint32_t ordinal) {
    const auto regions = Regions(scene);
    std::array<Result, 2> results {Result {regions[0]}, Result {regions[1]}};
    if (!Valid(image) || image.width != scene.Width() || image.height != scene.Height() ||
        factor < 2 || factor > 6 || ordinal < 1 || ordinal >= factor) return results;
    const double t = double(ordinal) / factor;
    const auto& expected = Reference(scene, factor, ordinal);
    const auto blend = Blend(scene, t);
    for (auto& result : results) {
        double absolute = 0; bool alpha = true;
        for (uint32_t y = 32; y < scene.Height() - 32; ++y)
            for (uint32_t x = 32; x < scene.Width() - 32; ++x) if (InMask(scene, result.region, x, y, t)) {
                const auto a = image.words[size_t(y) * image.stride + x], b = expected[size_t(y) * scene.Width() + x];
                alpha &= Byte(a, 3) == 255;
                for (uint32_t c = 0; c < 3; ++c) absolute += std::abs(int(Byte(a, c)) - int(Byte(b, c)));
                ++result.pixels;
            }
        if (!result.pixels) continue;
        const double mse = MaskMse(image, expected, scene, result.region, t);
        result.mae = absolute / (3 * result.pixels); result.rmse = std::sqrt(mse);
        result.blend_mse = MaskMse(image, blend, scene, result.region, t);
        const uint64_t minimum = result.region == Region::Foreground ? 10'000 : 25'000;
        result.okay = alpha && result.pixels >= minimum && result.mae <= 8 && result.rmse <= 16;
        if (result.region != Region::Background) {
            for (uint32_t other = 0; other <= factor; ++other) if (other != ordinal) {
                const auto& alternate = Reference(scene, factor, other);
                result.alternate_mse = std::min(result.alternate_mse,
                    MaskMse(image, alternate, scene, result.region, t));
            }
            result.okay &= mse < .60 * result.alternate_mse && mse + 4 < .80 * result.blend_mse;
        }
    }
    return results;
}
inline bool AllOkay(const std::array<Result, 2>& results) { return results[0].okay && results[1].okay; }
inline Pixel SamplePixels(View previous, double x, double y) {
    x = std::clamp(x, 0.0, double(previous.width - 1)); y = std::clamp(y, 0.0, double(previous.height - 1));
    const uint32_t ax = uint32_t(x), ay = uint32_t(y), bx = std::min(ax + 1, previous.width - 1),
                   by = std::min(ay + 1, previous.height - 1);
    const double fx = x - ax, fy = y - ay;
    std::array<double, 3> rgb {};
    for (uint32_t c = 0; c < 3; ++c)
        rgb[c] = ((1 - fx) * Byte(previous.words[size_t(ay) * previous.stride + ax], c) +
                  fx * Byte(previous.words[size_t(ay) * previous.stride + bx], c)) * (1 - fy) +
                 ((1 - fx) * Byte(previous.words[size_t(by) * previous.stride + ax], c) +
                  fx * Byte(previous.words[size_t(by) * previous.stride + bx], c)) * fy;
    return Pack(rgb[0], rgb[1], rgb[2]);
}
// CPU validation/mutants only. Per-pixel known region vectors never enter the emulator.
inline std::vector<Pixel> KnownWarp(const Scene& scene, double t, bool force_global = false,
                                  double global_dx = 0, double global_dy = 0) {
    const auto previous = Frame(scene, 11); const auto view = Packed(previous, scene);
    std::vector<Pixel> output(previous.size());
    for (uint32_t y = 0; y < scene.Height(); ++y)
        for (uint32_t x = 0; x < scene.Width(); ++x) {
            double dx = 0, dy = 0;
            if (force_global) { dx = global_dx; dy = global_dy; }
            else if (scene.kind == Kind::OppositePanels) { dx = x < 256 ? 12 : -12; dy = x < 256 ? -6 : 6; }
            else if (Rectangle(x + .5, y + .5, 11 + t)) { dx = 12; dy = -6; }
            output[size_t(y) * scene.Width() + x] = SamplePixels(view, x - t * dx, y - t * dy);
        }
    return output;
}
} // namespace MultiFgLocalOracle

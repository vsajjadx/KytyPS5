#pragma once
// Private synthetic-image diagnostic only. It cannot change the pass/fail oracle.
#include "MultiFrameGenerationOracle.h"
#include <cstdio>
#include <cstdlib>
#include <bit>
#include <filesystem>
#include <limits>
#include <system_error>

namespace MultiFgDiagnostic {
using namespace MultiFgOracle;
struct Error { double mae = 0, rmse = 0; uint64_t pixels = 0; };
struct Match { double x = 0, y = 0, mse = INFINITY; };

inline Error Region(View output, View reference, uint32_t x0, uint32_t y0,
                    uint32_t x1, uint32_t y1) {
    double absolute = 0, squares = 0;
    Error r;
    if (!Valid(output) || !Valid(reference) || output.width != reference.width ||
        output.height != reference.height || x0 >= x1 || y0 >= y1 ||
        x1 > output.width || y1 > output.height) return r;
    for (uint32_t y = y0; y < y1; ++y)
        for (uint32_t x = x0; x < x1; ++x) {
            for (uint32_t c = 0; c < 3; ++c) {
                const double d = double(Byte(output.words[size_t(y) * output.stride + x], c)) -
                    Byte(reference.words[size_t(y) * reference.stride + x], c);
                absolute += std::abs(d); squares += d * d;
            }
            ++r.pixels;
        }
    if (r.pixels) { r.mae = absolute / (3 * r.pixels); r.rmse = std::sqrt(squares / (3 * r.pixels)); }
    return r;
}

inline double SampleChannel(View source, double x, double y, uint32_t channel) {
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    if (ix < 0 || iy < 0 || ix + 1 >= int(source.width) || iy + 1 >= int(source.height)) return NAN;
    const double fx = x - ix, fy = y - iy;
    const auto a = source.words[size_t(iy) * source.stride + ix];
    const auto b = source.words[size_t(iy) * source.stride + ix + 1];
    const auto c = source.words[size_t(iy + 1) * source.stride + ix];
    const auto d = source.words[size_t(iy + 1) * source.stride + ix + 1];
    return ((1 - fx) * Byte(a, channel) + fx * Byte(b, channel)) * (1 - fy) +
           ((1 - fx) * Byte(c, channel) + fx * Byte(d, channel)) * fy;
}

// Independent fit to actual saved endpoint pixels: O(x,y) ~= P(x-shift_x,y-shift_y).
// Sparse interior samples keep the bounded search cheap. This is diagnostic, never a gate.
inline double TranslationMse(View output, View previous, double sx, double sy) {
    if (!Valid(output) || !Valid(previous) || output.width != previous.width ||
        output.height != previous.height || output.width <= 64 || output.height <= 64) return INFINITY;
    double squares = 0; uint64_t count = 0;
    for (uint32_t y = 32; y < output.height - 32; y += 4)
        for (uint32_t x = 32; x < output.width - 32; x += 4)
            for (uint32_t c = 0; c < 3; ++c) {
                const double ref = SampleChannel(previous, x - sx, y - sy, c);
                if (!std::isfinite(ref)) return INFINITY;
                const double d = Byte(output.words[size_t(y) * output.stride + x], c) - ref;
                squares += d * d; ++count;
            }
    return count ? squares / count : INFINITY;
}

inline Match BestTranslation(View output, View previous) {
    Match best;
    for (int y = -12; y <= 12; ++y)
        for (int x = -24; x <= 24; ++x) {
            const double mse = TranslationMse(output, previous, x, y);
            if (mse < best.mse) best = {double(x), double(y), mse};
        }
    const double coarse_x = best.x, coarse_y = best.y;
    for (int iy = -4; iy <= 4; ++iy)
        for (int ix = -4; ix <= 4; ++ix) {
            const double x = coarse_x + ix * .25, y = coarse_y + iy * .25;
            const double mse = TranslationMse(output, previous, x, y);
            if (mse < best.mse) best = {x, y, mse};
        }
    return best;
}

inline bool Ppm(const std::filesystem::path& path, View image) {
    if (!Valid(image)) return false;
    auto *f = std::fopen(path.string().c_str(), "wb");
    if (!f) return false;
    bool okay = std::fprintf(f, "P6\n%u %u\n255\n", image.width, image.height) > 0;
    for (uint32_t y = 0; y < image.height && okay; ++y)
        for (uint32_t x = 0; x < image.width && okay; ++x) {
            const auto p = image.words[size_t(y) * image.stride + x];
            const unsigned char rgb[] = {static_cast<unsigned char>(Byte(p, 0)),
                static_cast<unsigned char>(Byte(p, 1)), static_cast<unsigned char>(Byte(p, 2))};
            okay = std::fwrite(rgb, 1, 3, f) == 3;
        }
    return std::fclose(f) == 0 && okay;
}

inline bool Raw(const std::filesystem::path& path, View image) {
    if (!Valid(image)) return false;
    auto *f = std::fopen(path.string().c_str(), "wb");
    if (!f) return false;
    bool okay = true;
    for (uint32_t y = 0; y < image.height && okay; ++y)
        okay = std::fwrite(image.words.data() + size_t(y) * image.stride, sizeof(uint32_t),
                          image.width, f) == image.width;
    return std::fclose(f) == 0 && okay;
}
inline bool RawWords(const std::filesystem::path& path, std::span<const uint32_t> words) {
    auto *f = std::fopen(path.string().c_str(), "wb");
    if (!f) return false;
    const bool okay = std::fwrite(words.data(), sizeof(uint32_t), words.size(), f) == words.size();
    return std::fclose(f) == 0 && okay;
}

// Root explicitly chooses a NEW private G path. No game images, arbitrary reads,
// replacement files, GUI or external output are accessed by this helper.
inline std::filesystem::path EventDirectory(uint32_t factor, uint32_t variant, uint32_t ordinal) {
    const char *value = std::getenv("KYTY_MULTIFG_ORACLE_DUMP_DIR");
    if (!value || !*value) return {};
    static std::filesystem::path root;
    static bool attempted = false, ready = false;
    static uint32_t events = 0;
    std::error_code error;
    if (!attempted) {
        attempted = true;
        root = std::filesystem::path(value).lexically_normal();
        auto text = root.generic_string();
        for (char& c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        constexpr const char *prefix = "g:/ps4/emu/kyty-perf/";
        const auto prefix_size = std::char_traits<char>::length(prefix);
        if (!text.starts_with(prefix) || text.size() <= prefix_size ||
            text.find('/', prefix_size) != std::string::npos || text.find("..") != std::string::npos) return {};
        // One newly created directory is the no-overwrite boundary for all artifacts.
        ready = std::filesystem::create_directory(root, error) && !error;
        if (!ready) std::fprintf(stderr, "MULTIFG_DUMP refused_existing_or_invalid_root=%s\n", value);
    }
    if (!ready || events >= 30) return {};
    const auto directory = root / ("factor" + std::to_string(factor) + "-variant" +
        std::to_string(variant) + "-ordinal" + std::to_string(ordinal));
    if (!std::filesystem::create_directory(directory, error) || error) return {};
    ++events;
    return directory;
}

inline void Save(const std::filesystem::path& directory, View previous, View current,
                 View output, const Scene& scene, uint32_t factor, uint32_t ordinal,
                 const Result& oracle) {
    if (directory.empty() || !Valid(previous) || !Valid(current) || !Valid(output) ||
        previous.width != scene.width || previous.height != scene.height ||
        current.width != scene.width || current.height != scene.height ||
        output.width != scene.width || output.height != scene.height ||
        scene.width <= 64 || scene.height <= 64 ||
        uint64_t(scene.width) * scene.height > 1'048'576 || factor < 2 || factor > 6 ||
        ordinal < 1 || ordinal >= factor) return;
    const double t = double(ordinal) / factor;
    const auto ref = Frame(scene, scene.previous_frame + t);
    const auto p_ref = Frame(scene, scene.previous_frame);
    const auto c_ref = Frame(scene, scene.previous_frame + 1);
    const auto blend = Blend(scene, t);
    bool files = Ppm(directory / "P-gpu.ppm", previous) & Ppm(directory / "C-gpu.ppm", current) &
        Ppm(directory / "O-gpu.ppm", output) & Ppm(directory / "expected.ppm", Packed(ref, scene)) &
        Ppm(directory / "endpoint-blend.ppm", Packed(blend, scene));
    files &= Raw(directory / "P-gpu.rgba8.bin", previous) & Raw(directory / "C-gpu.rgba8.bin", current) &
        Raw(directory / "O-gpu.rgba8.bin", output) & Raw(directory / "expected.rgba8.bin", Packed(ref, scene));
    const auto fit = BestTranslation(output, previous);
    const auto p_error = Region(previous, Packed(p_ref, scene), 0, 0, scene.width, scene.height);
    const auto c_error = Region(current, Packed(c_ref, scene), 0, 0, scene.width, scene.height);
    auto *f = std::fopen((directory / "analysis.json").string().c_str(), "wb");
    if (f) {
        std::fprintf(f, "{\"width\":%u,\"height\":%u,\"raw_image_format\":\"RGBA8_UNORM_tight_little_endian\","
            "\"original_output_stride_pixels\":%u,\"factor\":%u,\"variant\":%u,\"ordinal\":%u,\"t\":%.9f,\"previous_frame\":%u,"
            "\"scene_dx\":%.9f,\"scene_dy\":%.9f,\"expected_relative_shift\":[%.9f,%.9f],"
            "\"fit_relative_to_P\":[%.9f,%.9f],\"fit_sparse_mse\":%.9f,"
            "\"P_gpu_cpu_mae\":%.9f,\"C_gpu_cpu_mae\":%.9f,"
            "\"oracle_mae\":%.9f,\"oracle_rmse\":%.9f,\"alternate_mse\":%.9f,\"blend_mse\":%.9f,\"oracle_okay\":%s,"
            "\"translation_fit_is_diagnostic_only\":true,\"raw_flow_unknown_until_separate_copy\":true}\n",
            scene.width, scene.height, output.stride, factor, scene.variant, ordinal, t, scene.previous_frame, scene.dx, scene.dy,
            t * scene.dx, t * scene.dy, fit.x, fit.y, fit.mse, p_error.mae, c_error.mae,
            oracle.mae_bytes, oracle.rmse_bytes, oracle.alternate_mse, oracle.blend_mse,
            oracle.okay ? "true" : "false");
        files &= std::fclose(f) == 0;
    } else files = false;
    f = std::fopen((directory / "roi-errors.csv").string().c_str(), "wb");
    if (f) {
        std::fprintf(f, "x0,y0,x1,y1,pixels,mae_byte,rmse_byte\n");
        for (uint32_t y = 32; y < scene.height - 32; y += 32)
            for (uint32_t x = 32; x < scene.width - 32; x += 32) {
                const auto x1 = std::min(x + 32, scene.width - 32), y1 = std::min(y + 32, scene.height - 32);
                const auto e = Region(output, Packed(ref, scene), x, y, x1, y1);
                std::fprintf(f, "%u,%u,%u,%u,%llu,%.9f,%.9f\n", x, y, x1, y1,
                    static_cast<unsigned long long>(e.pixels), e.mae, e.rmse);
            }
        files &= std::fclose(f) == 0;
    } else files = false;
    f = std::fopen((directory / "differences-float.csv").string().c_str(), "wb");
    if (f) {
        std::fprintf(f, "x,y,r_signed_normalized,g_signed_normalized,b_signed_normalized\n");
        for (uint32_t y = 0; y < scene.height; ++y)
            for (uint32_t x = 0; x < scene.width; ++x) {
                const auto a = output.words[size_t(y) * output.stride + x], b = ref[size_t(y) * scene.width + x];
                std::fprintf(f, "%u,%u,%.9f,%.9f,%.9f\n", x, y,
                    (int(Byte(a, 0)) - int(Byte(b, 0))) / 255.0,
                    (int(Byte(a, 1)) - int(Byte(b, 1))) / 255.0,
                    (int(Byte(a, 2)) - int(Byte(b, 2))) / 255.0);
            }
        files &= std::fclose(f) == 0;
    } else files = false;
    std::fprintf(stderr, "MULTIFG_DUMP path=%s files_okay=%u fit_shift=(%.3f,%.3f) fit_mse=%.6f expected=(%.3f,%.3f)\n",
        directory.string().c_str(), files ? 1u : 0u, fit.x, fit.y, fit.mse, t * scene.dx, t * scene.dy);
}

inline void SaveFlow(const std::filesystem::path& directory, std::span<const uint32_t> words,
                     std::span<const uint32_t> scd, View stored_previous, View stored_current,
                     View input_previous, View input_current, const Scene& scene) {
    const uint32_t width = (scene.width + 7) / 8, height = (scene.height + 7) / 8;
    if (directory.empty() || words.size() != uint64_t(width) * height || scd.size() < 2 ||
        !Valid(stored_previous) || !Valid(stored_current) || !Valid(input_previous) || !Valid(input_current)) return;
    Ppm(directory / "P-stored-history.ppm", stored_previous);
    Ppm(directory / "C-stored-history.ppm", stored_current);
    Raw(directory / "P-stored-history.rgba8.bin", stored_previous);
    Raw(directory / "C-stored-history.rgba8.bin", stored_current);
    RawWords(directory / "flow-rg16sint.bin", words);
    RawWords(directory / "scd-u32.bin", scd.first(2));
    auto *f = std::fopen((directory / "raw-flow.csv").string().c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "cell_x,cell_y,raw_dx_pixel,raw_dy_pixel,interior_roi\n");
    double sum_x = 0, sum_y = 0;
    uint32_t count = 0, expected = 0, opposite = 0, zero = 0;
    int min_x = 32767, max_x = -32768, min_y = 32767, max_y = -32768;
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            const auto word = words[size_t(y) * width + x];
            const int dx = std::bit_cast<int16_t>(static_cast<uint16_t>(word));
            const int dy = std::bit_cast<int16_t>(static_cast<uint16_t>(word >> 16));
            const bool inside = x * 8 >= 32 && (x + 1) * 8 <= scene.width - 32 &&
                y * 8 >= 32 && (y + 1) * 8 <= scene.height - 32;
            std::fprintf(f, "%u,%u,%d,%d,%u\n", x, y, dx, dy, inside ? 1u : 0u);
            if (inside) {
                ++count; sum_x += dx; sum_y += dy;
                min_x = std::min(min_x, dx); max_x = std::max(max_x, dx);
                min_y = std::min(min_y, dy); max_y = std::max(max_y, dy);
                expected += dx == -scene.dx && dy == -scene.dy;
                opposite += dx == scene.dx && dy == scene.dy;
                zero += dx == 0 && dy == 0;
            }
        }
    std::fclose(f);
    const auto p = Region(stored_previous, input_previous, 0, 0, scene.width, scene.height);
    const auto c = Region(stored_current, input_current, 0, 0, scene.width, scene.height);
    const auto history_fit = BestTranslation(stored_previous, input_previous);
    f = std::fopen((directory / "raw-flow-analysis.json").string().c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "{\"flow_width\":%u,\"flow_height\":%u,\"format\":\"R16G16_SINT\","
        "\"units_expected\":\"full_resolution_pixels\",\"expected_backward_flow\":[%.9f,%.9f],"
        "\"roi_cells\":%u,\"roi_mean\":[%.9f,%.9f],\"roi_min\":[%d,%d],\"roi_max\":[%d,%d],"
        "\"roi_exact_expected\":%u,\"roi_exact_opposite\":%u,\"roi_zero\":%u,"
        "\"scd_words\":[%u,%u],\"scd_slot1_low4\":%u,\"stored_P_vs_input_P_mae\":%.9f,"
        "\"stored_C_vs_input_C_mae\":%.9f,\"stored_P_relative_to_input_P_fit\":[%.9f,%.9f],\"fit_mse\":%.9f}\n",
        width, height, -scene.dx, -scene.dy, count, count ? sum_x / count : 0.0,
        count ? sum_y / count : 0.0, min_x, min_y, max_x, max_y, expected, opposite, zero,
        scd[0], scd[1], scd[1] & 15u, p.mae, c.mae, history_fit.x, history_fit.y, history_fit.mse);
    std::fclose(f);
}
} // namespace MultiFgDiagnostic

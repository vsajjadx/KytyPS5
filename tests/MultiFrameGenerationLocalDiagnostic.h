#pragma once
#include "MultiFrameGenerationLocalOracle.h"
#include "MultiFrameGenerationDiagnostic.h"
#include <map>

namespace MultiFgLocalDiagnostic {
namespace Local = MultiFgLocalOracle;
using namespace MultiFgOracle;
inline std::filesystem::path Directory(unsigned scene, uint32_t factor, uint32_t ordinal) {
    const char* value = std::getenv("KYTY_MULTIFG_LOCAL_DUMP_DIR");
    if (!value || !*value) return {};
    static bool attempted = false, ready = false;
    static std::filesystem::path root;
    static unsigned events = 0;
    std::error_code error;
    if (!attempted) {
        attempted = true;
        root = std::filesystem::path(value).lexically_normal();
        auto text = root.generic_string();
        for (char& c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        constexpr const char* prefix = "g:/ps4/emu/kyty-perf/";
        const auto n = std::char_traits<char>::length(prefix);
        if (!text.starts_with(prefix) || text.size() <= n || text.find('/', n) != std::string::npos ||
            text.find("..") != std::string::npos) return {};
        ready = std::filesystem::create_directory(root, error) && !error;
        if (!ready) std::fprintf(stderr, "MULTIFG_LOCAL_DUMP refused_root=%s\n", value);
    }
    if (!ready || events >= 30) return {};
    const auto path = root / ("scene" + std::to_string(scene) + "-factor" +
        std::to_string(factor) + "-ordinal" + std::to_string(ordinal));
    if (!std::filesystem::create_directory(path, error) || error) return {};
    ++events;
    return path;
}
inline double FitMse(View output, View previous, const Local::Scene& scene,
                     Local::Region region, double t, double sx, double sy) {
    double error = 0; uint64_t channels = 0;
    for (uint32_t y = 32; y < scene.Height() - 32; y += 4)
        for (uint32_t x = 32; x < scene.Width() - 32; x += 4) if (Local::InMask(scene, region, x, y, t))
            for (uint32_t c = 0; c < 3; ++c) {
                const auto b = MultiFgDiagnostic::SampleChannel(previous, x - sx, y - sy, c);
                if (!std::isfinite(b)) return INFINITY;
                const double d = Byte(output.words[size_t(y) * output.stride + x], c) - b;
                error += d * d; ++channels;
            }
    return channels ? error / channels : INFINITY;
}
inline MultiFgDiagnostic::Match Fit(View output, View previous, const Local::Scene& scene,
                                   Local::Region region, double t) {
    MultiFgDiagnostic::Match best;
    for (int y = -12; y <= 12; ++y)
        for (int x = -24; x <= 24; ++x) {
            const auto mse = FitMse(output, previous, scene, region, t, x, y);
            if (mse < best.mse) best = {double(x), double(y), mse};
        }
    const auto coarse = best;
    for (int y = -4; y <= 4; ++y)
        for (int x = -4; x <= 4; ++x) {
            const double sx = coarse.x + .25 * x, sy = coarse.y + .25 * y;
            const auto mse = FitMse(output, previous, scene, region, t, sx, sy);
            if (mse < best.mse) best = {sx, sy, mse};
        }
    return best;
}
inline void Save(const std::filesystem::path& path, const Local::Scene& scene,
                 uint32_t factor, uint32_t ordinal, View input_p, View input_c, View output,
                 const std::array<std::vector<uint32_t>, 5>& data,
                 const std::array<Local::Result, 2>& results) {
    if (path.empty() || factor < 2 || factor > 6 || ordinal < 1 || ordinal >= factor ||
        !Valid(input_p) || !Valid(input_c) || !Valid(output)) return;
    const uint32_t w = scene.Width(), h = scene.Height(), fw = (w + 7) / 8, fh = (h + 7) / 8;
    if (data[0].size() != uint64_t(fw) * fh || data[1].size() != 2 ||
        data[2].size() != uint64_t(w) * h || data[3].size() != uint64_t(w) * h ||
        data[4].size() != uint64_t(fw) * fh * 4) return;
    const View stored_p {data[2], w, h, w}, stored_c {data[3], w, h, w};
    const double t = double(ordinal) / factor;
    const auto& expected = Local::Reference(scene, factor, ordinal);
    bool files = true;
    for (const auto& [name, image] : std::array {
            std::pair {"P-input", input_p}, std::pair {"C-input", input_c},
            std::pair {"P-stored-history", stored_p}, std::pair {"C-stored-history", stored_c},
            std::pair {"O-gpu", output}, std::pair {"expected", Local::Packed(expected, scene)}}) {
        files &= MultiFgDiagnostic::Ppm(path / (std::string(name) + ".ppm"), image);
        files &= MultiFgDiagnostic::Raw(path / (std::string(name) + ".rgba8.bin"), image);
    }
    files &= MultiFgDiagnostic::RawWords(path / "flow-rg16sint.bin", data[0]);
    files &= MultiFgDiagnostic::RawWords(path / "scd-u32.bin", data[1]);
    files &= MultiFgDiagnostic::RawWords(path / "refined-flow-rgba32sint.bin", data[4]);
    auto* f = std::fopen((path / "analysis.json").string().c_str(), "wb");
    if (!f) return;
    const auto pe = MultiFgDiagnostic::Region(stored_p, input_p, 0, 0, w, h);
    const auto ce = MultiFgDiagnostic::Region(stored_c, input_c, 0, 0, w, h);
    std::fprintf(f, "{\"scene\":%u,\"width\":%u,\"height\":%u,\"flow_width\":%u,\"flow_height\":%u,"
        "\"factor\":%u,\"ordinal\":%u,\"t\":%.9f,\"previous_frame\":11,\"current_frame\":12,"
        "\"stored_P_input_P_mae\":%.9f,\"stored_C_input_C_mae\":%.9f,\"scd\":[%u,%u],\"regions\":[",
        static_cast<unsigned>(scene.kind), w, h, fw, fh, factor, ordinal, t, pe.mae, ce.mae, data[1][0], data[1][1]);
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i]; const auto fit = Fit(output, input_p, scene, r.region, t);
        std::fprintf(f, "%s{\"name\":\"%s\",\"valid_pixels\":%llu,\"mae\":%.9f,\"rmse\":%.9f,"
            "\"okay\":%s,\"fit_relative_P\":[%.9f,%.9f],\"fit_sparse_mse\":%.9f}",
            i ? "," : "", Local::Name(r.region), static_cast<unsigned long long>(r.pixels),
            r.mae, r.rmse, r.okay ? "true" : "false", fit.x, fit.y, fit.mse);
    }
    std::fprintf(f, "],\"excluded_boundary_or_unobserved\":%llu,\"files_okay\":%s}\n",
        static_cast<unsigned long long>(uint64_t(w) * h - results[0].pixels - results[1].pixels), files ? "true" : "false");
    std::fclose(f);
    f = std::fopen((path / "flow.csv").string().c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "cell_x,cell_y,region,raw_dx,raw_dy,refined_dx,refined_dy\n");
    for (uint32_t y = 0; y < fh; ++y)
        for (uint32_t x = 0; x < fw; ++x) {
            const size_t at = size_t(y) * fw + x; const auto raw = data[0][at];
            const int dx = std::bit_cast<int16_t>(static_cast<uint16_t>(raw));
            const int dy = std::bit_cast<int16_t>(static_cast<uint16_t>(raw >> 16));
            const auto rx = std::bit_cast<int32_t>(data[4][at * 4]), ry = std::bit_cast<int32_t>(data[4][at * 4 + 1]);
            const char* region = "excluded";
            for (const auto r : Local::Regions(scene)) if (Local::InMask(scene, r, x * 8 + 4, y * 8 + 4, t)) region = Local::Name(r);
            std::fprintf(f, "%u,%u,%s,%d,%d,%d,%d\n", x, y, region, dx, dy, rx, ry);
        }
    std::fclose(f);
    std::fprintf(stderr, "MULTIFG_LOCAL_DUMP path=%s files_okay=%u\n", path.string().c_str(), files ? 1u : 0u);
}
// The public status query reads the same completed GPU feedback word as the bool gate.
// Save before that unchanged gate can abort, alongside the same-pair raw SCD/images.
template<class Status>
inline void SaveSceneStatus(const std::filesystem::path& path, const Local::Scene& scene,
                            uint32_t factor, const Status& status) {
    if (path.empty()) return;
    auto* f = std::fopen((path / "scene-status.json").string().c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "{\"scene\":%u,\"factor\":%u,\"state\":%u,\"state_schema\":\"0=Unavailable,1=Pending,2=Clear,3=Changed\","
        "\"feedback_flags\":%u,\"recorded_tick\":%llu,\"completed_tick\":%llu,\"captured_after_submit_wait\":true}\n",
        static_cast<unsigned>(scene.kind), factor, static_cast<unsigned>(status.state), status.flags,
        static_cast<unsigned long long>(status.recorded_tick), static_cast<unsigned long long>(status.completed_tick));
    std::fclose(f);
}
// Failure-only, caller-owned synthetic diagnostic. No directory exists until Changed
// was already known, and the root itself must be a new immediate private Gperf child.
template<class Status>
inline void SavePostFailureScd(const Local::Scene& scene, uint32_t factor,
                              const Status& original, const Status& after,
                              uint64_t copy_tick, bool copied, std::span<const uint32_t> words) {
    const char* value = std::getenv("KYTY_MULTIFG_SCD_FAILURE_DIR");
    if (!value || !*value || static_cast<unsigned>(original.state) != 3 || !original.flags) return;
    auto path = std::filesystem::path(value).lexically_normal();
    auto text = path.generic_string();
    for (char& c: text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    constexpr const char* prefix = "g:/ps4/emu/kyty-perf/";
    const auto n = std::char_traits<char>::length(prefix);
    if (!text.starts_with(prefix) || text.size() <= n || text.find('/', n) != std::string::npos ||
        text.find("..") != std::string::npos) return;
    std::error_code error;
    if (!std::filesystem::create_directory(path, error) || error) {
        std::fprintf(stderr, "MULTIFG_POSTFAIL_SCD refused_existing_or_invalid_root=%s\n", value);
        return;
    }
    copied = copied && words.size() == 2;
    bool raw_okay = false;
    char float_value[64] = "null";
    const char* float_class = "unknown";
    if (copied) {
        raw_okay = MultiFgDiagnostic::RawWords(path / "scd-postfailure-u32.bin", words);
        const auto scalar = std::bit_cast<float>(words[0]);
        float_class = std::isnan(scalar) ? "nan" : std::isinf(scalar) ?
            (std::signbit(scalar) ? "negative_infinity" : "positive_infinity") : "finite";
        if (std::isfinite(scalar)) std::snprintf(float_value, sizeof(float_value), "%.9g", static_cast<double>(scalar));
    }
    auto* f = std::fopen((path / "postfailure-scd.json").string().c_str(), "wb");
    if (f) {
        std::fprintf(f, "{\"scene\":%u,\"factor\":%u,\"original_state\":%u,\"original_flags\":%u,"
            "\"original_recorded_tick\":%llu,\"original_completed_tick\":%llu,\"copy_tick\":%llu,"
            "\"after_state\":%u,\"after_flags\":%u,\"after_recorded_tick\":%llu,\"after_completed_tick\":%llu,"
            "\"copy_succeeded\":%s,\"raw_file_okay\":%s,\"scd0_f32\":%s,\"scd0_float_class\":\"%s\","
            "\"post_original_failure_only\":true,\"original_batch_had_no_capture\":true",
            static_cast<unsigned>(scene.kind), factor, static_cast<unsigned>(original.state), original.flags,
            static_cast<unsigned long long>(original.recorded_tick), static_cast<unsigned long long>(original.completed_tick),
            static_cast<unsigned long long>(copy_tick), static_cast<unsigned>(after.state), after.flags,
            static_cast<unsigned long long>(after.recorded_tick), static_cast<unsigned long long>(after.completed_tick),
            copied ? "true" : "false", raw_okay ? "true" : "false", float_value, float_class);
        if (copied) std::fprintf(f, ",\"scd_words\":[%u,%u],\"scd0_hex\":\"0x%08x\",\"scd1_hex\":\"0x%08x\",\"historybits1\":%u",
            words[0], words[1], words[0], words[1], words[1] & 15u);
        else std::fprintf(f, ",\"scd_words\":\"unknown\"");
        std::fprintf(f, "}\n");
        std::fclose(f);
    }
    std::fprintf(stderr, "MULTIFG_POSTFAIL_SCD path=%s copy_succeeded=%u raw_file_okay=%u\n",
        path.string().c_str(), copied ? 1u : 0u, raw_okay ? 1u : 0u);
}
} // namespace MultiFgLocalDiagnostic

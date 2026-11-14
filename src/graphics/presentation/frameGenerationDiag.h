#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATIONDIAG_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATIONDIAG_H_

// Default-off frame-generation presentation diagnostic (KYTY_FG_DIAG=1, exactly "1").
// Classifies each SUCCESSFUL vkQueuePresentKHR by the presenter call site that issued it; it says
// nothing about physical scanout, unique images or game FPS. No allocation, no GPU work, no policy.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::FrameGenerationDiag {

// Call-site identity of a present:
//   Real         Presenter::Present with a non-blank main frame shown immediately.
//   Generated    the interpolated frame shown ahead of a new real frame.
//   DelayedReal  the real frame a generated frame went ahead of, shown later (PresentPendingFrame).
//   Repeat       PresentLastFrame (overlay refresh of the frames already shown).
//   Blank        Presenter::Present with a blank main frame (or no main frame).
enum class PresentKind : uint8_t { Real, Generated, DelayedReal, Repeat, Blank, Count };
inline constexpr std::array<const char*, static_cast<size_t>(PresentKind::Count)> KindNames {
    "real", "generated", "delayed_real", "repeat", "blank"};

[[nodiscard]] inline bool ParseEnabled(const char* value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = ParseEnabled(std::getenv("KYTY_FG_DIAG"));
    return enabled;
}

class Stats {
public:
    static constexpr uint32_t MaxSamples = 512;
    static constexpr uint64_t WindowNs   = 2'000'000'000ull;

    // One per new frame through Presenter::Present: frame generation requested (toggle on) and
    // effectively eligible (the existing `wanted` predicate).
    void NoteRequest(bool requested, bool eligible) {
        ++m_requests;
        m_requested += requested ? 1u : 0u;
        m_eligible += eligible ? 1u : 0u;
    }
    // FrameGenerator::Generate recorded (true) or failed (false) for an eligible frame.
    void NoteGeneration(bool recorded) { (recorded ? m_generation_recorded : m_generation_failed)++; }
    // Three-way form for the present path. gen_recorded: Generate returned true, an interpolated
    // frame was recorded and goes to presentation. gen_seeded: it returned false after writing its
    // output (first frame after a reset seeds history; nothing generated is shown). gen_failed:
    // nothing was written. A seed is never counted as recorded or failed.
    void NoteGeneration(bool recorded, bool output_written) {
        if (!recorded && output_written) {
            ++m_generation_seeded;
        } else {
            NoteGeneration(recorded);
        }
    }
    // A pending delayed real frame was replaced by a newer frame before it was presented.
    void NoteSuperseded() { ++m_superseded_real; }
    // Swapchain recreated: the next interval would span a recreation, so it is not sampled.
    void NoteRecreate() {
        ++m_recreates;
        m_last_present_ns = 0;
    }
    // Presentation retries exhausted for a present of this kind (nothing was shown).
    void NoteFailed(PresentKind kind) { ++m_failed[static_cast<size_t>(kind)]; }

    // A successful present at CPU monotonic `now_ns`. Returns true when this closed a >= 2 s window;
    // `out` then holds one report line (or the truncation marker) and the window restarts.
    bool NotePresented(PresentKind kind, uint64_t now_ns, char* out, size_t size) {
        ++m_presented[static_cast<size_t>(kind)];
        if (m_last_present_ns != 0 && now_ns > m_last_present_ns) {
            const uint64_t us = (now_ns - m_last_present_ns) / 1000u;
            if (m_count < MaxSamples) {
                m_samples[m_count++] = static_cast<uint32_t>(std::min<uint64_t>(us, UINT32_MAX));
            } else {
                ++m_dropped;
            }
        }
        m_last_present_ns = now_ns;
        if (m_window_start_ns == 0) m_window_start_ns = now_ns;
        if (now_ns - m_window_start_ns < WindowNs) return false;
        Format(now_ns - m_window_start_ns, out, size);
        ResetWindow(now_ns);
        return true;
    }

private:
    // Nearest-rank quantile of the sorted samples; n > 0.
    static uint32_t Quantile(const std::array<uint32_t, MaxSamples>& sorted, uint32_t n, uint32_t percent) {
        const uint32_t rank = (n * percent + 99u) / 100u;
        return sorted[std::max(rank, 1u) - 1u];
    }

    void Format(uint64_t window_ns, char* out, size_t size) const {
        if (out == nullptr || size == 0) return;
        std::array<uint32_t, MaxSamples> sorted = m_samples;
        std::sort(sorted.begin(), sorted.begin() + m_count);
        char interval[96];
        if (m_count == 0) {
            std::snprintf(interval, sizeof(interval), "n=0 p50=na p90=na p99=na max=na");
        } else {
            std::snprintf(interval, sizeof(interval), "n=%u p50=%u p90=%u p99=%u max=%u", m_count,
                          Quantile(sorted, m_count, 50), Quantile(sorted, m_count, 90),
                          Quantile(sorted, m_count, 99), sorted[m_count - 1u]);
        }
        const auto p = [&](PresentKind k) { return static_cast<unsigned long long>(m_presented[static_cast<size_t>(k)]); };
        const auto f = [&](PresentKind k) { return static_cast<unsigned long long>(m_failed[static_cast<size_t>(k)]); };
        const int written = std::snprintf(out, size,
            "FG_DIAG v=1 window_ms=%llu frames=%llu requested=%llu eligible=%llu gen_recorded=%llu gen_failed=%llu "
            "superseded_real=%llu ok_real=%llu ok_generated=%llu ok_delayed_real=%llu ok_repeat=%llu ok_blank=%llu "
            "fail_real=%llu fail_generated=%llu fail_delayed_real=%llu fail_repeat=%llu fail_blank=%llu "
            "recreates=%llu interval_us %s dropped=%llu source=callsite scanout=unknown gen_seeded=%llu",
            static_cast<unsigned long long>(window_ns / 1'000'000u), static_cast<unsigned long long>(m_requests),
            static_cast<unsigned long long>(m_requested), static_cast<unsigned long long>(m_eligible),
            static_cast<unsigned long long>(m_generation_recorded),
            static_cast<unsigned long long>(m_generation_failed), static_cast<unsigned long long>(m_superseded_real),
            p(PresentKind::Real), p(PresentKind::Generated), p(PresentKind::DelayedReal), p(PresentKind::Repeat),
            p(PresentKind::Blank), f(PresentKind::Real), f(PresentKind::Generated), f(PresentKind::DelayedReal),
            f(PresentKind::Repeat), f(PresentKind::Blank), static_cast<unsigned long long>(m_recreates), interval,
            static_cast<unsigned long long>(m_dropped), static_cast<unsigned long long>(m_generation_seeded));
        if (written < 0 || static_cast<size_t>(written) >= size) {
            std::snprintf(out, size, "FG_DIAG v=1 error=truncated");
        }
    }

    void ResetWindow(uint64_t now_ns) {
        m_requests = m_requested = m_eligible = m_generation_recorded = m_generation_failed = 0;
        m_superseded_real = m_recreates = m_dropped = m_generation_seeded = 0;
        m_presented.fill(0);
        m_failed.fill(0);
        m_count           = 0;
        m_window_start_ns = now_ns; // m_last_present_ns stays: intervals continue across windows
    }

    uint64_t m_requests = 0, m_requested = 0, m_eligible = 0, m_generation_recorded = 0, m_generation_failed = 0;
    uint64_t m_superseded_real = 0, m_recreates = 0, m_dropped = 0, m_generation_seeded = 0;
    std::array<uint64_t, static_cast<size_t>(PresentKind::Count)> m_presented {}, m_failed {};
    std::array<uint32_t, MaxSamples> m_samples {};
    uint32_t m_count           = 0;
    uint64_t m_window_start_ns = 0, m_last_present_ns = 0;
};

} // namespace Libs::Graphics::FrameGenerationDiag

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATIONDIAG_H_

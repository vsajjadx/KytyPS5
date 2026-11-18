#ifndef KYTY_CAUSAL_DIAGNOSTICS_RUNTIME_H_
#define KYTY_CAUSAL_DIAGNOSTICS_RUNTIME_H_

#include "graphics/host_gpu/renderer/causalDiagnostics.h"
#include "common/threads.h"

#include <chrono>
#include <cstdlib>

namespace Libs::Graphics::CausalDiagnostics {
// Child-process opt-in only. Callers branch here before constructing records or
// touching diagnostic state. No diagnostic collector thread is created.
inline bool Enabled() noexcept {
    static const bool enabled = EnabledFor(std::getenv("KYTY_CAUSAL_DIAG"));
    return enabled;
}
inline uint64_t NowNs() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline uint64_t Domain(const void* pointer) noexcept {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
}
inline Recorder& RuntimeRecorder() noexcept {
    constinit static Recorder recorder(true);
    return recorder;
}
inline uint64_t NewId() noexcept {
    return Enabled() ? RuntimeRecorder().NewId() : 0;
}
inline Record Make(Event event, uint64_t domain, uint64_t id = 0, uint64_t tick = 0) noexcept {
    if (!Enabled()) return {};
    const auto context = CurrentContext();
    Record record{};
    record.timestamp_ns = NowNs();
    record.domain = domain; record.id = id; record.tick = tick; record.event = event;
    record.thread = static_cast<uint32_t>(Common::Thread::GetThreadIdUnique());
    record.SetParent(context.cause);
    record.data[3] = context.guest.sequence;
    record.data[4] = context.guest.domain;
    record.data[5] = context.guest.id;
    return record;
}
inline RecordResult RecordEvent(Family family, Record record,
                               bool coalesce = false, bool focus = false) noexcept {
    if (!Enabled()) return {};
    return RuntimeRecorder().TryRecord(family, record, coalesce, focus);
}

struct RuntimeReportState {
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> contention{0};
    ReportCursor cursor{};
};
inline RuntimeReportState& ReportingState() noexcept {
    constinit static RuntimeReportState state{};
    return state;
}
// Invoke only from a live renderer owner, outside engine locks. Capture takes
// an existing-mutex try-lock and samples already available atomics; it must not
// query a GPU, read guest memory, or wait. A failed capture remains explicit.
template <typename Capture, typename ExtraReport>
void MaybeReport(Capture&& capture, ExtraReport&& extra_report) {
    if (!Enabled()) return;
    auto& state = ReportingState();
    if (state.busy.test_and_set(std::memory_order_acquire)) {
        state.contention.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    struct Unlock {
        std::atomic_flag& flag;
        ~Unlock() { flag.clear(std::memory_order_release); }
    } unlock{state.busy};
    const auto now = NowNs();
    if (state.cursor.reported &&
        (now < state.cursor.last_ns || now - state.cursor.last_ns < ReportPeriodNs)) return;
    auto current = std::forward<Capture>(capture)();
    current.report_contention = state.contention.load(std::memory_order_relaxed);
    // The text sink may incur diagnostic-only I/O cost. It runs after the capture
    // returned and thus holds no scheduler, queue, Flip, or renderer mutex.
    Report(RuntimeRecorder(), state.cursor, now, [](std::string_view text) {
        std::fwrite(text.data(), 1, text.size(), stderr);
    }, &current);
    std::forward<ExtraReport>(extra_report)(current, now);
}
template <typename Capture>
void MaybeReport(Capture&& capture) {
    MaybeReport(std::forward<Capture>(capture), [](const CurrentState&, uint64_t) {});
}
} // namespace Libs::Graphics::CausalDiagnostics
#endif

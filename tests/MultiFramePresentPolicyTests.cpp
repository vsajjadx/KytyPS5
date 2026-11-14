#include "graphics/presentation/multiFramePresentPolicy.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace Libs::Graphics::MultiFramePresentPolicy;
namespace {
void Check(bool okay, const char* message) {
    if (!okay) { std::fprintf(stderr, "MultiFramePresentPolicyTests: %s\n", message); std::abort(); }
}
}
int main() {
    using Kind = Cadence::Kind;
    for (uint32_t factor = 2; factor <= 6; ++factor) {
        Cadence cadence;
        const uint64_t start = 1000, interval = 120 * factor, epoch = 9;
        Check(cadence.Begin(factor, start, interval, epoch), "valid factor did not begin");
        Check(!cadence.Begin(factor, start, interval, epoch), "active real silently overwritten");
        for (uint32_t i = 0; i < factor; ++i) {
            Check(cadence.Take(start + 120 * i - 1, epoch).kind == Kind::None,
                  "ticket became due early");
            const auto ticket = cadence.Take(start + 120 * i, epoch);
            Check(ticket.index == i && ticket.due == start + 120 * i &&
                  ticket.kind == (i + 1 == factor ? Kind::Real : Kind::Generated),
                  "ordered unique generated positions or final real missing");
            Check(Cadence::Current(ticket, epoch), "live ticket rejected");
            if (ticket.kind == Kind::Generated)
                Check(!Cadence::Current(ticket, epoch + 1), "stale generation survived cancellation epoch");
        }
        Check(!cadence.Pending() && cadence.Take(100000, epoch).kind == Kind::None,
              "completed batch emitted twice");
        Check(cadence.Cancel().real == 0, "completed real was canceled twice");

        Check(cadence.Begin(factor, start, interval, epoch), "restart failed");
        const auto forced = cadence.Take(start - 1, epoch + 1);
        Check(forced.kind == Kind::Real && cadence.GetCounts().superseded == factor - 1,
              "epoch cancellation did not preserve active real progress");
        Check(!cadence.Pending() && cadence.Cancel().real == 0, "forced real accounted twice");

        Check(cadence.Begin(factor, start, interval, epoch), "newer-real setup failed");
        cadence.Take(start, epoch);
        const auto newer = cadence.Take(start + 1, epoch, true);
        Check(newer.kind == Kind::Real && cadence.GetCounts().superseded == factor - 2,
              "new input failed to flush old active real before replacement");

        Check(cadence.Begin(factor, start, interval, epoch), "overrun setup failed");
        const auto overdue = cadence.Take(start + interval * 2, epoch);
        Check(overdue.kind == Kind::Real && cadence.GetCounts().late == factor - 1,
              "blocking presentation overrun replayed stale generated frames");

        Check(cadence.Begin(factor, start, interval, epoch), "stop setup failed");
        const auto stopped = cadence.Cancel();
        Check(stopped.generated == factor - 1 && stopped.real == 1 && !cadence.Pending(),
              "stop/cancel lost a live output or real reference");
        Check(cadence.Cancel().generated == 0 && cadence.Cancel().real == 0,
              "stop double-accounted canceled frames");
    }
    Cadence invalid;
    for (auto factor: {0u, 1u, 7u, std::numeric_limits<uint32_t>::max()})
        Check(!invalid.Begin(factor, 1, 1000, 1), "invalid factor admitted underflow/oversized output");
    Check(!invalid.Begin(6, 0, 5, 1), "zero-step interval accepted");
    Check(!invalid.Begin(6, std::numeric_limits<uint64_t>::max() - 2, 1000, 1), "deadline overflow accepted");

    // These are the actual queue ownership rules used by the production mailbox.
    Queue<uint32_t> queue;
    std::array<uint32_t, 20> releases {};
    for (uint32_t i = 1; i <= 10; ++i) {
        auto evicted = queue.Push(i);
        if (evicted) ++releases[*evicted];
        Check(queue.Size() <= InputCapacity, "queue grew beyond its production capacity");
    }
    Check(queue.Size() == InputCapacity, "bounded queue lost recent inputs");
    queue.Close();
    auto rejected = queue.Push(11);
    Check(rejected && *rejected == 11 && queue.Size() == InputCapacity, "closed mailbox took ownership silently");
    ++releases[*rejected];
    while (auto remaining = queue.Pop()) ++releases[*remaining];
    for (uint32_t i = 1; i <= 11; ++i) Check(releases[i] == 1, "evicted/queued/rejected input was not released exactly once");
    Check(queue.Size() == 0 && !queue.Pop(), "stopped mailbox did not drain");

    Check(AcceptedPresent(true, false, false) && AcceptedPresent(true, false, true),
          "ordinary successful WSI call was rejected");
    Check(AcceptedPresent(false, true, true) && !AcceptedPresent(false, true, false),
          "Multi suboptimal did not count once or legacy retry changed");
    Check(!AcceptedPresent(false, false, true) && !AcceptedPresent(false, false, false),
          "out-of-date/failed WSI was falsely counted successful");
    SuccessCounts success;
    Check(success.generated == 0 && success.real == 0, "recording/reserving counted as a WSI success");
    success.Presented(PresentedKind::Generated);
    success.Presented(PresentedKind::Repeat);
    success.Presented(PresentedKind::Real);
    Check(success.generated == 1 && success.real == 1 && success.repeat == 1,
          "repeat was relabelled generated or successful real lost");
    const auto generation_tick = RetainTick(0, 41);
    const auto presentation_tick = RetainTick(generation_tick, 45);
    Check(presentation_tick == 45 && RetainTick(presentation_tick, 42) == 45,
          "later generation or presentation use lost its required completion tick");
    std::puts("MultiFramePresentPolicy PASS (all factors, epochs, overrun, queue ownership, stop, success classification)");
}

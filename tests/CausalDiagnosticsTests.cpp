#include "graphics/host_gpu/renderer/causalDiagnostics.h"
#include "common/threadIdentityDiagnostics.h"
#include "graphics/host_gpu/renderer/causalOperationWitness.h"
#include <type_traits>

#include <algorithm>

#include <atomic>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

namespace D = Libs::Graphics::CausalDiagnostics;
namespace Libs::Graphics::CausalDiagnostics {
struct RecorderTestAccess {
    static bool Hold(Recorder& r, Family family) {
        return !r.m_family[static_cast<size_t>(family)].busy.test_and_set(std::memory_order_acquire);
    }
    static void Release(Recorder& r, Family family) {
        r.m_family[static_cast<size_t>(family)].busy.clear(std::memory_order_release);
    }
};
}

namespace {
int failures = 0;
void Check(bool ok, const char* test, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL %s: %s\n", test, message); ++failures; }
}
D::Record Make(uint64_t id, D::Event event = D::Event::GuestEnqueue) {
    D::Record r{};
    r.timestamp_ns = id + 1;
    r.domain = 0x42;
    r.id = id;
    r.event = event;
    r.data[0] = id ^ 0x19f543a284d168caull;
    r.data[1] = ~id;
    r.data[2] = id * 0x9e3779b97f4a7c15ull;
    return r;
}
bool Intact(const D::Record& r) {
    return r.data[0] == (r.id ^ 0x19f543a284d168caull) &&
           r.data[1] == ~r.id &&
           r.data[2] == r.id * 0x9e3779b97f4a7c15ull;
}
void ExactFlagAndDisabled() {
    Check(D::EnabledFor("1"), "flag", "exact one must enable");
    for (const char* value : std::array<const char*, 8>{nullptr, "", "0", "01", "true", "1x", " 1", "1 "})
        Check(!D::EnabledFor(value), "flag", "non-exact value enabled diagnostics");
    D::Recorder recorder(false);
    unsigned invoked = 0;
    const auto result = recorder.RecordLazy(D::Family::Guest, [&] {
        ++invoked; return Make(1);
    });
    Check(result.outcome == D::Outcome::Disabled && invoked == 0,
          "disabled", "disabled path evaluated the producer or recorded");
    Check(recorder.NewId() == 0, "disabled", "disabled path allocated an ID");
    D::Snapshot snapshot{};
    Check(!recorder.ReadSnapshot(snapshot), "disabled", "disabled path took a snapshot");
    D::ReportCursor cursor{};
    unsigned emitted = 0;
    const auto report = D::Report(recorder, cursor, 10, [&](std::string_view) { ++emitted; });
    Check(!report.emitted && emitted == 0, "disabled", "disabled path called the output sink");
}
void RolloverAndAncestry() {
    D::Recorder recorder(true);
    auto parent = recorder.TryRecord(D::Family::Guest, Make(1));
    Check(parent.outcome == D::Outcome::Stored && parent.parent.sequence != 0,
          "record", "enabled recorder did not commit the event");
    auto child = Make(2, D::Event::NativeSubmitResult);
    child.SetParent(parent.parent);
    recorder.TryRecord(D::Family::Native, child);
    D::Snapshot snapshot{};
    Check(recorder.ReadSnapshot(snapshot), "snapshot", "enabled snapshot unavailable");
    Check(D::Ancestry(snapshot, child) == D::ParentStatus::Retained,
          "ancestry", "retained parent was not recognized");
    for (uint64_t i = 2; i <= 100; ++i) recorder.TryRecord(D::Family::Guest, Make(i));
    recorder.ReadSnapshot(snapshot);
    const auto& guest = snapshot.family[static_cast<size_t>(D::Family::Guest)];
    Check(guest.valid && guest.count == 64 && guest.overwritten == 36 &&
              guest.attempted == 100 && guest.stored == 100,
          "rollover", "tail bound or cumulative overwrite accounting is incorrect");
    Check(guest.records[0].id == 37 && guest.records[63].id == 100,
          "rollover", "near-stall tail was not retained");
    Check(D::Ancestry(snapshot, child) == D::ParentStatus::OlderThanRetainedTail,
          "ancestry", "old parent was not classified as outside the retained tail");
    child.parent_sequence = UINT64_MAX;
    Check(D::Ancestry(snapshot, child) == D::ParentStatus::Missing,
          "ancestry", "unknown parent was represented as complete");
}
void CoalescingAndFocusedTarget() {
    D::Recorder recorder(true);
    auto state = Make(0, D::Event::FlipNoOwner); // No owner is a valid zero ID.
    const auto first = recorder.TryRecord(D::Family::Flip, state, true);
    for (uint64_t i = 0; i < 10000; ++i) {
        state.timestamp_ns = i + 100;
        recorder.TryRecord(D::Family::Flip, state, true);
    }
    auto target = Make(55, D::Event::TargetDispatchIndirectRecorded);
    target.data[3] = 0x1a04c58a9fbc2052ull;
    const auto marked = recorder.TryRecord(D::Family::Native, target, false, true);
    for (uint64_t i = 100; i < 1000; ++i)
        recorder.TryRecord(D::Family::Native, Make(i, D::Event::NativeSubmitResult));
    D::Snapshot snapshot{};
    recorder.ReadSnapshot(snapshot);
    const auto& flip = snapshot.family[static_cast<size_t>(D::Family::Flip)];
    Check(flip.count == 1 && flip.repeated == 10000 && flip.attempted == 10001 &&
              flip.last.sequence == first.parent.sequence && flip.last.timestamp_ns == 10099,
          "coalesce", "repeated rejection consumed tail or lost last observation");
    Check(snapshot.focus_valid && snapshot.focus.sequence == marked.parent.sequence &&
              snapshot.focus.id == 55 && snapshot.focus.data[3] == 0x1a04c58a9fbc2052ull,
          "focus", "scarce target was lost before first report after native rollover");
}
void ForcedContention() {
    D::Recorder recorder(true);
    Check(D::RecorderTestAccess::Hold(recorder, D::Family::Priority),
          "contention", "fixture could not hold the recorder slot");
    const auto dropped = recorder.TryRecord(D::Family::Priority, Make(1));
    D::Snapshot snapshot{};
    recorder.ReadSnapshot(snapshot);
    Check(dropped.outcome == D::Outcome::Dropped &&
              !snapshot.family[static_cast<size_t>(D::Family::Priority)].valid,
          "contention", "contended producer/snapshot waited or claimed coherent data");
    D::RecorderTestAccess::Release(recorder, D::Family::Priority);
    recorder.ReadSnapshot(snapshot);
    const auto& priority = snapshot.family[static_cast<size_t>(D::Family::Priority)];
    Check(priority.valid && priority.dropped == 1 && priority.attempted == 1 && priority.count == 0,
          "contention", "contention loss was silently omitted");
}
void DroppedParentThenRollover() {
    D::Recorder recorder(true);
    Check(D::RecorderTestAccess::Hold(recorder, D::Family::Guest),
          "dropped-parent", "fixture could not hold guest recorder");
    const auto dropped = recorder.TryRecord(D::Family::Guest, Make(500));
    D::RecorderTestAccess::Release(recorder, D::Family::Guest);
    Check(dropped.outcome == D::Outcome::Dropped, "dropped-parent", "parent was not dropped");
    auto child = Make(501, D::Event::PriorityQueued);
    child.SetParent(dropped.parent);
    recorder.TryRecord(D::Family::Priority, child);
    for (uint64_t i = 1; i <= 80; ++i)
        recorder.TryRecord(D::Family::Guest, Make(600 + i));
    D::Snapshot snapshot{};
    recorder.ReadSnapshot(snapshot);
    const auto& guest = snapshot.family[static_cast<size_t>(D::Family::Guest)];
    Check(guest.dropped == 1 && guest.stored == 80 && guest.overwritten == 16,
          "dropped-parent", "exact family loss counters were conflated");
    Check(D::Ancestry(snapshot, child) == D::ParentStatus::OlderThanRetainedTail,
          "dropped-parent", "dropped parent was incorrectly asserted to have been stored");
    D::ReportCursor cursor{};
    std::string output;
    D::Report(recorder, cursor, 1, [&](std::string_view text) { output.append(text); });
    Check(output.find("ancestry=older_than_retained_tail") != std::string::npos,
          "dropped-parent", "serialized ancestry implies a known overwrite cause");
}

void ConcurrentSnapshots() {
    D::Recorder recorder(true);
    std::atomic<unsigned> done{0};
    std::vector<std::thread> writers;
    for (unsigned t = 0; t < 4; ++t) writers.emplace_back([&, t] {
        for (uint64_t i = 1; i <= 2000; ++i)
            recorder.TryRecord(D::Family::Signal, Make(i + uint64_t{t} * 10000));
        done.fetch_add(1, std::memory_order_release);
    });
    bool coherent = true;
    D::Snapshot snapshot{};
    while (done.load(std::memory_order_acquire) != 4) {
        recorder.ReadSnapshot(snapshot);
        const auto& signal = snapshot.family[static_cast<size_t>(D::Family::Signal)];
        if (signal.valid) for (size_t i = 0; i < signal.count; ++i)
            coherent &= Intact(signal.records[i]) &&
                        (i == 0 || signal.records[i - 1].sequence < signal.records[i].sequence);
        std::this_thread::yield();
    }
    for (auto& writer : writers) writer.join();
    recorder.ReadSnapshot(snapshot);
    const auto& signal = snapshot.family[static_cast<size_t>(D::Family::Signal)];
    for (size_t i = 0; i < signal.count; ++i)
        coherent &= Intact(signal.records[i]) &&
                    (i == 0 || signal.records[i - 1].sequence < signal.records[i].sequence);
    Check(coherent, "concurrency", "snapshot contained a torn event");
    Check(signal.valid && signal.attempted == 8000 &&
              signal.stored + signal.repeated + signal.dropped == 8000,
          "concurrency", "concurrent attempts were lost without accounting");
}
void ReportDeltaAndBounds() {
    D::Recorder recorder(true);
    D::ReportCursor cursor{};
    for (uint64_t i = 1; i < 80; ++i)
        recorder.TryRecord(D::Family::Guest, Make(i));
    std::string output;
    auto sink = [&](std::string_view text) { output.append(text); };
    const auto first = D::Report(recorder, cursor, 1'000'000'000, sink);
    Check(first.emitted && first.records == 64 && first.bytes == output.size() &&
              first.bytes <= D::MaxReportBytes,
          "report", "first changed-tail report missing or exceeded byte bound");
    const size_t old_size = output.size();
    const auto throttled = D::Report(recorder, cursor, 1'500'000'000, sink);
    Check(!throttled.emitted && output.size() == old_size,
          "report", "more than one report per second");
    const auto quiet = D::Report(recorder, cursor, 2'000'000'000, sink);
    Check(quiet.emitted && quiet.records == 0 && quiet.bytes <= D::MaxReportBytes,
          "report", "unchanged history was dumped again");
    recorder.TryRecord(D::Family::Guest, Make(100));
    const auto next = D::Report(recorder, cursor, 3'000'000'000, sink);
    Check(next.records == 1, "report", "changed-tail delta includes old records");
}
void ContextLifetimeAndCallbacks() {
    const auto empty = D::CurrentContext();
    D::Context outer{};
    outer.cause = {11, 0x41, 7, D::Family::Guest};
    outer.guest = {9, 0x41, 3, D::Family::Guest};
    outer.point = 21; outer.request = 34;
    {
        D::ContextScope scope(outer);
        const auto observed = D::CurrentContext();
        Check(observed.cause.sequence == 11 && observed.guest.sequence == 9 &&
                  observed.point == 21 && observed.request == 34,
              "context", "scope did not install the complete parent capsule");
        D::Context inner = outer; inner.cause.sequence = 77; inner.point = 99;
        {
            D::ContextScope disabled(inner, false);
            Check(D::CurrentContext().cause.sequence == 11,
                  "context-disabled", "disabled scope changed thread context");
        }
        try {
            auto deferred = D::BindContext(inner, [&] {
                Check(D::CurrentContext().cause.sequence == 77 &&
                          D::CurrentContext().guest.sequence == 9 &&
                          D::CurrentContext().point == 99 && D::CurrentContext().request == 34,
                      "context-callback", "deferred callback lost originating context");
                auto recursive = D::BindContext(outer, [&] {
                    Check(D::CurrentContext().cause.sequence == 11,
                          "context-recursive", "recursive context was not installed");
                });
                recursive();
                Check(D::CurrentContext().cause.sequence == 77,
                      "context-recursive", "recursive callback did not restore its caller");
                throw std::runtime_error("intentional callback failure");
            });
            deferred();
            Check(false, "context-exception", "callback exception was swallowed");
        } catch (const std::runtime_error&) {
            Check(D::CurrentContext().cause.sequence == 11,
                  "context-exception", "exception did not restore the caller context");
        }
        std::thread independent([&] {
            Check(D::CurrentContext().cause.sequence == 0, "context-thread", "context leaked across threads");
            D::ContextScope other(inner);
            Check(D::CurrentContext().cause.sequence == 77, "context-thread", "thread context not installed");
        });
        independent.join();
        Check(D::CurrentContext().cause.sequence == 11, "context-thread", "another thread changed caller context");
    }
    Check(D::CurrentContext().cause.sequence == empty.cause.sequence &&
              D::CurrentContext().guest.sequence == empty.guest.sequence,
          "context", "scope did not restore the empty parent");
}
void CurrentWorkerSurvivesTailRollover() {
    D::Recorder recorder(true);
    for (uint64_t i = 1; i <= 900; ++i)
        recorder.TryRecord(D::Family::Priority, Make(i, D::Event::PriorityQueued));
    D::CurrentState current{};
    current.available = current.phase_stable = true;
    current.timestamp_ns = 12345; current.domain = 0x91; current.lifetime_sequence = 8;
    current.active_operation = 41; current.active_tick = 97;
    current.phase_before = current.phase_after = (17ull << 8) | static_cast<uint8_t>(D::WorkerPhase::NativeWait);
    current.cached_known_tick = 96; current.last_native_submit = 97; current.recording_tick = 98;
    current.priority_queued = 1; current.priority_front_tick = 98; current.parent_bridge_lost = 2;
    D::ReportCursor cursor{};
    std::string output;
    D::Report(recorder, cursor, 1'000'000'000, [&](std::string_view text) { output.append(text); }, &current);
    Check(output.find("CAUSAL_CURRENT available=1") != std::string::npos &&
              output.find("active_op=41 active_tick=97") != std::string::npos &&
              output.find("cached_known_tick=96") != std::string::npos &&
              output.find("parent_bridge_lost=2") != std::string::npos,
          "current-worker", "current wait was overwritten by unrelated queue tail events");
    Check(output.find("CAUSAL_CURRENT") < output.find("CAUSAL_EVENT"),
          "current-worker", "current worker state was not reserved ahead of high-rate tails");
    output.clear();
    current.available = false; current.phase_stable = false;
    D::Report(recorder, cursor, 2'000'000'000, [&](std::string_view text) { output.append(text); }, &current);
    Check(output.find("CAUSAL_CURRENT available=0") != std::string::npos,
          "current-worker", "failed try-lock was represented as known idle state");
}

void BoundedCallbackParentBridge() {
    D::CallbackParents bridge;
    D::Context context{};
    context.guest = {5, 0x123, 9, D::Family::Guest};
    context.point = 19; context.request = 23;
    for (uint64_t i = 1; i <= 35; ++i) {
        context.cause = {100 + i, 0x456, i, D::Family::Priority};
        const auto id = bridge.Enqueued(true, context);
        Check(id == i * 2 + 1, "parent-bridge", "priority ordinal changed");
    }
    context.cause.sequence = 999;
    const auto normal = bridge.Enqueued(false, context);
    Check(normal == 2 && bridge.Overwritten() == 3, "parent-bridge", "bounded overwrite accounting is incorrect");
    uint64_t id = 0;
    bool missing = false;
    const auto normal_context = bridge.Dequeued(false, 0x456, id, missing);
    Check(!missing && id == normal && normal_context.cause.sequence == 999,
          "parent-bridge", "normal and priority parents were confused");
    for (uint64_t i = 1; i <= 35; ++i) {
        const auto captured = bridge.Dequeued(true, 0x456, id, missing);
        if (i <= 3) {
            Check(missing && captured.cause.sequence == 0 && captured.cause.id == id &&
                      captured.cause.family == D::Family::Priority,
                  "parent-bridge", "missing parent was represented as complete or absent");
        } else {
            Check(!missing && captured.cause.sequence == 100 + i &&
                      captured.guest.sequence == 5 && captured.guest.domain == 0x123 &&
                      captured.guest.id == 9 && captured.point == 19 && captured.request == 23,
                  "parent-bridge", "retained callback parent capsule was torn or incomplete");
        }
    }
    Check(bridge.Lost() == 3 && bridge.Overwritten() == 3, "parent-bridge", "loss counters were silently omitted");
}

void FailedCallbackPreparationDoesNotAdvanceOrdinal() {
    D::CallbackParents bridge;
    D::Context context{};
    context.cause = {77, 0x100, 9, D::Family::Priority};
    const auto prepared = bridge.NextId(true);
    struct ThrowsOnMove {
        ThrowsOnMove() = default;
        ThrowsOnMove(ThrowsOnMove&&) { throw std::runtime_error("callback preparation failure"); }
        void operator()() {}
    };
    bool failed = false;
    try {
        auto callback = D::BindContext(context, ThrowsOnMove{});
        (void)callback;
        Check(false, "parent-prepare", "throwing callback preparation unexpectedly succeeded");
    } catch (const std::runtime_error&) {
        failed = true;
    }
    Check(failed && prepared == 3 && bridge.NextId(true) == prepared &&
              bridge.Lost() == 0 && bridge.Overwritten() == 0,
          "parent-prepare", "failed preparation advanced the actual queue ordinal");
    const auto committed = bridge.Enqueued(true, context);
    uint64_t dequeued = 0;
    bool missing = false;
    const auto observed = bridge.Dequeued(true, 0x100, dequeued, missing);
    Check(committed == prepared && dequeued == committed && !missing &&
              observed.cause.sequence == 77 && bridge.NextId(true) == 5,
          "parent-prepare", "next accepted callback lost its actual FIFO parent");
}

namespace I = Common::ThreadIdentityDiagnostics;
I::Record IdentitySample() {
    I::Record record{};
    record.phase = I::Phase::Main;
    record.process_id = 2222;
    record.process_creation_time = 9007199254740993ull;
    record.os_thread_id = 4242;
    record.thread_creation_time = 9007199254740997ull;
    record.internal_unique_id = 23;
    record.guest_thread_id = 112;
    record.object_token = 0x1234;
    record.entry = 0x903f37960ull;
    record.process_creation_valid = record.thread_creation_valid = true;
    return record;
}
constexpr std::string_view IdentitySampleLine =
    "THREAD_IDENTITY v=1 phase=0 process_id=2222 process_creation_time=9007199254740993 "
    "process_creation_valid=1 process_creation_error=0 os_tid=4242 "
    "thread_creation_time=9007199254740997 thread_creation_valid=1 thread_creation_error=0 "
    "internal_unique_id=23 guest_thread_id=112 object=0x0000000000001234 entry=0x0000000903f37960\n";

void IdentityExactFlagAndDisabled() {
    Check(I::EnabledFor("1"), "identity-flag", "exact one did not enable");
    for (const char* value : std::array<const char*, 8>{nullptr, "", "0", "01", "true", "1x", " 1", "1 "})
        Check(!I::EnabledFor(value), "identity-flag", "non-exact value enabled identity output");
    struct BudgetSpy {
        unsigned touched = 0;
        I::Permit Acquire() { ++touched; return I::Permit::Record; }
    } budget;
    unsigned supplied = 0, emitted = 0;
    const auto result = I::Emit(false, budget, [&] {
        ++supplied; return IdentitySample();
    }, [&](const char*, size_t) { ++emitted; });
    Check(result == I::Result::Disabled && budget.touched == 0 && supplied == 0 && emitted == 0,
          "identity-disabled", "disabled path consulted budget, evaluated metadata or called output");
}
void IdentityExplicitFieldsAndLifetimes() {
    auto record = IdentitySample();
    std::array<char, I::MaxLineBytes> text{};
    const auto length = I::FormatRecord(record, text.data(), text.size());
    Check(std::string_view(text.data(), length) == IdentitySampleLine,
          "identity-fields", "independent IDs, full-width native creation times or object/entry changed");
    record.phase = I::Phase::Begin;
    record.thread_creation_time = UINT64_MAX; // Same numeric OS TID, a different native lifetime.
    record.internal_unique_id = -1;
    record.guest_thread_id = INT32_MAX;
    const auto other_length = I::FormatRecord(record, text.data(), text.size());
    const std::string_view other(text.data(), other_length);
    Check(other.find("phase=1 ") != std::string_view::npos &&
              other.find("os_tid=4242 ") != std::string_view::npos &&
              other.find("thread_creation_time=18446744073709551615 ") != std::string_view::npos &&
              other.find("internal_unique_id=-1 guest_thread_id=2147483647 ") != std::string_view::npos,
          "identity-lifetime", "reused native TID was collapsed or signed IDs were inferred/equated");
    record.process_creation_valid = record.thread_creation_valid = false;
    record.process_creation_time = 111;
    record.thread_creation_time = 222; // Stale output from a failed native call must not look valid.
    record.process_creation_error = 5;
    record.thread_creation_error = 87;
    const auto unknown_length = I::FormatRecord(record, text.data(), text.size());
    const std::string_view unknown(text.data(), unknown_length);
    Check(unknown.find("process_creation_time=0 process_creation_valid=0 process_creation_error=5 ") != std::string_view::npos &&
              unknown.find("thread_creation_time=0 thread_creation_valid=0 thread_creation_error=87 ") != std::string_view::npos,
          "identity-unknown", "failed native time calls lost validity/error or leaked stale creation values");
    I::Budget budget;
    unsigned supplied = 0, emitted = 0;
    std::string output;
    const auto result = I::Emit(true, budget, [&] { ++supplied; return IdentitySample(); },
        [&](const char* data, size_t bytes) { ++emitted; output.append(data, bytes); });
    Check(result == I::Result::Emitted && supplied == 1 && emitted == 1 &&
              output == IdentitySampleLine && budget.Claimed() == 1,
          "identity-emission", "one permitted witness did not produce exactly one intact bounded line");
}
void IdentityFormatterBounds() {
    auto record = IdentitySample();
    record.process_id = record.process_creation_time = record.os_thread_id =
        record.thread_creation_time = record.object_token = record.entry = UINT64_MAX;
    record.internal_unique_id = INT32_MIN;
    record.guest_thread_id = INT32_MAX;
    record.process_creation_error = record.thread_creation_error = UINT32_MAX;
    std::array<char, I::MaxLineBytes + 2> storage{};
    storage.fill('Z');
    const auto length = I::FormatRecord(record, storage.data() + 1, I::MaxLineBytes);
    const std::string_view line(storage.data() + 1, length);
    Check(length > 0 && length < I::MaxLineBytes && storage.front() == 'Z' && storage.back() == 'Z' &&
              storage[length + 1] == '\0' && line.back() == '\n' &&
              std::count(line.begin(), line.end(), '\n') == 1,
          "identity-line-bound", "maximal numeric witness overflowed, truncated or emitted multiple lines");
    storage.fill('Z');
    const auto tiny = I::FormatRecord(record, storage.data() + 1, 12);
    Check(tiny == 0 && storage[1] == '\0' && storage.front() == 'Z' && storage[13] == 'Z',
          "identity-small-buffer", "insufficient output capacity exposed a partial line or crossed its bound");
    Check(I::FormatRecord(record, nullptr, I::MaxLineBytes) == 0 &&
              I::FormatRecord(record, storage.data(), 0) == 0 && storage.front() == 'Z',
          "identity-null-buffer", "null or zero output capacity was written");
}
void IdentityBudgetBoundary() {
    I::Budget budget;
    bool exact_records = true;
    for (unsigned i = 0; i < I::MaxRecords; ++i)
        exact_records &= budget.Acquire() == I::Permit::Record;
    Check(exact_records && budget.Claimed() == I::MaxRecords,
          "identity-budget", "record allowance differs from the fixed 256-record bound");
    const auto limit = budget.Acquire();
    bool suppressed = true;
    for (unsigned i = 0; i < 10000; ++i) suppressed &= budget.Acquire() == I::Permit::Suppressed;
    Check(limit == I::Permit::Limit && suppressed && budget.Claimed() == I::MaxRecords + 1,
          "identity-limit", "limit notice is missing/repeated or saturated budget kept advancing");
}
void IdentityConcurrentEmission() {
    I::Budget budget;
    std::atomic<unsigned> supplied{0}, records{0}, limits{0};
    std::array<std::atomic<unsigned>, 5> outcomes{};
    std::atomic<bool> go{false}, malformed{false};
    std::vector<std::thread> workers;
    for (unsigned worker = 0; worker < 8; ++worker) workers.emplace_back([&] {
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        for (unsigned attempt = 0; attempt < 128; ++attempt) {
            const auto result = I::Emit(true, budget, [&] {
                auto record = IdentitySample();
                record.guest_thread_id = static_cast<int32_t>(supplied.fetch_add(1, std::memory_order_relaxed) + 1);
                return record;
            }, [&](const char* data, size_t bytes) {
                const std::string_view line(data, bytes);
                if (line == I::LimitLine) limits.fetch_add(1, std::memory_order_relaxed);
                else if (line.starts_with("THREAD_IDENTITY v=1 ") && bytes < I::MaxLineBytes &&
                         !line.empty() && line.back() == '\n' && std::count(line.begin(), line.end(), '\n') == 1)
                    records.fetch_add(1, std::memory_order_relaxed);
                else malformed.store(true, std::memory_order_relaxed);
            });
            outcomes[static_cast<size_t>(result)].fetch_add(1, std::memory_order_relaxed);
        }
    });
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    Check(supplied == 256 && records == 256 && limits == 1 && !malformed &&
              outcomes[static_cast<size_t>(I::Result::Emitted)] == 256 &&
              outcomes[static_cast<size_t>(I::Result::LimitNotice)] == 1 &&
              outcomes[static_cast<size_t>(I::Result::Suppressed)] == 767 &&
              outcomes[static_cast<size_t>(I::Result::Disabled)] == 0 &&
              outcomes[static_cast<size_t>(I::Result::FormatError)] == 0 && budget.Claimed() == 257,
          "identity-concurrency", "concurrent writers exceeded/lost the fixed permits or evaluated suppressed metadata");
}

D::OperationWitness OperationSample(uint64_t id = 17, uint64_t tick = 77) {
    D::OperationWitness value{};
    value.native_domain = 0x200;
    value.lifetime_sequence = 11;
    value.operation_id = id;
    value.selected_tick = tick;
    value.queued_parent = {500 + id, 0x200, id, D::Family::Priority};
    value.enqueue_thread = 31;
    value.priority = false;
    return value;
}
bool SameOperation(const D::OperationWitness& a, const D::OperationWitness& b) {
    return a.native_domain == b.native_domain && a.lifetime_sequence == b.lifetime_sequence &&
           a.operation_id == b.operation_id && a.selected_tick == b.selected_tick &&
           a.queued_parent.sequence == b.queued_parent.sequence &&
           a.queued_parent.domain == b.queued_parent.domain && a.queued_parent.id == b.queued_parent.id &&
           a.queued_parent.family == b.queued_parent.family &&
           a.enqueue_thread == b.enqueue_thread && a.priority == b.priority;
}
struct AcceptedState {
    D::EnqueueReceipt* receipt = nullptr;
    D::OperationWitness observed{};
    unsigned calls = 0;
    unsigned stage = 0;
    bool receipt_was_committed = false;
};
void SaveAcceptedOperation(void* opaque, const D::OperationWitness& operation) noexcept {
    auto& state = *static_cast<AcceptedState*>(opaque);
    ++state.calls;
    state.observed = operation;
    state.stage = 1;
    state.receipt_was_committed = state.receipt && state.receipt->outcome == D::EnqueueOutcome::Queued &&
                                  SameOperation(state.receipt->operation, operation);
}
void OperationAcceptedOnce() {
    D::EnqueueReceipt receipt{D::EnqueueOutcome::Queued, OperationSample(91, 999)};
    AcceptedState sink{&receipt};
    const auto selected = OperationSample();
    {
        D::EnqueueWitnessScope scope(receipt, SaveAcceptedOperation, &sink);
        Check(receipt.outcome == D::EnqueueOutcome::NotAccepted,
              "operation-reset", "new scope retained stale accepted identity");
        auto capture = D::TakeEnqueueWitness();
        capture.PublishAccepted(selected);
        capture.PublishAccepted(OperationSample(19, 88));
        capture.PublishInline();
        Check(receipt.outcome == D::EnqueueOutcome::Queued && SameOperation(receipt.operation, selected) &&
                  sink.calls == 1 && sink.receipt_was_committed && SameOperation(sink.observed, selected),
              "operation-accept", "accepted exact identity was lost, published twice or overwritten");
        auto unrelated = D::TakeEnqueueWitness();
        unrelated.PublishAccepted(OperationSample(21, 99));
        Check(SameOperation(receipt.operation, selected) && sink.calls == 1,
              "operation-consume", "nested/unrelated enqueue reused a consumed stack witness");
    }
    auto expired = D::TakeEnqueueWitness();
    expired.PublishAccepted(OperationSample(23, 111));
    Check(SameOperation(receipt.operation, selected) && sink.calls == 1,
          "operation-expiry", "scope left a retained stack witness after return");
}
void OperationPreparationFailureAndNestedScopes() {
    D::EnqueueReceipt outer, inner;
    AcceptedState outer_sink{&outer}, inner_sink{&inner};
    {
        D::EnqueueWitnessScope outer_scope(outer, SaveAcceptedOperation, &outer_sink);
        try {
            D::EnqueueWitnessScope inner_scope(inner, SaveAcceptedOperation, &inner_sink);
            auto failed_preparation = D::TakeEnqueueWitness();
            (void)failed_preparation;
            throw std::runtime_error("insertion did not happen");
        } catch (const std::runtime_error&) {}
        Check(inner.outcome == D::EnqueueOutcome::NotAccepted && inner_sink.calls == 0,
              "operation-failure", "failed preparation manufactured an accepted queue entry");
        auto outer_capture = D::TakeEnqueueWitness();
        outer_capture.PublishAccepted(OperationSample(25, 123));
    }
    Check(outer.outcome == D::EnqueueOutcome::Queued &&
              outer.operation.operation_id == 25 && outer_sink.calls == 1,
          "operation-nested", "exceptional nested scope lost the outer pending witness");
}
void OperationInlineDisabledAndThreadIsolation() {
    D::EnqueueReceipt receipt, disabled{D::EnqueueOutcome::Queued, OperationSample(99, 999)};
    AcceptedState sink{&receipt};
    {
        D::EnqueueWitnessScope scope(receipt, SaveAcceptedOperation, &sink);
        {
            D::EnqueueWitnessScope off(disabled, SaveAcceptedOperation, &sink, false);
            auto no_capture = D::TakeEnqueueWitness(false);
            no_capture.PublishAccepted(OperationSample(101, 1001));
        }
        Check(disabled.outcome == D::EnqueueOutcome::Queued && disabled.operation.operation_id == 99 &&
                  receipt.outcome == D::EnqueueOutcome::NotAccepted && sink.calls == 0,
              "operation-disabled", "disabled scope/capture changed receipt or consumed live TLS");
        std::thread independent([&] {
            auto absent = D::TakeEnqueueWitness();
            absent.PublishAccepted(OperationSample(103, 1003));
        });
        independent.join();
        auto inline_capture = D::TakeEnqueueWitness();
        inline_capture.PublishInline();
        inline_capture.PublishAccepted(OperationSample(105, 1005));
    }
    Check(receipt.outcome == D::EnqueueOutcome::Inline &&
              receipt.operation.selected_tick == 0 && receipt.operation.operation_id == 0 && sink.calls == 0,
          "operation-inline", "inline invocation claimed a queued tick or inherited another thread's witness");
}
void CallbackExecutionScopes() {
    const auto before = D::CurrentContext();
    const auto first = OperationSample(27, 127);
    const auto second = OperationSample(29, 129);
    const D::ParentRef begin1{700, 0x200, 27, D::Family::Priority};
    const D::ParentRef begin2{701, 0x200, 29, D::Family::Priority};
    {
        D::CallbackExecutionScope outer(D::QueuedExecution(first, begin1));
        auto observed = D::CurrentCallbackExecution();
        Check(observed.valid && !observed.inline_call && SameOperation(observed.operation, first) &&
                  observed.callback_begin.sequence == 700,
              "execution-queued", "callback lost its actual queue identity/begin parent");
        {
            D::CallbackExecutionScope off(D::QueuedExecution(second, begin2), false);
            observed = D::CurrentCallbackExecution();
            const auto disabled = D::CurrentCallbackExecution(false);
            Check(observed.valid && SameOperation(observed.operation, first) && !disabled.valid &&
                      !disabled.inline_call && disabled.operation.operation_id == 0,
                  "execution-disabled", "disabled execution scope/read touched current metadata");
        }
        {
            D::CallbackExecutionScope inline_scope(D::InlineExecution());
            observed = D::CurrentCallbackExecution();
            Check(!observed.valid && observed.inline_call && observed.operation.selected_tick == 0 &&
                      observed.callback_begin.sequence == 0,
                  "execution-inline", "inline callback inherited an outer accepted native tick");
            try {
                D::CallbackExecutionScope nested(D::QueuedExecution(second, begin2));
                throw std::runtime_error("nested callback failure");
            } catch (const std::runtime_error&) {}
            observed = D::CurrentCallbackExecution();
            Check(!observed.valid && observed.inline_call && observed.operation.operation_id == 0,
                  "execution-exception", "nested exception did not restore explicit inline state");
        }
        bool isolated = false;
        std::thread independent([&] {
            const auto other = D::CurrentCallbackExecution();
            isolated = !other.valid && !other.inline_call && other.operation.operation_id == 0;
        });
        independent.join();
        observed = D::CurrentCallbackExecution();
        Check(isolated && observed.valid && SameOperation(observed.operation, first) &&
                  observed.callback_begin.sequence == 700,
              "execution-nested", "TLS crossed native threads or failed to restore outer callback");
    }
    const auto after = D::CurrentCallbackExecution();
    Check(!after.valid && !after.inline_call && after.operation.operation_id == 0 &&
              D::CurrentContext().cause.sequence == before.cause.sequence &&
              D::CurrentContext().guest.sequence == before.guest.sequence,
          "execution-exit", "execution scope escaped or modified the existing guest Context");
}
void OperationAcceptanceBeforeImmediateCompletion() {
    D::EnqueueReceipt receipt;
    AcceptedState sink{&receipt};
    const auto operation = OperationSample(33, 133);
    {
        D::EnqueueWitnessScope scope(receipt, SaveAcceptedOperation, &sink);
        auto capture = D::TakeEnqueueWitness();
        capture.PublishAccepted(operation); // Real hook must be after queue insertion and parent commit.
        {
            D::CallbackExecutionScope execution(D::QueuedExecution(operation,
                {733, operation.native_domain, operation.operation_id, D::Family::Priority}));
            const auto active = D::CurrentCallbackExecution();
            Check(sink.stage == 1 && sink.receipt_was_committed && active.valid &&
                      SameOperation(active.operation, operation),
                  "operation-before-completion", "callback could begin before synchronous accepted metadata");
            sink.stage = 2; // Completion can precede DeferOperation's caller regaining control.
        }
    }
    Check(sink.stage == 2 && sink.calls == 1 && receipt.outcome == D::EnqueueOutcome::Queued &&
              SameOperation(receipt.operation, operation),
          "operation-no-regression", "scope return regressed a completed operation or lost acceptance");
}

}
int main() {
    static_assert(sizeof(D::Record) == 128);
    static_assert(sizeof(D::Recorder) <= 64 * 1024);
    ExactFlagAndDisabled();
    RolloverAndAncestry();
    CoalescingAndFocusedTarget();
    ForcedContention();
    DroppedParentThenRollover();
    ConcurrentSnapshots();
    ReportDeltaAndBounds();
    ContextLifetimeAndCallbacks();
    CurrentWorkerSurvivesTailRollover();
    BoundedCallbackParentBridge();
    FailedCallbackPreparationDoesNotAdvanceOrdinal();
    IdentityExactFlagAndDisabled();
    IdentityExplicitFieldsAndLifetimes();
    IdentityFormatterBounds();
    IdentityBudgetBoundary();
    IdentityConcurrentEmission();
    static_assert(sizeof(D::Context) == 80);
    static_assert(!std::is_copy_constructible_v<D::EnqueueCapture> && !std::is_move_constructible_v<D::EnqueueCapture>);
    OperationAcceptedOnce();
    OperationPreparationFailureAndNestedScopes();
    OperationInlineDisabledAndThreadIsolation();
    CallbackExecutionScopes();
    OperationAcceptanceBeforeImmediateCompletion();
    std::printf("CausalDiagnosticsTests: %s (%d failures)\n", failures ? "FAILED" : "PASS", failures);
    return failures ? 1 : 0;
}

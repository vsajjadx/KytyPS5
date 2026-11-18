#ifndef KYTY_CAUSAL_DIAGNOSTICS_H_
#define KYTY_CAUSAL_DIAGNOSTICS_H_

// Diagnostic storage only. It never waits, queries the GPU, or reads guest memory.
// A false enable branch returns before atomics, clocks, producer evaluation, or output.
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>

namespace Libs::Graphics::CausalDiagnostics {
enum class Family : uint8_t { Guest, Native, Priority, Signal, Flip, Count };
inline constexpr size_t FamilyCount = static_cast<size_t>(Family::Count);
inline constexpr size_t Capacity = 64;
inline constexpr size_t MaxReportBytes = 128 * 1024;
inline constexpr uint64_t ReportPeriodNs = 1'000'000'000;
enum class Event : uint16_t {
    GuestEnqueue, GuestSliceBegin, GuestSliceEnd, GuestBlocked, GuestSchedule,
    SuspendEnter, SuspendAcquired, SuspendEnqueued, SuspendCreditReleased,
    NativeSubmitAttempt, NativeSubmitResult, TimelineObserved, TimelineWaitEnter, TimelineWaitReturn,
    PriorityQueued, PriorityDequeued, PriorityWait, PriorityCallbackBegin, PriorityCallbackEnd,
    PriorityState, SignalPublished, SignalObserved, InterruptCallback,
    FlipReserved, FlipPrepared, FlipReady, FlipCancelled, FlipReplaced,
    FlipNoOwner, FlipCpuMember, FlipNotReady, FlipNotDue, FlipDispatch, FlipGuestComplete,
    TargetDispatchIndirectRecorded,
    SchedulerCreated, SchedulerDestroyed, GuestFlush
};
enum class Outcome : uint8_t { Disabled, Stored, Repeated, Dropped };
enum class ParentStatus : uint8_t { None, Retained, OlderThanRetainedTail, Missing, Unavailable };
struct ParentRef {
    uint64_t sequence = 0, domain = 0, id = 0;
    Family family = Family::Count;
};
// Diagnostic context is copied through an enabled-only callback wrapper. It is never
// a scheduling token and never substitutes for a publication or timeline observation.
struct Context {
    ParentRef cause{}, guest{};
    uint64_t point = 0, request = 0;
};
inline thread_local Context current_context{};
inline Context CurrentContext() noexcept { return current_context; }
class ContextScope {
public:
    explicit ContextScope(Context context, bool enabled = true) noexcept : m_enabled(enabled) {
        if (m_enabled) { m_previous = current_context; current_context = context; }
    }
    ~ContextScope() { if (m_enabled) current_context = m_previous; }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
private:
    bool m_enabled;
    Context m_previous{};
};
template <typename Callback>
auto BindContext(Context context, Callback&& callback) {
    return [context, callback = std::forward<Callback>(callback)]() mutable -> decltype(auto) {
        ContextScope scope(context);
        return callback();
    };
}
// Called only while the scheduler's existing operation mutex is held. Fixed
// storage is independent of queue length; the callback owns its full context too.
// IDs encode normal/priority parity, leaving 32 retained parents per queue.
class CallbackParents {
public:
    static constexpr size_t Size = 64;
    uint64_t NextId(bool priority) const noexcept {
        return ((m_queued[priority ? 1 : 0] + 1) << 1) | (priority ? 1u : 0u);
    }
    // Commit only after the engine queue insertion succeeds. Preparation or
    // allocation failure must not advance the actual dequeue ordinal.
    uint64_t Enqueued(bool priority, const Context& context) noexcept {
        const auto id = NextId(priority);
        ++m_queued[priority ? 1 : 0];
        auto& slot = m_slots[id % Size];
        if (slot.id != 0) ++m_overwritten;
        slot = {id, context};
        return id;
    }
    Context Dequeued(bool priority, uint64_t domain, uint64_t& id, bool& missing) noexcept {
        id = (++m_dequeued[priority ? 1 : 0] << 1) | (priority ? 1u : 0u);
        auto& slot = m_slots[id % Size];
        missing = slot.id != id;
        if (missing) {
            ++m_lost;
            Context context{};
            context.cause = {0, domain, id, Family::Priority};
            return context;
        }
        const auto context = slot.context;
        slot = {};
        return context;
    }
    uint64_t Lost() const noexcept { return m_lost; }
    uint64_t Overwritten() const noexcept { return m_overwritten; }
private:
    struct Slot { uint64_t id = 0; Context context{}; };
    std::array<Slot, Size> m_slots{};
    uint64_t m_queued[2]{}, m_dequeued[2]{}, m_lost = 0, m_overwritten = 0;
};
static_assert(sizeof(CallbackParents) <= 8 * 1024);
enum class WorkerPhase : uint8_t { NotStarted, QueueWait, NativeWait, Callback, Idle, Stopped };
struct CurrentState {
    bool available = false, phase_stable = false;
    uint64_t timestamp_ns = 0, domain = 0, lifetime_sequence = 0, lifetime_id = 0;
    uint64_t active_operation = 0, active_tick = 0;
    uint64_t priority_queued = 0, priority_front_tick = 0, normal_queued = 0;
    uint64_t phase_before = 0, phase_after = 0, last_native_submit = 0;
    uint64_t cached_known_tick = 0, recording_tick = 0, operation_state = 0;
    uint64_t parent_bridge_lost = 0, parent_bridge_overwritten = 0, report_contention = 0;
};
struct Record {
    uint64_t sequence = 0, timestamp_ns = 0, domain = 0, id = 0;
    uint64_t parent_sequence = 0, parent_domain = 0, parent_id = 0;
    uint64_t tick = 0, related = 0;
    std::array<uint64_t, 6> data{};
    uint32_t thread = 0; // Emulator surrogate unless a producer explicitly documents OS TID.
    Event event = Event::GuestEnqueue;
    Family parent_family = Family::Count;
    uint8_t flags = 0;
    void SetParent(const ParentRef& p) noexcept {
        parent_sequence = p.sequence; parent_domain = p.domain; parent_id = p.id; parent_family = p.family;
    }
};
static_assert(sizeof(Record) == 128);
struct RecordResult { Outcome outcome = Outcome::Disabled; ParentRef parent{}; };
struct FamilySnapshot {
    bool valid = false; // Record storage is coherent for this family only.
    uint64_t attempted = 0, stored = 0, repeated = 0, dropped = 0, overwritten = 0;
    size_t count = 0;
    std::array<Record, Capacity> records{};
    Record last{}; // Last observation, not necessarily the state of an entire subsystem.
};
struct Snapshot {
    std::array<FamilySnapshot, FamilyCount> family{};
    bool focus_available = false, focus_valid = false;
    uint64_t focus_dropped = 0;
    Record focus{};
};
inline bool EnabledFor(const char* value) noexcept {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}
inline bool SameState(const Record& a, const Record& b) noexcept {
    return a.domain == b.domain && a.id == b.id && a.parent_sequence == b.parent_sequence &&
           a.parent_domain == b.parent_domain && a.parent_id == b.parent_id &&
           a.tick == b.tick && a.related == b.related && a.data == b.data &&
           a.thread == b.thread && a.event == b.event && a.parent_family == b.parent_family &&
           a.flags == b.flags;
}
struct RecorderTestAccess;
class Recorder {
public:
    constexpr explicit Recorder(bool enabled) noexcept : m_enabled(enabled) {}
    bool Enabled() const noexcept { return m_enabled; }
    uint64_t NewId() noexcept {
        if (!m_enabled) return 0;
        return m_id.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    template <typename Make>
    RecordResult RecordLazy(Family family, Make&& make, bool coalesce = false, bool focus = false) {
        if (!m_enabled) return {};
        return TryRecord(family, std::forward<Make>(make)(), coalesce, focus);
    }
    RecordResult TryRecord(Family family, Record record, bool coalesce = false, bool focus = false) noexcept {
        if (!m_enabled) return {};
        const auto index = static_cast<size_t>(family);
        if (index >= FamilyCount) return {Outcome::Dropped, {}};
        auto& state = m_family[index];
        state.attempted.fetch_add(1, std::memory_order_relaxed);
        RecordResult result{};
        if (state.busy.test_and_set(std::memory_order_acquire)) {
            record.sequence = m_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
            state.dropped.fetch_add(1, std::memory_order_relaxed);
            result = {Outcome::Dropped, {record.sequence, record.domain, record.id, family}};
        } else {
            if (coalesce && state.count != 0 && SameState(state.last, record)) {
                ++state.repeated;
                state.last.timestamp_ns = record.timestamp_ns;
                result = {Outcome::Repeated, {state.last.sequence, state.last.domain, state.last.id, family}};
                record.sequence = state.last.sequence;
            } else {
                // Assign at commit, under the family lock. Assigning before it could put a
                // delayed producer's old sequence behind a newer report cursor.
                record.sequence = m_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
                state.records[state.next] = record;
                state.next = (state.next + 1) % Capacity;
                if (state.count == Capacity) ++state.overwritten;
                else ++state.count;
                ++state.stored;
                state.last = record;
                result = {Outcome::Stored, {record.sequence, record.domain, record.id, family}};
            }
            state.busy.clear(std::memory_order_release);
        }
        if (focus) {
            if (m_focus_busy.test_and_set(std::memory_order_acquire)) {
                m_focus_dropped.fetch_add(1, std::memory_order_relaxed);
            } else {
                // A producer delayed before this lock must not overwrite a newer target.
                if (record.sequence > m_focus.sequence) m_focus = record;
                m_focus_busy.clear(std::memory_order_release);
            }
        }
        return result;
    }
    bool ReadSnapshot(Snapshot& out) noexcept {
        if (!m_enabled) return false;
        out = {};
        for (size_t i = 0; i < FamilyCount; ++i) {
            auto& state = m_family[i];
            auto& copy = out.family[i];
            // These lifetime attempt/loss counters are observations, not a global instant.
            copy.attempted = state.attempted.load(std::memory_order_relaxed);
            copy.dropped = state.dropped.load(std::memory_order_relaxed);
            if (state.busy.test_and_set(std::memory_order_acquire)) continue;
            copy.valid = true;
            copy.stored = state.stored; copy.repeated = state.repeated; copy.overwritten = state.overwritten;
            copy.count = state.count; copy.last = state.last;
            const auto first = (state.next + Capacity - state.count) % Capacity;
            for (size_t j = 0; j < state.count; ++j)
                copy.records[j] = state.records[(first + j) % Capacity];
            state.busy.clear(std::memory_order_release);
        }
        out.focus_dropped = m_focus_dropped.load(std::memory_order_relaxed);
        if (!m_focus_busy.test_and_set(std::memory_order_acquire)) {
            out.focus_available = true;
            out.focus_valid = m_focus.sequence != 0;
            out.focus = m_focus;
            m_focus_busy.clear(std::memory_order_release);
        }
        return true;
    }
private:
    struct FamilyState {
        std::atomic_flag busy = ATOMIC_FLAG_INIT;
        std::atomic<uint64_t> attempted{0}, dropped{0};
        uint64_t stored = 0, repeated = 0, overwritten = 0;
        size_t next = 0, count = 0;
        std::array<Record, Capacity> records{};
        Record last{};
    };
    bool m_enabled;
    std::array<FamilyState, FamilyCount> m_family{};
    std::atomic<uint64_t> m_sequence{0}, m_id{0};
    std::atomic_flag m_focus_busy = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> m_focus_dropped{0};
    Record m_focus{};
    friend struct RecorderTestAccess;
};
static_assert(sizeof(Recorder) <= 64 * 1024);
static_assert(std::atomic<uint64_t>::is_always_lock_free);
inline ParentStatus Ancestry(const Snapshot& snapshot, const Record& record) noexcept {
    if (record.parent_family == Family::Count && record.parent_sequence == 0 &&
        record.parent_domain == 0 && record.parent_id == 0) return ParentStatus::None;
    const auto index = static_cast<size_t>(record.parent_family);
    if (index >= FamilyCount || record.parent_sequence == 0) return ParentStatus::Missing;
    const auto matches = [&](const Record& parent) {
        return parent.sequence == record.parent_sequence && parent.domain == record.parent_domain &&
               parent.id == record.parent_id;
    };
    if (record.parent_family == Family::Native && snapshot.focus_available &&
        snapshot.focus_valid && matches(snapshot.focus)) return ParentStatus::Retained;
    const auto& family = snapshot.family[index];
    if (!family.valid) return ParentStatus::Unavailable;
    for (size_t i = 0; i < family.count; ++i)
        if (matches(family.records[i])) return ParentStatus::Retained;
    if (family.count != 0 && family.overwritten != 0 &&
        record.parent_sequence < family.records[0].sequence) return ParentStatus::OlderThanRetainedTail;
    return ParentStatus::Missing;
}
inline const char* ParentName(ParentStatus status) noexcept {
    switch (status) {
    case ParentStatus::None: return "none";
    case ParentStatus::Retained: return "retained";
    case ParentStatus::OlderThanRetainedTail: return "older_than_retained_tail";
    case ParentStatus::Missing: return "missing";
    case ParentStatus::Unavailable: return "unavailable";
    }
    return "invalid";
}
struct ReportCursor {
    bool reported = false;
    uint64_t last_ns = 0, focus_sequence = 0;
    std::array<uint64_t, FamilyCount> sequence{};
};
struct ReportResult { bool emitted = false; size_t bytes = 0, records = 0, truncated = 0; };

// The caller serializes cursor ownership and calls outside engine locks.
// Emit must not reenter this reporter. No dynamic formatting buffer is allocated.
template <typename Emit>
ReportResult Report(Recorder& recorder, ReportCursor& cursor, uint64_t now_ns, Emit&& emit,
                    const CurrentState* current = nullptr) {
    if (!recorder.Enabled()) return {};
    if (cursor.reported && (now_ns < cursor.last_ns || now_ns - cursor.last_ns < ReportPeriodNs))
        return {};
    Snapshot snapshot{};
    if (!recorder.ReadSnapshot(snapshot)) return {};
    cursor.reported = true; cursor.last_ns = now_ns;
    ReportResult result{true, 0, 0, 0};
    char line[1536];
    constexpr size_t footer_reserve = 192;
    const auto output = [&](int length, bool footer = false) {
        const auto limit = MaxReportBytes - (footer ? 0 : footer_reserve);
        if (length < 0 || static_cast<size_t>(length) >= sizeof(line) ||
            static_cast<size_t>(length) > limit - result.bytes) {
            ++result.truncated; return false;
        }
        emit(std::string_view(line, static_cast<size_t>(length)));
        result.bytes += static_cast<size_t>(length);
        return true;
    };
    output(std::snprintf(line, sizeof(line),
        "CAUSAL_BEGIN v=1 ns=%" PRIu64 " storage=per_family_not_global capacity=%zu max_bytes=%zu\n",
        now_ns, Capacity, MaxReportBytes));
    const auto event = [&](const Record& r, Family family, const char* tag) {
        const auto n = std::snprintf(line, sizeof(line),
            "%s seq=%" PRIu64 " ns=%" PRIu64 " f=%u k=%u d=%" PRIx64 " id=%" PRIu64
            " ps=%" PRIu64 " pd=%" PRIx64 " pi=%" PRIu64 " pf=%u tick=%" PRIu64
            " rel=%" PRIu64 " tid=%u flags=%u ancestry=%s x=%" PRIx64 ",%" PRIx64 ",%" PRIx64
            ",%" PRIx64 ",%" PRIx64 ",%" PRIx64 "\n",
            tag, r.sequence, r.timestamp_ns, static_cast<unsigned>(family), static_cast<unsigned>(r.event),
            r.domain, r.id, r.parent_sequence, r.parent_domain, r.parent_id, static_cast<unsigned>(r.parent_family),
            r.tick, r.related, r.thread, r.flags, ParentName(Ancestry(snapshot, r)),
            r.data[0], r.data[1], r.data[2], r.data[3], r.data[4], r.data[5]);
        return output(n);
    };
    // This independently sampled live state cannot be overwritten by enqueues or
    // normal callbacks in a history tail. Queue fields share the scheduler mutex;
    // phase/last-submit/timeline are atomic observations, not a global instant.
    if (current != nullptr) {
        const auto& c = *current;
        Record lifetime{};
        lifetime.SetParent({c.lifetime_sequence, c.domain, c.lifetime_id, Family::Native});
        output(std::snprintf(line, sizeof(line),
            "CAUSAL_CURRENT available=%u phase_stable=%u ns=%" PRIu64 " d=%" PRIx64
            " life=%" PRIu64 " life_id=%" PRIu64 " life_record=%s active_op=%" PRIu64 " active_tick=%" PRIu64
            " priority_queued=%" PRIu64 " priority_front_tick=%" PRIu64 " normal_queued=%" PRIu64
            " phase_before=%" PRIu64 " phase_after=%" PRIu64 " last_native_submit=%" PRIu64
            " cached_known_tick=%" PRIu64 " recording_tick=%" PRIu64 " operation_state=%" PRIu64
            " parent_bridge_lost=%" PRIu64 " parent_bridge_overwritten=%" PRIu64 " report_contention=%" PRIu64
            " scope=live_renderer_try_lock_plus_atomic_observations\n",
            c.available ? 1u : 0u, c.phase_stable ? 1u : 0u, c.timestamp_ns, c.domain,
            c.lifetime_sequence, c.lifetime_id, ParentName(Ancestry(snapshot, lifetime)),
            c.available ? c.active_operation : 0, c.available ? c.active_tick : 0,
            c.available ? c.priority_queued : 0, c.available ? c.priority_front_tick : 0,
            c.available ? c.normal_queued : 0, c.available ? c.phase_before : 0,
            c.available ? c.phase_after : 0, c.available ? c.last_native_submit : 0,
            c.available ? c.cached_known_tick : 0, c.available ? c.recording_tick : 0,
            c.available ? c.operation_state : 0, c.available ? c.parent_bridge_lost : 0,
            c.available ? c.parent_bridge_overwritten : 0, c.report_contention));
    }
    // The scarce target is emitted before high-rate tails. A busy snapshot is not "no target".
    output(std::snprintf(line, sizeof(line),
        "CAUSAL_TARGET_STATE available=%u valid=%u dropped=%" PRIu64 " last_seq=%" PRIu64
        " last_ns=%" PRIu64 " meaning=recorded_dispatch_indirect\n",
        snapshot.focus_available ? 1u : 0u, snapshot.focus_valid ? 1u : 0u,
        snapshot.focus_dropped, snapshot.focus.sequence, snapshot.focus.timestamp_ns));
    if (snapshot.focus_available && snapshot.focus_valid && snapshot.focus.sequence > cursor.focus_sequence &&
        event(snapshot.focus, Family::Native, "CAUSAL_TARGET")) cursor.focus_sequence = snapshot.focus.sequence;
    for (size_t i = 0; i < FamilyCount; ++i) {
        const auto& family = snapshot.family[i];
        output(std::snprintf(line, sizeof(line),
            "CAUSAL_FAMILY f=%zu valid=%u attempts=%" PRIu64 " stored=%" PRIu64
            " repeated=%" PRIu64 " dropped=%" PRIu64 " overwritten=%" PRIu64
            " first_seq=%" PRIu64 " last_seq=%" PRIu64 " last_seen_ns=%" PRIu64
            " last_k=%u last_id=%" PRIu64 " last_tick=%" PRIu64 "\n",
            i, family.valid ? 1u : 0u, family.attempted, family.stored, family.repeated,
            family.dropped, family.overwritten, family.count ? family.records[0].sequence : 0,
            family.last.sequence, family.last.timestamp_ns, static_cast<unsigned>(family.last.event),
            family.last.id, family.last.tick));
        if (!family.valid) continue;
        for (size_t j = 0; j < family.count; ++j) {
            const auto& record = family.records[j];
            if (record.sequence <= cursor.sequence[i]) continue;
            if (!event(record, static_cast<Family>(i), "CAUSAL_EVENT")) break;
            cursor.sequence[i] = record.sequence;
            ++result.records;
        }
    }
    output(std::snprintf(line, sizeof(line),
        "CAUSAL_END ns=%" PRIu64 " records=%zu truncated=%zu\n", now_ns, result.records, result.truncated), true);
    return result;
}
} // namespace Libs::Graphics::CausalDiagnostics
#endif

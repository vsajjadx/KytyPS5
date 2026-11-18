#ifndef KYTY_INDIRECT_ARGS_LEDGER_H
#define KYTY_INDIRECT_ARGS_LEDGER_H

// Private candidate: caller-owned, explicitly installed diagnostics. No activation policy.
// The journal records selected host actions and recorded Vulkan commands, not GPU execution.
// An entry's journal index, host time, or tracked-command ordinal is not cross-thread
// happens-before. A mapped snapshot is not proof that a later GPU read consumed those bytes.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

namespace Libs::Graphics::IndirectArgsLedger {

inline constexpr size_t MaxWatches = 8;
inline constexpr size_t EventCapacity = 4096;
inline constexpr size_t ReservationCapacity = 512;
inline constexpr uint64_t ArgumentBytes = 12;
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<void*>::is_always_lock_free);
static_assert(std::atomic<bool>::is_always_lock_free);

enum class Kind : uint8_t {
    CpuStoreCommitted,
    CpuCopyCommitted,
    CpuFillCommitted,
    HostPayloadSnapshot,
    TransferCopyRecorded,
    TransferFillRecorded,
    MigrationCopyRecorded,
    IndirectConsumerRecorded,
    DescriptorWriteCandidate,
    CpuDirtyUnknown,
};
enum class PayloadMeaning : uint8_t {
    None,
    CpuCommittedBytes,
    HostSnapshotBytes,
    EmbeddedCommandBytes,
};
enum class ReservationState : uint8_t {
    NotSuballocated,
    Known,
    Unknown,
    HistoryIncomplete,
};
struct ResourceIdentity {
    uint64_t capture = 0;
    uint64_t generation = 0; // assigned at successful native Buffer construction, not a cache slot
    uint64_t handle = 0;
    uint64_t bytes = 0;
};
struct NativeSpan {
    ResourceIdentity resource {};
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t reservation = 0; // new serial for each successful ring/prepared reservation
    ReservationState reservation_state = ReservationState::Unknown;
    bool identity_known = false;
};
struct Position {
    uint64_t native_domain = 0;
    uint64_t causal_lifetime = 0; // zero means unavailable; pointer alone is not a lifetime
    uint64_t command_buffer = 0;
    uint64_t tick = 0; // recording tick, never a completion witness
    uint64_t tracked_command_ordinal = 0; // only this thread's instrumented native actions
    bool owner_witness = false;
};
struct ShaderIdentity {
    uint64_t shader = 0;
    uint64_t module = 0;
    uint32_t stage = 0;
    uint32_t resource = 0;
    uint32_t descriptor = 0;
};
struct Event {
    uint64_t journal_sequence = 0;
    uint64_t host_thread_token = 0; // diagnostic thread lifetime token, not an OS thread id
    uint64_t host_action_ordinal = 0;
    Kind kind = Kind::CpuDirtyUnknown;
    PayloadMeaning payload_meaning = PayloadMeaning::None;
    uint8_t watch = 0;
    uint8_t word = 0;
    uint8_t byte_mask = 0; // intersection with this watched dword, bits 0..3
    uint8_t payload_mask = 0; // interpretation is ALWAYS given by payload_meaning and kind
    uint32_t payload = 0; // little-endian bytes in watched-dword positions; unmasked bytes zero
    uint64_t word_address = 0;
    uint64_t operation_guest_begin = 0;
    uint64_t operation_bytes = 0;
    uint64_t intersection_guest_begin = 0;
    uint64_t intersection_bytes = 0;
    uint64_t source_guest_begin = 0; // matching source intersection; zero means unknown/non-guest
    uint64_t guest_mapping_generation = 0; // deliberately unobserved in this minimum candidate
    NativeSpan source {};
    NativeSpan destination {};
    Position position {};
    ShaderIdentity shader {};
    uint64_t causal_dispatch_sequence = 0; // exact existing dispatch event, if emitted
};
struct Reservation {
    ResourceIdentity resource {};
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t serial = 0;
};
struct Coverage {
    uint64_t event_tickets = 0;
    uint64_t dropped_events = 0;
    uint64_t reservation_tickets = 0;
    uint64_t dropped_reservations = 0;
    bool event_history_full = false;
    bool reservation_history_full = false;
    bool arbitrary_guest_stores_observed = false;
    bool kernel_backing_writes_observed = false;
    bool all_native_commands_observed = false;
    bool native_execution_observed = false;
};
inline bool ValidRange(uint64_t begin, uint64_t bytes) noexcept {
    return bytes != 0 && begin <= std::numeric_limits<uint64_t>::max() - bytes;
}
inline uint64_t GuestOffset(uint64_t base, uint64_t offset, uint64_t bytes) noexcept {
    if (base == 0 || offset > std::numeric_limits<uint64_t>::max() - base) return 0;
    const auto address = base + offset;
    return ValidRange(address, bytes) ? address : 0;
}
inline bool Overlaps(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) noexcept {
    return ValidRange(a, an) && ValidRange(b, bn) && a < b + bn && b < a + an;
}
inline bool SameResource(const ResourceIdentity& a, const ResourceIdentity& b) noexcept {
    return a.capture != 0 && a.generation != 0 && a.capture == b.capture &&
           a.generation == b.generation && a.handle == b.handle;
}
inline NativeSpan Slice(NativeSpan span, uint64_t delta, uint64_t bytes) noexcept {
    if (delta > span.bytes || bytes > span.bytes - delta ||
        span.offset > std::numeric_limits<uint64_t>::max() - delta) return {};
    span.offset += delta;
    span.bytes = bytes;
    if (!ValidRange(span.offset, span.bytes)) return {};
    return span;
}
// NativeStorageBuffer aligns the Vulkan offset down and RebindBuffers packs the
// byte adjustment into already-final shader_data. Use that exact committed metadata;
// a cache lookup or the aligned descriptor offset alone would give the wrong lineage.
inline NativeSpan DescriptorSpan(uint64_t handle, uint64_t descriptor_offset,
                                 uint64_t descriptor_range, std::span<const uint32_t> shader_data,
                                 uint64_t memory_offset_dword, uint32_t descriptor_index) noexcept {
    if (handle == 0 || descriptor_index == UINT32_MAX ||
        descriptor_range == std::numeric_limits<uint64_t>::max() ||
        memory_offset_dword > shader_data.size()) return {};
    const uint64_t word = descriptor_index / 4u;
    if (word >= shader_data.size() - memory_offset_dword) return {};
    const uint64_t adjustment =
        (shader_data[static_cast<size_t>(memory_offset_dword + word)] >>
            ((descriptor_index % 4u) * 8u)) & 0xffu;
    if (adjustment >= descriptor_range ||
        descriptor_offset > std::numeric_limits<uint64_t>::max() - adjustment) return {};
    NativeSpan span {};
    span.resource.handle = handle;
    span.offset = descriptor_offset + adjustment;
    span.bytes = descriptor_range - adjustment;
    if (!ValidRange(span.offset, span.bytes)) return {};
    return span; // Logical offset known; allocation generation/reservation remain unknown.
}
template <typename T> inline uint64_t HandleBits(T value) noexcept {
    static_assert(sizeof(value) <= sizeof(uint64_t));
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

class Recorder;
inline std::atomic<Recorder*> g_recorder {nullptr};
inline std::atomic<uint64_t> g_capture_serial {1};
inline std::atomic<uint64_t> g_thread_serial {1};
inline thread_local uint64_t g_thread_token = g_thread_serial.fetch_add(1, std::memory_order_relaxed);
inline Recorder* Active() noexcept { return g_recorder.load(std::memory_order_acquire); }

class Recorder final {
public:
    explicit Recorder(std::span<const uint64_t> watch_begins) noexcept
        : m_capture(g_capture_serial.fetch_add(1, std::memory_order_relaxed)) {
        if (watch_begins.empty() || watch_begins.size() > MaxWatches || m_capture == 0) return;
        for (size_t i = 0; i < watch_begins.size(); ++i) {
            const auto begin = watch_begins[i];
            if (begin == 0 || (begin & 3u) != 0 || !ValidRange(begin, ArgumentBytes)) return;
            for (size_t j = 0; j < i; ++j) {
                if (Overlaps(begin, ArgumentBytes, m_watches[j], ArgumentBytes)) return;
            }
            m_watches[i] = begin;
        }
        m_watch_count = watch_begins.size();
        m_valid = true;
    }
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;
    [[nodiscard]] bool Valid() const noexcept { return m_valid; }
    [[nodiscard]] uint64_t Capture() const noexcept { return m_capture; }

    // The caller must install before resource creation/writers and remove only after all
    // participating producer threads and resources stop using it. Neither method waits.
    [[nodiscard]] bool Install() noexcept {
        if (!m_valid) return false;
        bool used = false;
        if (!m_installed_once.compare_exchange_strong(used, true, std::memory_order_acq_rel))
            return false; // never rearm a capture whose per-thread ordinals/history already exist
        Recorder* expected = nullptr;
        if (g_recorder.compare_exchange_strong(expected, this, std::memory_order_acq_rel))
            return true;
        m_installed_once.store(false, std::memory_order_release); // unsuccessful attempt is retryable
        return false;
    }
    [[nodiscard]] bool Uninstall() noexcept {
        Recorder* expected = this;
        return g_recorder.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
    }
    [[nodiscard]] ResourceIdentity NewResource(uint64_t handle, uint64_t bytes) noexcept {
        if (!m_valid || handle == 0 || bytes == 0) return {};
        return {m_capture, m_resource_serial.fetch_add(1, std::memory_order_relaxed), handle, bytes};
    }
    void Reserve(const ResourceIdentity& resource, uint64_t offset, uint64_t bytes) noexcept {
        if (resource.capture != m_capture || resource.generation == 0 ||
            offset > resource.bytes || bytes == 0 || bytes > resource.bytes - offset) return;
        const auto slot = Claim(m_reservation_next, m_reservation_full,
                                m_reservation_dropped, ReservationCapacity);
        if (slot >= ReservationCapacity) return;
        m_reservations[slot].value = {resource, offset, bytes, slot + 1};
        m_reservations[slot].ready.store(true, std::memory_order_release);
    }
    [[nodiscard]] NativeSpan Span(const ResourceIdentity& resource, uint64_t offset,
                                  uint64_t bytes, bool suballocated) const noexcept {
        NativeSpan out {resource, offset, bytes, 0, ReservationState::Unknown, false};
        if (offset > resource.bytes || bytes > resource.bytes - offset) return {};
        out.identity_known = resource.capture == m_capture && resource.generation != 0;
        if (!out.identity_known) return out; // resource predates this capture, or handle-only
        if (!suballocated) {
            out.reservation_state = ReservationState::NotSuballocated;
            return out;
        }
        if (m_reservation_full.load(std::memory_order_acquire)) {
            out.reservation_state = ReservationState::HistoryIncomplete;
            return out;
        }
        const auto count = std::min<uint64_t>(m_reservation_next.load(std::memory_order_acquire),
                                              ReservationCapacity);
        for (uint64_t i = count; i != 0; --i) {
            const auto& slot = m_reservations[i - 1];
            // A writer has reserved a slot but not published its identity yet. Fail closed.
            if (!slot.ready.load(std::memory_order_acquire)) return out;
            const auto& reservation = slot.value;
            if (!SameResource(resource, reservation.resource) ||
                !Overlaps(offset, bytes, reservation.offset, reservation.bytes)) continue;
            // A later partial overwrite invalidates a whole-span single-reservation claim.
            if (offset < reservation.offset || bytes > reservation.bytes ||
                offset - reservation.offset > reservation.bytes - bytes) return out;
            out.reservation = reservation.serial;
            out.reservation_state = ReservationState::Known;
            return out;
        }
        return out;
    }
    [[nodiscard]] uint64_t NextTrackedCommand() noexcept {
        return ++ThreadState().command_ordinal;
    }
    [[nodiscard]] bool ReadEvent(size_t index, Event& out) const noexcept {
        if (index >= EventCapacity || !m_events[index].ready.load(std::memory_order_acquire)) return false;
        out = m_events[index].value;
        return true;
    }
    [[nodiscard]] Coverage ReadCoverage() const noexcept {
        return {m_event_next.load(std::memory_order_acquire),
                m_event_dropped.load(std::memory_order_relaxed),
                m_reservation_next.load(std::memory_order_acquire),
                m_reservation_dropped.load(std::memory_order_relaxed),
                m_event_full.load(std::memory_order_acquire),
                m_reservation_full.load(std::memory_order_acquire)};
    }

    // Runtime callers use only the recorder returned by Active(), never a retained
    // pointer after Uninstall. Direct member calls below are also used by the isolated
    // CPU metadata contract, which is not a runtime writer/integration test.
    void CpuStore(uint64_t guest, const void* already_read_payload, uint64_t bytes) noexcept {
        Operation op {Kind::CpuStoreCommitted, guest, bytes};
        op.meaning = PayloadMeaning::CpuCommittedBytes;
        op.payload = static_cast<const uint8_t*>(already_read_payload);
        Emit(op);
    }
    void CpuCopy(uint64_t destination, uint64_t source, uint64_t bytes,
                 const void* owned_snapshot = nullptr) noexcept {
        Operation op {Kind::CpuCopyCommitted, destination, bytes};
        op.source_guest = source;
        op.payload = static_cast<const uint8_t*>(owned_snapshot);
        if (owned_snapshot != nullptr) op.meaning = PayloadMeaning::CpuCommittedBytes;
        Emit(op); // direct guest-to-guest memcpy records lineage; it never rereads the guest
    }
    void CpuFill(uint64_t guest, uint64_t bytes, uint32_t pattern) noexcept {
        Operation op {Kind::CpuFillCommitted, guest, bytes};
        op.meaning = PayloadMeaning::CpuCommittedBytes;
        op.pattern = pattern;
        op.repeated_pattern = true;
        Emit(op);
    }
    void HostSnapshot(uint64_t guest, uint64_t bytes, NativeSpan destination,
                      const void* owned_mapped_payload) noexcept {
        Operation op {Kind::HostPayloadSnapshot, guest, bytes};
        op.destination = destination;
        op.meaning = PayloadMeaning::HostSnapshotBytes;
        op.payload = static_cast<const uint8_t*>(owned_mapped_payload);
        Emit(op); // captured after the existing mapped copy, before any later GPU consumption
    }
    void TransferCopy(uint64_t guest, uint64_t bytes, NativeSpan source, NativeSpan destination,
                      Position position, uint64_t source_guest = 0, bool migration = false) noexcept {
        Operation op {migration ? Kind::MigrationCopyRecorded : Kind::TransferCopyRecorded,
                      guest, bytes};
        op.source = source;
        op.destination = destination;
        op.position = position;
        op.source_guest = source_guest;
        Emit(op); // no payload: a recorded copy is not execution/completion evidence
    }
    void TransferFill(uint64_t guest, uint64_t bytes, uint32_t pattern,
                      NativeSpan destination, Position position) noexcept {
        Operation op {Kind::TransferFillRecorded, guest, bytes};
        op.destination = destination;
        op.position = position;
        op.meaning = PayloadMeaning::EmbeddedCommandBytes;
        op.pattern = pattern;
        op.repeated_pattern = true;
        Emit(op);
    }
    void Consumer(uint64_t guest, NativeSpan arguments, Position position,
                  ShaderIdentity shader, uint64_t causal_sequence) noexcept {
        Operation op {Kind::IndirectConsumerRecorded, guest, ArgumentBytes};
        op.source = arguments;
        op.position = position;
        op.shader = shader;
        op.causal_sequence = causal_sequence;
        Emit(op); // records the exact bound input owner, never the argument values
    }
    void Candidate(uint64_t guest, uint64_t bytes, NativeSpan destination,
                   Position position, ShaderIdentity shader) noexcept {
        Operation op {Kind::DescriptorWriteCandidate, guest, bytes};
        op.destination = destination;
        op.position = position;
        op.shader = shader;
        Emit(op); // overlap and writable metadata do not prove a shader store
    }
    void DirtyUnknown(uint64_t guest, uint64_t bytes) noexcept {
        Emit({Kind::CpuDirtyUnknown, guest, bytes}); // the fault may precede the real store
    }

private:
    struct Operation {
        Kind kind;
        uint64_t guest;
        uint64_t bytes;
        uint64_t source_guest = 0;
        NativeSpan source {};
        NativeSpan destination {};
        Position position {};
        ShaderIdentity shader {};
        uint64_t causal_sequence = 0;
        PayloadMeaning meaning = PayloadMeaning::None;
        const uint8_t* payload = nullptr;
        uint32_t pattern = 0;
        bool repeated_pattern = false;
    };
    template <typename T> struct Slot {
        std::atomic<bool> ready {false};
        T value {};
    };
    struct ThreadCounters {
        uint64_t capture = 0;
        uint64_t action_ordinal = 0;
        uint64_t command_ordinal = 0;
    };
    ThreadCounters& ThreadState() const noexcept {
        static thread_local ThreadCounters state {};
        if (state.capture != m_capture) state = {m_capture, 0, 0};
        return state;
    }
    static uint64_t Claim(std::atomic<uint64_t>& next, std::atomic<bool>& full,
                          std::atomic<uint64_t>& dropped, uint64_t capacity) noexcept {
        if (full.load(std::memory_order_acquire)) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return capacity;
        }
        const auto ticket = next.fetch_add(1, std::memory_order_acq_rel);
        if (ticket >= capacity) {
            full.store(true, std::memory_order_release);
            dropped.fetch_add(1, std::memory_order_relaxed);
        }
        return ticket; // immutable slots are never recycled
    }
    void Emit(const Operation& op) noexcept {
        if (!m_valid || op.guest == 0 || !ValidRange(op.guest, op.bytes)) return;
        uint64_t action = 0;
        for (size_t w = 0; w < m_watch_count; ++w) {
            for (uint8_t word = 0; word < 3; ++word) {
                const auto word_address = m_watches[w] + uint64_t {word} * 4;
                if (!Overlaps(op.guest, op.bytes, word_address, 4)) continue;
                if (action == 0) action = ++ThreadState().action_ordinal;
                const auto begin = std::max(op.guest, word_address);
                const auto end = std::min(op.guest + op.bytes, word_address + 4);
                const auto delta = begin - op.guest;
                const auto count = end - begin;
                const auto byte_in_word = begin - word_address;
                const auto slot = Claim(m_event_next, m_event_full, m_event_dropped, EventCapacity);
                if (slot >= EventCapacity) continue;
                Event event {};
                event.journal_sequence = slot + 1;
                event.host_thread_token = g_thread_token;
                event.host_action_ordinal = action;
                event.kind = op.kind;
                event.payload_meaning = op.meaning;
                event.watch = static_cast<uint8_t>(w);
                event.word = word;
                event.byte_mask = static_cast<uint8_t>(((1u << count) - 1u) << byte_in_word);
                event.word_address = word_address;
                event.operation_guest_begin = op.guest;
                event.operation_bytes = op.bytes;
                event.intersection_guest_begin = begin;
                event.intersection_bytes = count;
                if (op.source_guest != 0 && ValidRange(op.source_guest, op.bytes))
                    event.source_guest_begin = op.source_guest + delta;
                event.source = Slice(op.source, delta, count);
                event.destination = Slice(op.destination, delta, count);
                event.position = op.position;
                event.shader = op.shader;
                event.causal_dispatch_sequence = op.causal_sequence;
                if (op.meaning != PayloadMeaning::None && (op.payload != nullptr || op.repeated_pattern)) {
                    event.payload_mask = event.byte_mask;
                    for (uint64_t b = 0; b < count; ++b) {
                        const auto byte = op.repeated_pattern
                            ? uint8_t(op.pattern >> (((delta + b) & 3u) * 8u))
                            : op.payload[delta + b];
                        event.payload |= uint32_t {byte} << ((byte_in_word + b) * 8u);
                    }
                }
                m_events[slot].value = event;
                m_events[slot].ready.store(true, std::memory_order_release);
            }
        }
    }

    uint64_t m_capture = 0;
    bool m_valid = false;
    std::atomic<bool> m_installed_once {false};
    size_t m_watch_count = 0;
    std::array<uint64_t, MaxWatches> m_watches {};
    std::atomic<uint64_t> m_resource_serial {1};
    std::array<Slot<Event>, EventCapacity> m_events {};
    std::atomic<uint64_t> m_event_next {0};
    std::atomic<uint64_t> m_event_dropped {0};
    std::atomic<bool> m_event_full {false};
    std::array<Slot<Reservation>, ReservationCapacity> m_reservations {};
    std::atomic<uint64_t> m_reservation_next {0};
    std::atomic<uint64_t> m_reservation_dropped {0};
    std::atomic<bool> m_reservation_full {false};
};

inline thread_local bool g_copy_is_migration = false;
inline thread_local uint64_t g_copy_source_guest = 0;
class CopyScope final {
public:
    explicit CopyScope(bool migration, uint64_t source_guest = 0) noexcept {
        if (Active() == nullptr) return;
        m_installed = true;
        m_migration = g_copy_is_migration;
        m_source = g_copy_source_guest;
        g_copy_is_migration = migration;
        g_copy_source_guest = source_guest;
    }
    ~CopyScope() {
        if (!m_installed) return;
        g_copy_is_migration = m_migration;
        g_copy_source_guest = m_source;
    }
    CopyScope(const CopyScope&) = delete;
    CopyScope& operator=(const CopyScope&) = delete;
private:
    bool m_installed = false;
    bool m_migration = false;
    uint64_t m_source = 0;
};
} // namespace Libs::Graphics::IndirectArgsLedger
#endif // KYTY_INDIRECT_ARGS_LEDGER_H

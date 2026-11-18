#ifndef KYTY_INDIRECT_ARGS_DIAGNOSTICS_H_
#define KYTY_INDIRECT_ARGS_DIAGNOSTICS_H_

#include "graphics/host_gpu/renderer/causalDiagnostics.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

namespace Libs::Graphics {
struct GraphicContext;
class CommandScheduler;
class Buffer;
struct IndirectArgsObserverControlTestAccess;

// Exact child-only filter: sixteen hexadecimal digits plus the existing exact
// CAUSAL=1 and FAULT_PEEK=1 gates. This is never a renderer policy input.
[[nodiscard]] bool IndirectArgsDiagnosticEnabled() noexcept;
[[nodiscard]] uint64_t IndirectArgsDiagnosticShader() noexcept;

class IndirectArgsDiagnostics {
public:
    static constexpr size_t Capacity = 8;
    static constexpr uint64_t ArgumentBytes = 3 * sizeof(uint32_t);
    enum class Status : uint8_t {
        Disabled = 0, NotObserved = 1, NoCopies = 2, LockUnavailable = 3,
        InvalidIdentity = 4, NativeIncomplete = 5, HostWaitIncomplete = 6,
        Observed = 7, AlreadyObserved = 8
    };
    struct Dispatch {
        bool valid = false;
        uint64_t ordinal = 0, sequence = 0, domain = 0, lifetime = 0, tick = 0;
        uint64_t shader = 0, guest_address = 0, native_buffer = 0, native_offset = 0;
        uint32_t recording_thread = 0; // Internal unique ID, never inferred OS identity.
        CausalDiagnostics::ParentRef cause {}, guest {};
        uint64_t point = 0, request = 0;
    };
    struct Observation {
        Dispatch dispatch {};
        bool valid = false; // Payload was read after the required real native host wait.
        std::array<uint32_t, 3> groups {};
        uint64_t observed_known_tick = 0, observed_host_waited_tick = 0;
    };
    struct Snapshot {
        bool enabled = false, available = false;
        Status status = Status::Disabled;
        uint64_t selected_shader = 0;
        uint64_t matched_total = 0, copied_total = 0, observed_total = 0;
        uint64_t capacity_skipped = 0, identity_errors = 0;
        uint64_t cached_known_tick = 0, host_wait_entered_tick = 0, host_waited_tick = 0;
        Dispatch latest_match {}; // Includes uncaptured matches after fixed capacity is exhausted.
        std::array<Observation, Capacity> slots {};
    };

    IndirectArgsDiagnostics(GraphicContext& graphics, CommandScheduler& scheduler);
    ~IndirectArgsDiagnostics();
    IndirectArgsDiagnostics(const IndirectArgsDiagnostics&) = delete;
    IndirectArgsDiagnostics& operator=(const IndirectArgsDiagnostics&) = delete;

    [[nodiscard]] bool Matches(uint64_t shader) const noexcept {
        return shader == m_selected_shader;
    }
    // Called only around the final, actually recorded native indirect dispatch.
    // No source pointer is retained. NoteRecordedDispatch joins the allocated
    // causal sequence, including an explicitly lost causal-history record.
    void CopyBeforeDispatch(vk::CommandBuffer command, const Buffer& source,
                            uint64_t offset, uint64_t guest_address, uint64_t shader) noexcept;
    void NoteRecordedDispatch(const CausalDiagnostics::Record& event,
                              const CausalDiagnostics::RecordResult& result) noexcept;
    [[nodiscard]] bool PeekCompleted() noexcept;
    [[nodiscard]] Snapshot Read() const noexcept;

private:
    friend struct IndirectArgsObserverControlTestAccess;
    enum class TestControlKind : uint8_t {
        ComposedBarrierOnly,
        ImmutableScratchCopy,
    };
    // Installed only by the native test friend before recording. No runtime
    // option selects this state. Its records are not typed argument observations.
    struct TestControlState {
        static constexpr uint64_t SourceBytes = 5 * sizeof(uint32_t);
        static constexpr uint64_t SourceOffset = sizeof(uint32_t);
        static constexpr uint64_t SlotBytes = 5 * sizeof(uint32_t);
        static constexpr uint64_t DestinationBytes = Capacity * SlotBytes;
        TestControlKind kind = TestControlKind::ComposedBarrierOnly;
        uint64_t recorded_total = 0;
        std::array<Dispatch, Capacity> dispatches {};
        std::unique_ptr<Buffer> source, destination;
    };
    std::unique_ptr<TestControlState> m_test_control;
    GraphicContext& m_graphics;
    CommandScheduler& m_scheduler;
    const uint64_t m_selected_shader;
    std::unique_ptr<Buffer> m_download;
    mutable std::mutex m_mutex;
    Snapshot m_state {};
    // GPU-owner-only bridge across the immediately adjacent copy/dispatch pair.
    Dispatch m_pending {};
    size_t m_pending_slot = Capacity;
    bool m_recording = false;
};
static_assert(sizeof(IndirectArgsDiagnostics::Snapshot) <= 4096);
} // namespace Libs::Graphics

#endif

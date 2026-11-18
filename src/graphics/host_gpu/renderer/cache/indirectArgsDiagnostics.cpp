#include "graphics/host_gpu/renderer/cache/indirectArgsDiagnostics.h"

#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/causalDiagnosticsRuntime.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <cstdlib>
#include <cstring>
#include <limits>

namespace Libs::Graphics {
namespace {
struct Filter {
    bool enabled = false;
    uint64_t shader = 0;
};
const Filter& RequestedFilter() noexcept {
    static const Filter filter = [] {
        Filter value;
        if (!FaultPeekEnabled()) return value;
        const char* text = std::getenv("KYTY_INDIRECT_ARGS_DIAG");
        if (text == nullptr) return value;
        for (size_t i = 0; i < 16; ++i) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            uint64_t digit = 0;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else return Filter {};
            value.shader = (value.shader << 4u) | digit;
        }
        if (text[16] != '\0') return Filter {};
        value.enabled = true;
        return value;
    }();
    return filter;
}
bool SameParent(const CausalDiagnostics::ParentRef& a,
                const CausalDiagnostics::ParentRef& b) noexcept {
    return a.sequence == b.sequence && a.domain == b.domain &&
           a.id == b.id && a.family == b.family;
}
bool SameDispatch(const IndirectArgsDiagnostics::Dispatch& a,
                  const IndirectArgsDiagnostics::Dispatch& b) noexcept {
    return a.valid == b.valid && a.ordinal == b.ordinal && a.sequence == b.sequence &&
           a.domain == b.domain && a.lifetime == b.lifetime && a.tick == b.tick &&
           a.shader == b.shader && a.guest_address == b.guest_address &&
           a.native_buffer == b.native_buffer && a.native_offset == b.native_offset &&
           a.recording_thread == b.recording_thread && SameParent(a.cause, b.cause) &&
           SameParent(a.guest, b.guest) && a.point == b.point && a.request == b.request;
}
void InvalidIdentity(IndirectArgsDiagnostics::Snapshot& snapshot) noexcept {
    if (snapshot.identity_errors != std::numeric_limits<uint64_t>::max())
        ++snapshot.identity_errors;
    snapshot.status = IndirectArgsDiagnostics::Status::InvalidIdentity;
}
} // namespace

bool IndirectArgsDiagnosticEnabled() noexcept { return RequestedFilter().enabled; }
uint64_t IndirectArgsDiagnosticShader() noexcept { return RequestedFilter().shader; }

IndirectArgsDiagnostics::IndirectArgsDiagnostics(GraphicContext& graphics,
                                                 CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_selected_shader(IndirectArgsDiagnosticShader()),
      m_download(std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
          vk::BufferUsageFlagBits::eTransferDst, Capacity * ArgumentBytes)) {
    m_state.enabled = true;
    m_state.available = true;
    m_state.selected_shader = m_selected_shader;
    m_state.status = Status::NoCopies;
    // BufferCache creates this optional object before any binding/dispatch. Its
    // native allocation lives with that cache; ordinary renderer teardown drains
    // native work before destroying the cache's native buffers.
}
IndirectArgsDiagnostics::~IndirectArgsDiagnostics() = default;

void IndirectArgsDiagnostics::CopyBeforeDispatch(vk::CommandBuffer command, const Buffer& source,
                                                 uint64_t offset, uint64_t guest_address,
                                                 uint64_t shader) noexcept {
    if (!GuestGpu::IsGpuThreadFor(m_scheduler.Context())) {
        std::scoped_lock lock(m_mutex);
        InvalidIdentity(m_state);
        return;
    }
    if (m_recording) {
        std::scoped_lock lock(m_mutex);
        InvalidIdentity(m_state);
        return;
    }
    m_recording = true;
    m_pending = {};
    m_pending_slot = Capacity;
    auto& master = m_scheduler.GetMasterSemaphore();
    const auto context = CausalDiagnostics::CurrentContext();
    m_pending.domain = CausalDiagnostics::Domain(&master);
    m_pending.lifetime = m_scheduler.CausalLifetimeSequence();
    m_pending.tick = m_scheduler.CurrentTick();
    m_pending.shader = shader;
    m_pending.guest_address = guest_address;
    const VkBuffer handle = static_cast<VkBuffer>(source.Handle());
    static_assert(sizeof(handle) <= sizeof(m_pending.native_buffer));
    std::memcpy(&m_pending.native_buffer, &handle, sizeof(handle));
    m_pending.native_offset = offset;
    m_pending.recording_thread = static_cast<uint32_t>(Common::Thread::GetThreadIdUnique());
    m_pending.cause = context.cause;
    m_pending.guest = context.guest;
    m_pending.point = context.point;
    m_pending.request = context.request;
    const bool valid = command != nullptr && source.Handle() != nullptr &&
        source.Size() >= ArgumentBytes && offset <= source.Size() - ArgumentBytes &&
        (offset & 3u) == 0 && guest_address != 0 && (guest_address & 3u) == 0 &&
        guest_address <= std::numeric_limits<uint64_t>::max() - ArgumentBytes &&
        shader == m_selected_shader && m_pending.domain != 0 && m_pending.lifetime != 0 &&
        m_pending.tick != 0 && m_pending.recording_thread != 0;
    {
        std::scoped_lock lock(m_mutex);
        const auto recorded = m_test_control != nullptr
            ? m_test_control->recorded_total : m_state.copied_total;
        const bool scratch_valid = m_test_control == nullptr ||
            m_test_control->kind == TestControlKind::ComposedBarrierOnly ||
            (m_test_control->kind == TestControlKind::ImmutableScratchCopy &&
             m_test_control->source && m_test_control->destination &&
             m_test_control->source->Size() == TestControlState::SourceBytes &&
             m_test_control->destination->Size() == TestControlState::DestinationBytes &&
             m_test_control->source->Handle() != nullptr &&
             m_test_control->destination->Handle() != nullptr &&
             m_test_control->source->Handle() != source.Handle() &&
             m_test_control->destination->Handle() != source.Handle() &&
             m_test_control->source->Handle() != m_test_control->destination->Handle());
        if (!valid || !scratch_valid || m_state.identity_errors != 0 || recorded > Capacity) {
            InvalidIdentity(m_state);
            return; // Actual dispatch still proceeds; no diagnostic copy is recorded.
        }
        m_pending.valid = true;
        if (recorded < Capacity)
            m_pending_slot = static_cast<size_t>(recorded);
    }
    if (m_pending_slot == Capacity) return; // Never overwrite a prior capture.

    // Real-copy path: prior GPU argument writes become visible to the exact
    // 12-byte read. Controls keep the explicitly selected argument scopes;
    // final arguments/bindings have already been acquired and validated.
    vk::BufferMemoryBarrier2 source_ready {};
    source_ready.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
    source_ready.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
    source_ready.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
    source_ready.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
    source_ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_ready.buffer = source.Handle();
    source_ready.offset = offset;
    source_ready.size = ArgumentBytes;
    vk::DependencyInfo ready {};
    ready.bufferMemoryBarrierCount = 1;
    ready.pBufferMemoryBarriers = &source_ready;
    if (m_test_control != nullptr &&
        m_test_control->kind == TestControlKind::ComposedBarrierOnly) {
        // Explicit broad-ordering control. It is not described as an exact
        // equivalent of deleting a transfer between two barriers.
        source_ready.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        source_ready.dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
                                     vk::AccessFlagBits2::eMemoryWrite;
        command.pipelineBarrier2(ready);
        return;
    }
    // Scratch mode retains the actual argument-buffer barrier pair. Its real
    // transfer reads an independent immutable source; it does not capture args.
    const Buffer& copy_source = m_test_control != nullptr ? *m_test_control->source : source;
    Buffer& copy_destination = m_test_control != nullptr ? *m_test_control->destination : *m_download;
    const uint64_t copy_source_offset = m_test_control != nullptr
        ? TestControlState::SourceOffset : offset;
    const uint64_t copy_destination_offset = m_test_control != nullptr
        ? m_pending_slot * TestControlState::SlotBytes + sizeof(uint32_t)
        : m_pending_slot * ArgumentBytes;
    command.pipelineBarrier2(ready);
    command.copyBuffer(copy_source.Handle(), copy_destination.Handle(),
        vk::BufferCopy{copy_source_offset, copy_destination_offset, ArgumentBytes});

    std::array<vk::BufferMemoryBarrier2, 2> after {};
    // Real-copy path: order its argument read before later consumption/writes.
    // Scratch control retains this same argument range and execution scopes;
    // its transfer payload comes from the independent immutable source.
    after[0] = source_ready;
    after[0].srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    after[0].srcAccessMask = vk::AccessFlagBits2::eTransferRead;
    after[0].dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
    after[0].dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
                             vk::AccessFlagBits2::eMemoryWrite;
    after[1].srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    after[1].srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    after[1].dstStageMask = vk::PipelineStageFlagBits2::eHost;
    after[1].dstAccessMask = vk::AccessFlagBits2::eHostRead;
    after[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after[1].buffer = copy_destination.Handle();
    after[1].offset = copy_destination_offset;
    after[1].size = ArgumentBytes;
    vk::DependencyInfo copied {};
    copied.bufferMemoryBarrierCount = static_cast<uint32_t>(after.size());
    copied.pBufferMemoryBarriers = after.data();
    command.pipelineBarrier2(copied);
    // No Vulkan or allocation operation above runs while m_mutex is held.
}

void IndirectArgsDiagnostics::NoteRecordedDispatch(const CausalDiagnostics::Record& event,
                                                   const CausalDiagnostics::RecordResult& result) noexcept {
    if (!GuestGpu::IsGpuThreadFor(m_scheduler.Context())) {
        std::scoped_lock lock(m_mutex);
        InvalidIdentity(m_state);
        return;
    }
    const CausalDiagnostics::ParentRef event_parent {
        event.parent_sequence, event.parent_domain, event.parent_id, event.parent_family};
    const bool exact = m_recording && m_pending.valid &&
        event.event == CausalDiagnostics::Event::TargetDispatchIndirectRecorded &&
        event.domain == m_pending.domain && event.tick == m_pending.tick &&
        event.thread == m_pending.recording_thread &&
        event.data[0] == m_pending.shader && event.data[1] == m_pending.native_offset &&
        event.data[2] == m_pending.lifetime && event.data[3] == m_pending.guest.sequence &&
        event.data[4] == m_pending.guest.domain && event.data[5] == m_pending.guest.id &&
        SameParent(event_parent, m_pending.cause) &&
        result.outcome != CausalDiagnostics::Outcome::Disabled &&
        result.parent.sequence != 0 && result.parent.domain == event.domain &&
        result.parent.id == event.id && result.parent.family == CausalDiagnostics::Family::Native;
    {
        std::scoped_lock lock(m_mutex);
        if (!exact || m_state.matched_total == std::numeric_limits<uint64_t>::max())
            InvalidIdentity(m_state);
        if (m_state.matched_total != std::numeric_limits<uint64_t>::max())
            ++m_state.matched_total;
        m_pending.ordinal = m_state.matched_total;
        m_pending.sequence = result.parent.sequence; // Allocated even if history storage was dropped.
        m_pending.valid = exact;
        m_state.latest_match = m_pending;
        if (m_pending_slot < Capacity) {
            // Retain each argument copy or explicit test control, including
            // invalid identities. An error forbids every later payload read.
            auto& recorded_total = m_test_control != nullptr
                ? m_test_control->recorded_total : m_state.copied_total;
            auto& recorded_dispatch = m_test_control != nullptr
                ? m_test_control->dispatches[m_pending_slot]
                : m_state.slots[m_pending_slot].dispatch;
            if (m_pending_slot != recorded_total || recorded_dispatch.valid) {
                InvalidIdentity(m_state);
            } else {
                recorded_dispatch = m_pending;
                ++recorded_total;
            }
        } else if (exact) {
            ++m_state.capacity_skipped;
        }
        if (m_state.identity_errors == 0) m_state.status = Status::NotObserved;
    }
    m_pending = {};
    m_pending_slot = Capacity;
    m_recording = false;
}

bool IndirectArgsDiagnostics::PeekCompleted() noexcept {
    if (!GuestGpu::IsGpuThreadFor(m_scheduler.Context()) ||
        CommandScheduler::InDeferredOperation() || m_recording) return false;
    // Control metadata/raw scratch bytes never become typed argument payloads.
    if (m_test_control != nullptr) return false;
    auto& master = m_scheduler.GetMasterSemaphore();
    const auto domain = CausalDiagnostics::Domain(&master);
    const auto lifetime = m_scheduler.CausalLifetimeSequence();
    const auto known = master.KnownGpuTick();
    const auto waited = master.ReadFaultPeekHostWait();
    std::array<Dispatch, Capacity> captured {};
    std::array<bool, Capacity> unobserved {};
    size_t copied = 0;
    bool any_unobserved = false;
    {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        m_state.cached_known_tick = known;
        m_state.host_wait_entered_tick = waited.entered_tick;
        m_state.host_waited_tick = waited.completed_tick;
        if (m_state.identity_errors != 0 || domain == 0 || lifetime == 0 ||
            m_state.copied_total > Capacity ||
            m_download->Size() != Capacity * ArgumentBytes ||
            m_download->Mapped().size() < m_download->Size()) {
            m_state.status = Status::InvalidIdentity;
            return false;
        }
        copied = static_cast<size_t>(m_state.copied_total);
        if (copied == 0) {
            m_state.status = Status::NoCopies;
            return false;
        }
        for (size_t i = 0; i < copied; ++i) {
            const auto& slot = m_state.slots[i];
            const auto& dispatch = slot.dispatch;
            if (!dispatch.valid || dispatch.ordinal != i + 1 || dispatch.sequence == 0 ||
                dispatch.domain != domain || dispatch.lifetime != lifetime ||
                dispatch.tick == 0 || dispatch.shader != m_selected_shader ||
                dispatch.native_buffer == 0 || dispatch.guest_address == 0 ||
                dispatch.recording_thread == 0) {
                m_state.status = Status::InvalidIdentity;
                return false;
            }
            // ALL written slots, including already-observed ones, must be host
            // synchronized before invalidation can touch a shared noncoherent atom.
            if (dispatch.tick > known) {
                m_state.status = Status::NativeIncomplete;
                return false;
            }
            if (dispatch.tick > waited.completed_tick) {
                m_state.status = Status::HostWaitIncomplete;
                return false;
            }
            captured[i] = dispatch;
            unobserved[i] = !slot.valid;
            any_unobserved |= unobserved[i];
        }
        if (!any_unobserved) {
            m_state.status = Status::AlreadyObserved;
            return false;
        }
    }

    // The exact renderer owner cannot begin another capture while here. No slot
    // is ever reused. VMA separates noncoherent allocations by atom boundaries;
    // whole-allocation invalidation follows HOST_READ availability and an acquired
    // witness from a successful actual native wait, never just query/entered state.
    m_download->Invalidate(0, m_download->Size());
    const auto mapped = m_download->Mapped();
    std::array<std::array<uint32_t, 3>, Capacity> payloads {};
    for (size_t i = 0; i < copied; ++i) {
        if (unobserved[i])
            std::memcpy(payloads[i].data(), mapped.data() + i * ArgumentBytes, ArgumentBytes);
    }

    std::scoped_lock lock(m_mutex);
    if (m_state.identity_errors != 0 || m_state.copied_total != copied) {
        m_state.status = Status::InvalidIdentity;
        return false;
    }
    for (size_t i = 0; i < copied; ++i) {
        if (!SameDispatch(m_state.slots[i].dispatch, captured[i])) {
            InvalidIdentity(m_state);
            return false;
        }
    }
    for (size_t i = 0; i < copied; ++i) {
        if (!unobserved[i] || m_state.slots[i].valid) continue;
        auto& slot = m_state.slots[i];
        slot.groups = payloads[i];
        slot.observed_known_tick = known;
        slot.observed_host_waited_tick = waited.completed_tick;
        slot.valid = true;
        ++m_state.observed_total;
    }
    m_state.status = Status::Observed;
    return true;
}

IndirectArgsDiagnostics::Snapshot IndirectArgsDiagnostics::Read() const noexcept {
    std::unique_lock lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        Snapshot snapshot;
        snapshot.enabled = true;
        snapshot.selected_shader = m_selected_shader;
        snapshot.status = Status::LockUnavailable;
        return snapshot;
    }
    return m_state;
}
} // namespace Libs::Graphics

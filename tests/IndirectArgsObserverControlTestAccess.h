#pragma once

// Private native-fixture access. No environment variable or runtime caller
// installs these controls. Default observer behavior remains real-argument copy.
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/indirectArgsDiagnostics.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/causalDiagnosticsRuntime.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <array>
#include <cstring>
#include <memory>

namespace Libs::Graphics {
struct IndirectArgsObserverControlTestAccess {
    enum class Mode : uint8_t {
        RealArgumentCopy,
        ComposedBarrierOnly,
        ImmutableScratchCopy,
    };
    using Observer = IndirectArgsDiagnostics;
    struct Report {
        bool available = false;
        bool raw_scratch_read = false; // Never an Observation.groups payload.
        uint64_t recorded_total = 0, cached_known_tick = 0, actual_host_waited_tick = 0;
        uint64_t scratch_source_handle = 0, scratch_destination_handle = 0;
        std::array<Observer::Dispatch, Observer::Capacity> dispatches {};
        std::array<uint32_t, 5> source_words {};
        std::array<std::array<uint32_t, 5>, Observer::Capacity> slots {};
    };

    static uint64_t NativeHandle(const Buffer& buffer) noexcept {
        const VkBuffer native = static_cast<VkBuffer>(buffer.Handle());
        uint64_t value = 0;
        static_assert(sizeof(native) <= sizeof(value));
        std::memcpy(&value, &native, sizeof(native));
        return value;
    }

    // Called on the actual GPU owner, before any producer/consumer is recorded.
    // Ordinary memcpy + Flush happens before the existing queue submission.
    // No command, submit, query, native wait, finish or callback drain is added.
    static bool Install(Observer& observer, Mode mode,
                        const std::array<uint32_t, 3>& payload, uint32_t canary) {
        if (!GuestGpu::IsGpuThreadFor(observer.m_scheduler.Context()) ||
            CommandScheduler::InDeferredOperation() || observer.m_recording ||
            observer.m_test_control != nullptr)
            return false;
        const auto snapshot = observer.Read();
        if (!snapshot.available || snapshot.matched_total != 0 ||
            snapshot.copied_total != 0 || snapshot.observed_total != 0 ||
            snapshot.identity_errors != 0)
            return false;
        if (mode == Mode::RealArgumentCopy) return true;
        if (mode != Mode::ComposedBarrierOnly && mode != Mode::ImmutableScratchCopy)
            return false;
        auto control = std::make_unique<Observer::TestControlState>();
        control->kind = mode == Mode::ComposedBarrierOnly
            ? Observer::TestControlKind::ComposedBarrierOnly
            : Observer::TestControlKind::ImmutableScratchCopy;
        if (mode == Mode::ImmutableScratchCopy) {
            control->source = std::make_unique<Buffer>(
                observer.m_graphics, observer.m_scheduler, MemoryUsage::Upload, 0,
                vk::BufferUsageFlagBits::eTransferSrc, Observer::TestControlState::SourceBytes);
            control->destination = std::make_unique<Buffer>(
                observer.m_graphics, observer.m_scheduler, MemoryUsage::Download, 0,
                vk::BufferUsageFlagBits::eTransferDst, Observer::TestControlState::DestinationBytes);
            if (control->source->Mapped().size() < control->source->Size() ||
                control->destination->Mapped().size() < control->destination->Size() ||
                control->source->Handle() == control->destination->Handle() ||
                control->source->Handle() == observer.m_download->Handle() ||
                control->destination->Handle() == observer.m_download->Handle())
                return false;
            const std::array<uint32_t, 5> initialized {
                canary, payload[0], payload[1], payload[2], canary};
            std::memcpy(control->source->Mapped().data(), initialized.data(), sizeof(initialized));
            control->source->Flush(0, control->source->Size());
            // All eight payload ranges must change from canaries to these
            // bytes. Guards detect range corruption; equal payloads do not
            // independently detect an identical repeat into a completed slot.
            for (size_t slot = 0; slot < Observer::Capacity; ++slot) {
                std::array<uint32_t, 5> initial {};
                initial.fill(canary);
                std::memcpy(control->destination->Mapped().data() +
                                slot * Observer::TestControlState::SlotBytes,
                            initial.data(), sizeof(initial));
            }
            control->destination->Flush(0, control->destination->Size());
        }
        observer.m_test_control = std::move(control);
        return true;
    }

    // Raw fixture control bytes have their own host-wait guard. They never flow
    // through the runtime typed argument publisher or into guest CPU backing.
    static Report Read(Observer& observer) noexcept {
        Report report;
        if (!GuestGpu::IsGpuThreadFor(observer.m_scheduler.Context()) ||
            CommandScheduler::InDeferredOperation() || observer.m_recording ||
            observer.m_test_control == nullptr)
            return report;
        const auto snapshot = observer.Read();
        if (!snapshot.available || snapshot.identity_errors != 0)
            return report;
        auto& control = *observer.m_test_control;
        if (control.recorded_total > Observer::Capacity)
            return report;
        report.recorded_total = control.recorded_total;
        report.dispatches = control.dispatches;
        report.available = true;
        if (control.kind == Observer::TestControlKind::ComposedBarrierOnly)
            return report; // No copy or mapped read exists in this mode.
        auto& master = observer.m_scheduler.GetMasterSemaphore();
        report.cached_known_tick = master.KnownGpuTick();
        report.actual_host_waited_tick = master.ReadFaultPeekHostWait().completed_tick;
        const auto domain = CausalDiagnostics::Domain(&master);
        const auto lifetime = observer.m_scheduler.CausalLifetimeSequence();
        if (control.recorded_total == 0 || !control.source || !control.destination ||
            control.source->Size() != Observer::TestControlState::SourceBytes ||
            control.destination->Size() != Observer::TestControlState::DestinationBytes ||
            control.source->Mapped().size() < control.source->Size() ||
            control.destination->Mapped().size() < control.destination->Size())
            return report;
        for (size_t i = 0; i < control.recorded_total; ++i) {
            const auto& dispatch = control.dispatches[i];
            if (!dispatch.valid || dispatch.ordinal != i + 1 || dispatch.sequence == 0 ||
                dispatch.domain != domain || dispatch.lifetime != lifetime ||
                dispatch.shader != observer.m_selected_shader || dispatch.tick == 0 ||
                dispatch.recording_thread == 0 || dispatch.native_buffer == 0 ||
                dispatch.guest_address == 0 ||
                dispatch.tick > report.cached_known_tick ||
                dispatch.tick > report.actual_host_waited_tick)
                return report;
        }
        // All native writes in this independent allocation are completed and
        // host-synchronized before whole-allocation invalidation.
        control.destination->Invalidate(0, control.destination->Size());
        report.scratch_source_handle = NativeHandle(*control.source);
        report.scratch_destination_handle = NativeHandle(*control.destination);
        std::memcpy(report.source_words.data(), control.source->Mapped().data(),
                    sizeof(report.source_words));
        for (size_t i = 0; i < Observer::Capacity; ++i)
            std::memcpy(report.slots[i].data(),
                        control.destination->Mapped().data() +
                            i * Observer::TestControlState::SlotBytes,
                        sizeof(report.slots[i]));
        report.raw_scratch_read = true;
        return report;
    }
};
} // namespace Libs::Graphics

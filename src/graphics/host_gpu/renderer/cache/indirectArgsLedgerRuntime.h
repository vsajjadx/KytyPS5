#ifndef KYTY_INDIRECT_ARGS_LEDGER_RUNTIME_H
#define KYTY_INDIRECT_ARGS_LEDGER_RUNTIME_H

#include "graphics/host_gpu/renderer/cache/indirectArgsLedger.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/causalDiagnosticsRuntime.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

namespace Libs::Graphics::IndirectArgsLedger {

// Only cached owner/context/tick metadata is read. This does not start a command,
// query a semaphore, wait, or establish a dependency. Non-owner recording is retained
// with unknown scheduler identity, rather than inventing a CPU/native causal link.
inline Position RecordingPosition(CommandScheduler& scheduler, vk::CommandBuffer command) noexcept {
    auto* recorder = Active();
    if (recorder == nullptr) return {};
    Position out {};
    out.command_buffer = HandleBits(static_cast<VkCommandBuffer>(command));
    out.tracked_command_ordinal = recorder->NextTrackedCommand();
    if (command == nullptr || !GuestGpu::IsGpuThreadFor(scheduler.Context())) return out;
    if (scheduler.Current().Handle() != command) return out;
    out.native_domain = CausalDiagnostics::Domain(&scheduler.GetMasterSemaphore());
    out.causal_lifetime = scheduler.CausalLifetimeSequence();
    out.tick = scheduler.CurrentTick();
    out.owner_witness = true;
    return out;
}
} // namespace Libs::Graphics::IndirectArgsLedger
#endif // KYTY_INDIRECT_ARGS_LEDGER_RUNTIME_H

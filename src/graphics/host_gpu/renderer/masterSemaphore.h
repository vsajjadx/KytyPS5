#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>

namespace Libs::Graphics {

struct GraphicContext;

// Small in-memory history of the most recent GPU submits. It is dumped when the Vulkan device is
// lost so the log shows what the GPU was working on right before the failure.
void RecordSubmitHistory(uint64_t tick, uint32_t debug_op, uint64_t submit_id, uint32_t arg0,
                         uint32_t arg1, uint32_t arg2, uint32_t arg3, uint64_t arg4);
void DumpSubmitHistory();
// True when the KYTY_GPU_SYNC environment variable is set: every submit is waited on so a GPU
// hang is attributed to the exact submit that caused it.
bool GpuSyncDebugEnabled();
// Exact child opt-in for passive staged-fault observation; never a scheduling input.
[[nodiscard]] bool FaultPeekEnabled() noexcept;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	// Entered is only a branch-selection witness. Only completed authorizes a host
	// read, and only after an existing successful vkWaitSemaphores(requested tick).
	struct HostWaitWitness {
		uint64_t entered_tick = 0;
		uint64_t completed_tick = 0;
	};
	[[nodiscard]] HostWaitWitness ReadFaultPeekHostWait() const noexcept;

	void Refresh();
	void Wait(uint64_t tick);

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	// Fixed footprint only when disabled; no accesses on the default-off path.
	std::atomic<uint64_t> m_fault_peek_wait_entered {0};
	std::atomic<uint64_t> m_fault_peek_wait_completed {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

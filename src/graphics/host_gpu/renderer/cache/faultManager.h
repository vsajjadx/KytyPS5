#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>

namespace Libs::Graphics {

class BufferCache;
namespace CausalDiagnostics { struct OperationWitness; }

class FaultManager {
	static constexpr size_t MaxPendingFaults = 8;

public:
	FaultManager(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache);
	~FaultManager();
	KYTY_CLASS_NO_COPY(FaultManager);

	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return &m_fault_buffer; }
	void                  ProcessFaultBuffer();

	// Table-call misses recorded by shaders in the fault buffer trailer (null, out-of-range or
	// unknown loaded pointer). Reported as failures; never treated as successful rendering.
	struct CallTableMiss {
		uint64_t shader_hash = 0;
		uint64_t target      = 0;
		uint32_t pc          = 0;
	};
	// Opt-in observation of the real fault-harvest callbacks, never a scheduling input.
	// Queue identities come from the actual accepted operation. Enqueue sequence/thread
	// witness host recording order after copy/clear; they are not GPU timestamps.
	enum class HarvestPhase : uint8_t { Empty, Queued, Running, Completed };
	struct HarvestRecord {
		bool valid = false;
		uint32_t area = 0;
		HarvestPhase phase = HarvestPhase::Empty;
		uint64_t domain = 0, lifetime = 0, operation_id = 0, tick = 0;
		uint64_t enqueue_sequence = 0;
		uint32_t enqueue_thread = 0; // Internal unique ID, not a native OS thread ID.
		uint64_t miss_invocations = 0, raw_page_faults = 0;
		uint32_t page_faults = 0; // Clamped to the number of records in the download area.
		CallTableMiss call_miss {};
	};
	struct HarvestSnapshot {
		bool enabled = false, available = false;
		uint64_t queued_total = 0, completed_total = 0, identity_errors = 0;
		uint64_t cumulative_call_table_misses = 0;
		std::array<HarvestRecord, MaxPendingFaults> areas {};
		HarvestRecord last_completed {};
	};
	[[nodiscard]] HarvestSnapshot ReadHarvestDiagnostics() const noexcept;

	// A passive read of completed staged bytes is separate from actual callback
	// processing. In particular, staged.phase stays Queued and no host fault count,
	// owner area, normal callback or last-completed harvest is changed by a peek.
	enum class PeekStatus : uint8_t {
		Disabled = 0, NotObserved = 1, NoPending = 2, LockUnavailable = 3,
		InvalidIdentity = 4, NativeIncomplete = 5, HostWaitIncomplete = 6,
		Observed = 7, AlreadyObserved = 8
	};
	struct PeekRecord {
		bool valid = false;
		HarvestRecord staged {};
		uint64_t observed_known_tick = 0, observed_host_waited_tick = 0;
	};
	struct PeekSnapshot {
		bool enabled = false, available = false;
		PeekStatus status = PeekStatus::Disabled;
		uint64_t observed_total = 0;
		uint64_t cached_known_tick = 0, host_wait_entered_tick = 0, host_waited_tick = 0;
		std::array<PeekRecord, MaxPendingFaults> areas {};
	};
	[[nodiscard]] PeekSnapshot ReadFaultPeekDiagnostics() const noexcept;
	// Must run on this renderer's GPU owner, outside command recording/callbacks.
	[[nodiscard]] bool PeekCompletedFaultDownloads() noexcept;

	[[nodiscard]] uint64_t CallTableMisses() const noexcept { return m_call_table_misses.load(); }
	[[nodiscard]] CallTableMiss LastCallTableMiss() const {
		std::scoped_lock lock(m_call_table_mutex);
		return m_last_call_table_miss;
	}

private:
	void QueueHarvestDiagnostics(uint32_t area,
	                             const CausalDiagnostics::OperationWitness& operation) noexcept;
	void BeginHarvestDiagnostics(uint32_t area) noexcept;
	void CompleteHarvestDiagnostics(uint32_t area, const std::array<uint32_t, 8>& miss,
	                                uint64_t raw_page_faults, uint32_t page_faults) noexcept;

	GraphicContext&                            m_graphics;
	CommandScheduler&                          m_scheduler;
	BufferCache&                               m_buffer_cache;
	Buffer                                     m_fault_buffer;
	Buffer                                     m_download_buffer;
	std::array<uint64_t, MaxPendingFaults>      m_fault_areas {};
	uint32_t                                   m_current_area = 0;
	std::atomic<uint64_t>                      m_call_table_misses {0};
	mutable std::mutex                         m_call_table_mutex;
	CallTableMiss                              m_last_call_table_miss {};
	// Disengaged when disabled: fixed storage, no diagnostic heap allocation.
	// All engaged-state accesses use m_call_table_mutex; never read engine areas from the reporter.
	std::optional<HarvestSnapshot>              m_harvest_diagnostics;
	std::optional<PeekSnapshot>                 m_peek_diagnostics;
	// Only this renderer's GPU owner reads/writes this flag. It rejects a peek
	// during an in-progress ProcessFaultBuffer, including nested processing.
	bool                                       m_fault_peek_recording = false;
	vk::DescriptorSetLayout                    m_fault_process_desc_layout = nullptr;
	vk::Pipeline                               m_fault_process_pipeline = nullptr;
	vk::PipelineLayout                         m_fault_process_pipeline_layout = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

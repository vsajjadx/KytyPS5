#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>

namespace Libs::Graphics {

class BufferCache;

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
	[[nodiscard]] uint64_t CallTableMisses() const noexcept { return m_call_table_misses.load(); }
	[[nodiscard]] CallTableMiss LastCallTableMiss() const {
		std::scoped_lock lock(m_call_table_mutex);
		return m_last_call_table_miss;
	}

private:
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
	vk::DescriptorSetLayout                    m_fault_process_desc_layout = nullptr;
	vk::Pipeline                               m_fault_process_pipeline = nullptr;
	vk::PipelineLayout                         m_fault_process_pipeline_layout = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

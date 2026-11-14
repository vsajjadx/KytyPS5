#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <memory>
#include <atomic>
#include <shared_mutex>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext&              graphics,
	                       const std::filesystem::path& shader_store = ShaderStorePath());
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	ImageHeap&          GetImageHeap() { return m_image_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }
	OcclusionQueries&   GetOcclusionQueries() { return m_occlusion_queries; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	// GPU thread only: reads `size` bytes of a protected page straight from guest backing when
	// no pending GPU write or image write-back covers them, so no download or sync is needed.
	[[nodiscard]] bool TryReadCpuCurrent(uint64_t vaddr, void* data, uint64_t size) noexcept;
	[[nodiscard]] bool TryReadActivityCurrent(uint64_t vaddr, void* data, uint64_t size) noexcept;
	[[nodiscard]] bool ReadActivityEpoch(ShaderRecompiler::IR::ActivityEpoch& epoch) const noexcept;
	// Optional observational output from the ORIGINAL first failing guard. No resource reads.
	struct ActivityTargetFailure {
		enum class Reason : uint32_t { None, WrongThread, EmptyTargets, ZeroSize, GpuUnmapped, PhysicalNotUnique };
		uint64_t index = UINT64_MAX, address = 0, size = 0;
		Reason reason = Reason::None;
	};
	[[nodiscard]] bool ValidateActivityTargets(
	    std::span<const std::pair<uint64_t, uint64_t>> targets,
	    ActivityTargetFailure* failure = nullptr) const noexcept;
	// GPU thread only. A dword the GPU keeps rewriting with the same value (buffer headers,
	// sizes) is served from its last download for a few reads before it is downloaded again.
	[[nodiscard]] bool TryPredictGpuRead(uint64_t vaddr, uint32_t& value) noexcept;
	void               NoteGpuRead(uint64_t vaddr) noexcept;
	void               NoteReadCheck(uint64_t vaddr, uint32_t expected, uint32_t actual) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               PrepareBda();
	// Schedules the next fault-buffer pass (BDA faults and table-call miss records).
	void               RequestFaultProcess() noexcept { m_fault_process_pending = true; }
	void               RunGarbageCollector();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	// Waits for the image write-backs over the range's pages; returns whether any overlapped.
	bool WaitWriteBacks(uint64_t vaddr, uint64_t size);

	GraphicContext&           m_graphics;
	const uint64_t            m_activity_context_id;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	ImageHeap                 m_image_heap;
	OcclusionQueries          m_occlusion_queries;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

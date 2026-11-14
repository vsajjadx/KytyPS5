#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <unordered_map>

namespace Libs::Graphics {

namespace {
std::atomic<uint64_t> g_activity_contexts {1};
}

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_graphics(graphics), m_activity_context_id(g_activity_contexts.fetch_add(1u)),
      m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics, &m_image_heap, shader_store, this), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager), m_occlusion_queries(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::TryReadActivityCurrent(uint64_t vaddr, void* data, uint64_t size) noexcept {
	// Never honor KYTY_STALE_WALKER or prediction in a binding-validity certificate.
	if (!GuestGpu::IsGpuThread() || !IsMapped(vaddr, size) ||
	    !LibKernel::Memory::IsBackingUniquelyMapped(vaddr, size) ||
	    m_buffer_cache.IsRangeGpuWritten(vaddr, size) ||
	    m_texture_cache.IsRegionRegistered(vaddr, size) ||
	    m_texture_cache.WriteBackTick(vaddr, size) != 0u) return false;
	return LibKernel::Memory::TryReadBacking(vaddr, data, size);
}

bool RenderContext::ReadActivityEpoch(ShaderRecompiler::IR::ActivityEpoch& epoch) const noexcept {
	if (!GuestGpu::IsGpuThread() || !ShaderRecompiler::IR::HostActivityProofEnabled()) return false;
	const auto protection = LibKernel::Memory::ProtectionGeneration();
	const auto configuration = Config::Revision();
	if (protection == 0u || configuration == 0u) return false;
	epoch = {m_activity_context_id, LibKernel::Memory::MappingGeneration(), protection,
	         m_buffer_cache.BindingEpoch(), Image::activity_epoch.load(), configuration};
	return true;
}

bool RenderContext::ValidateActivityTargets(
    std::span<const std::pair<uint64_t, uint64_t>> targets,
    ActivityTargetFailure* failure) const noexcept {
	using Reason = ActivityTargetFailure::Reason;
	const auto reject = [&](Reason reason, uint64_t index = UINT64_MAX, uint64_t address = 0,
	                        uint64_t size = 0) {
		if (failure != nullptr) *failure = {index, address, size, reason};
		return false;
	};
	// Preserve the original short-circuit order and exactly one evaluation of each guard.
	if (!GuestGpu::IsGpuThread()) return reject(Reason::WrongThread);
	if (targets.empty()) return reject(Reason::EmptyTargets);
	uint64_t index = 0;
	for (const auto [address, size]: targets) {
		if (size == 0u) return reject(Reason::ZeroSize, index, address, size);
		if (!IsMapped(address, size)) return reject(Reason::GpuUnmapped, index, address, size);
		if (!LibKernel::Memory::IsBackingUniquelyMapped(address, size))
			return reject(Reason::PhysicalNotUnique, index, address, size);
		++index;
	}
	return true;
}

bool RenderContext::TryReadCpuCurrent(uint64_t vaddr, void* data, uint64_t size) noexcept {
	if (!GuestGpu::IsGpuThread() || !IsMapped(vaddr, size)) {
		return false;
	}
	static const bool stale_experiment = std::getenv("KYTY_STALE_WALKER") != nullptr;
	if (!stale_experiment && m_buffer_cache.IsRangeGpuWritten(vaddr, size) ||
	    m_texture_cache.WriteBackTick(vaddr, size) != 0) {
		return false;
	}
	return LibKernel::Memory::TryReadBacking(vaddr, data, size);
}

namespace {
// SRT-walker reads of GPU-written bytes. A dword that read back the same value on several
// downloads is served from that value; an asynchronous copy of the GPU bytes checks it, and a
// changed value takes the address off speculation for good.
struct PredictedRead {
	uint32_t value    = 0;
	uint32_t stable   = 0; // consecutive downloads that returned `value`
	uint32_t changes  = 0;
	bool     unstable = false;
};
constexpr uint32_t PredictStableDownloads = 30;
bool               PredictEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_PREDICT_GPU_READS");
		return text != nullptr && text[0] == '1';
	}();
	return enabled;
}
std::mutex& PredictMutex() {
	static std::mutex mutex;
	return mutex;
}
std::unordered_map<uint64_t, PredictedRead>& PredictedReads() {
	static std::unordered_map<uint64_t, PredictedRead> reads;
	return reads;
}
std::atomic<uint64_t> g_predicted {0}, g_mispredicted {0};
} // namespace

bool RenderContext::TryPredictGpuRead(uint64_t vaddr, uint32_t& value) noexcept {
	if (!PredictEnabled() || !GuestGpu::IsGpuThread() || !IsMapped(vaddr, sizeof(value)) ||
	    m_texture_cache.WriteBackTick(vaddr, sizeof(value)) != 0) {
		return false;
	}
	{
		std::scoped_lock lock {PredictMutex()};
		const auto it = PredictedReads().find(vaddr);
		if (it == PredictedReads().end() || it->second.unstable ||
		    it->second.stable < PredictStableDownloads) {
			return false;
		}
		value = it->second.value;
	}
	m_buffer_cache.QueueReadCheck(vaddr, value);
	const auto n = ++g_predicted;
	if (std::getenv("KYTY_DL_DIAG") != nullptr && n % 20000 == 0) {
		std::printf("PREDICT: %llu walker reads served from stable values, %llu mispredicted\n",
		            static_cast<unsigned long long>(n),
		            static_cast<unsigned long long>(g_mispredicted.load()));
		std::fflush(stdout);
	}
	return true;
}

void RenderContext::NoteGpuRead(uint64_t vaddr) noexcept {
	if (!PredictEnabled() || !GuestGpu::IsGpuThread()) {
		return;
	}
	uint32_t backing = 0;
	if (!LibKernel::Memory::TryReadBacking(vaddr, &backing, sizeof(backing))) {
		return;
	}
	std::scoped_lock lock {PredictMutex()};
	auto&            reads = PredictedReads();
	if (reads.size() > 65536) {
		reads.clear();
	}
	auto& read = reads[vaddr];
	if (read.stable != 0 && read.value == backing) {
		++read.stable;
	} else {
		// Values that keep changing are GPU-produced data, never speculated.
		if (read.stable != 0 && ++read.changes >= 2) {
			read.unstable = true;
		}
		read.value  = backing;
		read.stable = 1;
	}
}

void RenderContext::NoteReadCheck(uint64_t vaddr, uint32_t expected, uint32_t actual) noexcept {
	if (expected == actual) {
		return;
	}
	++g_mispredicted;
	std::scoped_lock lock {PredictMutex()};
	PredictedReads()[vaddr].unstable = true;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	LibKernel::Memory::NoteActivityMappingChange();
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		LibKernel::Memory::NoteActivityMappingChange();
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics

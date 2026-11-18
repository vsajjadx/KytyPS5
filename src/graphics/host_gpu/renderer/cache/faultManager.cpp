#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/frameStats.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/causalDiagnosticsRuntime.h"
#include "graphics/host_gpu/renderer/causalOperationWitness.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include "common/assert.h"
#include "common/logging/log.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>

namespace Libs::Graphics {

namespace {

namespace Causal = CausalDiagnostics;

constexpr size_t MaxPageFaults    = 1024;
constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(uint64_t);
// The page scan dispatches whole 64-invocation groups of four bitmap words, so it may read up to
// 16 KiB past the bitmap; the zero pad absorbs that. The table-call miss record is the last 32
// bytes (shaders address it as OpArrayLength - 8 DWORDs).
constexpr size_t FaultBitmapSize   = BufferCache::CACHING_NUMPAGES / 8;
constexpr size_t FaultScanPad      = 16 * 1024;
constexpr size_t CallMissSize      = 8 * sizeof(uint32_t);
constexpr size_t CallMissOffset    = FaultBitmapSize + FaultScanPad;
static_assert(((BufferCache::CACHING_NUMPAGES / 128 + 63) / 64) * 64 * 16 <= CallMissOffset);

bool ValidHarvestOperation(const Causal::OperationWitness& operation,
                           const CommandScheduler& scheduler) noexcept {
	// The domain is the scheduler's native timeline, never a guest-origin tuple.
	return !operation.priority && operation.native_domain != 0 &&
	       operation.lifetime_sequence == scheduler.CausalLifetimeSequence() &&
	       operation.lifetime_sequence != 0 && operation.operation_id != 0 &&
	       operation.selected_tick != 0 && operation.queued_parent.sequence != 0 &&
	       operation.queued_parent.domain == operation.native_domain &&
	       operation.queued_parent.id == operation.operation_id &&
	       operation.queued_parent.family == Causal::Family::Priority &&
	       operation.enqueue_thread != 0;
}

bool SameHarvestOperation(const FaultManager::HarvestRecord& record,
                          const Causal::OperationWitness& operation) noexcept {
	return record.valid && record.domain == operation.native_domain &&
	       record.lifetime == operation.lifetime_sequence &&
	       record.operation_id == operation.operation_id && record.tick == operation.selected_tick &&
	       record.enqueue_sequence == operation.queued_parent.sequence &&
	       record.enqueue_thread == operation.enqueue_thread;
}

bool ValidHarvestExecution(const Causal::CallbackExecution& execution,
                           const CommandScheduler& scheduler) noexcept {
	return execution.valid && !execution.inline_call &&
	       ValidHarvestOperation(execution.operation, scheduler) &&
	       execution.callback_begin.sequence > execution.operation.queued_parent.sequence &&
	       execution.callback_begin.domain == execution.operation.native_domain &&
	       execution.callback_begin.id == execution.operation.operation_id &&
	       execution.callback_begin.family == Causal::Family::Priority;
}

class FaultPeekRecordingScope {
public:
	explicit FaultPeekRecordingScope(bool* recording) noexcept
	    : m_recording(recording), m_previous(recording != nullptr && *recording) {
		if (m_recording != nullptr) *m_recording = true;
	}
	~FaultPeekRecordingScope() {
		if (m_recording != nullptr) *m_recording = m_previous;
	}
	FaultPeekRecordingScope(const FaultPeekRecordingScope&) = delete;
	FaultPeekRecordingScope& operator=(const FaultPeekRecordingScope&) = delete;
private:
	bool* m_recording;
	bool m_previous;
};

bool SameRetainedHarvest(const FaultManager::HarvestRecord& first,
                       const FaultManager::HarvestRecord& second) noexcept {
	return first.valid && second.valid && first.area == second.area &&
	       first.phase == second.phase &&
	       first.domain == second.domain && first.lifetime == second.lifetime &&
	       first.operation_id == second.operation_id && first.tick == second.tick &&
	       first.enqueue_sequence == second.enqueue_sequence &&
	       first.enqueue_thread == second.enqueue_thread;
}

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     CallMissOffset + CallMissSize),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * (PageFaultAreaSize + CallMissSize)) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	if (CausalDiagnostics::Enabled()) {
		auto& diagnostics = m_harvest_diagnostics.emplace();
		diagnostics.enabled = true;
		diagnostics.available = true;
	}
	if (FaultPeekEnabled()) {
		auto& diagnostics = m_peek_diagnostics.emplace();
		diagnostics.enabled = true;
		diagnostics.available = true;
		diagnostics.status = PeekStatus::NotObserved;
	}

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	m_fault_process_pipeline =
	    CreateComputePipeline(m_graphics.device, FAULT_BUFFER_PROCESS_SPV,
		                      m_fault_process_pipeline_layout, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

FaultManager::HarvestSnapshot FaultManager::ReadHarvestDiagnostics() const noexcept {
	HarvestSnapshot out {};
	if (!m_harvest_diagnostics) return out;
	out.enabled = true;
	std::unique_lock lock(m_call_table_mutex, std::try_to_lock);
	if (!lock.owns_lock()) return out;
	out = *m_harvest_diagnostics;
	// This is the existing cumulative atomic observation, not an additional GPU read
	// or a claim that all fields describe one global instant.
	out.cumulative_call_table_misses = m_call_table_misses.load(std::memory_order_relaxed);
	return out;
}

FaultManager::PeekSnapshot FaultManager::ReadFaultPeekDiagnostics() const noexcept {
	PeekSnapshot out {};
	if (!m_peek_diagnostics) return out;
	out.enabled = true;
	std::unique_lock lock(m_call_table_mutex, std::try_to_lock);
	if (!lock.owns_lock()) {
		out.status = PeekStatus::LockUnavailable;
		return out;
	}
	return *m_peek_diagnostics;
}

bool FaultManager::PeekCompletedFaultDownloads() noexcept {
	if (!m_peek_diagnostics || !GuestGpu::IsGpuThreadFor(m_scheduler.Context()) ||
	    CommandScheduler::InDeferredOperation() || m_fault_peek_recording) return false;

	// Only the owner reads the live engine-area array. The reporter reads the
	// separately locked fixed snapshot. No queued command or normal callback can
	// record, free or reuse an area while this owner-only operation is executing.
	auto& master = m_scheduler.GetMasterSemaphore();
	const auto domain = CausalDiagnostics::Domain(&master);
	const auto lifetime = m_scheduler.CausalLifetimeSequence();
	const auto known = master.KnownGpuTick();
	const auto waited = master.ReadFaultPeekHostWait();
	std::array<HarvestRecord, MaxPendingFaults> captured {};
	std::array<HarvestRecord, MaxPendingFaults> retained {};
	std::array<uint64_t, MaxPendingFaults> area_ticks {};
	bool has_pending = false;
	bool has_unobserved = false;
	{
		std::unique_lock lock(m_call_table_mutex, std::try_to_lock);
		if (!lock.owns_lock()) return false;
		auto& diagnostics = *m_peek_diagnostics;
		diagnostics.cached_known_tick = known;
		diagnostics.host_wait_entered_tick = waited.entered_tick;
		diagnostics.host_waited_tick = waited.completed_tick;
		if (!m_harvest_diagnostics || m_harvest_diagnostics->identity_errors != 0 ||
		    lifetime == 0 || domain == 0 ||
		    m_download_buffer.Mapped().size() < m_download_buffer.Size()) {
			diagnostics.status = PeekStatus::InvalidIdentity;
			return false;
		}
		// A 32-byte miss trailer can share a noncoherent atom with another area.
		// Check ALL retained writes, including freed Completed slots, before the
		// complete-buffer invalidate. A callback completed by query-only progress
		// does not imply that a later actual host wait has covered that slot.
		for (uint32_t area = 0; area < MaxPendingFaults; ++area) {
			const auto tick = m_fault_areas[area];
			area_ticks[area] = tick;
			const auto& source = m_harvest_diagnostics->areas[area];
			if (tick == 0 && !source.valid && source.phase == HarvestPhase::Empty &&
			    source.tick == 0 && source.domain == 0 && source.lifetime == 0 &&
			    source.operation_id == 0 && source.enqueue_sequence == 0 &&
			    source.enqueue_thread == 0) continue; // Never written.
			const bool live = source.phase == HarvestPhase::Queued && tick != 0 &&
			                  source.tick == tick;
			const bool freed = source.phase == HarvestPhase::Completed && tick == 0;
			if (!source.valid || source.area != area || (!live && !freed) ||
			    source.domain != domain || source.lifetime != lifetime || source.tick == 0 ||
			    source.operation_id == 0 || source.enqueue_sequence == 0 ||
			    source.enqueue_thread == 0) {
				diagnostics.status = PeekStatus::InvalidIdentity;
				return false;
			}
			retained[area] = source;
			if (source.tick > known) {
				diagnostics.status = PeekStatus::NativeIncomplete;
				return false;
			}
			// Entered and query-only completion are deliberately insufficient.
			if (source.tick > waited.completed_tick) {
				diagnostics.status = PeekStatus::HostWaitIncomplete;
				return false;
			}
			if (freed) continue;
			has_pending = true;
			const auto& observed = diagnostics.areas[area];
			if (observed.valid && SameRetainedHarvest(observed.staged, source)) continue;
			captured[area] = source;
			has_unobserved = true;
		}
		if (!has_pending || !has_unobserved) {
			diagnostics.status = has_pending ? PeekStatus::AlreadyObserved : PeekStatus::NoPending;
			return false;
		}
	}

	// Each recorded parser/copy has its child-gated HOST_READ availability barrier.
	// The acquired completed-wait witness follows an actual successful native host
	// wait, not a counter query. Every retained writer is complete and the owner
	// cannot start a new one here. VMA aligns noncoherent allocations to separate atom
	// lines, so its outward rounding cannot reach another buffer allocation.
	m_download_buffer.Invalidate(0, m_download_buffer.Size());
	const auto mapped = m_download_buffer.Mapped();
	for (uint32_t area = 0; area < MaxPendingFaults; ++area) {
		auto& record = captured[area];
		if (!record.valid) continue;
		const auto offset = area * PageFaultAreaSize;
		const auto miss_offset = MaxPendingFaults * PageFaultAreaSize + area * CallMissSize;
		std::array<uint32_t, 8> miss {};
		uint64_t raw_count = 0;
		std::memcpy(miss.data(), mapped.data() + miss_offset, CallMissSize);
		std::memcpy(&raw_count, mapped.data() + offset, sizeof(raw_count));
		record.miss_invocations = miss[0];
		record.raw_page_faults = raw_count;
		record.page_faults = static_cast<uint32_t>(std::min<uint64_t>(raw_count, MaxPageFaults - 1));
		record.call_miss = {};
		if (miss[0] != 0u) {
			record.call_miss = {
			    .shader_hash = (uint64_t {miss[5]} << 32u) | miss[4],
			    .target = (uint64_t {miss[3]} << 32u) | miss[2],
			    .pc = miss[1]};
		}
		// Keep phase Queued: these reads did not run the harvest callback.
	}

	// No queue/allocator/Vulkan operation runs under this metadata-only lock.
	// Publication completes once for each accepted identity, even if the reporter
	// briefly held the same existing mutex while the bytes were read.
	std::scoped_lock lock(m_call_table_mutex);
	auto& diagnostics = *m_peek_diagnostics;
	if (m_harvest_diagnostics->identity_errors != 0) {
		diagnostics.status = PeekStatus::InvalidIdentity;
		return false;
	}
	for (uint32_t area = 0; area < MaxPendingFaults; ++area) {
		const auto& source = m_harvest_diagnostics->areas[area];
		if (m_fault_areas[area] != area_ticks[area] || source.valid != retained[area].valid ||
		    (retained[area].valid && !SameRetainedHarvest(retained[area], source))) {
			diagnostics.status = PeekStatus::InvalidIdentity;
			return false;
		}
	}
	for (uint32_t area = 0; area < MaxPendingFaults; ++area) {
		if (!captured[area].valid) continue;
		diagnostics.areas[area] = {.valid = true, .staged = captured[area],
		    .observed_known_tick = known, .observed_host_waited_tick = waited.completed_tick};
		++diagnostics.observed_total;
	}
	diagnostics.status = PeekStatus::Observed;
	return true;
}

void FaultManager::QueueHarvestDiagnostics(
    uint32_t area, const CausalDiagnostics::OperationWitness& operation) noexcept {
	// Called synchronously after accepted insertion while the scheduler queue lock
	// excludes dequeue. No scheduler, memory, or GPU operation occurs under this lock.
	const bool identity_valid = ValidHarvestOperation(operation, m_scheduler) &&
	    operation.native_domain == CausalDiagnostics::Domain(&m_scheduler.GetMasterSemaphore());
	std::scoped_lock lock(m_call_table_mutex);
	auto& diagnostics = *m_harvest_diagnostics;
	++diagnostics.queued_total;
	if (area >= diagnostics.areas.size()) {
		++diagnostics.identity_errors;
		return;
	}
	auto& record = diagnostics.areas[area];
	if (m_peek_diagnostics) m_peek_diagnostics->areas[area] = {};
	const bool reusable = !record.valid || record.phase == HarvestPhase::Empty ||
	                      record.phase == HarvestPhase::Completed;
	record = {};
	record.valid = identity_valid && reusable;
	record.area = area;
	record.phase = HarvestPhase::Queued;
	record.domain = operation.native_domain;
	record.lifetime = operation.lifetime_sequence;
	record.operation_id = operation.operation_id;
	record.tick = operation.selected_tick;
	record.enqueue_sequence = operation.queued_parent.sequence;
	record.enqueue_thread = operation.enqueue_thread;
	if (!record.valid) ++diagnostics.identity_errors;
}

void FaultManager::BeginHarvestDiagnostics(uint32_t area) noexcept {
	const auto execution = CausalDiagnostics::CurrentCallbackExecution();
	const bool identity_valid = ValidHarvestExecution(execution, m_scheduler) &&
	    execution.operation.native_domain ==
	        CausalDiagnostics::Domain(&m_scheduler.GetMasterSemaphore());
	std::scoped_lock lock(m_call_table_mutex);
	auto& diagnostics = *m_harvest_diagnostics;
	if (area >= diagnostics.areas.size() || !identity_valid ||
	    diagnostics.areas[area].phase != HarvestPhase::Queued ||
	    !SameHarvestOperation(diagnostics.areas[area], execution.operation)) {
		++diagnostics.identity_errors;
		return;
	}
	diagnostics.areas[area].phase = HarvestPhase::Running;
}

void FaultManager::CompleteHarvestDiagnostics(uint32_t area,
    const std::array<uint32_t, 8>& miss, uint64_t raw_page_faults, uint32_t page_faults) noexcept {
	const auto execution = CausalDiagnostics::CurrentCallbackExecution();
	const bool identity_valid = ValidHarvestExecution(execution, m_scheduler) &&
	    execution.operation.native_domain ==
	        CausalDiagnostics::Domain(&m_scheduler.GetMasterSemaphore());
	std::scoped_lock lock(m_call_table_mutex);
	auto& diagnostics = *m_harvest_diagnostics;
	++diagnostics.completed_total; // The real callback has finished, even if its identity is unknown.
	if (area >= diagnostics.areas.size() || !identity_valid ||
	    diagnostics.areas[area].phase != HarvestPhase::Running ||
	    !SameHarvestOperation(diagnostics.areas[area], execution.operation)) {
		++diagnostics.identity_errors;
		return; // Inline/unknown execution cannot invent a last-completed native tick.
	}
	auto& record = diagnostics.areas[area];
	record.miss_invocations = miss[0];
	record.raw_page_faults = raw_page_faults;
	record.page_faults = page_faults;
	record.call_miss = {};
	if (miss[0] != 0u) {
		record.call_miss = {
		    .shader_hash = (uint64_t {miss[5]} << 32u) | miss[4],
		    .target = (uint64_t {miss[3]} << 32u) | miss[2],
		    .pc = miss[1]};
	}
	record.phase = HarvestPhase::Completed;
	diagnostics.last_completed = record;
}

void FaultManager::ProcessFaultBuffer() {
	FaultPeekRecordingScope peek_recording(
	    m_peek_diagnostics ? &m_fault_peek_recording : nullptr);
	if (const auto wait_tick = m_fault_areas[m_current_area]; wait_tick != 0) {
		Libs::Graphics::WaitSiteScope wait_scope(Libs::Graphics::WaitSite::FaultManager);
		m_scheduler.Wait(wait_tick);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = m_current_area * PageFaultAreaSize;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	std::memset(mapped, 0, PageFaultAreaSize);
	m_download_buffer.Flush(offset, PageFaultAreaSize);

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = 0;
	pre_barrier.size           = m_fault_buffer.Size();
	auto post_barrier         = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), 0, m_fault_buffer.Size()},
	    {m_download_buffer.Handle(), offset, PageFaultAreaSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0, writes);
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 128; // Four words per invocation.
	const auto num_workgroups = (num_threads + 63) / 64;
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);

	// Move the table-call miss record out and clear it in order with the shader writes.
	const auto miss_offset = MaxPendingFaults * PageFaultAreaSize + m_current_area * CallMissSize;
	vk::BufferMemoryBarrier2 miss_barrier {};
	miss_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	miss_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	miss_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	miss_barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	miss_barrier.buffer        = m_fault_buffer.Handle();
	miss_barrier.offset        = CallMissOffset;
	miss_barrier.size          = CallMissSize;
	dependency.pBufferMemoryBarriers = &miss_barrier;
	command.pipelineBarrier2(dependency);
	const vk::BufferCopy miss_copy {CallMissOffset, miss_offset, CallMissSize};
	command.copyBuffer(m_fault_buffer.Handle(), m_download_buffer.Handle(), 1, &miss_copy);
	// The clear overwrites the bytes just copied: order the copy's read before the fill's write
	// (write-after-read), or the clear could lose the harvested record.
	miss_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	miss_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	miss_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	miss_barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
	command.pipelineBarrier2(dependency);
	command.fillBuffer(m_fault_buffer.Handle(), CallMissOffset, CallMissSize, 0);
	miss_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	miss_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	miss_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	miss_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	command.pipelineBarrier2(dependency);


	// Diagnostic-only device-to-host availability for the two staged ranges.
	// The later passive reader still requires a successful existing native host
	// wait plus all live areas complete before whole-allocation invalidation.
	if (FaultPeekEnabled()) {
		std::array<vk::BufferMemoryBarrier2, 2> host_barriers {};
		for (auto& host : host_barriers) {
			host.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
			                    vk::PipelineStageFlagBits2::eTransfer;
			host.srcAccessMask = vk::AccessFlagBits2::eShaderWrite |
			                     vk::AccessFlagBits2::eTransferWrite;
			host.dstStageMask = vk::PipelineStageFlagBits2::eHost;
			host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
			host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			host.buffer = m_download_buffer.Handle();
		}
		host_barriers[0].offset = offset;
		host_barriers[0].size = PageFaultAreaSize;
		host_barriers[1].offset = miss_offset;
		host_barriers[1].size = CallMissSize;
		vk::DependencyInfo host_dependency {};
		host_dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(host_barriers.size());
		host_dependency.pBufferMemoryBarriers = host_barriers.data();
		command.pipelineBarrier2(host_dependency);
	}

	const auto area = m_current_area;
	auto operation = [this, mapped, offset, area, miss_offset] {
		if (m_harvest_diagnostics) BeginHarvestDiagnostics(area);
		m_download_buffer.Invalidate(miss_offset, CallMissSize);
		std::array<uint32_t, 8> miss {};
		std::memcpy(miss.data(), m_download_buffer.Mapped().data() + miss_offset, CallMissSize);
		if (miss[0] != 0u) {
			const auto total = m_call_table_misses.fetch_add(miss[0]) + miss[0];
			CallTableMiss record {
			    .shader_hash = (uint64_t {miss[5]} << 32u) | miss[4],
			    .target      = (uint64_t {miss[3]} << 32u) | miss[2],
			    .pc          = miss[1]};
			{
				std::scoped_lock lock(m_call_table_mutex);
				m_last_call_table_miss = record;
			}
			// A failed callable: the selected entry was null, out of range or not in the closed set.
			LOGF_COLOR(Log::Color::BrightRed,
			           "SHADER CALL TABLE MISS shader=0x%016" PRIx64 " pc=0x%08x target=0x%016" PRIx64
			           " invocations=%u total=%" PRIu64 " (callable skipped; rendering incomplete)\n",
			           record.shader_hash, record.pc, record.target, miss[0], total);
		}
		m_download_buffer.Invalidate(offset, PageFaultAreaSize);
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		// An overflowing scan keeps counting past the MaxPageFaults - 1 records the area holds.
		const auto raw_count = faults[0];
		const auto count = std::min<uint64_t>(raw_count, MaxPageFaults - 1);
		for (uint32_t index = 1; index <= count; ++index) {
			const auto address = BufferCache::GuestAddress(faults[index]);
			fault_ranges.Add(address, BufferCache::CACHING_PAGESIZE);
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			(void)m_buffer_cache.FindBuffer(start, end - start);
		});
		m_fault_areas[area] = 0;
		if (m_harvest_diagnostics)
			CompleteHarvestDiagnostics(area, miss, raw_count, static_cast<uint32_t>(count));
	};
	if (m_harvest_diagnostics) {
		struct EnqueueState { FaultManager* owner; uint32_t area; };
		EnqueueState state {this, area};
		CausalDiagnostics::EnqueueReceipt receipt;
		CausalDiagnostics::EnqueueWitnessScope capture(receipt,
		    [](void* opaque, const CausalDiagnostics::OperationWitness& accepted) noexcept {
			    const auto& state = *static_cast<const EnqueueState*>(opaque);
			    state.owner->QueueHarvestDiagnostics(state.area, accepted);
		    }, &state);
		m_scheduler.DeferOperation(std::move(operation));
		// The synchronous sink already published accepted state before dequeue. Do not
		// publish after return: a callback may already be completed, or may run inline.
	} else {
		m_scheduler.DeferOperation(std::move(operation));
	}

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

} // namespace Libs::Graphics

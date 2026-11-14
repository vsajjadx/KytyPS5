#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/frameStats.h"

#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include "common/logging/log.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstring>

namespace Libs::Graphics {

namespace {

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

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     CallMissOffset + CallMissSize),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * (PageFaultAreaSize + CallMissSize)) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

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

void FaultManager::ProcessFaultBuffer() {
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

	const auto area = m_current_area;
	m_scheduler.DeferOperation([this, mapped, offset, area, miss_offset] {
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
		const auto count = std::min<uint64_t>(faults[0], MaxPageFaults - 1);
		for (uint32_t index = 1; index <= count; ++index) {
			fault_ranges.Add(BufferCache::GuestAddress(faults[index]),
			                 BufferCache::CACHING_PAGESIZE);
		}
		m_buffer_cache.MarkDmaAccess(fault_ranges);
		m_fault_areas[area] = 0;
	});

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

} // namespace Libs::Graphics

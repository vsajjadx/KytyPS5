#include "graphics/presentation/frameGeneration.h"
#include "graphics/presentation/multiFrameGeneration.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <cstdio>
#include <span>

#if defined(_WIN32)
#include "ffx_vk_fsr3_3_1_5_bridge.h"
extern "C" FfxVkFsr3_3_1_6FrameGenerationResult kytyFsrCreateSynchronizedMultiContext(
    const FfxVkFsr3_3_1_6FrameGenerationCreateInfo*,FfxVkFsr3_3_1_6FrameGenerationContext**);
#endif

namespace Libs::Graphics {

namespace FrameGeneration {

static std::atomic_bool g_enabled {false};
static std::atomic<Backend> g_backend {Backend::Fsr};
static std::atomic<uint32_t> g_multiplier {2};
void SetBackend(Backend b) { if(b==Backend::Fsr||b==Backend::Multi)g_backend=b; }
Backend GetBackend() { return g_backend.load(); }
void SetMultiplier(uint32_t n) { if(n>=2&&n<=6)g_multiplier=n; }
uint32_t GetMultiplier() { return g_multiplier.load(); }
#if defined(_WIN32)
static std::atomic_bool g_available {true};
#else
static std::atomic_bool g_available {false};
#endif
static std::atomic_bool g_retry_requested {false};

void SetEnabled(bool enabled) {
	g_enabled = enabled;
	if (enabled) {
		// Turning it on again also resumes a generator that paused after a failed frame.
		g_retry_requested = true;
	}
}

void Toggle() {
	SetEnabled(!g_enabled.load());
	std::printf("Frame generation %s\n", g_enabled.load() ? "on" : "off");
}

bool IsEnabled() {
	return g_enabled && g_available;
}

bool IsAvailable() {
	return g_available;
}

} // namespace FrameGeneration

void FrameGenerator::Reset() { m_reset=true;m_multi_prepares=0;if(m_multi)m_multi->Reset(); }
MultiFrameSceneStatus FrameGenerator::GetMultipleSceneStatus() const {
    return m_multi ? m_multi->SceneStatus() : MultiFrameSceneStatus{};
}
bool FrameGenerator::CopyMultipleSceneForDiagnostics(vk::CommandBuffer command,
    vk::Buffer destination,uint64_t capacity) {
    return m_multi&&m_multi->CopySceneForDiagnostics(command,destination,capacity);
}
bool FrameGenerator::CopyMultipleFineForDiagnostics(vk::CommandBuffer command,
    const MultiFrameDiagnosticBuffer& full_motion,const MultiFrameDiagnosticBuffer& static_layer) {
    return m_multi&&m_multi->CopyFineForDiagnostics(command,full_motion,static_layer);
}
bool FrameGenerator::MultipleSceneChanged() const {
    return GetMultipleSceneStatus().state!=MultiFrameSceneState::Clear;
}

#if defined(_WIN32)

namespace {

constexpr vk::Format ColorFormat  = vk::Format::eR8G8B8A8Unorm;
constexpr vk::Format DepthFormat  = vk::Format::eR32Sfloat;
constexpr vk::Format MotionFormat = vk::Format::eR16G16Sfloat;

constexpr vk::ImageUsageFlags ColorUsage        = vk::ImageUsageFlagBits::eStorage |
                                                  vk::ImageUsageFlagBits::eSampled |
                                                  vk::ImageUsageFlagBits::eTransferDst;
constexpr vk::ImageUsageFlags InterpolatedUsage = vk::ImageUsageFlagBits::eStorage |
                                                  vk::ImageUsageFlagBits::eSampled |
                                                  vk::ImageUsageFlagBits::eTransferSrc;
constexpr vk::ImageUsageFlags InputUsage        = vk::ImageUsageFlagBits::eStorage |
                                                  vk::ImageUsageFlagBits::eSampled |
                                                  vk::ImageUsageFlagBits::eTransferDst;

constexpr uint32_t SignatureMaxWidth = 48;
constexpr float    SceneCutThreshold = 0.08f;

const vk::ImageSubresourceRange  ColorRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
const vk::ImageSubresourceLayers ColorLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};

FfxVkFsr3_3_1_6FrameGenerationImage MakeImage(vk::Image image, vk::Format format, uint32_t width,
                                              uint32_t height, vk::ImageUsageFlags usage) {
	FfxVkFsr3_3_1_6FrameGenerationImage result {};
	result.image  = static_cast<VkImage>(image);
	result.format = static_cast<VkFormat>(format);
	result.width  = width;
	result.height = height;
	result.layout = VK_IMAGE_LAYOUT_GENERAL;
	result.usage  = static_cast<VkImageUsageFlags>(usage);
	return result;
}

vk::ImageMemoryBarrier2 Barrier(vk::Image image, vk::PipelineStageFlags2 src_stage,
                                vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
                                vk::AccessFlags2 dst_access, vk::ImageLayout old_layout,
                                vk::ImageLayout new_layout) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = src_stage;
	barrier.srcAccessMask       = src_access;
	barrier.dstStageMask        = dst_stage;
	barrier.dstAccessMask       = dst_access;
	barrier.oldLayout           = old_layout;
	barrier.newLayout           = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = ColorRange;
	return barrier;
}

void PipelineBarriers(vk::CommandBuffer                        command,
                      std::span<const vk::ImageMemoryBarrier2> barriers) {
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command.pipelineBarrier2(dependency);
}

// A blit, not a copy: game frames can be 10-bit while the interpolation works in 8-bit RGBA.
void CopyImage(vk::CommandBuffer command, vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
               vk::ImageLayout dst_layout, uint32_t width, uint32_t height) {
	vk::ImageBlit region {};
	region.srcSubresource = ColorLayers;
	region.dstSubresource = ColorLayers;
	region.srcOffsets[1] =
	    vk::Offset3D {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
	region.dstOffsets[1] = region.srcOffsets[1];
	command.blitImage(src, src_layout, dst, dst_layout, region, vk::Filter::eNearest);
}

} // namespace

FrameGenerator::FrameGenerator(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {}

FrameGenerator::~FrameGenerator() {
	Destroy();
}

void FrameGenerator::Fail(const char* what) {
	std::printf("FSR frame generation disabled: %s\n", what);
	m_failed                     = true;
	FrameGeneration::g_available = false;
}

void FrameGenerator::Suspend(const char* what) {
	// A failed pass or an allocation that ran out of video memory can pass, unlike a GPU that
	// cannot run the effect at all. Pause, and try again from a clean state later.
	std::printf("FSR frame generation paused: %s\n", what);
	m_failed                           = true;
	m_failed_tick                      = m_scheduler.CurrentTick();
	FrameGeneration::g_retry_requested = false;
	m_retry_at                         = std::chrono::steady_clock::now() + RetryDelay;
	if (++m_consecutive_failures >= MaxAutomaticRetries) {
		// Keep it paused until it is turned on again.
		m_retry_at = std::chrono::steady_clock::time_point::max();
	}
}

bool FrameGenerator::TryResume() {
	if (!FrameGeneration::IsAvailable()) {
		return false;
	}
	const bool requested = FrameGeneration::g_retry_requested;
	if (!requested && std::chrono::steady_clock::now() < m_retry_at) {
		return false;
	}
	// The commands of the failed frame may still be recording or running on the GPU.
	if (!m_scheduler.IsFree(m_failed_tick)) {
		return false;
	}
	FrameGeneration::g_retry_requested = false;
	if (requested) {
		m_consecutive_failures = 0;
	}
	Destroy();
	m_failed = false;
	return true;
}

bool FrameGenerator::CreateTarget(Target& target, vk::Format format, vk::ImageUsageFlags usage,
                                  uint32_t width, uint32_t height, uint32_t mip_levels) {
	vk::ImageCreateInfo info {};
	info.imageType                     = vk::ImageType::e2D;
	info.format                        = format;
	info.extent                        = vk::Extent3D {width, height, 1};
	info.mipLevels                     = mip_levels;
	info.arrayLayers                   = 1;
	info.samples                       = vk::SampleCountFlagBits::e1;
	info.tiling                        = vk::ImageTiling::eOptimal;
	info.usage                         = usage;
	info.sharingMode                   = vk::SharingMode::eExclusive;
	info.initialLayout                 = vk::ImageLayout::eUndefined;
	const auto              image_info = static_cast<VkImageCreateInfo>(info);
	VmaAllocationCreateInfo alloc_info {};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	VkImage image    = nullptr;
	if (vmaCreateImage(m_graphics.allocator, &image_info, &alloc_info, &image, &target.allocation,
	                   nullptr) != VK_SUCCESS) {
		target = {};
		return false;
	}
	target.image = vk::Image {image};
	return true;
}

void FrameGenerator::DestroyTarget(Target& target) {
	if (target.image) {
		vmaDestroyImage(m_graphics.allocator, static_cast<VkImage>(target.image),
		                target.allocation);
	}
	target = {};
}

void FrameGenerator::Destroy() {
	m_multi.reset();
	m_multi_prepares=0;
	if (m_context != nullptr || m_color.image) {
		const auto tick = m_scheduler.CurrentTick();
		if (tick > 1) {
			m_scheduler.Wait(tick - 1);
		}
	}
	if (m_context != nullptr) {
		ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(m_context, m_frame_id);
		ffxVkFsr3_3_1_6FrameGenerationContextDestroy(m_context);
		m_context = nullptr;
	}
	m_pending.clear();
    m_context_is_multi = false;
	DestroySignature();
	DestroyTarget(m_color);
	DestroyTarget(m_interpolated);
	DestroyTarget(m_depth);
	DestroyTarget(m_motion);
	m_width  = 0;
	m_height = 0;
}

bool FrameGenerator::CreateSignature(uint32_t frame_width, uint32_t frame_height) {
	uint32_t   level_width  = std::max(frame_width / 2, 1u);
	uint32_t   level_height = std::max(frame_height / 2, 1u);
	const auto base_width   = level_width;
	const auto base_height  = level_height;
	uint32_t   levels       = 1;
	while (level_width > SignatureMaxWidth) {
		level_width  = std::max(level_width / 2, 1u);
		level_height = std::max(level_height / 2, 1u);
		++levels;
	}
	if (!CreateTarget(m_signature, ColorFormat,
	                  vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
	                  base_width, base_height, levels)) {
		return false;
	}
	m_signature_levels = levels;
	m_signature_width  = level_width;
	m_signature_height = level_height;

	vk::BufferCreateInfo buffer {};
	buffer.size        = static_cast<vk::DeviceSize>(level_width) * level_height * 4;
	buffer.usage       = vk::BufferUsageFlagBits::eTransferDst;
	buffer.sharingMode = vk::SharingMode::eExclusive;
	const auto              buffer_info = static_cast<VkBufferCreateInfo>(buffer);
	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags =
	    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	std::scoped_lock lock(m_signature_mutex);
	for (auto& slot: m_signature_slots) {
		VkBuffer          handle = nullptr;
		VmaAllocationInfo info {};
		if (vmaCreateBuffer(m_graphics.allocator, &buffer_info, &alloc_info, &handle,
		                    &slot.allocation, &info) != VK_SUCCESS) {
			slot = {};
			return false;
		}
		slot.buffer   = vk::Buffer {handle};
		slot.mapped   = info.pMappedData;
		slot.frame_id = 0;
	}
	m_previous_signature.clear();
	m_previous_signature_id = 0;
	return true;
}

void FrameGenerator::DestroySignature() {
	std::scoped_lock lock(m_signature_mutex);
	for (auto& slot: m_signature_slots) {
		if (slot.buffer) {
			vmaDestroyBuffer(m_graphics.allocator, static_cast<VkBuffer>(slot.buffer),
			                 slot.allocation);
		}
		slot = {};
	}
	m_previous_signature.clear();
	m_previous_signature_id = 0;
	DestroyTarget(m_signature);
	m_signature_levels = 0;
}

void FrameGenerator::RecordSignature(vk::CommandBuffer command, vk::Image frame,
                                     uint32_t frame_width, uint32_t frame_height, uint64_t id) {
	using Stage  = vk::PipelineStageFlagBits2;
	using Access = vk::AccessFlagBits2;
	using Layout = vk::ImageLayout;

	const auto mip_barrier = [&](uint32_t mip, vk::AccessFlags2 src_access,
	                             vk::AccessFlags2 dst_access, Layout old_layout, Layout new_layout,
	                             uint32_t level_count) {
		auto barrier = Barrier(m_signature.image, Stage::eAllCommands, src_access, Stage::eTransfer,
		                       dst_access, old_layout, new_layout);
		barrier.subresourceRange.baseMipLevel = mip;
		barrier.subresourceRange.levelCount   = level_count;
		return barrier;
	};

	const std::array to_blit {mip_barrier(0, Access::eTransferRead | Access::eTransferWrite,
	                                      Access::eTransferWrite, Layout::eUndefined,
	                                      Layout::eTransferDstOptimal, m_signature_levels)};
	PipelineBarriers(command, to_blit);

	const uint32_t base_width    = std::max(frame_width / 2, 1u);
	const uint32_t base_height   = std::max(frame_height / 2, 1u);
	uint32_t       source_width  = frame_width;
	uint32_t       source_height = frame_height;
	for (uint32_t level = 0; level < m_signature_levels; ++level) {
		const uint32_t level_width  = std::max(base_width >> level, 1u);
		const uint32_t level_height = std::max(base_height >> level, 1u);
		if (level > 0) {
			const std::array to_source {
			    mip_barrier(level - 1, Access::eTransferWrite, Access::eTransferRead,
				            Layout::eTransferDstOptimal, Layout::eTransferSrcOptimal, 1)};
			PipelineBarriers(command, to_source);
		}
		vk::ImageBlit blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, level > 0 ? level - 1 : 0, 0, 1};
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(source_width),
		                                    static_cast<int32_t>(source_height), 1};
		blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, level, 0, 1};
		blit.dstOffsets[1] =
		    vk::Offset3D {static_cast<int32_t>(level_width), static_cast<int32_t>(level_height), 1};
		command.blitImage(level > 0 ? m_signature.image : frame,
		                  level > 0 ? Layout::eTransferSrcOptimal : Layout::eGeneral,
		                  m_signature.image, Layout::eTransferDstOptimal, blit,
		                  vk::Filter::eLinear);
		source_width  = level_width;
		source_height = level_height;
	}

	const uint32_t   last = m_signature_levels - 1;
	const std::array to_readback {mip_barrier(last, Access::eTransferWrite, Access::eTransferRead,
	                                          Layout::eTransferDstOptimal,
	                                          Layout::eTransferSrcOptimal, 1)};
	PipelineBarriers(command, to_readback);

	SignatureSlot&      slot = m_signature_slots[id % SignatureSlots];
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, last, 0, 1};
	copy.imageExtent      = vk::Extent3D {m_signature_width, m_signature_height, 1};
	command.copyImageToBuffer(m_signature.image, Layout::eTransferSrcOptimal, slot.buffer, copy);

	vk::MemoryBarrier2 to_host {};
	to_host.srcStageMask  = Stage::eTransfer;
	to_host.srcAccessMask = Access::eTransferWrite;
	to_host.dstStageMask  = Stage::eHost;
	to_host.dstAccessMask = Access::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &to_host;
	command.pipelineBarrier2(dependency);

	std::scoped_lock lock(m_signature_mutex);
	slot.frame_id = id;
}

bool FrameGenerator::IsSceneCut(uint64_t id) {
	std::scoped_lock lock(m_signature_mutex);
	if (id == 0) {
		return false;
	}
	const SignatureSlot& slot = m_signature_slots[id % SignatureSlots];
	if (slot.mapped == nullptr || slot.frame_id != id) {
		return false;
	}
	const size_t size = static_cast<size_t>(m_signature_width) * m_signature_height * 4;
	vmaInvalidateAllocation(m_graphics.allocator, slot.allocation, 0, size);
	const auto* data = static_cast<const uint8_t*>(slot.mapped);

	bool cut = false;
	if (m_previous_signature_id + 1 == id && m_previous_signature.size() == size) {
		uint64_t difference = 0;
		for (size_t i = 0; i < size; i += 4) {
			for (size_t channel = 0; channel < 3; ++channel) {
				const int delta = static_cast<int>(data[i + channel]) -
				                  static_cast<int>(m_previous_signature[i + channel]);
				difference += static_cast<uint64_t>(delta < 0 ? -delta : delta);
			}
		}
		const double mean = static_cast<double>(difference) / (size / 4 * 3 * 255.0);
		cut               = mean > SceneCutThreshold;
	}
	m_previous_signature.assign(data, data + size);
	m_previous_signature_id = id;
	if (cut) {
		m_reset = true;
	}
	return cut;
}

bool FrameGenerator::Recreate(vk::CommandBuffer command, uint32_t width, uint32_t height, bool multi) {
	Destroy();
	if (!CreateTarget(m_color, ColorFormat, ColorUsage, width, height) ||
	    !CreateTarget(m_interpolated, ColorFormat, InterpolatedUsage, width, height) ||
	    !CreateTarget(m_depth, DepthFormat, InputUsage, width, height) ||
	    !CreateTarget(m_motion, MotionFormat, InputUsage, width, height)) {
		Destroy();
		Suspend("out of video memory for its images");
		return false;
	}
	if (!CreateSignature(width, height)) {
		Destroy();
		Suspend("out of video memory for its scene change signature");
		return false;
	}

	FfxVkFsr3_3_1_6FrameGenerationCreateInfo create_info {};
	create_info.physicalDevice  = static_cast<VkPhysicalDevice>(m_graphics.physical_device);
	create_info.device          = static_cast<VkDevice>(m_graphics.device);
	create_info.maxRenderWidth  = width;
	create_info.maxRenderHeight = height;
	create_info.displayWidth    = width;
	create_info.displayHeight   = height;
	create_info.colorFormat     = static_cast<VkFormat>(ColorFormat);
	const auto result = multi
        ? kytyFsrCreateSynchronizedMultiContext(&create_info, &m_context)
        : ffxVkFsr3_3_1_6FrameGenerationContextCreate(&create_info, &m_context);
	if (result != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
		m_context = nullptr;
		Destroy();
		std::printf("FSR frame generation context creation failed: %d\n", static_cast<int>(result));
		Fail("its context could not be created on this GPU");
		return false;
	}
	m_context_is_multi = multi;
	m_width  = width;
	m_height = height;
	m_reset  = true;

	// Without depth and motion from the game, the interpolation follows the optical flow alone.
	const std::array to_clear {
	    Barrier(m_depth.image, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
		        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral),
	    Barrier(m_motion.image, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
		        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral)};
	PipelineBarriers(command, to_clear);
	const vk::ClearColorValue zero {};
	command.clearColorImage(m_depth.image, vk::ImageLayout::eGeneral, zero, ColorRange);
	command.clearColorImage(m_motion.image, vk::ImageLayout::eGeneral, zero, ColorRange);
	const std::array cleared {
	    Barrier(m_depth.image, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
		        vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
		        vk::ImageLayout::eGeneral),
	    Barrier(m_motion.image, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
		        vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
		        vk::ImageLayout::eGeneral)};
	PipelineBarriers(command, cleared);

	FfxVkFsr3_3_1_6FrameGenerationMemoryUsage usage {};
	ffxVkFsr3_3_1_6FrameGenerationContextGetMemoryUsage(m_context, &usage);
	std::printf("FSR 3.1 frame generation ready at %ux%u (%.1f MB)\n", m_width, m_height,
	            static_cast<double>(usage.totalUsageInBytes) / 1e6);
	return true;
}

bool FrameGenerator::Generate(vk::CommandBuffer command, vk::Image frame, vk::Image output,
                              uint32_t frame_width, uint32_t frame_height) {
	m_signature_id = 0;
	m_output_written = false;
    // Switching back must use the untouched legacy SDK shader, never a Multi context.
    // Refuse an unsubmitted mode switch rather than submitting the caller's live command.
    if(m_context_is_multi&&((!m_pending.empty()&&m_pending.back().tick==m_scheduler.CurrentTick())||
        (m_multi&&m_multi->SceneReadbackPending())))return false;
	if (frame_width == 0 || frame_height == 0) {
		return false;
	}
	if (m_failed && !TryResume()) {
		return false;
	}

	const auto now      = std::chrono::steady_clock::now();
	float      frame_ms = std::chrono::duration<float, std::milli>(now - m_last_generate).count();
	m_last_generate     = now;
	if (!(frame_ms >= 1.0f && frame_ms <= 200.0f)) {
		frame_ms = 33.3f;
		m_reset  = true;
	}

	while (m_context != nullptr && !m_pending.empty() &&
	       m_scheduler.IsFree(m_pending.front().tick)) {
		ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(m_context, m_pending.front().frame_id);
		m_pending.pop_front();
	}

	if (m_context == nullptr || m_context_is_multi || m_width != frame_width || m_height != frame_height) {
		if (!Recreate(command, frame_width, frame_height)) {
			return false;
		}
	}

	using Stage  = vk::PipelineStageFlagBits2;
	using Access = vk::AccessFlagBits2;
	using Layout = vk::ImageLayout;

	const std::array to_copy {
	    Barrier(frame, Stage::eAllCommands,
		        Access::eColorAttachmentWrite | Access::eShaderWrite | Access::eTransferWrite,
		        Stage::eTransfer, Access::eTransferRead, Layout::eGeneral, Layout::eGeneral),
	    Barrier(m_color.image, Stage::eAllCommands, Access::eNone, Stage::eTransfer,
		        Access::eTransferWrite, Layout::eUndefined, Layout::eGeneral)};
	PipelineBarriers(command, to_copy);
	CopyImage(command, frame, Layout::eGeneral, m_color.image, Layout::eGeneral, m_width, m_height);
	// The scene-cut signature (RecordSignature/IsSceneCut/SignatureId) has no consumer in src or
	// tests, so its per-frame downsample chain and host copy are no longer recorded.

	const std::array to_interpolate {
	    Barrier(m_color.image, Stage::eTransfer, Access::eTransferWrite, Stage::eAllCommands,
		        Access::eShaderRead, Layout::eGeneral, Layout::eGeneral),
	    Barrier(m_interpolated.image, Stage::eAllCommands, Access::eNone, Stage::eAllCommands,
		        Access::eShaderWrite | Access::eShaderRead, Layout::eUndefined, Layout::eGeneral)};
	PipelineBarriers(command, to_interpolate);

	const bool     was_reset = m_reset;
	const uint64_t id        = ++m_frame_id;

	FfxVkFsr3_3_1_6FrameGenerationPrepareInfo prepare {};
	prepare.commandBuffer = static_cast<VkCommandBuffer>(command);
	prepare.color         = MakeImage(m_color.image, ColorFormat, m_width, m_height, ColorUsage);
	prepare.depth         = MakeImage(m_depth.image, DepthFormat, m_width, m_height, InputUsage);
	prepare.motionVectors = MakeImage(m_motion.image, MotionFormat, m_width, m_height, InputUsage);
	prepare.renderWidth   = m_width;
	prepare.renderHeight  = m_height;
	prepare.motionVectorScaleX       = 1.0f;
	prepare.motionVectorScaleY       = 1.0f;
	prepare.frameTimeMilliseconds    = frame_ms;
	prepare.minLuminance             = 0.0f;
	prepare.maxLuminance             = 1.0f;
	prepare.transferFunction         = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
	prepare.cameraNear               = 0.1f;
	prepare.cameraFar                = 1000.0f;
	prepare.viewSpaceToMeters        = 1.0f;
	prepare.cameraVerticalFovRadians = 1.0f;
	prepare.cameraUp[1]              = 1.0f;
	prepare.cameraRight[0]           = 1.0f;
	prepare.cameraForward[2]         = 1.0f;
	prepare.frameId                  = id;
	prepare.reset                    = was_reset ? VK_TRUE : VK_FALSE;
	if (ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(m_context, &prepare) !=
	    FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
		m_pending.push_back({id, m_scheduler.CurrentTick()});
		Suspend("its prepare pass failed");
		return false;
	}

	FfxVkFsr3_3_1_6FrameGenerationDispatchInfo dispatch {};
	dispatch.commandBuffer = prepare.commandBuffer;
	dispatch.color         = prepare.color;
	dispatch.output =
	    MakeImage(m_interpolated.image, ColorFormat, m_width, m_height, InterpolatedUsage);
	dispatch.displayWidth             = m_width;
	dispatch.displayHeight            = m_height;
	dispatch.frameTimeMilliseconds    = frame_ms;
	dispatch.cameraNear               = prepare.cameraNear;
	dispatch.cameraFar                = prepare.cameraFar;
	dispatch.viewSpaceToMeters        = prepare.viewSpaceToMeters;
	dispatch.cameraVerticalFovRadians = prepare.cameraVerticalFovRadians;
	dispatch.minLuminance             = prepare.minLuminance;
	dispatch.maxLuminance             = prepare.maxLuminance;
	dispatch.transferFunction         = prepare.transferFunction;
	dispatch.frameId                  = id;
	dispatch.reset                    = prepare.reset;
	const auto result = ffxVkFsr3_3_1_6FrameGenerationContextRecordDispatch(m_context, &dispatch);
	m_pending.push_back({id, m_scheduler.CurrentTick()});
	if (result != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
		Suspend("its interpolation pass failed");
		return false;
	}
	m_reset                = false;
	m_consecutive_failures = 0;

	const std::array to_output {
	    Barrier(m_interpolated.image, Stage::eAllCommands, Access::eShaderWrite, Stage::eTransfer,
		        Access::eTransferRead, Layout::eGeneral, Layout::eGeneral),
	    Barrier(output, Stage::eAllCommands, Access::eNone, Stage::eTransfer,
		        Access::eTransferWrite, Layout::eUndefined, Layout::eTransferDstOptimal)};
	PipelineBarriers(command, to_output);
	CopyImage(command, m_interpolated.image, Layout::eGeneral, output, Layout::eTransferDstOptimal,
	          m_width, m_height);
	m_output_written = true; // the barrier below leaves it in the transfer source layout

	const std::array to_present {
	    Barrier(output, Stage::eTransfer, Access::eTransferWrite, Stage::eTransfer,
		        Access::eTransferRead, Layout::eTransferDstOptimal, Layout::eTransferSrcOptimal),
	    Barrier(frame, Stage::eTransfer, Access::eTransferRead, Stage::eAllCommands,
		        Access::eShaderRead | Access::eTransferRead, Layout::eGeneral, Layout::eGeneral)};
	PipelineBarriers(command, to_present);
	return !was_reset;
}

MultiFrameResult FrameGenerator::GenerateMultiple(vk::CommandBuffer command,vk::Image frame,
    std::span<const vk::Image> outputs,uint32_t width,uint32_t height,uint32_t factor,
    const MultiFrameDiagnostics* diagnostics) {
    if(!command||!frame||!width||!height||width>m_graphics.physical_device_properties.limits.maxImageDimension2D||height>m_graphics.physical_device_properties.limits.maxImageDimension2D||factor<2||factor>6||outputs.size()!=factor-1)return {};
    for(size_t i=0;i<outputs.size();++i){if(!outputs[i]||outputs[i]==frame)return {};for(size_t j=0;j<i;++j)if(outputs[i]==outputs[j])return {};}
    if(diagnostics) {
        const uint64_t rgba=uint64_t(width)*height*4, flow_bytes=uint64_t((width+7)/8)*((height+7)/8)*4;
        const MultiFrameDiagnosticBuffer targets[]={diagnostics->optical_flow,diagnostics->scene_change,diagnostics->previous,diagnostics->current};
        const uint64_t sizes[]={flow_bytes,8,rgba,rgba};
        for(size_t i=0;i<4;++i){if(!targets[i].buffer||targets[i].capacity<sizes[i])return {};for(size_t j=0;j<i;++j)if(targets[i].buffer==targets[j].buffer)return {};}
        if(diagnostics->refined_flow.buffer) {
            if(diagnostics->refined_flow.capacity<flow_bytes*4)return {};
            for(const auto& target:targets)if(diagnostics->refined_flow.buffer==target.buffer)return {};
        }
        // Diagnostics require an already-warmed pair; rejected requests do not advance it.
        if(!m_multi||m_multi_prepares<6||m_reset)return {};
    }
    // A diagnostic read retains the SDK-owned SCD image through its own submission.
    if(m_multi&&m_multi->SceneReadbackPending())return {};
    // Invalid validation calls do not alter the timing/history. Reconfiguration cannot submit
    // the caller's live command buffer; a caller must first submit its preceding batch.
    if((width!=m_width||height!=m_height||m_failed||(m_context&&!m_context_is_multi))&&!m_pending.empty()&&m_pending.back().tick==m_scheduler.CurrentTick())return {};
    if(m_failed&&!TryResume())return {};
    while(m_context&&!m_pending.empty()&&m_scheduler.IsFree(m_pending.front().tick)) {
        ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(m_context,m_pending.front().frame_id);m_pending.pop_front();
    }
    if(!m_context||!m_context_is_multi||m_width!=width||m_height!=height)if(!Recreate(command,width,height,true))return {};
    if(!m_multi){auto candidate=std::make_unique<MultiFrameInterpolator>(m_graphics,m_scheduler);if(!candidate->Initialize(width,height))return {};m_multi=std::move(candidate);}
    const auto now=std::chrono::steady_clock::now();
    float frame_ms=std::chrono::duration<float,std::milli>(now-m_last_generate).count();m_last_generate=now;
    if(!(frame_ms>=1.0f&&frame_ms<=200.0f)){frame_ms=33.3f;Reset();}
    const bool was_reset=m_reset;
    using Stage=vk::PipelineStageFlagBits2;using Access=vk::AccessFlagBits2;using Layout=vk::ImageLayout;
    const std::array before{Barrier(frame,Stage::eAllCommands,Access::eMemoryWrite,Stage::eTransfer,Access::eTransferRead,Layout::eGeneral,Layout::eGeneral),
        Barrier(m_color.image,Stage::eAllCommands,Access::eMemoryRead|Access::eMemoryWrite,Stage::eTransfer,Access::eTransferWrite,Layout::eUndefined,Layout::eGeneral)};
    PipelineBarriers(command,before);CopyImage(command,frame,Layout::eGeneral,m_color.image,Layout::eGeneral,width,height);
    const std::array ready{Barrier(m_color.image,Stage::eTransfer,Access::eTransferWrite,Stage::eAllCommands,Access::eShaderRead,Layout::eGeneral,Layout::eGeneral)};PipelineBarriers(command,ready);
    const uint64_t id=++m_frame_id;
	FfxVkFsr3_3_1_6FrameGenerationPrepareInfo prepare {};
	prepare.commandBuffer = static_cast<VkCommandBuffer>(command);
	prepare.color         = MakeImage(m_color.image, ColorFormat, m_width, m_height, ColorUsage);
	prepare.depth         = MakeImage(m_depth.image, DepthFormat, m_width, m_height, InputUsage);
	prepare.motionVectors = MakeImage(m_motion.image, MotionFormat, m_width, m_height, InputUsage);
	prepare.renderWidth   = m_width;
	prepare.renderHeight  = m_height;
	prepare.motionVectorScaleX       = 1.0f;
	prepare.motionVectorScaleY       = 1.0f;
	prepare.frameTimeMilliseconds    = frame_ms;
	prepare.minLuminance             = 0.0f;
	prepare.maxLuminance             = 1.0f;
	prepare.transferFunction         = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
	prepare.cameraNear               = 0.1f;
	prepare.cameraFar                = 1000.0f;
	prepare.viewSpaceToMeters        = 1.0f;
	prepare.cameraVerticalFovRadians = 1.0f;
	prepare.cameraUp[1]              = 1.0f;
	prepare.cameraRight[0]           = 1.0f;
	prepare.cameraForward[2]         = 1.0f;
	prepare.frameId                  = id;
	prepare.reset                    = was_reset ? VK_TRUE : VK_FALSE;
	if (ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(m_context, &prepare) !=
	    FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
		m_pending.push_back({id, m_scheduler.CurrentTick()});
		Suspend("its optical flow prepare failed");
		Reset();
		return {};
	}

    KytyFsrOpticalFlow flow{};
    const bool finished=kytyFsrFinishOpticalFlow(m_context,&flow);
    m_pending.push_back({id,m_scheduler.CurrentTick()});
    if(!finished||flow.frame_id!=id){Suspend("its optical flow completion failed");Reset();return {};}
    m_reset=false;m_consecutive_failures=0;
    if(was_reset)m_multi_prepares=0;
    ++m_multi_prepares;
    auto result=m_multi->Record(command,m_color.image,flow,outputs,m_multi_prepares>=7,diagnostics);
    if(result.status==MultiFrameStatus::Failed)Reset();
    return result;
}

#else

FrameGenerator::FrameGenerator(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {}
FrameGenerator::~FrameGenerator() = default;
bool FrameGenerator::Generate(vk::CommandBuffer, vk::Image, vk::Image, uint32_t, uint32_t) {
	return false;
}
MultiFrameResult FrameGenerator::GenerateMultiple(vk::CommandBuffer,vk::Image,std::span<const vk::Image>,uint32_t,uint32_t,uint32_t,const MultiFrameDiagnostics*) { return {}; }
bool FrameGenerator::IsSceneCut(uint64_t) {
	return false;
}

#endif

} // namespace Libs::Graphics

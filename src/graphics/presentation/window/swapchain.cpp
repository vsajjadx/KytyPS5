#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_video_out_overlay_spv.h"
#include "graphics/host_gpu/frameStats.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameGeneration.h"
#include "graphics/presentation/frameGenerationDiag.h"
#include "graphics/presentation/frameGenerationPacing.h"
#include "graphics/presentation/multiFramePresentPolicy.h"
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <charconv>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage   image;
	vk::ImageView view         = nullptr;
	uint64_t      present_tick = 0;
	bool          busy         = false;
	bool          hdr          = false;
	bool          blank        = false;
	uint32_t      references   = 0; // FramePool mutex; each queued/layer/last-shown owner holds one.
	bool          clear_pending = false;
	vk::ClearColorValue clear_color {};

	void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	bool TryConfigure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
	void CopyFrom(CommandBuffer& command, Image& source);
	void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
	FramePool(WindowContext& window, CommandScheduler& scheduler)
	    : m_window(window), m_scheduler(scheduler) {}
	~FramePool() {
		m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		for (auto& frame: m_frames) {
			m_window.graphic_ctx.device.destroyImageView(frame->view, nullptr);
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
		}
	}
	KYTY_CLASS_NO_COPY(FramePool);

	void Initialize(uint32_t count, vk::Format format) {
		Common::LockGuard lock(m_mutex);
		m_format = format;
		m_frames.reserve(count);
		m_free.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
	}

	void SetFormat(vk::Format format) {
		Common::LockGuard lock(m_mutex);
		m_format = format;
	}

	vk::Format GetFormat() {
		Common::LockGuard lock(m_mutex);
		return m_format;
	}

	Presenter::Frame* Acquire(vk::Extent2D extent, vk::Format format) {
		m_mutex.Lock();
		// A synchronized flip may need more frames than the swapchain has images.
		// Queue capacity bounds growth; waiting here can prevent its master from recording.
		if (m_free.empty()) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
		const auto compatible = std::ranges::find_if(m_free, [&](const auto* frame) {
			return frame->image.extent.width == extent.width &&
			       frame->image.extent.height == extent.height && frame->image.format == format;
		});
		auto*      frame      = compatible == m_free.end() ? m_free.back() : *compatible;
		if (compatible != m_free.end()) {
			*compatible = m_free.back();
		}
		m_free.pop_back();
		frame->busy = true;
		frame->references = 1;
		frame->clear_pending = false;
		m_mutex.Unlock();

		Libs::Graphics::WaitSiteScope wait_scope(Libs::Graphics::WaitSite::SwapchainFrame);
		// A released frame only names submitted uses. Pool reuse must not run scheduler
		// callbacks or record/submit from a guest preparation thread.
		m_scheduler.GetMasterSemaphore().Wait(frame->present_tick);
		return frame;
	}

	void Retain(Presenter::Frame* frame) {
		Common::LockGuard lock(m_mutex);
		EXIT_IF(!frame->busy || frame->references == 0);
		++frame->references;
	}

	void Release(Presenter::Frame* frame) {
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		EXIT_IF(frame->references == 0);
		if (--frame->references != 0) return;
		frame->busy = false;
		m_free.push_back(frame);
	}

private:
	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::vector<Presenter::Frame*>                 m_free;
	vk::Format                                     m_format = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}

	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferSrc |
	    vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	if (dst.image != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	               vk::ImageUsageFlagBits::eSampled;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
}

bool Presenter::Frame::TryConfigure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		return false;
	}
	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return true;
	}

	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferSrc |
	    vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		return false;
	}

	if (dst.image != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	               vk::ImageUsageFlagBits::eSampled;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		return false;
	}
	return true;
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
	const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
	                                access == vk::AccessFlagBits2::eTransferWrite
	                            ? vk::PipelineStageFlagBits2::eTransfer
	                            : vk::PipelineStageFlagBits2::eAllCommands;
	constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
	if (image.state.layout == layout && image.state.access_mask == access &&
	    !static_cast<bool>(image.state.access_mask & writes)) {
		return;
	}
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
	copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
	                       std::min(source.backing.extent.height, image.extent.height), 1};
	EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
}

// Captures the emulator's own video-out image, without desktop/window automation.
// A request is one monotonically increasing integer in KYTY_MENU_TEST/capture.txt.
// The only extra GPU work and synchronization occurs for an explicit new request.
static void CaptureMenuTestFrame(CommandBuffer& command, Presenter::Frame& frame) {
	const char* directory = std::getenv("KYTY_MENU_TEST");
	if (directory == nullptr || directory[0] == '\0') return;
	static auto next_poll = std::chrono::steady_clock::time_point {};
	static uint64_t sequence = 0;
	const auto now = std::chrono::steady_clock::now();
	if (now < next_poll) return;
	next_poll = now + std::chrono::milliseconds(200);
	const auto folder = std::filesystem::path(directory);
	std::ifstream request(folder / "capture.txt");
	uint64_t requested = 0;
	if (!(request >> requested) || requested <= sequence) return;
	sequence = requested;
	const auto format = frame.image.format;
	const bool rgba = format == vk::Format::eR8G8B8A8Unorm ||
	                  format == vk::Format::eR8G8B8A8Srgb;
	const bool bgra = format == vk::Format::eB8G8R8A8Unorm ||
	                  format == vk::Format::eB8G8R8A8Srgb;
	const bool rgb10 = format == vk::Format::eA2B10G10R10UnormPack32;
	const bool bgr10 = format == vk::Format::eA2R10G10B10UnormPack32;
	if (!rgba && !bgra && !rgb10 && !bgr10) {
		std::printf("MENU_TEST: unsupported capture format=%d\n", static_cast<int>(format));
		std::fflush(stdout);
		return;
	}
	const auto width = frame.image.extent.width;
	const auto height = frame.image.extent.height;
	const uint64_t size = uint64_t {width} * height * 4;
	auto& scheduler = command.GetContext().GetCommandScheduler();
	auto download = std::make_unique<Buffer>(command.GetGraphics(), scheduler,
	    MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, size);
	frame.Transit(command.Handle(), vk::ImageLayout::eTransferSrcOptimal,
	              vk::AccessFlagBits2::eTransferRead);
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.imageExtent = {width, height, 1};
	command.Handle().copyImageToBuffer(frame.image.image, vk::ImageLayout::eTransferSrcOptimal,
	                                  download->Handle(), copy);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = download->Handle();
	barrier.size = size;
	command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	    vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &barrier, 0, nullptr);
	const auto path = folder / ("capture-" + std::to_string(requested) + ".ppm");
	scheduler.DeferOperation([download = std::move(download), size, width, height, rgba,
	                         bgra, rgb10, path, requested] {
		download->Invalidate(0, size);
		const auto bytes = download->Mapped();
		std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
		for (size_t pixel = 0; pixel < rgb.size() / 3; ++pixel) {
			const auto* in = bytes.data() + pixel * 4;
			auto* out = rgb.data() + pixel * 3;
			if (rgba || bgra) {
				out[0] = in[rgba ? 0 : 2]; out[1] = in[1]; out[2] = in[rgba ? 2 : 0];
			} else {
				const auto word = uint32_t {in[0]} | uint32_t {in[1]} << 8u |
				                  uint32_t {in[2]} << 16u | uint32_t {in[3]} << 24u;
				out[0] = static_cast<uint8_t>(((word >> (rgb10 ? 0u : 20u)) & 1023u) * 255u / 1023u);
				out[1] = static_cast<uint8_t>(((word >> 10u) & 1023u) * 255u / 1023u);
				out[2] = static_cast<uint8_t>(((word >> (rgb10 ? 20u : 0u)) & 1023u) * 255u / 1023u);
			}
		}
		std::ofstream output(path, std::ios::binary);
		output << "P6\n" << width << " " << height << "\n255\n";
		output.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
		std::printf("MENU_TEST: capture sequence=%llu size=%ux%u ok=%d\n",
		            static_cast<unsigned long long>(requested), width, height,
		            static_cast<int>(output.good()));
		std::fflush(stdout);
	});
}

// Optional owned-menu-test capture of a committed Multi generated source. This
// is pre-composition/pre-Present evidence, never proof of successful WSI/scanout.
struct MenuGeneratedCapture {
    MenuGeneratedCapture() {
        const char* directory = std::getenv("KYTY_MENU_TEST");
        if (directory != nullptr && directory[0] != '\0') folder = std::filesystem::path(directory);
    }
    std::filesystem::path folder;
    std::chrono::steady_clock::time_point next_poll {};
    uint64_t seen_sequence = 0, pending_sequence = 0;
    uint32_t requested_ordinal = 0; // 0 selects the next committed generated ticket.
};
static void CaptureMenuTestGeneratedFrame(WindowContext& window, CommandScheduler& scheduler,
    FrameGenerator* generator, Presenter::Frame& frame, uint32_t ordinal, uint32_t factor,
    uint64_t epoch, MenuGeneratedCapture& state) {
    if (state.folder.empty()) return;
    const auto now = std::chrono::steady_clock::now();
    const auto& folder = state.folder;
    if (now >= state.next_poll) {
        state.next_poll = now + std::chrono::milliseconds(200);
        std::ifstream request(folder / "capture-generated.txt", std::ios::binary);
        std::array<char, 64> bytes {};
        request.read(bytes.data(), bytes.size());
        const auto count = request.gcount();
        if (count > 0 && count < static_cast<std::streamsize>(bytes.size())) {
            const char* at = bytes.data(); const char* end = at + count;
            const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
            const auto skip = [&] { while (at != end && space(*at)) ++at; };
            skip();
            uint64_t sequence = 0; uint32_t requested_ordinal = 0;
            const auto sequence_parse = std::from_chars(at, end, sequence);
            bool valid = sequence_parse.ec == std::errc {} && sequence != 0;
            at = sequence_parse.ptr;
            valid = valid && (at == end || space(*at));
            skip();
            if (valid && at != end) {
                const auto ordinal_parse = std::from_chars(at, end, requested_ordinal);
                valid = ordinal_parse.ec == std::errc {} && requested_ordinal >= 1 && requested_ordinal <= 5;
                at = ordinal_parse.ptr; skip();
            }
            valid = valid && at == end;
            if (valid && sequence > state.seen_sequence) {
                state.seen_sequence = sequence; state.pending_sequence = sequence;
                state.requested_ordinal = requested_ordinal;
                std::printf("MENU_TEST_GENERATED: requested sequence=%llu ordinal=%u (0=next committed)\n",
                    static_cast<unsigned long long>(sequence), requested_ordinal);
                std::fflush(stdout);
            }
        }
    }
    if (!state.pending_sequence || (state.requested_ordinal && state.requested_ordinal != ordinal)) return;
    if (!generator || factor < 2 || factor > 6 || ordinal == 0 || ordinal >= factor) return;
    const auto scene = generator->GetMultipleSceneStatus();
    if (scene.state != MultiFrameSceneState::Clear || scene.flags != 0 || scene.recorded_tick == 0 ||
        !scheduler.IsFree(scene.recorded_tick) || frame.present_tick >= scheduler.CurrentTick() ||
        !scheduler.IsFree(frame.present_tick)) return;
    const auto sequence = std::exchange(state.pending_sequence, uint64_t {0});
    const auto width = frame.image.extent.width, height = frame.image.extent.height;
    if (frame.image.format != vk::Format::eR8G8B8A8Unorm || width == 0 || height == 0 ||
        width > 4096 || height > 4096 || frame.image.extent.depth != 1) {
        std::printf("MENU_TEST_GENERATED: rejected sequence=%llu format=%d size=%ux%u\n",
            static_cast<unsigned long long>(sequence), static_cast<int>(frame.image.format), width, height);
        std::fflush(stdout); return;
    }
    const uint64_t size = uint64_t {width} * height * 4;
    const auto path = folder / ("capture-generated-" + std::to_string(sequence) +
                               "-ordinal-" + std::to_string(ordinal) + ".ppm");
    // Exclusive creation occurs BEFORE allocation/recording. Existing evidence is
    // never opened for truncation, even after a process restart with the same sequence.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
    auto* raw_output = _wfopen(path.c_str(), L"wbx");
#else
    auto* raw_output = std::fopen(path.c_str(), "wbx");
#endif
    if (!raw_output) {
        std::printf("MENU_TEST_GENERATED: exclusive open failed sequence=%llu path=%s\n",
            static_cast<unsigned long long>(sequence), path.string().c_str());
        std::fflush(stdout); return;
    }
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> output(raw_output, &std::fclose);
    Buffer download(window.graphic_ctx, scheduler, MemoryUsage::Download, 0,
                    vk::BufferUsageFlagBits::eTransferDst, size);
    // The worker reaches this committed-ticket boundary with no live command:
    // BeginMulti submitted/waited its generation; the preceding Present also submitted.
    // BeginCommand enforces that precondition instead of implicitly flushing any work.
    auto& command = scheduler.BeginCommand();
    frame.Transit(command.Handle(), vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
    vk::BufferImageCopy copy {};
    copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copy.imageExtent = {width, height, 1}; // rowLength/imageHeight0: tight RGBA8.
    command.Handle().copyImageToBuffer(frame.image.image, vk::ImageLayout::eTransferSrcOptimal,
                                      download.Handle(), copy);
    vk::BufferMemoryBarrier barrier {};
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = download.Handle(); barrier.size = size;
    command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &barrier, 0, nullptr);
    const auto copy_tick = scheduler.Submit();
    frame.present_tick = MultiFramePresentPolicy::RetainTick(frame.present_tick, copy_tick);
    scheduler.Wait(copy_tick); // Submitted tick only. Buffer/source survive through completion.
    download.Invalidate(0, size);
    const auto bytes = download.Mapped();
    if (bytes.size() < size) {
        std::printf("MENU_TEST_GENERATED: mapped download too small sequence=%llu\n",
            static_cast<unsigned long long>(sequence));
        std::fflush(stdout); return; // GPU copy completed; retain the incomplete evidence file.
    }
    bool okay = std::fprintf(output.get(),
        "P6\n# MENU_TEST_GENERATED sequence=%llu ordinal=%u factor=%u epoch=%llu scene_flags=0 generation_tick=%llu copy_tick=%llu source=multi_generated_pre_present scanout=unknown\n%u %u\n255\n",
        static_cast<unsigned long long>(sequence), ordinal, factor,
        static_cast<unsigned long long>(epoch), static_cast<unsigned long long>(scene.recorded_tick),
        static_cast<unsigned long long>(copy_tick), width, height) > 0;
    std::vector<uint8_t> row(size_t(width) * 3);
    for (uint32_t y = 0; okay && y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const auto* source = bytes.data() + (uint64_t {y} * width + x) * 4;
            row[size_t(x) * 3] = source[0]; row[size_t(x) * 3 + 1] = source[1]; row[size_t(x) * 3 + 2] = source[2];
        }
        okay = std::fwrite(row.data(), 1, row.size(), output.get()) == row.size();
    }
    okay = std::fflush(output.get()) == 0 && okay;
    okay = std::fclose(output.release()) == 0 && okay;
    std::printf("MENU_TEST_GENERATED: capture sequence=%llu ordinal=%u factor=%u scene_flags=0 generation_tick=%llu copy_tick=%llu size=%ux%u ok=%u source=multi_generated_pre_present path=%s\n",
        static_cast<unsigned long long>(sequence), ordinal, factor,
        static_cast<unsigned long long>(scene.recorded_tick), static_cast<unsigned long long>(copy_tick),
        width, height, okay ? 1u : 0u, path.string().c_str());
    std::fflush(stdout);
}

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost, PresentedRecreate };

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	void                 Create();
	void                 Recreate(bool surface_lost = false);
	[[nodiscard]] bool   NeedsResize() const;
	[[nodiscard]] Status AcquireNextImage(CommandScheduler& scheduler);
	[[nodiscard]] bool   PrepareSystemOverlay();
	void     RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
	                               const Presenter::Layer& overlay, bool draw_system_overlay);
	uint64_t Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }
	[[nodiscard]] bool IsHdrSupported() const noexcept { return m_hdr_supported.load(); }
	bool SetHdr(bool hdr) {
		const bool changed = m_hdr_active != (hdr && IsHdrSupported());
		m_hdr_requested = hdr;
		return changed;
	}

private:
	vk::SurfaceFormatKHR SelectFormat();
	void Destroy();
	void DrawLayer(vk::CommandBuffer command, const Presenter::Layer& layer,
	               const vk::Rect2D& rect);

	WindowContext&   m_window;
	vk::SwapchainKHR m_handle = nullptr;
	vk::Format       m_format = vk::Format::eUndefined;
	std::atomic_bool m_hdr_supported {false};
	bool             m_hdr_requested = false;
	bool             m_hdr_active    = false;
	vk::Extent2D     m_extent {};
	// Drawable pixel size observed when this swapchain was created.
	vk::Extent2D                   m_window_extent {};
	std::vector<vk::Image>         m_images;
	std::vector<vk::ImageView>     m_image_views;
	std::vector<vk::Semaphore>     m_image_acquired;
	std::vector<uint64_t>          m_image_acquired_ticks;
	std::vector<vk::Semaphore>     m_render_complete;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	vk::DescriptorSetLayout        m_overlay_descriptors = nullptr;
	vk::PipelineLayout             m_overlay_layout      = nullptr;
	vk::Pipeline                   m_overlay_pipeline    = nullptr;
	vk::Sampler                    m_overlay_sampler     = nullptr;
	uint32_t                       m_image_index         = 0;
	uint32_t                       m_frame_index         = 0;
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
		swapchain.Create();
		frames.Initialize(swapchain.ImageCount(), swapchain.Format());
		FrameGeneration::SetBackend(Config::FrameGenerationBackend() == "multi"
		    ? FrameGeneration::Backend::Multi : FrameGeneration::Backend::Fsr);
		FrameGeneration::SetMultiplier(Config::FrameGenerationMultiplier());
		const char* env = std::getenv("KYTY_FRAME_GENERATION");
		FrameGeneration::SetEnabled(Config::FrameGenerationEnabled() ||
		                            (env != nullptr && env[0] == '1'));
	}
	~Impl() {
		StopAsync();
		frame_generator.reset();
		present_scheduler.Wait(present_scheduler.CurrentTick() - 1);
		for (auto& set: multi_outputs) for (auto& frame: set) {
			window.graphic_ctx.device.destroyImageView(frame.view, nullptr);
			if (frame.image.image) window.graphic_ctx.DeleteImage(frame.image);
		}
		if (generated.image.image != nullptr) {
			present_scheduler.Wait(present_scheduler.CurrentTick() - 1);
			window.graphic_ctx.device.destroyImageView(generated.view, nullptr);
			window.graphic_ctx.DeleteImage(generated.image);
		}
	}

	// Shows a generated frame between the previous game frame and `real`, and keeps `real` for
	// a later vblank. Returns false when the frame should be shown right away instead.
	bool PresentWithGeneratedFrame(Frame* real) {
		using namespace std::chrono_literals;
		if (real->blank) {
			// Not a game frame: it neither samples the game frame interval nor keeps a pending real.
			if (frame_generator != nullptr) {
				frame_generator->Reset();
			}
			pending_real = false;
			return false;
		}
		const auto now = std::chrono::steady_clock::now();
		if (last_real_frame.time_since_epoch().count() != 0) {
			const auto interval = now - last_real_frame;
			real_interval = real_interval.count() == 0 || interval > real_interval * 3
			                    ? std::chrono::duration_cast<std::chrono::nanoseconds>(interval)
			                    : (real_interval * 7 + interval) / 8;
		}
		last_real_frame = now;
		const auto vblank =
		    std::chrono::nanoseconds(1'000'000'000 / std::max(Config::GetVblankFrequency(), 1u));
		const auto format           = real->image.format;
		const bool generated_format = format == vk::Format::eR8G8B8A8Unorm ||
		                              format == vk::Format::eB8G8R8A8Unorm ||
		                              format == vk::Format::eA2R10G10B10UnormPack32 ||
		                              format == vk::Format::eA2B10G10R10UnormPack32;
		// A game already at the display rate gains nothing, and a slide show is no animation.
		const bool wanted = FrameGeneration::IsEnabled() && !real->blank && !real->hdr &&
		                    generated_format && layers[1].frame == nullptr &&
		                    real_interval >= vblank * 9 / 5 && real_interval <= 150ms;
		if (FrameGenerationDiag::Enabled()) fg_diag.NoteRequest(FrameGeneration::IsEnabled(), wanted);
		if (!wanted) {
			if (frame_generator != nullptr) {
				frame_generator->Reset();
			}
			pending_real = false;
			return false;
		}
		if (frame_generator == nullptr) {
			frame_generator =
			    std::make_unique<FrameGenerator>(window.graphic_ctx, present_scheduler);
		}
		if (generated.present_tick != 0) {
			present_scheduler.Wait(generated.present_tick);
		}
		const vk::Extent2D extent {real->image.extent.width, real->image.extent.height};
		generated.Configure(window.graphic_ctx, extent, format);
		auto& command = present_scheduler.BeginCommand();
		real->Transit(command.Handle(), vk::ImageLayout::eGeneral,
		              vk::AccessFlagBits2::eTransferRead);
		const bool ok =
		    frame_generator->Generate(command.Handle(), real->image.image, generated.image.image,
			                          extent.width, extent.height);
		real->image.state = {vk::PipelineStageFlagBits2::eAllCommands,
		                     vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
		                     vk::ImageLayout::eGeneral};
		// Generate may write the output and still decline it (first frame after a reset).
		generated.image.state = {
		    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
		    frame_generator->OutputWritten() ? vk::ImageLayout::eTransferSrcOptimal
		                                     : generated.image.state.layout};
		// This submission reads `real` and writes `generated`: neither may be reused (pool reuse
		// and the next Configure wait on present_tick) before it completes, whatever happens next.
		const auto generation_tick = present_scheduler.Submit();
		real->present_tick      = FrameGenerationPacing::RetainTick(real->present_tick, generation_tick);
		generated.present_tick  = FrameGenerationPacing::RetainTick(generated.present_tick, generation_tick);
		// A reset seed (output written, declined) is not a generation failure.
		if (FrameGenerationDiag::Enabled()) fg_diag.NoteGeneration(ok, frame_generator->OutputWritten());
		if (!ok) {
			pending_real = false;
			return false;
		}
		layers[0].frame = &generated;
		Present(FrameGenerationDiag::PresentKind::Generated);
		layers[0].frame = real;
		if (++generated_count % 300 == 1) {
			std::printf("Frame generation: %llu frames generated, game frame every %.1f ms\n",
			            static_cast<unsigned long long>(generated_count),
			            std::chrono::duration<double, std::milli>(real_interval).count());
		}
		pending_real = true;
		real_due     = now + FrameGenerationPacing::PendingRealDelay(real_interval, vblank);
		return true;
	}

	// Shows the game frame a generated frame went ahead of, once its time has come.
	bool PresentPendingRealFrame() {
		if (!pending_real || std::chrono::steady_clock::now() < real_due) {
			return false;
		}
		pending_real = false;
		Present(FrameGenerationDiag::PresentKind::DelayedReal);
		return true;
	}

	void RecoverSwapchain(Swapchain::Status status) {
		if (async_owner.load(std::memory_order_acquire)) {
			async_epoch.fetch_add(1, std::memory_order_release);
			const auto cancelled = multi_cadence.Cancel();
			multi_superseded += cancelled.generated;
			force_real_after_recovery |= cancelled.real != 0;
			active_output_set = -1;
			if (frame_generator) frame_generator->Reset();
		}
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		Common::DebugCounters::Add(Common::DebugCounters::Counter::SwapchainRecreations);
		if (FrameGenerationDiag::Enabled()) fg_diag.NoteRecreate();
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		frames.SetFormat(swapchain.Format());
	}

	Image& ResolveSurface(const ImageInfo& info) {
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;

		auto&      cache      = renderer.GetTextureCache();
		const auto image_id   = cache.FindImage(desc);
		auto&      image      = cache.GetImage(image_id);
		image.usage.video_out = true;
		cache.UpdateImage(image_id);
		return image;
	}
	// `kind` is the calling site's identity, used only by the default-off KYTY_FG_DIAG statistics.
	bool Present(FrameGenerationDiag::PresentKind kind = FrameGenerationDiag::PresentKind::Real);
	static uint64_t AsyncNow() {
		return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
		    std::chrono::steady_clock::now().time_since_epoch()).count());
	}
	struct Job { std::array<Layer, 2> layers {}; size_t count = 0; uint64_t arrived = 0; };
	bool UseAsync();
	bool IsPooled(Frame* frame) const;
	void SetLastShown(Frame* frame);
	void DropJob(Job job);
	void Enqueue(std::span<const Layer> incoming);
	void RequestRepeat();
	void RequestClear(int bus);
	void StopAsync();
	void CancelMulti(bool show_real);
	void ProcessJob(Job job);
	bool BeginMulti(Frame* real, uint64_t arrived);
	void RunAsync();
	void ReportMulti(bool force = false);
	void ResetMultiWindow();

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	Common::Mutex         present_mutex;
	std::array<Layer, 2>  layers {};
	std::atomic<uint64_t> presented_overlay_revision {0};
	Frame                                 generated;
	std::unique_ptr<FrameGenerator>       frame_generator;
	std::chrono::steady_clock::time_point last_real_frame {};
	std::chrono::steady_clock::time_point real_due {};
	std::chrono::nanoseconds              real_interval {};
	bool                                  pending_real    = false;
	uint64_t                              generated_count = 0;
	FrameGenerationDiag::Stats            fg_diag;
	std::atomic_bool async_owner {false}, async_pending {false}, async_visible {false};
	std::atomic<uint64_t> async_epoch {1}, async_real_dropped {0};
	std::mutex async_mutex;
	std::condition_variable async_wake;
	std::thread async_worker;
	MultiFramePresentPolicy::Queue<Job> async_jobs;
	bool async_stopping = false, async_repeat = false;
	uint32_t async_clear_mask = 0, async_queue_peak = 0;
	Frame* last_shown = nullptr;
	std::array<std::array<Frame, MultiFramePresentPolicy::MaxGenerated>, MultiFramePresentPolicy::OutputSets> multi_outputs {};
	MultiFramePresentPolicy::Cadence multi_cadence;
	MenuGeneratedCapture menu_generated_capture; // Worker-owned; disabled without KYTY_MENU_TEST.
	int active_output_set = -1;
	uint64_t batch_overlay_revision = 0, last_multi_arrival = 0;
	std::chrono::nanoseconds multi_interval {};
	bool emitting_multi_generated = false, force_real_after_recovery = false;
	bool main_real_accounted = true;
	bool async_multi_mode = false, multi_window_enabled = false;
	uint32_t multi_window_factor = 2;
	MultiFramePresentPolicy::SuccessCounts multi_success;
	uint64_t multi_requested = 0, multi_eligible = 0, multi_batches = 0, multi_recorded = 0;
	uint64_t multi_seeded = 0, multi_failed = 0, multi_scene = 0, multi_late = 0, multi_superseded = 0;
	uint64_t multi_present_failed = 0, multi_report_start = 0, multi_last_present = 0, multi_sample_dropped = 0;
	std::array<uint64_t, 512> multi_samples {};
	uint32_t multi_sample_count = 0;
};

// One owner after activation. Mailbox locks protect only bounded CPU ownership transfers.
bool Presenter::Impl::UseAsync() {
    if (async_owner.load(std::memory_order_acquire)) return true;
    if (FrameGeneration::GetBackend() != FrameGeneration::Backend::Multi ||
        !FrameGeneration::IsAvailable()) return false;
    Common::LockGuard legacy(present_mutex);
    if (!async_owner.load(std::memory_order_relaxed)) {
        async_owner.store(true, std::memory_order_release);
        async_worker = std::thread([this] { RunAsync(); });
    }
    return true;
}

bool Presenter::Impl::IsPooled(Frame* frame) const {
    if (!frame || frame == &generated) return false;
    for (const auto& set: multi_outputs)
        for (const auto& output: set) if (frame == &output) return false;
    return true;
}
void Presenter::Impl::SetLastShown(Frame* frame) {
    if (last_shown == frame) {
        async_visible.store(frame != nullptr || layers[1].frame != nullptr, std::memory_order_release);
        return;
    }
    if (IsPooled(frame)) frames.Retain(frame);
    auto* previous = last_shown;
    last_shown = frame;
    if (IsPooled(previous)) frames.Release(previous);
    async_visible.store(frame != nullptr || layers[1].frame != nullptr, std::memory_order_release);
}
void Presenter::Impl::DropJob(Job job) {
    for (size_t i = 0; i < job.count; ++i) {
        if (job.layers[i].bus == 0 && job.layers[i].frame && !job.layers[i].frame->blank)
            async_real_dropped.fetch_add(1, std::memory_order_relaxed);
        if (job.layers[i].frame) frames.Release(job.layers[i].frame);
    }
}
void Presenter::Impl::Enqueue(std::span<const Layer> incoming) {
    // FlipQueue hands off only Ready groups, after its producer-GPU completion callback.
    // Deferred blank clears have no producer GPU work; the worker records them below.
    EXIT_IF(incoming.empty() || incoming.size() > layers.size());
    Job job;
    job.count = incoming.size();
    job.arrived = AsyncNow();
    for (size_t i = 0; i < incoming.size(); ++i) {
        EXIT_IF(incoming[i].bus < 0 || incoming[i].bus >= int(layers.size()));
        job.layers[i] = incoming[i];
        // One prepared lease may intentionally feed both buses. Each layer owns a reference.
        if (incoming[i].frame) for (size_t j = 0; j < i; ++j)
            if (incoming[j].frame == incoming[i].frame) { frames.Retain(incoming[i].frame); break; }
    }
    std::optional<Job> evicted;
    {
        std::lock_guard lock(async_mutex);
        evicted = async_jobs.Push(std::move(job));
        async_epoch.fetch_add(1, std::memory_order_release);
        async_pending.store(true, std::memory_order_release);
        async_queue_peak = std::max(async_queue_peak, async_jobs.Size());
    }
    if (evicted) DropJob(std::move(*evicted));
    async_wake.notify_one();
}
void Presenter::Impl::RequestRepeat() {
    { std::lock_guard lock(async_mutex); async_repeat = true; }
    async_wake.notify_one();
}
void Presenter::Impl::RequestClear(int bus) {
    std::array<Job, MultiFramePresentPolicy::InputCapacity> retained {};
    std::array<Job, MultiFramePresentPolicy::InputCapacity * 2> discarded {};
    size_t retained_count = 0, discarded_count = 0;
    {
        std::lock_guard lock(async_mutex);
        while (auto job = async_jobs.Pop()) {
            Job keep, remove;
            keep.arrived = remove.arrived = job->arrived;
            for (size_t i = 0; i < job->count; ++i) {
                if (job->layers[i].bus == bus) remove.layers[remove.count++] = job->layers[i];
                else keep.layers[keep.count++] = job->layers[i];
            }
            if (keep.count) retained[retained_count++] = std::move(keep);
            if (remove.count) discarded[discarded_count++] = std::move(remove);
        }
        for (size_t i = 0; i < retained_count; ++i)
            if (auto rejected = async_jobs.Push(std::move(retained[i])))
                discarded[discarded_count++] = std::move(*rejected);
        async_clear_mask |= 1u << bus;
        async_epoch.fetch_add(1, std::memory_order_release);
        async_pending.store(true, std::memory_order_release);
    }
    for (size_t i = 0; i < discarded_count; ++i) DropJob(std::move(discarded[i]));
    async_wake.notify_one();
}
void Presenter::Impl::StopAsync() {
    if (!async_worker.joinable()) return;
    {
        std::lock_guard lock(async_mutex);
        async_stopping = true;
        async_jobs.Close();
        async_epoch.fetch_add(1, std::memory_order_release);
    }
    async_wake.notify_all();
    // Never join while holding either the mailbox or the legacy presentation mutex.
    async_worker.join();
}
void Presenter::Impl::CancelMulti(bool show_real) {
    const auto cancelled = multi_cadence.Cancel();
    multi_superseded += cancelled.generated;
    active_output_set = -1;
    if (cancelled.real && layers[0].frame) {
        if (show_real) Present(FrameGenerationDiag::PresentKind::Real);
        else if (!layers[0].frame->blank && !main_real_accounted) {
            async_real_dropped.fetch_add(1, std::memory_order_relaxed);
            main_real_accounted = true;
        }
    }
}
void Presenter::Impl::ProcessJob(Job job) {
    bool main_updated = false;
    std::array<Frame*, 2> cleared {};
    size_t clear_count = 0;
    CommandBuffer* clear_command = nullptr;
    for (size_t i = 0; i < job.count; ++i) {
        auto* frame = job.layers[i].frame;
        if (frame && frame->clear_pending) {
            if (!clear_command) clear_command = &present_scheduler.BeginCommand();
            frame->Clear(*clear_command, frame->clear_color);
            frame->clear_pending = false;
            cleared[clear_count++] = frame;
        }
    }
    if (clear_count) {
        const auto tick = present_scheduler.Submit();
        for (size_t i = 0; i < clear_count; ++i)
            cleared[i]->present_tick = MultiFramePresentPolicy::RetainTick(cleared[i]->present_tick, tick);
    }
    for (size_t i = 0; i < job.count; ++i) {
        auto& old = layers[job.layers[i].bus];
        if (job.layers[i].bus == 0) {
            if (old.frame && !old.frame->blank && !main_real_accounted)
                async_real_dropped.fetch_add(1, std::memory_order_relaxed);
            main_real_accounted = false;
        }
        if (old.frame) frames.Release(old.frame);
        old = job.layers[i]; // Transfer the job's existing reference into the layer.
        main_updated |= old.bus == 0;
    }
    auto* real = layers[0].frame;
    if (!main_updated) {
        layers[0].frame = last_shown;
        Present(last_shown ? FrameGenerationDiag::PresentKind::Repeat : FrameGenerationDiag::PresentKind::Blank);
        layers[0].frame = real;
        return;
    }
    if (main_updated && FrameGeneration::GetBackend() == FrameGeneration::Backend::Multi) {
        if (BeginMulti(real, job.arrived)) return;
    } else if (main_updated && real && FrameGeneration::GetBackend() == FrameGeneration::Backend::Fsr) {
        // The default FSR algorithm remains the existing implementation. Do not overwrite a
        // generated image which is still the last successful source after a failed real present.
        if (last_shown != &generated && PresentWithGeneratedFrame(real)) return;
    }
    Present(real && !real->blank ? FrameGenerationDiag::PresentKind::Real
                                : FrameGenerationDiag::PresentKind::Blank);
}
bool Presenter::Impl::BeginMulti(Frame* real, uint64_t arrived) {
    using namespace std::chrono_literals;
    if (last_multi_arrival && arrived > last_multi_arrival) {
        const auto interval = std::chrono::nanoseconds(arrived - last_multi_arrival);
        multi_interval = multi_interval.count() == 0 || interval > multi_interval * 3
            ? interval : (multi_interval * 7 + interval) / 8;
    }
    last_multi_arrival = arrived;
    const auto factor = FrameGeneration::GetMultiplier();
    const auto format = real ? real->image.format : vk::Format::eUndefined;
    const bool supported = format == vk::Format::eR8G8B8A8Unorm ||
        format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eA2R10G10B10UnormPack32 ||
        format == vk::Format::eA2B10G10R10UnormPack32;
    const bool eligible = real && !real->blank && !real->hdr && supported &&
        layers[1].frame == nullptr && FrameGeneration::IsEnabled() && factor >= 2 && factor <= 6 &&
        multi_interval >= 1ms && multi_interval <= 150ms;
    ++multi_requested;
    if (!eligible) { if (frame_generator) frame_generator->Reset(); return false; }
    ++multi_eligible;
    int set_index = -1;
    for (size_t set = 0; set < multi_outputs.size(); ++set) {
        bool shown = false;
        for (auto& frame: multi_outputs[set]) shown |= last_shown == &frame;
        if (!shown) { set_index = int(set); break; }
    }
    if (set_index < 0) { ++multi_failed; return false; }
    auto& outputs = multi_outputs[set_index];
    const vk::Extent2D extent {real->image.extent.width, real->image.extent.height};
    std::array<vk::Image, MultiFramePresentPolicy::MaxGenerated> handles {};
    for (uint32_t i = 0; i < factor - 1; ++i) {
        present_scheduler.Wait(outputs[i].present_tick);
        if (!outputs[i].TryConfigure(window.graphic_ctx, extent, vk::Format::eR8G8B8A8Unorm)) {
            ++multi_failed; if (frame_generator) frame_generator->Reset(); return false;
        }
        handles[i] = outputs[i].image.image;
    }
    if (!frame_generator) frame_generator = std::make_unique<FrameGenerator>(window.graphic_ctx, present_scheduler);
    auto& command = present_scheduler.BeginCommand();
    real->Transit(command.Handle(), vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eTransferRead);
    for (uint32_t i = 0; i < factor - 1; ++i)
        outputs[i].Transit(command.Handle(), vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    const auto result = frame_generator->GenerateMultiple(command.Handle(), real->image.image,
        std::span<const vk::Image>(handles.data(), factor - 1), extent.width, extent.height, factor);
    real->image.state = {vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eGeneral};
    const auto generation_tick = present_scheduler.Submit();
    real->present_tick = MultiFramePresentPolicy::RetainTick(real->present_tick, generation_tick);
    for (uint32_t i = 0; i < factor - 1; ++i) {
        outputs[i].present_tick = MultiFramePresentPolicy::RetainTick(outputs[i].present_tick, generation_tick);
        if (result.written_mask & (1u << i)) outputs[i].image.state = {
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
            vk::ImageLayout::eTransferSrcOptimal};
    }
    // No mailbox, renderer or queue mutex is held during this wait.
    present_scheduler.Wait(generation_tick);
    if (result.status == MultiFrameStatus::Seeded) { ++multi_seeded; return false; }
    if (result.status != MultiFrameStatus::Generated || result.count != factor - 1 ||
        result.written_mask != (1u << (factor - 1)) - 1u) { ++multi_failed; return false; }
    ++multi_batches; multi_recorded += result.count;
    if (frame_generator->MultipleSceneChanged()) { ++multi_scene; return false; }
    active_output_set = set_index;
    batch_overlay_revision = GetSystemOverlayVisualState().revision;
    const auto now = AsyncNow();
    if (!multi_cadence.Begin(factor, now, uint64_t(multi_interval.count()),
                              async_epoch.load(std::memory_order_acquire))) {
        active_output_set = -1; ++multi_failed; return false;
    }
    async_pending.store(true, std::memory_order_release);
    return true;
}

void Presenter::Impl::RunAsync() {
    if (frame_generator) frame_generator->Reset();
    auto backend = FrameGeneration::GetBackend();
    auto factor = FrameGeneration::GetMultiplier();
    auto enabled = FrameGeneration::IsEnabled();
    async_multi_mode = backend == FrameGeneration::Backend::Multi;
    multi_window_factor = factor; multi_window_enabled = enabled;
    ResetMultiWindow();
    while (true) {
        uint32_t clear_mask = 0;
        bool stopping = false;
        {
            std::lock_guard lock(async_mutex);
            stopping = async_stopping;
            clear_mask = std::exchange(async_clear_mask, 0u);
        }
        if (stopping) break;
        const auto next_backend = FrameGeneration::GetBackend();
        const auto next_factor = FrameGeneration::GetMultiplier();
        const auto next_enabled = FrameGeneration::IsEnabled();
        if (next_backend != backend || next_factor != factor || next_enabled != enabled) {
            const bool previous_multi_mode = async_multi_mode;
            async_epoch.fetch_add(1, std::memory_order_release);
            CancelMulti(true);
            if (pending_real) { pending_real = false; Present(FrameGenerationDiag::PresentKind::DelayedReal); }
            if (frame_generator) frame_generator->Reset();
            ReportMulti(true); // Close the old settings window explicitly, before changing its labels.
            backend = next_backend; factor = next_factor; enabled = next_enabled;
            async_multi_mode = backend == FrameGeneration::Backend::Multi;
            multi_window_factor = factor; multi_window_enabled = enabled;
            if (!previous_multi_mode) async_real_dropped.exchange(0, std::memory_order_relaxed);
            ResetMultiWindow();
            last_multi_arrival = 0; multi_interval = {};
        }
        if (clear_mask) {
            const auto cancelled = multi_cadence.Cancel();
            multi_superseded += cancelled.generated;
            active_output_set = -1; pending_real = false;
            if (frame_generator) frame_generator->Reset();
            for (size_t bus = 0; bus < layers.size(); ++bus) if (clear_mask & (1u << bus)) {
                if (bus == 0) {
                    if (layers[0].frame && !layers[0].frame->blank && !main_real_accounted)
                        async_real_dropped.fetch_add(1, std::memory_order_relaxed);
                    main_real_accounted = true;
                    SetLastShown(nullptr);
                }
                if (layers[bus].frame) frames.Release(layers[bus].frame);
                layers[bus] = {};
            }
            if (cancelled.real && !(clear_mask & 1u) && layers[0].frame)
                Present(FrameGenerationDiag::PresentKind::Real);
            async_visible.store(last_shown != nullptr || layers[1].frame != nullptr, std::memory_order_release);
        }
        bool newer = false;
        {
            std::lock_guard lock(async_mutex);
            newer = async_jobs.Size() != 0;
        }
        if (multi_cadence.Pending()) {
            if (swapchain.NeedsResize() || GetSystemOverlayVisualState().revision != batch_overlay_revision) {
                async_epoch.fetch_add(1, std::memory_order_release);
                if (frame_generator) frame_generator->Reset();
            }
            const auto before = multi_cadence.GetCounts();
            const auto ticket = multi_cadence.Take(AsyncNow(), async_epoch.load(std::memory_order_acquire), newer);
            const auto after = multi_cadence.GetCounts();
            multi_late += after.late - before.late;
            multi_superseded += after.superseded - before.superseded;
            if (ticket.kind == MultiFramePresentPolicy::Cadence::Kind::Generated) {
                // This is the commit boundary: a subsequent mailbox update cancels later tickets,
                // while this already committed GPU/WSI use retains its own image through its tick.
                if (!MultiFramePresentPolicy::Cadence::Current(ticket, async_epoch.load(std::memory_order_acquire))) {
                    ++multi_superseded; CancelMulti(true); continue;
                }
                auto* real = layers[0].frame;
                layers[0].frame = &multi_outputs[active_output_set][ticket.index];
                emitting_multi_generated = true;
                if (!menu_generated_capture.folder.empty())
                    CaptureMenuTestGeneratedFrame(window, present_scheduler, frame_generator.get(),
                        *layers[0].frame, ticket.index + 1, factor, ticket.epoch, menu_generated_capture);
                Present(FrameGenerationDiag::PresentKind::Generated);
                emitting_multi_generated = false;
                layers[0].frame = real;
            } else if (ticket.kind == MultiFramePresentPolicy::Cadence::Kind::Real) {
                active_output_set = -1;
                Present(FrameGenerationDiag::PresentKind::Real);
            }
            if (ticket.kind != MultiFramePresentPolicy::Cadence::Kind::None) {
                if (force_real_after_recovery) { force_real_after_recovery = false; Present(FrameGenerationDiag::PresentKind::Real); }
                ReportMulti(); continue;
            }
        } else if (pending_real && (newer || std::chrono::steady_clock::now() >= real_due)) {
            pending_real = false; Present(FrameGenerationDiag::PresentKind::DelayedReal); continue;
        }
        if (!multi_cadence.Pending() && !pending_real) {
            std::optional<Job> job;
            bool repeat = false;
            {
                std::lock_guard lock(async_mutex);
                job = async_jobs.Pop();
                if (!job) repeat = std::exchange(async_repeat, false);
            }
            if (job) { ProcessJob(std::move(*job)); ReportMulti(); continue; }
            if (repeat && (last_shown || layers[1].frame)) {
                auto* real = layers[0].frame;
                layers[0].frame = last_shown;
                Present(FrameGenerationDiag::PresentKind::Repeat);
                layers[0].frame = real;
                ReportMulti(); continue;
            }
        }
        ReportMulti();
        std::unique_lock lock(async_mutex);
        const bool pending = multi_cadence.Pending() || pending_real || async_jobs.Size() != 0;
        async_pending.store(pending, std::memory_order_release);
        auto due = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
        if (multi_cadence.Pending()) due = std::min(due, std::chrono::steady_clock::time_point(std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::nanoseconds(multi_cadence.NextDue()))));
        else if (pending_real) due = std::min(due, real_due);
        // condition_variable releases the mailbox lock while waiting.
        async_wake.wait_until(lock, due);
    }
    CancelMulti(false); pending_real = false;
    while (true) {
        std::optional<Job> job;
        { std::lock_guard lock(async_mutex); job = async_jobs.Pop(); }
        if (!job) break;
        DropJob(std::move(*job));
    }
    if (layers[0].frame && !layers[0].frame->blank && !main_real_accounted)
        async_real_dropped.fetch_add(1, std::memory_order_relaxed);
    main_real_accounted = true;
    for (auto& layer: layers) { if (layer.frame) frames.Release(layer.frame); layer = {}; }
    SetLastShown(nullptr);
    present_scheduler.Finish();
    ReportMulti(true);
    async_pending.store(false, std::memory_order_release);
}
void Presenter::Impl::ResetMultiWindow() {
    multi_requested = multi_eligible = multi_batches = multi_recorded = multi_seeded = multi_failed = multi_scene = 0;
    multi_late = multi_superseded = multi_present_failed = multi_sample_dropped = 0;
    multi_success = {}; multi_sample_count = 0; multi_last_present = 0;
    multi_report_start = AsyncNow();
}
void Presenter::Impl::ReportMulti(bool force) {
    if (!FrameGenerationDiag::Enabled() || !async_multi_mode) return;
    const auto now = AsyncNow();
    if (!multi_report_start) { multi_report_start = now; return; }
    const auto duration = now - multi_report_start;
    if (!force && duration < 2'000'000'000ull) return;
    uint32_t depth = 0, peak = 0;
    { std::lock_guard lock(async_mutex); depth = async_jobs.Size(); peak = std::exchange(async_queue_peak, depth); }
    std::sort(multi_samples.begin(), multi_samples.begin() + multi_sample_count);
    const auto percentile = [&](uint32_t pct) -> uint64_t {
        return multi_sample_count ? multi_samples[(uint64_t(multi_sample_count - 1) * pct + 99) / 100] / 1000 : 0;
    };
    const auto dropped = async_real_dropped.exchange(0, std::memory_order_relaxed);
    std::printf("FG_DIAG v=2 window_ms=%llu window_ns=%llu partial=%u enabled=%u requested_backend=multi requested_factor=%u requested=%llu eligible=%llu batches_generated=%llu outputs_recorded=%llu generated_presented=%llu real_presented=%llu repeat_presented=%llu blank_presented=%llu seeded=%llu failed=%llu scene=%llu generated_late=%llu generated_superseded=%llu real_dropped=%llu present_failed=%llu queue_depth=%u queue_peak=%u queue_capacity=%u effective_num=%llu effective_den=%llu pending=%u sample_n=%u sample_dropped=%llu p50_us=%llu p90_us=%llu p99_us=%llu max_us=%llu scanout=unknown\n",
        (unsigned long long)(duration / 1'000'000), (unsigned long long)duration, force ? 1u : 0u,
        multi_window_enabled ? 1u : 0u, multi_window_factor,
        (unsigned long long)multi_requested, (unsigned long long)multi_eligible,
        (unsigned long long)multi_batches, (unsigned long long)multi_recorded,
        (unsigned long long)multi_success.generated, (unsigned long long)multi_success.real,
        (unsigned long long)multi_success.repeat, (unsigned long long)multi_success.blank,
        (unsigned long long)multi_seeded, (unsigned long long)multi_failed, (unsigned long long)multi_scene,
        (unsigned long long)multi_late, (unsigned long long)multi_superseded, (unsigned long long)dropped,
        (unsigned long long)multi_present_failed, depth, peak, MultiFramePresentPolicy::InputCapacity,
        (unsigned long long)(multi_success.real + multi_success.generated), (unsigned long long)multi_success.real,
        multi_cadence.Pending() ? 1u : 0u, multi_sample_count, (unsigned long long)multi_sample_dropped,
        (unsigned long long)percentile(50), (unsigned long long)percentile(90), (unsigned long long)percentile(99),
        (unsigned long long)(multi_sample_count ? multi_samples[multi_sample_count - 1] / 1000 : 0));
    multi_requested = multi_eligible = multi_batches = multi_recorded = multi_seeded = multi_failed = multi_scene = 0;
    multi_late = multi_superseded = multi_present_failed = multi_sample_dropped = 0;
    multi_success = {}; multi_sample_count = 0; multi_report_start = now;
}

vk::SurfaceFormatKHR Swapchain::SelectFormat() {
	const auto& formats    = m_window.surface_capabilities.formats;
	auto&       graphics   = m_window.graphic_ctx;
	const auto  hdr_format = std::ranges::find_if(formats, [&](const auto& candidate) {
		return candidate.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT &&
		       IsHdrSwapchainFormat(candidate.format) &&
		       static_cast<bool>(
		           graphics.GetFormatProperties(candidate.format).optimalTilingFeatures &
		           vk::FormatFeatureFlagBits::eBlitDst);
	});
	m_hdr_supported        = Config::HdrEnabled() && hdr_format != formats.end();
	m_hdr_active           = m_hdr_requested && IsHdrSupported();
	if (m_hdr_active) {
		return *hdr_format;
	}
	if (Config::HdrEnabled() && !IsHdrSupported()) {
		LOGF("HDR10 surface unavailable; using SDR output\n");
	}
	if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined) {
		return {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	}
	for (const auto& candidate: formats) {
		if (candidate.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
			continue;
		}
		switch (candidate.format) {
			case vk::Format::eB8G8R8A8Unorm:
			case vk::Format::eR8G8B8A8Unorm: return candidate;
			default: break;
		}
	}
	EXIT("no supported UNORM swapchain format\n");
}

void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	// Capture the size before querying surface capabilities. If the window changes
	// during creation, the next presentation will detect the newer size.
	{
		Common::LockGuard lock(m_window.mutex);
		m_window_extent = {graphics.screen_width, graphics.screen_height};
	}
	EXIT_IF(m_window_extent.width == 0);
	EXIT_IF(m_window_extent.height == 0);
	m_window.RefreshSurfaceCapabilities();
	const auto& surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		m_extent.width =
		    std::clamp(m_window_extent.width, surface.capabilities.minImageExtent.width,
		               surface.capabilities.maxImageExtent.width);
		m_extent.height =
		    std::clamp(m_window_extent.height, surface.capabilities.minImageExtent.height,
		               surface.capabilities.maxImageExtent.height);
	}
	uint32_t image_count = surface.capabilities.minImageCount + 1;
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	const auto transform =
	    surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
	        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
	        : surface.capabilities.currentTransform;
	const auto composite =
	    surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
	        ? vk::CompositeAlphaFlagBitsKHR::eOpaque
	        : vk::CompositeAlphaFlagBitsKHR::eInherit;

	const auto format = SelectFormat();
	m_format                      = format.format;
	const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
	if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
		EXIT("swapchain format cannot be a blit destination: format=%d\n",
		     static_cast<int>(m_format));
	}

	vk::SwapchainCreateInfoKHR create_info {};
	create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
	create_info.surface          = m_window.surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = format.format;
	create_info.imageColorSpace  = format.colorSpace;
	create_info.imageExtent      = m_extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	create_info.imageSharingMode = vk::SharingMode::eExclusive;
	create_info.preTransform     = transform;
	create_info.compositeAlpha   = composite;
	switch (Config::GetPresentMode()) {
		case Config::PresentMode::Mailbox:
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
			break;
		case Config::PresentMode::Immediate:
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
			break;
		case Config::PresentMode::Fifo:
		default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
	}
	if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
	              create_info.presentMode) == surface.present_modes.end()) {
		LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
		create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	create_info.clipped = VK_TRUE;
	RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
	                     "vkCreateSwapchainKHR");
	EXIT_IF(m_handle == nullptr);
	LOGF("Vulkan presentation: %s\n", m_hdr_active ? "HDR10 (PQ)" : "SDR");

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return graphics.device.getSwapchainImagesKHR(m_handle, count, images);
	    });
	EXIT_NOT_IMPLEMENTED(m_images.empty());

	m_image_views.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		vk::ImageViewCreateInfo view {};
		view.sType                           = vk::StructureType::eImageViewCreateInfo;
		view.image                           = m_images[i];
		view.viewType                        = vk::ImageViewType::e2D;
		view.format                          = m_format;
		view.components                      = {};
		view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.baseMipLevel   = 0;
		view.subresourceRange.layerCount     = 1;
		view.subresourceRange.levelCount     = 1;
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
		                     "vkCreateImageView");
		EXIT_IF(m_image_views[i] == nullptr);
	}

	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
	m_image_acquired.resize(m_images.size());
	m_image_acquired_ticks.assign(m_images.size(), 0);
	m_render_complete.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
		    "create swapchain image-acquired semaphore");
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
		    "create swapchain render-complete semaphore");
	}
}

Swapchain::~Swapchain() {
	Destroy();
}

void Swapchain::Destroy() {
	if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
	    m_image_views.empty()) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;

	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}
	graphics.device.destroyPipeline(m_overlay_pipeline, nullptr);
	graphics.device.destroyPipelineLayout(m_overlay_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_overlay_descriptors, nullptr);
	graphics.device.destroySampler(m_overlay_sampler, nullptr);
	m_overlay_pipeline    = nullptr;
	m_overlay_layout      = nullptr;
	m_overlay_descriptors = nullptr;
	m_overlay_sampler     = nullptr;

	for (const auto semaphore: m_image_acquired) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto semaphore: m_render_complete) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto view: m_image_views) {
		if (view != nullptr) {
			graphics.device.destroyImageView(view, nullptr);
		}
	}
	if (m_handle != nullptr) {
		graphics.device.destroySwapchainKHR(m_handle, nullptr);
	}

	m_handle        = nullptr;
	m_format        = vk::Format::eUndefined;
	m_extent        = {};
	m_window_extent = {};
	m_frame_index   = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_image_acquired_ticks.clear();
	m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		EXIT_IF(!SDL_RunOnMainThread(
		    [](void* window) { static_cast<WindowContext*>(window)->RecreateSurface(); }, &m_window,
		    true));
#else
		m_window.RecreateSurface();
#endif
	}
	Create();
}

bool Swapchain::NeedsResize() const {
	Common::LockGuard lock(m_window.mutex);
	return m_window_extent.width != m_window.graphic_ctx.screen_width ||
	       m_window_extent.height != m_window.graphic_ctx.screen_height;
}

Swapchain::Status Swapchain::AcquireNextImage(CommandScheduler& scheduler) {
	// Repeated frames can wrap the semaphore ring without acquiring a new source frame.
	// Wait for the submission that consumed this slot before signaling it again.
	scheduler.Wait(m_image_acquired_ticks[m_frame_index]);
	const auto result = m_window.graphic_ctx.device.acquireNextImageKHR(
	    m_handle, std::numeric_limits<uint64_t>::max(), m_image_acquired[m_frame_index], nullptr,
	    &m_image_index);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorUnknown:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
	}
	return Status::Success;
}

bool Swapchain::PrepareSystemOverlay() {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount());
}

static vk::Rect2D FitAspect(const vk::Extent3D& source, const vk::Extent2D& target) {
	const uint64_t src_w = std::max(source.width, 1u);
	const uint64_t src_h = std::max(source.height, 1u);
	uint32_t       w     = target.width;
	uint32_t       h     = target.height;
	if (src_w * h > src_h * w) {
		h = std::max<uint32_t>(1, static_cast<uint32_t>(src_h * w / src_w));
	} else {
		w = std::max<uint32_t>(1, static_cast<uint32_t>(src_w * h / src_h));
	}
	return {{static_cast<int32_t>((target.width - w) / 2),
	         static_cast<int32_t>((target.height - h) / 2)},
	        {w, h}};
}

void Swapchain::DrawLayer(vk::CommandBuffer command, const Presenter::Layer& layer,
                          const vk::Rect2D& rect) {
	auto device = m_window.graphic_ctx.device;
	if (m_overlay_pipeline == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_overlay_sampler),
		                     "create video-out overlay sampler");
		const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eCombinedImageSampler,
		                                              1, vk::ShaderStageFlagBits::eFragment};
		vk::DescriptorSetLayoutCreateInfo    descriptors {};
		descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptors.bindingCount = 1;
		descriptors.pBindings    = &binding;
		RequireVulkanSuccess(
		    device.createDescriptorSetLayout(&descriptors, nullptr, &m_overlay_descriptors),
		    "create video-out overlay descriptor layout");
		const vk::PushConstantRange alpha {vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_overlay_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &alpha;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_overlay_layout),
		                     "create video-out overlay pipeline layout");
		const auto vertex   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
		const auto fragment = CompileSPV(GPU_VIDEO_OUT_OVERLAY_SPV, device);
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
		stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
		stages[0].module = vertex;
		stages[0].pName  = "main";
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = fragment;
		stages[1].pName  = "main";
		vk::PipelineVertexInputStateCreateInfo   vertex_input {};
		vk::PipelineInputAssemblyStateCreateInfo assembly {};
		assembly.topology = vk::PrimitiveTopology::eTriangleList;
		vk::PipelineViewportStateCreateInfo viewport {};
		viewport.viewportCount = 1;
		viewport.scissorCount  = 1;
		vk::PipelineRasterizationStateCreateInfo rasterization {};
		rasterization.lineWidth = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisample {};
		multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
		vk::PipelineColorBlendAttachmentState attachment {};
		attachment.blendEnable         = VK_TRUE;
		attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
		attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
		                            vk::ColorComponentFlagBits::eG |
		                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		vk::PipelineColorBlendStateCreateInfo blend {};
		blend.attachmentCount = 1;
		blend.pAttachments    = &attachment;
		const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates    = dynamic_states.data();
		vk::PipelineRenderingCreateInfo rendering {};
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &m_format;
		vk::GraphicsPipelineCreateInfo create {};
		create.pNext               = &rendering;
		create.stageCount          = static_cast<uint32_t>(stages.size());
		create.pStages             = stages.data();
		create.pVertexInputState   = &vertex_input;
		create.pInputAssemblyState = &assembly;
		create.pViewportState      = &viewport;
		create.pRasterizationState = &rasterization;
		create.pMultisampleState   = &multisample;
		create.pColorBlendState    = &blend;
		create.pDynamicState       = &dynamic;
		create.layout              = m_overlay_layout;
		RequireVulkanSuccess(
		    device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &m_overlay_pipeline),
		    "create video-out overlay pipeline");
		device.destroyShaderModule(fragment, nullptr);
		device.destroyShaderModule(vertex, nullptr);
	}
	auto& frame = *layer.frame;
	if (frame.view == nullptr) {
		vk::ImageViewCreateInfo view {};
		view.image            = frame.image.image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = frame.image.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(device.createImageView(&view, nullptr, &frame.view),
		                     "create video-out overlay image view");
	}
	const vk::DescriptorImageInfo image {m_overlay_sampler, frame.view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_overlay_layout, 0, 1, &write);
	// The main bus ignores alpha.
	const uint32_t flags = (layer.premultiplied_alpha ? 1u : 0u) | (m_hdr_active ? 2u : 0u) |
	                       (frame.hdr ? 4u : 0u) |
	                       (frame.image.format == vk::Format::eR16G16B16A16Sfloat ? 8u : 0u) |
	                       (layer.bus == 0 ? 16u : 0u);
	command.pushConstants(m_overlay_layout, vk::ShaderStageFlagBits::eFragment, 0,
	                      sizeof(flags), &flags);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_overlay_pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = m_image_views[m_image_index];
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = m_extent;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {static_cast<float>(rect.offset.x),
	                             static_cast<float>(rect.offset.y),
	                             static_cast<float>(rect.extent.width),
	                             static_cast<float>(rect.extent.height),
	                             0.0f,
	                             1.0f};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &rect);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
                                      const Presenter::Layer& overlay, bool draw_system_overlay) {
	auto       vk_command      = command.Handle();
	const bool convert_source  = source != nullptr && source->hdr && !m_hdr_active;
	const bool blit_source     = source != nullptr && !convert_source;
	const bool draw_overlay    = overlay.frame != nullptr;
	const bool draw_attachment = convert_source || draw_overlay || draw_system_overlay;
	if (blit_source) {
		source->Transit(vk_command, vk::ImageLayout::eTransferSrcOptimal,
		                vk::AccessFlagBits2::eTransferRead);
	}
	if (convert_source) {
		source->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
		                vk::AccessFlagBits2::eShaderRead);
	}
	if (draw_overlay) {
		overlay.frame->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
		                       vk::AccessFlagBits2::eShaderRead);
	}

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask                   = vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
	to_transfer.newLayout                       = vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	// Match the acquire wait stage so the layout transition cannot precede acquisition.
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
	                           nullptr, 0, nullptr, 1, &to_transfer);

	const auto* aspect_source = source != nullptr ? source : overlay.frame;
	const auto  game_rect     = !Config::StretchEnabled() && aspect_source != nullptr
	                               ? FitAspect(aspect_source->image.extent, m_extent)
	                               : vk::Rect2D {{0, 0}, m_extent};
	if (source == nullptr || game_rect.extent != m_extent) {
		const vk::ClearColorValue black {};
		vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal,
		                           &black, 1, &to_transfer.subresourceRange);
		if (blit_source) {
			vk::ImageMemoryBarrier after_clear = to_transfer;
			after_clear.srcAccessMask          = vk::AccessFlagBits::eTransferWrite;
			after_clear.oldLayout              = vk::ImageLayout::eTransferDstOptimal;
			vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                           vk::PipelineStageFlagBits::eTransfer,
			                           vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 1,
			                           &after_clear);
		}
	}
	if (blit_source) {
		vk::ImageBlit region {};
		region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.srcSubresource.mipLevel       = 0;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.layerCount     = 1;
		region.srcOffsets[1].x               = static_cast<int>(source->image.extent.width);
		region.srcOffsets[1].y               = static_cast<int>(source->image.extent.height);
		region.srcOffsets[1].z               = 1;
		region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.dstSubresource.mipLevel       = 0;
		region.dstSubresource.baseArrayLayer = 0;
		region.dstSubresource.layerCount     = 1;
		region.dstOffsets[0].x = game_rect.offset.x;
		region.dstOffsets[0].y = game_rect.offset.y;
		region.dstOffsets[1].x = game_rect.offset.x + static_cast<int>(game_rect.extent.width);
		region.dstOffsets[1].y = game_rect.offset.y + static_cast<int>(game_rect.extent.height);
		region.dstOffsets[1].z               = 1;
		vk_command.blitImage(source->image.image, vk::ImageLayout::eTransferSrcOptimal,
		                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1,
		                     &region, vk::Filter::eLinear);
	}

	vk::ImageMemoryBarrier to_present {};
	to_present.sType               = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask       = draw_attachment ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                       vk::AccessFlagBits::eColorAttachmentWrite
	                                                 : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout           = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout           = draw_attachment ? vk::ImageLayout::eColorAttachmentOptimal
	                                                 : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.image               = m_images[m_image_index];
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           draw_attachment ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                                           : vk::PipelineStageFlagBits::eAllCommands,
	                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
	                           &to_present);
	if (draw_attachment) {
		// Each pass blends over the color writes of the previous one.
		to_present.srcAccessMask      = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.oldLayout          = vk::ImageLayout::eColorAttachmentOptimal;
		const auto attachment_barrier = [&] {
			vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
			                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
			                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
			                           &to_present);
		};
		if (convert_source) {
			DrawLayer(vk_command, {source, 0, false}, game_rect);
			if (draw_overlay || draw_system_overlay) {
				attachment_barrier();
			}
		}
		if (draw_overlay) {
			DrawLayer(vk_command, overlay, game_rect);
			if (draw_system_overlay) {
				attachment_barrier();
			}
		}
		if (draw_system_overlay) {
			m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
		}
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
	SubmitInfo submit;
	submit.AddWait(m_image_acquired[m_frame_index], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(m_render_complete[m_image_index]);
	const auto tick                       = scheduler.Submit(submit);
	m_image_acquired_ticks[m_frame_index] = tick;
	return tick;
}

Swapchain::Status Swapchain::Present() {
	const auto         ready = m_render_complete[m_image_index];
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &m_handle;
	present.pImageIndices      = &m_image_index;
	present.pWaitSemaphores    = &ready;
	present.waitSemaphoreCount = 1;

	vk::Result result;
	{
		Common::LockGuard lock(m_window.graphic_ctx.queue_mutex);
		result = m_window.graphic_ctx.queue.presentKHR(&present);
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::PresentedRecreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
	}
	m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
	return Status::Success;
}

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

Presenter::~Presenter() = default;

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info, bool hdr,
                                          bool headset) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid());
	auto frame_format = info.pixel_format;
	switch (frame_format) {
		case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
		case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
		default: break;
	}
	auto* frame = m_impl->frames.Acquire({info.extent.width, info.extent.height}, frame_format);
	frame->hdr   = hdr;
	frame->blank = false;
	Common::LockGuard render_lock(m_impl->renderer.GetMutex());
	auto&             image = m_impl->ResolveSurface(info);
	if (image.backing.format == vk::Format::eUndefined) {
		EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
	}
	frame->Configure(m_impl->window.graphic_ctx,
	                 {image.backing.extent.width, image.backing.extent.height}, frame_format);
	frame->CopyFrom(buffer, image);
	CaptureMenuTestFrame(buffer, *frame);
	if (headset) {
		// The frame holds the raw display-encoded pixels the headset needs.
		frame->Transit(buffer.Handle(), vk::ImageLayout::eTransferSrcOptimal,
		               vk::AccessFlagBits2::eTransferRead);
		Vr::OpenXrQueueEyeSource(buffer, frame->image);
	}
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	auto              format = m_impl->frames.GetFormat();
	auto*             frame  = m_impl->frames.Acquire({width, height}, format);
	frame->hdr   = false;
	frame->blank = true;
	frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	if (producer != nullptr) {
		EXIT_IF(producer->IsInvalid());
		frame->Clear(*producer, clear);
	} else if (m_impl->UseAsync()) {
		frame->clear_pending = true;
		frame->clear_color = clear;
	} else {
		Common::LockGuard lock(m_impl->present_mutex);
		if (m_impl->async_owner.load(std::memory_order_acquire)) {
			frame->clear_pending = true;
			frame->clear_color = clear;
		} else {
			auto& command = m_impl->present_scheduler.BeginCommand();
			frame->Clear(command, clear);
			frame->present_tick = m_impl->present_scheduler.Submit();
		}
	}
	return *frame;
}

bool Presenter::PresentLastFrame() {
	if (m_impl->UseAsync()) {
		const bool available = m_impl->async_visible.load(std::memory_order_acquire) ||
		    m_impl->async_pending.load(std::memory_order_acquire);
		if (available) m_impl->RequestRepeat();
		return available;
	}
	Common::LockGuard lock(m_impl->present_mutex);
	if (m_impl->async_owner.load(std::memory_order_acquire)) {
		m_impl->RequestRepeat(); return true;
	}
	if (m_impl->layers[0].frame == nullptr && m_impl->layers[1].frame == nullptr) {
		return false;
	}
	// layers[0] already holds a waiting real frame (PresentWithGeneratedFrame): this overlay
	// refresh shows it now, so it is that frame's one delayed-real present, not a repeat.
	auto kind = FrameGenerationDiag::PresentKind::Repeat;
	if (m_impl->pending_real) {
		m_impl->pending_real = false;
		kind                 = FrameGenerationDiag::PresentKind::DelayedReal;
	}
	m_impl->Present(kind);
	return true;
}

bool Presenter::IsHdrSupported() const noexcept {
	return m_impl->swapchain.IsHdrSupported();
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

void Presenter::Present(Frame& frame) {
	const Layer layer {&frame, 0, false};
	Present(std::span(&layer, 1));
}

void Presenter::Present(std::span<const Layer> layers) {
	if (m_impl->UseAsync()) { m_impl->Enqueue(layers); return; }
	Common::LockGuard lock(m_impl->present_mutex);
	if (m_impl->async_owner.load(std::memory_order_acquire)) { m_impl->Enqueue(layers); return; }
	bool main_updated = false;
	for (const auto& layer: layers) {
		EXIT_IF(layer.bus < 0 || layer.bus >= static_cast<int>(m_impl->layers.size()));
		main_updated |= layer.bus == 0;
		auto& previous = m_impl->layers[layer.bus];
		if (previous.frame != nullptr) {
			if (layer.bus == 0 && m_impl->pending_real) {
				// The waiting real frame is about to be released: show it first when it is due
				// (the flip was only handled first in this vblank); otherwise it is superseded.
				if (FrameGenerationPacing::FlushPendingReal(true, std::chrono::steady_clock::now(),
				                                            m_impl->real_due)) {
					m_impl->pending_real = false;
					m_impl->Present(FrameGenerationDiag::PresentKind::DelayedReal);
				} else if (FrameGenerationDiag::Enabled()) {
					m_impl->fg_diag.NoteSuperseded();
				}
			}
			m_impl->frames.Release(previous.frame);
		}
		previous = layer;
	}
	auto* main_frame = m_impl->layers[0].frame;
	// Only a new main-bus frame is a game frame; an overlay-only update must not sample the frame
	// interval, regenerate from the same frame or reset the generator.
	if (main_updated && main_frame != nullptr && m_impl->PresentWithGeneratedFrame(main_frame)) {
		return;
	}
	auto kind = main_frame == nullptr || main_frame->blank ? FrameGenerationDiag::PresentKind::Blank
	                                                       : FrameGenerationDiag::PresentKind::Real;
	if (!main_updated && m_impl->pending_real) {
		// This present shows the waiting real frame (with the new overlay) now.
		m_impl->pending_real = false;
		kind                 = FrameGenerationDiag::PresentKind::DelayedReal;
	}
	m_impl->Present(kind);
}

bool Presenter::PresentPendingFrame() {
	if (m_impl->UseAsync()) return m_impl->async_pending.load(std::memory_order_acquire);
	Common::LockGuard lock(m_impl->present_mutex);
	if (m_impl->async_owner.load(std::memory_order_acquire)) return m_impl->async_pending.load(std::memory_order_acquire);
	return m_impl->PresentPendingRealFrame();
}

void Presenter::ClearLayer(int bus) {
	if (static_cast<size_t>(bus) >= m_impl->layers.size()) {
		return;
	}
	if (m_impl->UseAsync()) { m_impl->RequestClear(bus); return; }
	Common::LockGuard lock(m_impl->present_mutex);
	if (m_impl->async_owner.load(std::memory_order_acquire)) { m_impl->RequestClear(bus); return; }
	auto& layer = m_impl->layers[bus];
	if (layer.frame != nullptr) {
		m_impl->frames.Release(layer.frame);
		layer = {};
	}
}

bool Presenter::Impl::Present(FrameGenerationDiag::PresentKind kind) {
	KYTY_PROFILER_FUNCTION();
	using Common::DebugCounters::Counter;
	using Common::DebugCounters::ScopedTimer;
	ScopedTimer present_time(Counter::PresentNs);

	const auto overlay_visual = GetSystemOverlayVisualState();
	// Keep black/blank flips in the current color space; only real main-bus frames switch it.
	const auto* main_frame = layers[0].frame;
	const bool hdr_changed = main_frame != nullptr && !main_frame->blank &&
	                         swapchain.SetHdr(main_frame->hdr);
	// Some window systems keep presenting an old swapchain after a resize.
	if (hdr_changed || swapchain.NeedsResize()) {
		RecoverSwapchain(Swapchain::Status::Recreate);
		if (emitting_multi_generated) return false;
	}
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		auto status = [&] {
			ScopedTimer acquire_time(Counter::PresentAcquireNs);
			return swapchain.AcquireNextImage(present_scheduler);
		}();
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			if (emitting_multi_generated) return false;
			continue;
		}
		// present_mutex owns the prepared frames and this scheduler. Only queue submission
		// is shared with rendering, and it has its own queue_mutex.
		auto&      command = present_scheduler.BeginCommand();
		const bool draw_system_overlay = [&] {
			ScopedTimer overlay_time(Counter::OverlayNs);
			return (overlay_visual.active || PerformanceOverlayEnabled()) &&
			       swapchain.PrepareSystemOverlay();
		}();
		swapchain.RecordPresentCommands(command, layers[0].frame, layers[1],
		                                draw_system_overlay);
		const auto tick = swapchain.Submit(present_scheduler);
		for (const auto& layer: layers) {
			if (layer.frame != nullptr) {
				layer.frame->present_tick = MultiFramePresentPolicy::RetainTick(layer.frame->present_tick, tick);
			}
		}
		status = swapchain.Present();
		const bool accepted = MultiFramePresentPolicy::AcceptedPresent(
		    status == Swapchain::Status::Success,
		    status == Swapchain::Status::PresentedRecreate, async_multi_mode);
		if (!accepted) {
			if (async_multi_mode) ++multi_present_failed;
			RecoverSwapchain(status);
			if (emitting_multi_generated) return false;
			continue;
		}

		presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
		window.loop.presented_frames.fetch_add(1, std::memory_order_relaxed);
		if (async_owner.load(std::memory_order_acquire)) {
			SetLastShown(layers[0].frame);
			if (kind == FrameGenerationDiag::PresentKind::Real || kind == FrameGenerationDiag::PresentKind::DelayedReal)
				main_real_accounted = true;
			if (async_multi_mode) {
				using K = MultiFramePresentPolicy::PresentedKind;
				const auto classified = kind == FrameGenerationDiag::PresentKind::Generated ? K::Generated :
				    kind == FrameGenerationDiag::PresentKind::Repeat ? K::Repeat :
				    kind == FrameGenerationDiag::PresentKind::Blank ? K::Blank : K::Real;
				multi_success.Presented(classified); // Only after actual successful vkQueuePresentKHR.
				const auto now = AsyncNow();
				if (multi_last_present && now >= multi_last_present) {
					if (multi_sample_count < multi_samples.size()) multi_samples[multi_sample_count++] = now - multi_last_present;
					else ++multi_sample_dropped;
				}
				multi_last_present = now;
			}
		}
		if (FrameGenerationDiag::Enabled() && (!async_owner.load(std::memory_order_acquire) || !async_multi_mode)) {
			// Successful vkQueuePresentKHR only; CPU monotonic time, no GPU query or wait.
			char       line[1024];
			const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
			    std::chrono::steady_clock::now().time_since_epoch());
			if (fg_diag.NotePresented(kind, static_cast<uint64_t>(now.count()), line, sizeof(line))) {
				std::printf("%s\n", line);
			}
		}
		// Suboptimal still accepted this image. Multi counts it once and recreates;
		// the default FSR path above preserves its existing retry behavior.
		if (status == Swapchain::Status::PresentedRecreate) RecoverSwapchain(Swapchain::Status::Recreate);
		return true;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DroppedPresents);
	if (FrameGenerationDiag::Enabled() && (!async_owner.load(std::memory_order_acquire) || !async_multi_mode)) fg_diag.NoteFailed(kind);
	return false;
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics

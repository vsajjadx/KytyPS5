#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATION_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATION_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

struct FfxVkFsr3_3_1_6FrameGenerationContext;
VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

namespace FrameGeneration {
void               SetEnabled(bool enabled);
void               Toggle();
[[nodiscard]] bool IsEnabled();
[[nodiscard]] bool IsAvailable();
} // namespace FrameGeneration

// Interpolates a frame between the two last presented frames with AMD FSR 3.1 frame generation.
// The game provides no depth or motion vectors, so the interpolation follows the optical flow of
// the presented images alone.
class FrameGenerator final {
public:
	FrameGenerator(GraphicContext& graphics, CommandScheduler& scheduler);
	~FrameGenerator();
	KYTY_CLASS_NO_COPY(FrameGenerator);

	// Records the frame between the previous `frame` and this one into `output`. Both images are
	// 32-bit RGBA or BGRA of the given size; `frame` is in the general layout and stays there,
	// `output` ends in the transfer source layout when it was written.
	[[nodiscard]] bool Generate(vk::CommandBuffer command, vk::Image frame, vk::Image output,
	                            uint32_t width, uint32_t height);

	// The next frame starts a new sequence (scene cut, resize, frame generation toggled).
	void Reset() { m_reset = true; }

	[[nodiscard]] uint64_t SignatureId() const { return m_signature_id; }

	// Whether the frame with this signature id differs so much from the previous one that
	// interpolating between them would blend two unrelated shots. Needs its GPU work complete.
	[[nodiscard]] bool IsSceneCut(uint64_t id);

private:
	struct Target {
		VmaAllocation allocation = nullptr;
		vk::Image     image      = nullptr;
	};
	struct PendingFrame {
		uint64_t frame_id;
		uint64_t tick;
	};
	struct SignatureSlot {
		VmaAllocation allocation = nullptr;
		vk::Buffer    buffer     = nullptr;
		void*         mapped     = nullptr;
		uint64_t      frame_id   = 0;
	};
	static constexpr uint32_t SignatureSlots      = 8;
	static constexpr uint32_t MaxAutomaticRetries = 3;
	static constexpr auto     RetryDelay          = std::chrono::seconds {5};

	bool Recreate(vk::CommandBuffer command, uint32_t width, uint32_t height);
	void Destroy();
	void DestroyTarget(Target& target);
	bool CreateTarget(Target& target, vk::Format format, vk::ImageUsageFlags usage, uint32_t width,
	                  uint32_t height, uint32_t mip_levels = 1);
	bool CreateSignature(uint32_t frame_width, uint32_t frame_height);
	void DestroySignature();
	void RecordSignature(vk::CommandBuffer command, vk::Image frame, uint32_t frame_width,
	                     uint32_t frame_height, uint64_t id);
	void Fail(const char* what);
	void Suspend(const char* what);
	bool TryResume();

	GraphicContext&                           m_graphics;
	CommandScheduler&                         m_scheduler;
	FfxVkFsr3_3_1_6FrameGenerationContext*    m_context = nullptr;
	Target                                    m_color;
	Target                                    m_interpolated;
	Target                                    m_depth;
	Target                                    m_motion;
	Target                                    m_signature;
	uint32_t                                  m_signature_levels = 0;
	uint32_t                                  m_signature_width  = 0;
	uint32_t                                  m_signature_height = 0;
	std::array<SignatureSlot, SignatureSlots> m_signature_slots {};
	std::vector<uint8_t>                      m_previous_signature;
	uint64_t                                  m_previous_signature_id = 0;
	uint64_t                                  m_signature_id          = 0;
	std::mutex                                m_signature_mutex;
	uint32_t                                  m_width    = 0;
	uint32_t                                  m_height   = 0;
	uint64_t                                  m_frame_id = 0;
	std::atomic_bool                          m_reset {true};
	bool                                      m_failed               = false;
	uint32_t                                  m_consecutive_failures = 0;
	uint64_t                                  m_failed_tick          = 0;
	std::chrono::steady_clock::time_point     m_retry_at {};
	std::deque<PendingFrame>                  m_pending;
	std::chrono::steady_clock::time_point     m_last_generate {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATION_H_

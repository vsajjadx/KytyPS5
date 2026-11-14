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
#include <memory>
#include <span>
#include <vector>

struct FfxVkFsr3_3_1_6FrameGenerationContext;
VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

namespace FrameGeneration {
enum class Backend : uint8_t { Fsr, Multi };
void SetBackend(Backend backend);
[[nodiscard]] Backend GetBackend();
void SetMultiplier(uint32_t factor);
[[nodiscard]] uint32_t GetMultiplier();
void               SetEnabled(bool enabled);
void               Toggle();
[[nodiscard]] bool IsEnabled();
[[nodiscard]] bool IsAvailable();
} // namespace FrameGeneration

enum class MultiFrameStatus : uint8_t { Failed, Seeded, Generated };
struct MultiFrameResult {
	MultiFrameStatus status = MultiFrameStatus::Failed;
	uint32_t count = 0;
	uint32_t written_mask = 0;
};
// Opt-in synthetic-test diagnostics only. Caller owns distinct TransferDst buffers and
// retains them through the recording submission. Tight RGBA8 P/C, signed RG16 flow,
// and the first two R32_UINT SCD words. Null diagnostics records no extra work.
struct MultiFrameDiagnosticBuffer { vk::Buffer buffer = nullptr; uint64_t capacity = 0; };
struct MultiFrameDiagnostics {
    MultiFrameDiagnosticBuffer optical_flow, scene_change, previous, current;
    // Optional: final locally refined flow, tight RGBA32_SINT (xy motion; zw reserved).
    MultiFrameDiagnosticBuffer refined_flow;
};
enum class MultiFrameSceneState : uint8_t { Unavailable, Pending, Clear, Changed };
struct MultiFrameSceneStatus {
    MultiFrameSceneState state = MultiFrameSceneState::Unavailable;
    uint32_t flags = 0;
    uint64_t recorded_tick = 0, completed_tick = 0;
};
class MultiFrameInterpolator;

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

	// Multi output: one SDK history/flow update per real pair, ascending times (i+1)/factor.
	// Input/output images enter General; outputs are at least width/height. Generated outputs
	// end TransferSrcOptimal. Failed/Seeded publish no outputs and keep external layouts General.
	// Generated means recorded: after completion, MultipleSceneChanged MUST be false before
	// displaying any subframes. A scene flag invalidates that batch (present the real frame).
	// Validate all handles/counts/aliases before advancing history. Caller retains the images
	// through the submission tick; this method records only and never submits or waits on WSI.
	[[nodiscard]] MultiFrameResult GenerateMultiple(vk::CommandBuffer command, vk::Image frame,
	    std::span<const vk::Image> outputs, uint32_t width, uint32_t height, uint32_t factor,
        const MultiFrameDiagnostics* diagnostics = nullptr);

	// The next frame starts a new sequence (scene cut, resize, frame generation toggled).
	void Reset();
	// After completing the recording submission: true for a scene fallback or not-yet-ready.
	[[nodiscard]] bool MultipleSceneChanged() const;
    // Diagnostic classification of the same existing feedback query; no new GPU work.
    [[nodiscard]] MultiFrameSceneStatus GetMultipleSceneStatus() const;
    // Post-failure synthetic diagnostic: after the original submission completes, record
    // an 8-byte SCD readback only. Submit+wait this command before any further generation;
    // caller retains destination through its tick. No flow/history/warp work is repeated.
    [[nodiscard]] bool CopyMultipleSceneForDiagnostics(vk::CommandBuffer command,
        vk::Buffer destination, uint64_t capacity);

    // Post-completion synthetic diagnostic only: copies the last generated pair's
    // tight RGBA32_SINT motion (primary.xy, alternate.zw) and RGBA32_SFLOAT layer
    // (K.rgb,beta; beta<0 means unavailable), 16 bytes/pixel each. The caller owns
    // distinct TransferDst buffers and must submit+wait this copy before generating
    // again. No prepare, history update or warp is repeated; null/default calls add no work.
    [[nodiscard]] bool CopyMultipleFineForDiagnostics(vk::CommandBuffer command,
        const MultiFrameDiagnosticBuffer& full_motion,
        const MultiFrameDiagnosticBuffer& static_layer);

	[[nodiscard]] uint64_t SignatureId() const { return m_signature_id; }

	// Whether the last Generate call recorded writes to its `output` image (it can, and still
	// return false for the first frame after a reset). Its layout is then transfer source.
	[[nodiscard]] bool OutputWritten() const { return m_output_written; }

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

	bool Recreate(vk::CommandBuffer command, uint32_t width, uint32_t height, bool multi = false);
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

	std::unique_ptr<MultiFrameInterpolator> m_multi;
	uint32_t m_multi_prepares = 0;
    bool m_context_is_multi = false;
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
	bool                                      m_output_written = false;
	bool                                      m_failed               = false;
	uint32_t                                  m_consecutive_failures = 0;
	uint64_t                                  m_failed_tick          = 0;
	std::chrono::steady_clock::time_point     m_retry_at {};
	std::deque<PendingFrame>                  m_pending;
	std::chrono::steady_clock::time_point     m_last_generate {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATION_H_

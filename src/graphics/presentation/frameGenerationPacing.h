#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATIONPACING_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATIONPACING_H_

// Pure pacing/lifetime rules of the frame-generation presenter (std-only, unit tested).
#include <chrono>
#include <cstdint>

namespace Libs::Graphics::FrameGenerationPacing {

// Delay from presenting a generated frame to presenting the real frame it went ahead of. The
// ideal time is the midpoint interval/2, but pending frames are polled once per vblank, and the
// first vblank at or after interval/2 - vblank/2 is the vblank NEAREST that midpoint. (A delay of
// interval/2 rounds up: at 25-30 FPS on 60 Hz it held the real frame a second vblank, where the
// next game flip could already replace it unpresented.)
constexpr std::chrono::nanoseconds PendingRealDelay(std::chrono::nanoseconds interval,
                                                    std::chrono::nanoseconds vblank) {
	return interval > vblank ? (interval - vblank) / 2 : std::chrono::nanoseconds {0};
}

// A new game frame replaces the main bus while the previous real frame still waits. When the
// pending frame is already due (the flip was only handled first in this vblank), present it before
// it is released; otherwise it is superseded. Never delays the guest flip itself.
constexpr bool FlushPendingReal(bool pending, std::chrono::steady_clock::time_point now,
                                std::chrono::steady_clock::time_point due) {
	return pending && now >= due;
}

// Submission ticks of one scheduler only grow; a frame keeps the latest submission that uses it.
constexpr uint64_t RetainTick(uint64_t current, uint64_t tick) {
	return current > tick ? current : tick;
}

} // namespace Libs::Graphics::FrameGenerationPacing

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEGENERATIONPACING_H_

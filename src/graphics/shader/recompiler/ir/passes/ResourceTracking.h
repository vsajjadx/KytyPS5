#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

// A rejected resource-tracking step. Thrown ONLY when the caller opted in (soft_failure, used for
// TABLE-linked translation); otherwise the historical fatal exit is kept. The partially tracked
// program must be discarded: it is never compiled or materialized.
struct ResourceTrackingFailure {
	uint32_t    pc = 0;   // translation (linked) PC of the rejected instruction
	std::string reason;   // tracker reason, including any bounded operation/memory details
	std::string message;  // the exact historical fatal message
};

// Resolves native descriptor sources, plans their scalar reads, and assigns dense resource bindings.
void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg,
                    ShaderStageInputInfo input, bool soft_failure = false);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_ */

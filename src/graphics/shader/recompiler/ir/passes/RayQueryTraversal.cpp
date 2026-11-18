#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

uint32_t TraceTraversalsWithRayQueries(Program&) {
	// Automatic guest-loop conversion is quarantined: ray updates, traversal liveness,
	// stack effects/bounds and equal-distance outputs are not yet proved equivalent.
	// Preserve the software IR even when the device supports native ray queries. There
	// is deliberately no runtime override; native backend/mirror tests are independent.
	return 0;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Automatic conversion of guest BVH traversal loops is quarantined. The former matcher
// does not certify all ray updates, stack effects/bounds, traversal liveness or equal-distance
// outputs. This entry point leaves the software IR unchanged and returns zero, including when
// native ray-query capability is enabled. There is no runtime override. Direct native backend
// and mirror tests are independent of guest-loop conversion.
[[nodiscard]] uint32_t TraceTraversalsWithRayQueries(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

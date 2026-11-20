#include "graphics/shader/recompiler/ir/ufc6WhiteRecovery.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

// In this guest shader, the first two reciprocals normalize horizontal tap weights.
 // Check the exact lowered zero-or-weight sequence before editing it; leave final vertical
 // reciprocals and all other shaders untouched.
bool GuardHorizontalWeights(std::vector<uint32_t>& words) {
	std::vector<size_t> def(words[3], 0);
	std::array<size_t, 4> divisions {};
	size_t count = 0;
	uint32_t float_type = 0, one = 0, zero = 0;
	for (size_t i = 5; i < words.size();) {
		const auto n = words[i] >> 16u;
		if (!n || i + n > words.size()) return false;
		const auto op = words[i] & 0xffff;
		if (op == spv::OpTypeFloat && n == 3 && words[i + 2] == 32) float_type = words[i+1];
		if (op == spv::OpConstant && n == 4 && words[i+1] == float_type) {
			if (words[i+3] == 0x3f800000u) one = words[i+2];
			if (words[i+3] == 0u) zero = words[i+2];
		}
		if ((op == spv::OpFMul || op == spv::OpSelect) && n >= 3 && words[i+2] < def.size())
			def[words[i+2]] = i;
		if (op == spv::OpFDiv && n == 5 && words[i+1] == float_type && one && words[i+3] == one) {
			if (count == divisions.size()) return false;
			divisions[count++] = i;
		}
		i += n;
	}
	if (count != divisions.size() || !one || !zero) return false;
	std::array<size_t, 2> selects {};
	for (size_t k = 0; k < selects.size(); ++k) {
		const auto div = divisions[k];
		if (words[div+4] >= def.size()) return false;
		const auto s = def[words[div+4]];
		if (!s || (words[s] & 0xffff) != spv::OpSelect || (words[s] >> 16u) != 6 ||
		    words[s+4] >= def.size()) return false;
		const auto mul = def[words[s+4]];
		if (!mul || (words[mul] & 0xffff) != spv::OpFMul ||
		    words[mul+3] != words[s+5] || words[mul+4] != zero) return false;
		selects[k] = s;
	}
	for (const auto s : selects) words[s+4] = one;
	return true;
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	auto words = state.builder.Build();
	if (profile.ufc6_post_process && program.shader_hash == KytyUfc6WhiteRecovery::HorizontalWeight) {
		const bool guarded = GuardHorizontalWeights(words);
		if (std::getenv("KYTY_UFC6_WHITE_DIAG")) {
			std::printf("UFC6_WHITE horizontal divisor guard: %s\n", guarded ? "matched" : "not matched");
		}
	}
	return words;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

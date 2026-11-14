#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bitset>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        emit_debug_info            = false;
	const Spirv::Profile*       profile                    = &Spirv::DefaultProfile;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
};

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

// Decoded source and typed scalar call queries are retained once per cache source.
struct ShaderSource {
	struct Call {
		// Uniform: one host-walkable target. Table: the target pair is one entry of a scalar
		// buffer table selected on the GPU; its closed callee set is enumerated per dispatch.
		enum class Kind { Uniform, Table };
		uint32_t instruction;
		std::array<IR::Value, 2> target;
		std::bitset<128> preserved_registers;
		uint64_t address = 0;
		Kind kind = Kind::Uniform;
		std::array<IR::Value, 4> table_descriptor;
		uint32_t table_stride = 0;
		uint32_t table_addend = 0;
		uint32_t table_word_offset = 0;
	};
	struct Linked {
		// One leaf copy per uniform site, or per (table site, callee slot).
		// offset/size index RAW DWORDs of `code`; synthetic_pc is this copy's first translation PC
		// (the caller keeps its original PCs; copies start after the caller's last instruction).
		struct Function {
			uint32_t offset; uint32_t size; size_t site = 0; uint32_t slot = 0;
			uint32_t synthetic_pc = UINT32_MAX;
		};
		std::vector<uint32_t> code;
		Decoder::Program decoded;
		std::vector<Function> functions;
		std::vector<uint32_t> observed_function;
		std::vector<std::vector<uint64_t>> site_targets;
	};
	std::vector<uint32_t> code;
	Decoder::Program decoded;
	IR::ResourcePlan call_targets;
	std::vector<Call> calls;
	uint32_t unsupported_call_pc = UINT32_MAX;
	const char* unsupported_call_reason = nullptr;
	uint32_t wave_size = 64;
	std::optional<Linked> linked;
	uint64_t revision = 0;
	std::vector<std::pair<uint64_t, uint64_t>> reads;
	// Last table enumeration summary (bounded diagnostics only).
	std::string call_diagnostic;
	// Callee code snapshotted for table sites; must stay immutable for the dispatch.
	std::vector<std::pair<uint64_t, uint64_t>> call_code_ranges;
	// Table ranges of this dispatch; must be uniquely backed so same-dispatch writes are seen.
	std::vector<std::pair<uint64_t, uint64_t>> call_table_ranges;
};

[[nodiscard]] const char* StageName(ShaderType stage);
[[nodiscard]] ShaderSource PrepareShaderSource(std::span<const uint32_t> code,
	                                           const CompileOptions& options);
const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime);
// Links the calls for this dispatch. nullptr (with a reason) when a table call site cannot be
// enumerated, read or validated; the source stays reusable for later dispatches.
[[nodiscard]] const Decoder::Program* TryRefreshShaderSource(ShaderSource& source,
                                                             const IR::SrtRuntime& runtime,
                                                             std::string& failure);
// soft_tracking_failure (TABLE-linked translation only) throws IR::ResourceTrackingFailure from
// resource tracking instead of exiting; every other translation keeps the fatal behavior.
[[nodiscard]] TranslateResult TranslateProgram(const Decoder::Program& decoded,
	                                           const CompileOptions& options,
	                                           bool soft_tracking_failure = false);
// Diagnostic only: where a linked translation PC came from (no memory reads, current link only).
struct LinkedPcOrigin {
	enum class Kind { Unknown, Caller, Callee, Stub };
	Kind     kind     = Kind::Unknown;
	size_t   site     = 0;
	uint32_t slot     = 0;
	uint64_t callee   = 0;
	uint32_t local_pc = 0; // caller: original PC; callee: PC inside the callee
};
[[nodiscard]] LinkedPcOrigin DescribeLinkedPc(const ShaderSource& source, uint32_t linked_pc);

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */

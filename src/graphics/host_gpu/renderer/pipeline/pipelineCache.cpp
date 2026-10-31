#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <cstdlib>
#include "graphics/host_gpu/frameStats.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_set>
#include <fmt/format.h>
#include <limits>
#include <ranges>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

void DumpShader(std::span<const uint32_t> words, const std::string& name) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	const auto path = Config::GetShaderLogFolder() / name;
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(words.data(), words.size_bytes());
}

void ValidateShaderSpirv(const char* stage, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF("%s\n", text.c_str());
	EXIT("ShaderRecompiler %s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", stage,
	     shader_hash, messages.c_str());
}

// Formats with a narrower host variant drop the components the shader does not read.
vk::Format NarrowInputFormat(vk::Format format, uint32_t used_components) {
	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				case 3: return vk::Format::eR32G32B32Sfloat;
				default: return format;
			}
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				default: return format;
			}
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR16Sfloat;
				case 2: return vk::Format::eR16G16Sfloat;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: return vk::Format::eR8Unorm;
				case 2: return vk::Format::eR8G8Unorm;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Snorm:
			return used_components == 2 ? vk::Format::eR8G8Snorm : format;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: return vk::Format::eR8Uint;
				case 2: return vk::Format::eR8G8Uint;
				default: return format;
			}
		default: return format;
	}
}

vk::Format InputFormat(const ShaderBufferResource& resource, uint32_t used_components) {
	switch (resource.RawFormat()) {
		// Temporary: invalid PS5 buffer format 113 (VertexAttribFormat::k16_16SInt).
		case 113: return NarrowInputFormat(vk::Format::eR32G32B32A32Sfloat, used_components);
		case 121: return vk::Format::eR16G16Sfloat;
		default: break;
	}
	const auto format = VulkanFormat(resource.Format());
	if (format == vk::Format::eUndefined ||
	    ShaderRecompiler::Format::GetFormatInfo(resource.Format()).component_count == 0) {
		EXIT("unknown vertex format: fmt = %u\n", resource.RawFormat());
	}
	return NarrowInputFormat(format, used_components);
}

// The host format of each vertex input attribute, narrowed to the components the stage reads.
std::vector<vk::Format> VertexInputFormats(const ShaderVertexInputInfo&                    input_info,
                                           const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	std::vector<vk::Format> formats(input_info.resources_num);
	for (int index = 0; index < input_info.resources_num; index++) {
		const auto registers_num = input_info.resources_dst[index].registers_num;
		EXIT_IF(registers_num < 1 || registers_num > 4);
		const auto fetched = program.info.vertex_fetch_components[index];
		formats[index] = InputFormat(input_info.resources[index],
		                             fetched > 0 ? fetched : static_cast<uint32_t>(registers_num));
	}
	return formats;
}

} // namespace

ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics) {
	// AMD GPUs (vendor 0x1002) run NMin/NMax as v_min/v_max, which give the guest's NaN and
	// signed-zero results; GLSL.std.450 alone does not order -0 below +0.
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == 0x1002u,
	        .device_clock_period_ps = graphics.device_clock_period_ps};
}

std::filesystem::path ShaderStorePath() {
	if (KYTY_BUILD != KYTY_BUILD_RELEASE || !Config::ShaderCacheEnabled() ||
	    Config::GraphicsDebugDumpEnabled()) {
		return {};
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return {};
	}
	return std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
}

bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& other) const {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	return XXH3_64bits(&key, sizeof(key));
}

std::span<const uint32_t> PipelineCache::Program::FunctionCode() const {
	return call_source
	    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
	    : std::span<const uint32_t>{};
}

bool PipelineCache::Program::RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
	const auto revision = call_source->revision;
	ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
	resource_plan.source_reads = call_source->reads;
	return revision != call_source->revision;
}

template <bool HasCalls>
ShaderModule* PipelineCache::Program::FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
                                                 uint32_t push_data_cursor) {
	EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(resource_plan, runtime, resources,
	                                                    specialization));
	for (auto& candidate: modules) {
		const auto& layout = candidate.info.bindings;
		if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
		        push_data_cursor, layout.ShaderDataDwords()) ||
		    candidate.specialization != specialization) {
			continue;
		}
		if constexpr (HasCalls) {
			if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
		}
		return &candidate;
	}
	return nullptr;
}

ShaderModule& PipelineCache::CompileModule(Program& program, const ProgramKey& key,
                                           ShaderRecompiler::CompileOptions  options,
                                           ShaderRecompiler::TranslateResult translated,
                                           uint32_t                          push_data_cursor) {
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
	                                         Common::DebugCounters::Counter::ShaderCompileNs,
	                                         Common::DebugCounters::Peak::ShaderCompileNs);
	auto built = BuildModule(options, std::move(translated), program.specialization,
	                         push_data_cursor);
	return AdoptModule(program, key, options, program.specialization, std::move(built));
}

PipelineCache::BuiltModule
PipelineCache::BuildModule(ShaderRecompiler::CompileOptions                    options,
                           ShaderRecompiler::TranslateResult                   translated,
                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                           uint32_t                                            push_data_cursor) {
	const auto compile_begin = std::chrono::steady_clock::now();
	const auto profile = SpirvProfile(m_graphics);
	options.profile    = &profile;
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization,
	                                               push_data_cursor);
	BuiltModule built {
	    .info  = std::move(result.program).TakeCompiledInfo(),
	    .spirv = std::move(result.spirv),
	};
	ValidateShaderSpirv(ShaderRecompiler::StageName(options.stage), options.shader_hash,
	                    built.spirv, profile.maintenance8);
	built.module = CompileSPV(built.spirv, m_graphics.device);
	EXIT_IF(built.module == nullptr);
	built.compile_us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
	                                                          compile_begin)
	        .count());
	return built;
}

ShaderModule& PipelineCache::AdoptModule(Program& program, const ProgramKey& key,
                                         const ShaderRecompiler::CompileOptions&             options,
                                         const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                         BuiltModule built) {
	// The same inputs give the same module id across runs.
	const auto function_code = program.FunctionCode();
	ShaderRecompiler::IR::Writer inputs;
	ShaderRecompiler::IR::IoAll(inputs, key.stage, key.hash, key.user_data_count, key.static_state,
	                            built.info.bindings.push_data_start_dword, specialization);
	inputs.Raw(function_code.data(), function_code.size_bytes()); // Last, so it needs no size.
	const auto id = XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());

	const auto* stage = ShaderRecompiler::StageName(key.stage);
	DumpShader(built.spirv, fmt::format("{}_{:016x}_{:016x}.spv", stage, key.hash, id));
	auto& module = program.modules.emplace_back(ShaderModule {
	    .id             = id,
	    .module         = built.module,
	    .info           = std::move(built.info),
	    .specialization = specialization,
	    .function_code  = {function_code.begin(), function_code.end()},
	});
	m_compiled[static_cast<size_t>(key.stage)]++;
	const auto compile_us = built.compile_us;
	g_frame_stats.compiles.fetch_add(1, std::memory_order_relaxed);
	g_frame_stats.compile_us.fetch_add(compile_us, std::memory_order_relaxed);
	auto previous_max = g_frame_stats.compile_us_max.load(std::memory_order_relaxed);
	while (compile_us > previous_max &&
	       !g_frame_stats.compile_us_max.compare_exchange_weak(previous_max, compile_us)) {
	}
	// One line per compile: which guest shader, how many variants of it exist now, and how
	// long this one took. A shader with dozens of variants is being re-specialized per draw.
	std::printf("Compiled %s %016llx variant #%zu in %.1f ms\n", stage,
	            static_cast<unsigned long long>(key.hash), program.modules.size(),
	            static_cast<double>(compile_us) / 1000.0);
	// Guest geometry shaders are compiled through the host mesh stage.
	std::printf("Shaders: VS %u | PS %u | CS %u | GS %u | LS %u | HS %u | TES %u\n",
	            m_compiled[static_cast<size_t>(ShaderType::Vertex)],
	            m_compiled[static_cast<size_t>(ShaderType::Pixel)],
	            m_compiled[static_cast<size_t>(ShaderType::Compute)],
	            m_compiled[static_cast<size_t>(ShaderType::Mesh)],
	            m_compiled[static_cast<size_t>(ShaderType::Local)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationControl)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
	// The inputs of these come from the stage's static key.
	if (key.stage == ShaderType::Vertex || key.stage == ShaderType::Local) {
		module.vertex_formats = VertexInputFormats(*options.input_info.vertex, module.info);
	} else if (key.stage == ShaderType::Pixel &&
	           options.input_info.pixel->parameter_mode == ShaderPixelParameterMode::Rectangle) {
		module.rect_parameters = RectListParameters(*options.input_info.pixel, module.info);
	}
	// Programs with GPU-selected calls depend on their callees' current code.
	if (program.call_source == nullptr) {
		StoreModule(key, program, module, built.spirv);
	}
	return module;
}

// --skip-shaders (or KYTY_SKIP_SHADERS): guest shader hashes, as the "Compiled vs <hash>" lines
// print them, whose draws and dispatches are skipped. Either the hashes themselves (separated by commas, spaces or new
// lines) or a file holding them, re-read when it changes so shaders can be added while the game
// runs. PORTHEON skips UFC 6's crowd this way: vs c342cc1c86e12a66. GPU thread only.
static bool ShaderSkipped(uint64_t hash) {
	static const std::string setting_text = [] {
		const char* env = std::getenv("KYTY_SKIP_SHADERS");
		return env != nullptr ? std::string(env) : Config::GetSkipShaders();
	}();
	if (setting_text.empty()) {
		return false;
	}
	const char* const setting = setting_text.c_str();
	static std::unordered_set<uint64_t>       skipped;
	static std::unordered_set<uint64_t>       reported;
	static std::filesystem::file_time_type    file_time {};
	static auto                               next_check = std::chrono::steady_clock::time_point {};
	static bool                               loaded     = false;
	const auto parse = [](std::string_view text) {
		std::unordered_set<uint64_t> hashes;
		size_t                       pos = 0;
		while (pos < text.size()) {
			const auto end   = text.find_first_of(", \t\r\n", pos);
			const auto token = text.substr(pos, end == std::string_view::npos ? text.npos : end - pos);
			if (!token.empty()) {
				uint64_t value = 0;
				if (std::from_chars(token.data(), token.data() + token.size(), value, 16).ec ==
				    std::errc {}) {
					hashes.insert(value);
				}
			}
			if (end == std::string_view::npos) break;
			pos = end + 1;
		}
		return hashes;
	};
	std::error_code error;
	const std::filesystem::path path(setting);
	const auto now = std::chrono::steady_clock::now();
	if (!loaded || now >= next_check) {
		next_check = now + std::chrono::milliseconds(500);
		if (std::filesystem::is_regular_file(path, error)) {
			const auto time = std::filesystem::last_write_time(path, error);
			if (!error && (!loaded || time != file_time)) {
				std::ifstream file(path, std::ios::binary);
				const std::string text((std::istreambuf_iterator<char>(file)),
				                       std::istreambuf_iterator<char>());
				skipped   = parse(text);
				file_time = time;
				std::printf("KYTY_SKIP_SHADERS: %zu shaders from %s\n", skipped.size(), setting);
			}
		} else if (!loaded) {
			skipped = parse(setting);
			std::printf("KYTY_SKIP_SHADERS: %zu shaders\n", skipped.size());
		}
		loaded = true;
	}
	if (!skipped.contains(hash)) {
		return false;
	}
	if (reported.insert(hash).second) {
		std::printf("KYTY_SKIP_SHADERS: skipping draws and dispatches of shader %016llx\n",
		            static_cast<unsigned long long>(hash));
	}
	return true;
}

// Owned copies of everything the render thread's compile options pointed at.
struct PipelineCache::ModuleJob {
	Program*                                         program     = nullptr;
	const ProgramKey*                                program_key = nullptr;
	ShaderRecompiler::CompileOptions                 options;
	std::optional<ShaderRecompiler::TranslateResult> translated;
	std::vector<uint32_t>                            code;
	std::vector<uint32_t>                            back_code;
	std::vector<uint32_t>                            user_data;
	std::optional<ShaderVertexInputInfo>             vertex_input;
	std::optional<ShaderPixelInputInfo>              pixel_input;
	std::optional<ShaderComputeInputInfo>            compute_input;
	ShaderRecompiler::IR::ResourceSpecialization     specialization;
	uint32_t                                         push_data_cursor = 0;

	// Points `options` at this job's own copies.
	void BindOptions();
};

void PipelineCache::ModuleJob::BindOptions() {
	options.user_data = user_data;
	options.back_code = back_code;
	options.input_info = {};
	if (vertex_input) {
		options.input_info.vertex = &*vertex_input;
	} else if (pixel_input) {
		options.input_info.pixel = &*pixel_input;
	} else if (compute_input) {
		options.input_info.compute = &*compute_input;
	}
}

template <typename InputInfo>
void PipelineCache::QueueModule(Program& program, const ProgramKey& key,
                                const ShaderRecompiler::CompileOptions&          options,
                                std::optional<ShaderRecompiler::TranslateResult> translated,
                                std::span<const uint32_t> code, const InputInfo& input_info,
                                uint32_t push_data_cursor) {
	auto module_job = std::make_shared<ModuleJob>();
	auto& job          = *module_job;
	job.program          = &program;
	job.program_key      = &key;
	job.options          = options;
	job.translated       = std::move(translated);
	job.code.assign(code.begin(), code.end());
	job.back_code.assign(options.back_code.begin(), options.back_code.end());
	job.user_data.assign(options.user_data.begin(), options.user_data.end());
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		job.vertex_input = input_info;
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		job.pixel_input = input_info;
	} else {
		job.compute_input = input_info;
	}
	job.specialization   = program.specialization;
	job.push_data_cursor = push_data_cursor;
	program.pending.push_back({program.specialization, push_data_cursor});
	{
		std::unique_lock lock(m_compile_mutex);
		m_compile_queue.push(CompileJob {.kind = JobKind::Module, .module = std::move(module_job)});
	}
	m_compile_cv.notify_one();
	EnsureWorker();
}

void PipelineCache::DrainAsyncModules() {
	if (!m_async_shaders) {
		return;
	}
	std::vector<CompletedModule> completed;
	{
		std::unique_lock lock(m_compile_mutex);
		completed.swap(m_completed_modules);
	}
	for (auto& [compile_job, built]: completed) {
		auto& job     = *compile_job.module;
		auto& program = *job.program;
		std::erase_if(program.pending, [&](const Program::PendingModule& pending) {
			return pending.push_data_cursor == job.push_data_cursor &&
			       pending.specialization == job.specialization;
		});
		job.BindOptions();
		AdoptModule(program, *job.program_key, job.options, job.specialization, std::move(built));
	}
}

template <typename InputInfo>
const ShaderModule* PipelineCache::GetProgram(const ShaderParams& params, InputInfo& input_info,
                                              uint32_t& push_data_cursor) {
	if (ShaderSkipped(params.hash)) {
		input_info.stage = {};
		return nullptr;
	}
	const auto user_data = std::span(params.user_data).first(params.user_data_count);
	ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .shader_base                = params.Base(),
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .image_heap                 = m_image_heap,
	};
	ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		options.stage            = input_info.logical_stage;
		options.input_info.vertex = &input_info;
		options.user_data_base   = 8;
		if (options.stage == ShaderType::Mesh ||
		    options.stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
			options.wave_size =
			    options.stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
		}
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		options.stage            = ShaderType::Pixel;
		options.input_info.pixel = &input_info;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		options.stage              = ShaderType::Compute;
		options.input_info.compute = &input_info;
		runtime.workgroup_counts   = input_info.workgroup_counts;
		runtime.indirect_args_address = input_info.indirect_args_address;
	}
	m_lookup_key.stage           = options.stage;
	m_lookup_key.hash            = params.hash;
	m_lookup_key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, m_lookup_key.static_state);
	auto entry = m_program_cache.find(m_lookup_key);
	const auto bind = [&](const ShaderModule& module) {
		input_info.stage = {.module = &module, .resources = &entry->second.resources};
		module.info.bindings.AdvancePushData(push_data_cursor);
		return &module;
	};
	bool source_changed = false;
	if (entry != m_program_cache.end()) {
		auto& cached = entry->second;
		if (cached.skip_dispatch) return nullptr;
		ShaderModule* module = nullptr;
		if (cached.call_source) {
			source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) module = cached.FindModule<true>(runtime, push_data_cursor);
		} else {
			module = cached.FindModule<false>(runtime, push_data_cursor);
		}
		if (module != nullptr) return bind(*module);
	}
	const bool async_module =
	    m_async_shaders && (options.stage != ShaderType::Compute || m_async_compute);
	if (async_module && entry != m_program_cache.end() && !entry->second.call_source) {
		// FindModule left the wanted variant in program.specialization.
		auto& program = entry->second;
		const bool pending = std::ranges::any_of(program.pending, [&](const auto& queued) {
			return queued.push_data_cursor == push_data_cursor &&
			       queued.specialization == program.specialization;
		});
		input_info.stage = {};
		if (pending) {
			NoteSkipped();
			return nullptr;
		}
	}

	options.shader_hash     = params.hash;
	options.user_data       = user_data;
	options.back_code       = params.back_code;
	options.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	if (async_module && entry != m_program_cache.end() && !entry->second.call_source) {
		// A new variant of a known program: translate and compile it on a worker.
		QueueModule(entry->second, entry->first, options, std::nullopt, params.code, input_info,
		            push_data_cursor);
		NoteSkipped();
		return nullptr;
	}
	ShaderRecompiler::ShaderSource source;
	if (entry == m_program_cache.end()) {
		DumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
		                                    ShaderRecompiler::StageName(options.stage), params.hash));
		source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX) {
			PipelineCacheLog("!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			                 "reason=unsupported GPU-selected callable target; rendering is incomplete",
			                 params.hash, source.unsupported_call_pc);
			m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
			return nullptr;
		}
	}
	auto* current = entry == m_program_cache.end() ? &source : entry->second.call_source.get();
	ShaderRecompiler::TranslateResult translated;
	if (current != nullptr) {
		if (!current->calls.empty()) {
			if (entry == m_program_cache.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
			translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
			translated.program.source_reads = current->reads;
		} else {
			translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
		}
	} else {
		translated = ShaderRecompiler::TranslateProgram(params.code, options);
	}
	if (entry == m_program_cache.end()) {
		auto retained = !source.calls.empty()
		                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
		                    : nullptr;
		entry = m_program_cache.emplace(m_lookup_key, Program {
		    .call_source   = std::move(retained),
		    .resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
		}).first;
		auto& program = entry->second;
		EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
		    program.resource_plan, runtime, program.resources, program.specialization));
	} else if (source_changed) {
		entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (const auto* module = entry->second.FindModule<true>(runtime, push_data_cursor)) {
			return bind(*module);
		}
	}
	if (async_module && !entry->second.call_source && !source_changed) {
		// Compile this variant on a worker and skip the draw/dispatch until it is ready.
		QueueModule(entry->second, entry->first, options, std::move(translated), params.code,
		            input_info, push_data_cursor);
		input_info.stage = {};
		NoteSkipped();
		return nullptr;
	}
	return bind(CompileModule(entry->second, entry->first, options, std::move(translated),
	                          push_data_cursor));
}

PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap,
                             const std::filesystem::path&           shader_store)
    : m_graphics(graphics), m_image_heap(image_heap),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_async_shaders = Config::AsyncShadersEnabled();
	m_async_compute = Config::AsyncComputeEnabled();
	m_async_enabled = Config::AsyncPipelinesEnabled();
	// The environment overrides the options (KYTY_ASYNC_*=0 or 1).
	const auto env_flag = [](const char* name, bool& flag) {
		if (const char* value = std::getenv(name)) {
			flag = value[0] != '\0' && value[0] != '0';
		}
	};
	env_flag("KYTY_ASYNC_PIPELINES", m_async_enabled);
	env_flag("KYTY_ASYNC_SHADERS", m_async_shaders);
	env_flag("KYTY_ASYNC_COMPUTE", m_async_compute);
	std::printf("Async compilation: pipelines %s, shaders %s, compute shaders %s\n",
	            m_async_enabled ? "on" : "off", m_async_shaders ? "on" : "off",
	            m_async_shaders && m_async_compute ? "on" : "off");
	InitializeDriverCache();
	WarmUp(shader_store);
}

PipelineCache::~PipelineCache() {
	m_shutdown.store(true, std::memory_order_relaxed);
	m_compile_cv.notify_all();
	for (auto& thread: m_compile_threads) {
		thread.join();
	}
	// Modules built after the last draw still own host shader modules.
	DrainAsyncModules();
	DrainAsyncPipelines();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& pipeline: pipelines | std::views::values) {
			m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			m_graphics.device.destroyShaderModule(module.module, nullptr);
		}
	}
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty = true;
	if (std::chrono::steady_clock::now() >= m_next_driver_cache_save) {
		Save();
	}
}

void PipelineCache::Save() {
	if (m_driver_cache == nullptr || !m_driver_cache_dirty) {
		return;
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
	m_driver_cache_dirty = false;
}

void PipelineCache::GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state) {
	// Between draws, like a synchronous compile: no module pointer of this draw is held yet.
	DrainAsyncModules();
	const auto& context           = command.GetRegisters();
	const auto& user_config       = command.GetUserConfig();
	const auto& sh                = context.GetShaderRegisters();
	const auto& vertex_regs       = command.GetShaders().GetVs();
	const auto& pixel_regs        = command.GetShaders().GetPs();
	auto&       vertex_info       = state.vertex_info;
	auto&       pixel_info        = state.ps_input_info;
	const bool  object_primitive_id = context.HasObjectPrimitiveIdOutput();
	const auto  write_mask          = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    color_output_mask   = object_primitive_id ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			color_output_mask |= 1u << slot;
		}
	}
	const auto& db = sh.db_shader_control;
	// Without color outputs, a pixel shader still runs for its depth or coverage side effects.
	const bool pixel_active =
	    pixel_regs.ps_regs.data_addr != 0 &&
	    (color_output_mask != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	     db.shader_mask_export_enable || db.shader_dual_export_enable ||
	     db.shader_execute_on_noop || sh.AlphaToMask());
	state.ps_active = pixel_active;
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
		// Reflection decodes both programs; the key also covers the VGT fields in tess.
		const uint64_t layout_key[] = {vertex_params[0].hash, vertex_params[1].hash,
		                               uint64_t {sh.m_vgtLsHsConfig & 0xfff00u} << 32u |
		                                   sh.m_vgtTfParam};
		const auto [layout, inserted] = m_tess_layouts.try_emplace(
		    XXH3_64bits(layout_key, sizeof(layout_key)), vertex_info[0].tess);
		if (inserted) {
			ShaderRecompiler::AnalyzeTessellationPrograms(vertex_params[0].code,
			                                              vertex_params[1].code, layout->second);
			StoreLayout(layout->first, layout->second);
		}
		for (auto& stage: vertex_info) {
			stage.tess = layout->second;
		}
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_offsets = state.indirect_offsets;
		vertex_info[0].indirect_draw    = state.indirect_draw;
	}
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: dropping mesh draws the host does not support (shader 0x{:016x}).\n",
				    vertex_params[0].hash));
			}
			return;
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
		    target_export_mapping {};
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << export_index;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << export_index;
			}
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0; blended targets are not integer.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
			pixel_info.target_uint_mask &= ~2u;
			pixel_info.target_sint_mask &= ~2u;
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id, last_stage.user_clip_planes);
	if (pixel_active) {
		const auto* pixel_program = GetProgram(pixel_params, pixel_info, push_data_cursor);
		if (pixel_program == nullptr) {
			// Still compiling (or not compilable): a null vertex module skips the draw.
			vertex_info[0].stage = {};
			return;
		}
		const auto& pixel_module = *pixel_program;
		for (const auto& output: pixel_module.info.info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				state.mrt_mask |= 1u << output.location;
			}
		}
		state.mrt_mask &= color_output_mask;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		if (GetProgram(vertex_params[i], vertex_info[i], push_data_cursor) == nullptr) {
			vertex_info[0].stage = {};
			return;
		}
	}
}

const ShaderModule* PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                                     ShaderComputeInputInfo&      input_info) {
	// Between dispatches, like a synchronous compile: no module pointer of this one is held yet.
	DrainAsyncModules();
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return GetProgram(params, input_info, push_data_cursor);
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(const CommandBuffer&   command,
                                                            const DrawRenderState& state) {
	DrainAsyncPipelines();
	const auto* colors        = state.color_info;
	const auto& depth         = state.depth_info;
	const auto  vertex_info   = state.VertexStages();
	const auto& ps_input_info = state.ps_input_info;
	const auto& vs_input_info = vertex_info.front();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	const bool ps_active    = state.ps_active;
	const bool tessellation = vertex_info.size() == 3;
	std::array<const ShaderModule*, 4> stages {};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		stages[i] = vertex_info[i].stage.module;
	}
	if (ps_active) {
		stages[3] = ps_input_info.stage.module;
	}
	EXIT_IF(stages[0] == nullptr || (ps_active && stages[3] == nullptr));
	const bool mesh        = stages[0]->info.stage == ShaderType::Mesh;
	const auto color_count = state.color_count;

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < stages.size(); i++) {
		if (stages[i] != nullptr) {
			key.stage_ids[i] = stages[i]->id;
		}
	}
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_active) {
			alpha_source = ps_input_info.alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info.ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = state.topology;
	static_params.primitive_restart_enable = state.primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info.ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	if (tessellation) {
		static_params.patch_control_points = vs_input_info.tess.input_control_points;
	} else if (!mesh && static_params.topology == vk::PrimitiveTopology::ePatchList) {
		// Rectangle lists expand through generated tessellation stages.
		static_params.patch_control_points = 3;
	}

	if (!mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (m_async_enabled) {
		// Create the pipeline on the background worker; the caller skips this draw until it is
		// ready instead of stalling on vkCreateGraphicsPipelines.
		std::unique_lock lock(m_compile_mutex);
		const auto       ready = [&] { return m_graphics_pipelines.contains(key); };
		const auto       wait  = [&]() -> Pipeline* {
			if (WaitForPipeline(lock, ready)) {
				return &m_graphics_pipelines.find(key)->second;
			}
			NoteSkipped();
			return nullptr;
		};
		if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
			return &iter->second;
		}
		if (m_pending_graphics.contains(key)) {
			return wait();
		}
		CompileJob job {.kind = JobKind::GraphicsPipeline, .graphics_key = key};
		for (size_t i = 0; i < stages.size(); i++) {
			if (stages[i] != nullptr) {
				job.stages[i] = *stages[i];
			}
		}
		m_pending_graphics.insert(key);
		m_compile_queue.push(std::move(job));
		lock.unlock();
		m_compile_cv.notify_one();
		EnsureWorker();
		lock.lock();
		return wait();
	}

	auto [iter, inserted] = m_graphics_pipelines.try_emplace(key);
	auto& pipeline        = iter->second;
	if (!inserted) {
		return &pipeline;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     key.stage_ids[0], static_cast<void*>(stages[0]->module), key.stage_ids[3],
		     ps_active ? static_cast<void*>(stages[3]->module) : nullptr);
	}

	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, pipeline, key, stages, m_driver_cache);
	}
	StorePipeline(key);
	CheckpointDriverCache();

	return &pipeline;
}

PipelineCache::Pipeline* PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info) {
	DrainAsyncPipelines();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(input_info.stage.module == nullptr);
	const auto& module = *input_info.stage.module;

	if (m_async_enabled) {
		std::unique_lock lock(m_compile_mutex);
		const auto       ready = [&] { return m_compute_pipelines.contains(module.id); };
		const auto       wait  = [&]() -> Pipeline* {
			if (WaitForPipeline(lock, ready)) {
				return &m_compute_pipelines.find(module.id)->second;
			}
			NoteSkipped();
			return nullptr;
		};
		if (auto iter = m_compute_pipelines.find(module.id); iter != m_compute_pipelines.end()) {
			return &iter->second;
		}
		if (m_pending_compute.contains(module.id)) {
			return wait();
		}
		if (graphics_debug_dump_enabled()) {
			ShaderDbgDumpInputInfo(input_info);
		}
		m_pending_compute.insert(module.id);
		m_compile_queue.push(
		    CompileJob {.kind = JobKind::ComputePipeline, .compute_module = module});
		lock.unlock();
		m_compile_cv.notify_one();
		EnsureWorker();
		lock.lock();
		return wait();
	}

	auto [iter, inserted] = m_compute_pipelines.try_emplace(module.id);
	auto& pipeline        = iter->second;
	if (!inserted) {
		return &pipeline;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, pipeline, module, m_driver_cache);
	}
	CheckpointDriverCache();

	return &pipeline;
}

void PipelineCache::EnsureWorker() {
	std::unique_lock lock(m_compile_mutex);
	if (m_compile_threads.empty()) {
		// Leave most cores to the guest and render threads.
		const auto count = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 3u);
		for (uint32_t i = 0; i < count; i++) {
			m_compile_threads.emplace_back([this]() { WorkerLoop(); });
		}
	}
}

void PipelineCache::WorkerLoop() {
	while (true) {
		CompileJob job;
		{
			std::unique_lock lock(m_compile_mutex);
			m_compile_cv.wait(lock, [this] {
				return m_shutdown.load(std::memory_order_relaxed) || !m_compile_queue.empty();
			});
			if (m_shutdown && m_compile_queue.empty()) {
				return;
			}
			job = std::move(m_compile_queue.front());
			m_compile_queue.pop();
		}
		if (job.kind == JobKind::Module) {
			auto& module = *job.module;
			module.BindOptions();
			Common::DebugCounters::ScopedTimer timer(
			    Common::DebugCounters::Counter::ShadersCompiled,
			    Common::DebugCounters::Counter::ShaderCompileNs,
			    Common::DebugCounters::Peak::ShaderCompileNs);
			auto translated = module.translated
			                      ? std::move(*module.translated)
			                      : ShaderRecompiler::TranslateProgram(module.code, module.options);
			module.translated.reset();
			auto built = BuildModule(module.options, std::move(translated), module.specialization,
			                         module.push_data_cursor);
			std::unique_lock lock(m_compile_mutex);
			m_completed_modules.push_back({std::move(job), std::move(built)});
			continue;
		}
		Pipeline created {};
		{
			Common::DebugCounters::ScopedTimer timer(
			    Common::DebugCounters::Counter::PipelinesCreated,
			    Common::DebugCounters::Counter::PipelineCreateNs,
			    Common::DebugCounters::Peak::PipelineCreateNs);
			if (job.kind == JobKind::GraphicsPipeline) {
				std::array<const ShaderModule*, 4> stages {};
				for (size_t i = 0; i < stages.size(); i++) {
					stages[i] = job.stages[i] ? &*job.stages[i] : nullptr;
				}
				CreatePipelineInternal(m_graphics, created, job.graphics_key, stages, m_driver_cache);
			} else {
				CreatePipelineInternal(m_graphics, created, *job.compute_module, m_driver_cache);
			}
		}
		EXIT_NOT_IMPLEMENTED(created.pipeline == nullptr);
		EXIT_NOT_IMPLEMENTED(created.pipeline_layout == nullptr);
		std::unique_lock lock(m_compile_mutex);
		if (job.kind == JobKind::GraphicsPipeline) {
			m_pending_graphics.erase(job.graphics_key);
			m_graphics_pipelines.emplace(job.graphics_key, created);
			m_created_graphics.push_back(job.graphics_key);
		} else {
			m_pending_compute.erase(job.compute_module->id);
			m_compute_pipelines.emplace(job.compute_module->id, created);
		}
		m_ready_cv.notify_all();
	}
}

// A pipeline that the driver builds in a few milliseconds is worth waiting for: the draw is not
// skipped, which matters for passes a game renders only once (PORTHEON saw UFC 6's tutorial stay
// white with background pipelines). The waits draw on an allowance that grows with wall time at
// 15%, up to 6 ms, so they can never add up to a stall; past it, the draw is skipped.
template <typename Ready>
bool PipelineCache::WaitForPipeline(std::unique_lock<std::mutex>& lock, Ready ready) {
	constexpr int64_t MaxBudgetNs = 6'000'000;
	const auto        now         = std::chrono::steady_clock::now();
	if (m_wait_refill.time_since_epoch().count() == 0) {
		m_wait_refill = now;
	}
	const auto elapsed =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_wait_refill).count();
	m_wait_budget_ns = std::min(MaxBudgetNs, m_wait_budget_ns + elapsed * 15 / 100);
	m_wait_refill    = now;
	if (m_wait_budget_ns <= 0) {
		return ready();
	}
	const bool done = m_ready_cv.wait_for(lock, std::chrono::nanoseconds(m_wait_budget_ns), ready);
	const auto after = std::chrono::steady_clock::now();
	m_wait_budget_ns -= std::chrono::duration_cast<std::chrono::nanoseconds>(after - now).count();
	m_wait_refill = after;
	return done;
}

void PipelineCache::NoteSkipped() {
	m_skipped_draws++;
	const auto now = std::chrono::steady_clock::now();
	if (now < m_skip_report) {
		return;
	}
	m_skip_report = now + std::chrono::seconds(2);
	std::printf("Async compilation: %llu draws/dispatches skipped so far while shaders or "
	            "pipelines were building\n",
	            static_cast<unsigned long long>(m_skipped_draws));
}

void PipelineCache::DrainAsyncPipelines() {
	if (!m_async_enabled) {
		return;
	}
	std::vector<GraphicsPipelineKey> created;
	{
		std::unique_lock lock(m_compile_mutex);
		created.swap(m_created_graphics);
	}
	for (const auto& key: created) {
		StorePipeline(key);
	}
	if (!created.empty()) {
		CheckpointDriverCache();
	}
}

} // namespace Libs::Graphics

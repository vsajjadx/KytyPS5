#include "graphics/shader/recompiler/ShaderCalls.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

namespace {

// Closed-set limits for a GPU-selected (table) call site. Tables with at most MaxCallTableSlots
// potential entries keep their exact per-DWORD reads.
constexpr uint64_t MaxCallTableSlots   = 256;
constexpr size_t   MaxCallTableTargets = 32;
constexpr uint32_t MaxLeafWords        = 1u << 16u;
// Larger tables still enumerate EVERY potential entry, through bounded batched reads. Per refresh,
// across all table sites: span bytes from the first to the last in-range pointer DWORD (gaps
// included), checked before a site reads anything; and single-DWORD reads replacing a failed batch.
constexpr uint64_t MaxCallTableScanBytes      = uint64_t {8} << 20u;
constexpr uint64_t CallTableChunkBytes        = uint64_t {64} << 10u;
constexpr uint64_t MaxCallTableFallbackDwords = 16384;
// Dependency (source.reads -> specialization_reads) ranges published by large sites per refresh.
// Pointer DWORDs only, never transport gaps: FindBuffers bounds sentinel writers by these spans.
// One range per row at stride >= 16, so this equals the most rows the scan budget admits there.
constexpr uint64_t MaxCallTableDependencyRanges = MaxCallTableScanBytes / 16u;
// Only when a refresh has ANY large table site (pure-small refreshes keep their old behavior):
// every table read request (bulk, failed bulk, fallback and <=256-entry DWORDs, earlier sites
// included) is charged before its attempt against this strict aggregate; unique callee decoding
// and the whole linked program (caller, stubs, every per-site leaf copy) are bounded too.
constexpr uint64_t MaxCallTableAttemptBytes = MaxCallTableScanBytes;
constexpr uint64_t MaxCallDecodedBytes      = uint64_t {8} << 20u;
constexpr uint64_t MaxCallLinkedBytes       = uint64_t {8} << 20u;
// S_CALL_SELECT_B64 is never decoded: its raw words are {marker, target lo, target hi} so the
// linked function code, which keys compiled modules, names every compared callee.
constexpr uint32_t CallSelectMarker = 0xbf800000u;
// S_CALL_TABLE_MISS_B64 raw words: {marker, call pc, 0}.
constexpr uint32_t CallMissMarker = 0xbf800001u;
// S_CALL_ENTER_B64 raw words: {marker, call pc, 0}; heads each table callee copy.
constexpr uint32_t CallEnterMarker = 0xbf800002u;

// A target pair that is not host-walkable may still be one aligned 64-bit entry of a scalar
// buffer table: lo/hi = ReadConstBuffer(V#, addend + X * 2^k) at adjacent DWORDs, with a
// host-walkable V#. X is GPU data (e.g. a waterfall's readfirstlane). Returns a reason on failure.
const char* MatchCallTable(const IR::Program& query, ShaderSource::Call& site) {
	using namespace IR;
	const auto* lo = site.target[0].TryInstruction();
	const auto* hi = site.target[1].TryInstruction();
	if (lo == nullptr || hi == nullptr || lo->GetOpcode() != ValueOpcode::ReadConstBuffer ||
	    hi->GetOpcode() != ValueOpcode::ReadConstBuffer || lo->NumArgs() != 2u || hi->NumArgs() != 2u)
		return "target words are not both scalar buffer loads";
	const auto handle  = lo->Arg(0).Resolve();
	const auto dynamic = lo->Arg(1).Resolve();
	if (hi->Arg(0).Resolve() != handle || hi->Arg(1).Resolve() != dynamic)
		return "target words use different descriptors or offsets";
	const auto lo_index = lo->Flags<MemoryFlags>().index;
	const auto hi_index = hi->Flags<MemoryFlags>().index;
	if (lo_index >= query.memory_info.size() || hi_index >= query.memory_info.size())
		return "target words have no memory metadata";
	const auto& lo_memory = query.memory_info[lo_index];
	const auto& hi_memory = query.memory_info[hi_index];
	if (lo_memory.kind != ResourceKind::ScalarBuffer || hi_memory.kind != ResourceKind::ScalarBuffer ||
	    lo_memory.data_dwords != 1u || hi_memory.data_dwords != 1u)
		return "target words are not raw scalar buffer DWORDs";
	// SMEM immediates are signed (sign-extended 21 bits). A negative displacement would need
	// signed range enumeration; reject it explicitly instead of misreading it as +4 GiB.
	if (static_cast<int32_t>(lo_memory.offset) < 0 || static_cast<int32_t>(hi_memory.offset) < 0)
		return "negative table displacement is unsupported";
	if ((lo_memory.offset & 3u) != 0u || hi_memory.offset != lo_memory.offset + 4u)
		return "target words are not one aligned 64-bit entry";
	const auto* resource = handle.TryInstruction();
	if (resource == nullptr || resource->GetOpcode() != ValueOpcode::GetBufferResource ||
	    resource->NumArgs() != 4u)
		return "table descriptor is not a plain V#";
	for (uint32_t word = 0; word < 4u; ++word) {
		const auto value = resource->Arg(word).Resolve();
		if (!ValidateRuntimeValue(query, value, RuntimeValueType::Integer))
			return "table descriptor is not host-walkable";
		site.table_descriptor[word] = value;
	}
	// dynamic = addend + X * 2^k (u32 wrap keeps every reachable offset congruent to addend).
	uint32_t addend = 0;
	auto     value  = dynamic;
	for (uint32_t depth = 0; depth < 8u; ++depth) {
		const auto* add = value.TryInstruction();
		if (add == nullptr || add->GetOpcode() != ValueOpcode::IAdd32) break;
		const auto a = add->Arg(0).Resolve();
		const auto b = add->Arg(1).Resolve();
		if (a.IsImmediate()) {
			addend += a.U32();
			value = b;
		} else if (b.IsImmediate()) {
			addend += b.U32();
			value = a;
		} else {
			break;
		}
	}
	const auto* scale = value.TryInstruction();
	uint32_t    shift = UINT32_MAX;
	if (scale != nullptr && scale->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
	    scale->Arg(1).Resolve().IsImmediate()) {
		shift = scale->Arg(1).Resolve().U32();
	} else if (scale != nullptr && scale->GetOpcode() == ValueOpcode::IMul32) {
		for (uint32_t arg = 0; arg < 2u; ++arg) {
			const auto factor = scale->Arg(arg).Resolve();
			if (factor.IsImmediate() && std::has_single_bit(factor.U32())) {
				shift = static_cast<uint32_t>(std::countr_zero(factor.U32()));
			}
		}
	}
	if (shift == UINT32_MAX) return "table offset is not an index scaled by a power of two";
	if (shift < 3u || shift > 31u) return "table entries are not at least 8 bytes apart";
	if ((addend & 3u) != 0u) return "table offset is not DWORD aligned";
	site.table_stride      = 1u << shift;
	site.table_addend      = addend;
	site.table_word_offset = lo_memory.offset;
	return nullptr;
}

// Prove companion values with the same typed IR relations used by the target.
// The caller query only follows plain moves from the already translated wide SMEM
// producer, so probing unrelated SGPRs cannot introduce unsupported instructions.
bool MatchCallContext(const IR::Program& query, const ShaderSource::Call& site,
                      const std::array<IR::Value, 2>& values) {
	using namespace IR;
	const auto* target = site.target[0].TryInstruction();
	if (target == nullptr || target->GetOpcode() != ValueOpcode::ReadConstBuffer ||
	    site.table_stride < 16u || site.table_word_offset > UINT32_MAX - 12u) return false;
	for (uint32_t half = 0; half < 2u; ++half) {
		const auto* value = values[half].Resolve().TryInstruction();
		if (value == nullptr || value->GetOpcode() != ValueOpcode::ReadConstBuffer ||
		    value->NumArgs() != 2u || value->Arg(0).Resolve() != target->Arg(0).Resolve() ||
		    value->Arg(1).Resolve() != target->Arg(1).Resolve()) return false;
		const auto index = value->Flags<MemoryFlags>().index;
		if (index >= query.memory_info.size()) return false;
		const auto& memory = query.memory_info[index];
		if (memory.kind != ResourceKind::ScalarBuffer || memory.data_dwords != 1u ||
		    memory.offset != site.table_word_offset + 8u + half * 4u) return false;
	}
	return true;
}

// Geometry of one table site, already validated by the caller: entry offsets are
// first + n * stride (n < slots, all < reach <= 2^32), the pointer is at word_offset in each.
struct CallTableGeometry {
	uint64_t base = 0, size = 0, first = 0, slots = 0, stride = 0, word_offset = 0;
};

// Same V# semantics as the walker's scalar buffer reads: 48-bit base with its low two bits clear,
// byte size = stride ? stride * records : records, out-of-range DWORDs are 0. Every reachable
// offset is (addend + n * entry_stride) mod 2^32, an unsigned 32-bit value.
struct CallTableDescriptor {
	uint64_t base = 0, raw_stride = 0, size = 0, reach = 0, first = 0, slots = 0;
};

CallTableDescriptor DescribeCallTable(const std::array<uint32_t, 4>& words, const ShaderSource::Call& site) {
	CallTableDescriptor d;
	d.base       = ((uint64_t {words[1] & 0xffffu} << 32u) | words[0]) & ~uint64_t {3};
	d.raw_stride = (words[1] >> 16u) & 0x3fffu;
	d.size       = d.raw_stride != 0u ? d.raw_stride * words[2] : words[2];
	d.reach      = std::min<uint64_t>(d.size, uint64_t {1} << 32u);
	d.first      = site.table_addend % site.table_stride;
	d.slots      = d.first < d.reach ? (d.reach - d.first + site.table_stride - 1u) / site.table_stride : 0u;
	return d;
}

// Strict aggregate of table read requests (see MaxCallTableAttemptBytes). Always accumulated;
// enforced only when the refresh has a large site.
struct CallTableAttempts {
	bool     enforced = false;
	uint64_t bytes    = 0;
	// Charges a request BEFORE it is attempted (a failed attempt stays charged).
	[[nodiscard]] bool Charge(uint64_t request) {
		if (enforced && (bytes > MaxCallTableAttemptBytes || request > MaxCallTableAttemptBytes - bytes))
			return false;
		bytes += request;
		return true;
	}
	[[nodiscard]] std::string Refusal(uint64_t request) const {
		return fmt::format("exceeds the {}-byte table attempt budget ({} bytes requested before this {}-byte read)",
		                   MaxCallTableAttemptBytes, bytes, request);
	}
};

// Optional descriptor-data proof; unsupported proofs preserve the original loads.
constexpr uint64_t MaxCallContextBytes = uint64_t {8} << 20u;
constexpr size_t MaxCallContexts = 4096u;
constexpr size_t MaxCallContextRanges = 16384u;
// Optional reaching-definition analysis must remain bounded independently of linking.
constexpr size_t MaxContextFlowInstructions = 8192u;
constexpr size_t MaxContextFlowBlocks = 512u;
constexpr uint32_t ContextLoadMarker = 0xbf800003u;

void AppendCaptured(std::vector<ShaderSource::CapturedData>& ranges, uint64_t address,
                    std::span<const uint32_t> words) {
	if (words.empty()) return;
	if (!ranges.empty() && ranges.back().address + ranges.back().words.size() * 4u == address) {
		ranges.back().words.insert(ranges.back().words.end(), words.begin(), words.end());
	} else {
		ranges.push_back({address, {words.begin(), words.end()}});
	}
}

struct CallTableContext {
	bool usable = true;
	size_t unique_contexts = 0;
	uint64_t bytes = 0;
	std::vector<ShaderSource::CapturedData> records;
	std::map<uint64_t, std::set<uint64_t>> contexts;
	struct Pending { uint64_t address; std::array<uint32_t, 4> entry; uint32_t available; };
	std::vector<Pending> pending;
	void Defer(uint64_t address, const std::array<uint32_t, 4>& entry, uint32_t available) {
		if (!usable) return;
		// A third in-range DWORD is real companion data even for a null target.
		// Read it before capture instead of publishing its zero-initialized slot.
		if (available <= 2u) { Observe(address, entry, available); return; }
		if (pending.size() >= MaxCallContextRanges) { usable = false; return; }
		pending.push_back({address, entry, available});
	}
	void Observe(uint64_t address, const std::array<uint32_t, 4>& entry, uint32_t available) {
		if (!usable) return;
		const auto target = ((uint64_t {entry[1]} << 32u) | entry[0]) & ~uint64_t {3};
		if (available > 4u || (target != 0u && available != 4u) ||
		    bytes > MaxCallContextBytes - available * 4u || records.size() >= MaxCallContextRanges) {
			usable = false; return;
		}
		AppendCaptured(records, address, std::span(entry).first(available));
		bytes += available * 4u;
		if (target == 0u) return;
		const auto context = (uint64_t {entry[3]} << 32u) | entry[2];
		if (context == 0u || (context & 3u) != 0u || (context >> 48u) != 0u) {
			usable = false; return;
		}
		if (contexts[target].insert(context).second && ++unique_contexts > MaxCallContexts) usable = false;
	}
};

bool NormalizeCaptured(std::vector<ShaderSource::CapturedData>& ranges) {
	std::ranges::sort(ranges, {}, &ShaderSource::CapturedData::address);
	std::vector<ShaderSource::CapturedData> merged;
	uint64_t bytes = 0;
	for (auto& range: ranges) {
		const auto size = uint64_t {range.words.size()} * 4u;
		if (range.words.empty() || (range.address & 3u) != 0u || range.address > UINT64_MAX - size) return false;
		if (!merged.empty() && range.address <= merged.back().address + merged.back().words.size() * 4u) {
			auto& previous = merged.back();
			const auto offset = (range.address - previous.address) / 4u;
			const auto overlap = std::min<uint64_t>(range.words.size(), previous.words.size() - offset);
			if (!std::equal(range.words.begin(), range.words.begin() + overlap, previous.words.begin() + offset)) return false;
			previous.words.insert(previous.words.end(), range.words.begin() + overlap, range.words.end());
			bytes += (range.words.size() - overlap) * 4u;
		} else {
			bytes += size;
			merged.push_back(std::move(range));
		}
		if (bytes > MaxCallContextBytes || merged.size() > MaxCallContextRanges) return false;
	}
	ranges = std::move(merged);
	return true;
}

struct CallTableScan {
	uint64_t null_entries = 0, span_bytes = 0, chunks = 0, fallback_dwords = 0, ranges = 0;
};

using CallTableReader = std::function<bool(uint64_t, std::span<uint32_t>)>;

// Enumerates every potential entry of a large table in first-appearance order.
// Transport: a batch reads one contiguous span of whole in-range DWORDs through the dispatch's
// ordered reader (same reader and synchronization as single DWORDs). It is NOT a dependency.
// Dependencies: exactly the in-range pointer DWORDs, as the small path records them, merged only
// with the immediately preceding record of THIS site when byte-adjacent (no gap is ever bridged).
// All of it lands before the caller's query_reads checkpoint. DWORDs with at + 4 > size read zero
// and are never read. A failed batch falls back to exact single-DWORD reads. Every bound
// (scan bytes, dependency ranges) is checked before the site's first read; fallback DWORDs are
// charged before each attempt. Returns a failure reason, or an empty string.
std::string ScanLargeCallTable(const CallTableGeometry& table, std::vector<uint64_t>& targets,
                               CallTableScan& scan, uint64_t& scan_budget, uint64_t& range_budget,
                               uint64_t& fallback_budget, CallTableAttempts& attempts,
                               const CallTableReader& read_transport,
                               std::vector<std::pair<uint64_t, uint64_t>>& reads,
                               CallTableContext* context = nullptr) {
	static thread_local std::vector<uint32_t> chunk(CallTableChunkBytes / sizeof(uint32_t));
	// Every pointer DWORD offset is a multiple of 4 (base, addend, stride and word offset are), so
	// "at + 4 <= size" is exactly "at < limit".
	const uint64_t limit      = table.size & ~uint64_t {3};
	const uint64_t span_begin = table.first + table.word_offset; // < 2^32 + 2^31
	const uint64_t rows       = span_begin < limit
	                                ? std::min(table.slots, (limit - span_begin - 1u) / table.stride + 1u)
	                                : 0u;
	scan.null_entries += table.slots - rows; // both halves out of range: the pair reads zero
	if (rows == 0u) return {};
	// (rows - 1) * stride <= limit - span_begin - 1, so no term below can wrap.
	const uint64_t span_end = std::min(span_begin + (rows - 1u) * table.stride + 8u, limit);
	scan.span_bytes         = span_end - span_begin;
	// lo/hi of a row are adjacent; rows are adjacent only at stride 8.
	scan.ranges = table.stride == 8u ? 1u : rows;
	if (scan.span_bytes > scan_budget)
		return fmt::format("needs {} scan bytes for {} entries (remaining budget {} of {})", scan.span_bytes,
		                   table.slots, scan_budget, MaxCallTableScanBytes);
	if (scan.ranges > range_budget)
		return fmt::format("needs {} dependency ranges (remaining budget {} of {})", scan.ranges,
		                   range_budget, MaxCallTableDependencyRanges);
	if (table.base > UINT64_MAX - span_end) return "runs past the address space";
	scan_budget -= scan.span_bytes;
	range_budget -= scan.ranges;
	const size_t own_records = reads.size();
	reads.reserve(own_records + scan.ranges);
	const auto depend = [&](uint64_t address) {
		if (reads.size() > own_records && reads.back().first + reads.back().second == address) {
			reads.back().second += sizeof(uint32_t);
			return true;
		}
		if (reads.size() - own_records >= scan.ranges) return false; // exceeds the pre-checked count
		reads.emplace_back(address, sizeof(uint32_t));
		return true;
	};
	// Rows per batch: the last row's 8 pointer bytes still end inside one chunk.
	const uint64_t per_chunk = (CallTableChunkBytes - 8u) / table.stride + 1u;
	std::array<uint64_t, MaxCallTableTargets> sorted {};
	uint64_t last = 0;
	for (uint64_t row = 0; row < rows; row += per_chunk) {
		const uint64_t count   = std::min(per_chunk, rows - row);
		const uint64_t begin   = span_begin + row * table.stride;
		const uint64_t end     = std::min(begin + (count - 1u) * table.stride + 8u, limit);
		const auto     words   = std::span(chunk).first((end - begin) / sizeof(uint32_t));
		if (!attempts.Charge(words.size_bytes())) return attempts.Refusal(words.size_bytes());
		const bool batched = read_transport(table.base + begin, words);
		++scan.chunks;
		for (uint64_t index = 0; index < count; ++index) {
			const uint64_t relative = index * table.stride;
			std::array<uint32_t, 4> entry {};
			for (uint32_t half = 0; half < 2u; ++half) {
				const uint64_t at = begin + relative + half * sizeof(uint32_t);
				if (at >= limit) continue; // Out of range reads zero.
				if (batched) {
					entry[half] = words[(relative + half * sizeof(uint32_t)) / sizeof(uint32_t)];
				} else {
					if (fallback_budget == 0u)
						return fmt::format("exhausted its single-DWORD fallback budget ({}) at 0x{:x}",
						                   MaxCallTableFallbackDwords, table.base + at);
					if (!attempts.Charge(sizeof(uint32_t))) return attempts.Refusal(sizeof(uint32_t));
					--fallback_budget;
					++scan.fallback_dwords;
					if (!read_transport(table.base + at, std::span(entry).subspan(half, 1)))
						return fmt::format("entry at 0x{:x} is unreadable", table.base + at);
				}
				if (!depend(table.base + at))
					return fmt::format("exceeded its {} pre-checked dependency ranges", scan.ranges);
			}
			// Companion bytes in an existing transport gap cost no extra read.
			// Missing words wait until every mandatory read has completed.
			if (context != nullptr && context->usable) {
				const auto available = static_cast<uint32_t>(std::min<uint64_t>(16u, limit - begin - relative) / 4u);
				if (batched && relative + available * 4u <= words.size_bytes()) {
					for (uint32_t half = 2u; half < available; ++half) entry[half] = words[relative / 4u + half];
					context->Observe(table.base + begin + relative, entry, available);
				} else {
					context->Defer(table.base + begin + relative, entry, available);
				}
			}
			const auto target = ((uint64_t {entry[1]} << 32u) | entry[0]) & ~uint64_t {3};
			if (target == 0u) {
				++scan.null_entries;
				continue;
			}
			if (target == last) continue;
			uint64_t* const known_begin = sorted.data();
			uint64_t* const known_end   = known_begin + targets.size();
			uint64_t* const position    = std::lower_bound(known_begin, known_end, target);
			last                        = target;
			if (position != known_end && *position == target) continue;
			if (targets.size() == MaxCallTableTargets)
				return fmt::format("names more than {} callees", MaxCallTableTargets);
			std::copy_backward(position, known_end, known_end + 1);
			*position = target;
			targets.push_back(target);
		}
	}
	return {};
}

} // namespace

void PrepareCallTargets(ShaderSource& source, const CompileOptions& options) {
	using namespace IR;
	EXIT_NOT_IMPLEMENTED(options.stage != ShaderType::Compute);
	source.wave_size = options.wave_size;
	for (uint32_t i = 0; i < source.decoded.instructions.size(); ++i) {
		if (source.decoded.instructions[i].opcode != Decoder::Opcode::S_SWAPPC_B64) continue;
		source.calls.push_back({.instruction = i});
	}
	const auto unsupported = [&](uint32_t pc, const char* reason) {
		source.unsupported_call_pc     = pc;
		source.unsupported_call_reason = reason;
		source.calls.clear();
	};
	const auto graph = CFG::BuildGraph(source.decoded);
	Program query;
	query.stage = options.stage;
	query.user_data_base = options.user_data_base;
	query.user_data_count = static_cast<uint32_t>(options.user_data.size());
	query.wave_size = options.wave_size;
	auto& block = *query.block_storage.emplace_back(std::make_unique<Block>());
	query.blocks.push_back(&block);
	struct Producer { bool ready = false; std::map<uint32_t, Value> values; };
	std::map<uint32_t, Producer> producers;
	using ContextQuery = std::pair<uint32_t, std::array<Value, 2>>;
	std::map<uint32_t, std::vector<ContextQuery>> context_queries;
	std::function<Value(uint32_t, uint32_t)> Register;
	std::function<Value(uint32_t, Value)> Resolve = [&](uint32_t before, Value value) -> Value {
		value = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr) return value;
		uint32_t code = UINT32_MAX;
		switch (inst->GetOpcode()) {
			case ValueOpcode::GetScalarRegister: code = static_cast<uint32_t>(inst->Arg(0).ScalarRegister()); break;
			case ValueOpcode::GetVccLo: code = 106u; break;
			case ValueOpcode::GetVccHi: code = 107u; break;
			default: break;
		}
		if (code != UINT32_MAX) {
			const auto result = Register(before, code);
			inst->ReplaceUsesWith(result);
			return result;
		}
		for (size_t i = 0; i < inst->NumArgs(); ++i) inst->SetArg(i, Resolve(before, inst->Arg(i)));
		return value;
	};
	Register = [&](uint32_t before, uint32_t code) -> Value {
		std::vector<uint32_t> crossed_calls;
		const auto definition = CFG::FindScalarDefinition(source.decoded, graph, before, code,
		                                                  source.wave_size, crossed_calls);
		// Several reaching definitions select the value on the GPU, so no SRT walk evaluates it.
		if (!definition) return Value(&block.AppendNewInst(ValueOpcode::UndefU32));
		const auto index = *definition;
		for (const auto crossed: crossed_calls) {
			std::ranges::find(source.calls, crossed, &ShaderSource::Call::instruction)->preserved_registers.set(code);
		}
		if (index == UINT32_MAX) {
			EXIT_IF(code < query.user_data_base || code - query.user_data_base >= query.user_data_count);
			return IREmitter(&block).GetUserData(static_cast<ScalarReg>(code));
		}
		auto [entry, inserted] = producers.try_emplace(index);
		auto& producer = entry->second;
		if (inserted) {
			Block translated;
			Frontend::Translator translator(query, &translated, 1u, Frontend::NoLaneSpills,
			    (options.input_info.compute->float_mode & 0x10u) == 0u,
			    !options.input_info.compute->async_compute);
			translator.TranslateInstruction(source.decoded.instructions[index]);
			for (const auto& inst: translated) {
				uint32_t destination = UINT32_MAX;
				Value value;
				switch (inst.GetOpcode()) {
					case ValueOpcode::SetScalarRegister:
						destination = static_cast<uint32_t>(inst.Arg(0).ScalarRegister()); value = inst.Arg(1); break;
					case ValueOpcode::SetVccLo: destination = 106u; value = inst.Arg(0); break;
					case ValueOpcode::SetVccHi: destination = 107u; value = inst.Arg(0); break;
					default: break;
				}
				if (destination != UINT32_MAX) producer.values.insert_or_assign(destination, value);
			}
			for (auto& [destination, value]: producer.values) value = Resolve(index, value);
			for (auto& inst: translated) inst.SetParent(&block);
			block.Instructions().splice(block.end(), translated.Instructions());
			producer.ready = true;
		}
		EXIT_IF(!producer.ready || !producer.values.contains(code));
		return producer.values.at(code).Resolve();
	};
	for (auto& site: source.calls) {
		const auto& call = source.decoded.instructions[site.instruction];
		EXIT_NOT_IMPLEMENTED(call.src0.kind != Decoder::OperandKind::Sgpr ||
		                     call.dst.kind != Decoder::OperandKind::Sgpr ||
		                     call.src0.reg != call.dst.reg || call.dst.reg >= 105u);
		std::array<Value, 2> values;
		bool walkable = true;
		for (uint32_t i = 0; i < 2u; ++i) {
			values[i] = Register(site.instruction, call.src0.reg + i);
			walkable &= ValidateRuntimeValue(query, values[i], RuntimeValueType::Integer);
		}
		if (!walkable) {
			// A GPU-selected target is only supported as an entry of a closed buffer table,
			// enumerated per dispatch. Anything else stays an explicit unsupported call.
			const bool buffer_loads = std::ranges::all_of(values, [](Value value) {
				const auto* inst = value.Resolve().TryInstruction();
				return inst != nullptr && inst->GetOpcode() == ValueOpcode::ReadConstBuffer;
			});
			if (!buffer_loads) {
				unsupported(call.pc, "target is neither host-walkable nor a scalar buffer table entry");
				return;
			}
			site.kind = ShaderSource::Call::Kind::Table;
		}
		if (site.kind == ShaderSource::Call::Kind::Table) {
			const auto* target = values[0].Resolve().TryInstruction();
			const auto flags = target->Flags<MemoryFlags>();
			const auto found = std::ranges::find(source.decoded.instructions, flags.pc, &Decoder::Instruction::pc);
			if (found != source.decoded.instructions.end() && found->dst.kind == Decoder::OperandKind::Sgpr &&
			    found->family == Decoder::Family::SMEM && flags.index < query.memory_info.size()) {
				const auto offset = query.memory_info[flags.index].offset;
				const auto component = (offset - found->offset) / 4u;
				const auto producer_index = static_cast<uint32_t>(found - source.decoded.instructions.begin());
				if (offset >= found->offset && (offset - found->offset) % 4u == 0u &&
				    component + 3u < found->data_dwords) {
					const auto originates = [&](const auto& self, uint32_t before, uint32_t reg,
					                             uint32_t word, uint32_t depth) -> bool {
						if (depth == 32u) return false;
						// The mandatory definition API fatally rejects crossing a call
						// that writes this register. Optional probing must never enter it.
						for (const auto& other: source.calls) {
							const auto& destination = source.decoded.instructions[other.instruction].dst;
							if (destination.kind == Decoder::OperandKind::Sgpr && reg >= destination.reg &&
							    reg - destination.reg < 2u) return false;
						}
						std::vector<uint32_t> crossed;
						const auto definition = CFG::FindScalarDefinition(source.decoded, graph, before, reg,
						                                                   source.wave_size, crossed);
						// Optional provenance must not impose new preservation requirements
						// on an earlier call through Register().
						if (!definition || *definition == UINT32_MAX || !crossed.empty()) return false;
						if (*definition == producer_index) return reg == found->dst.reg + component + word;
						const auto& move = source.decoded.instructions[*definition];
						if (move.src0.kind != Decoder::OperandKind::Sgpr || move.dst.kind != Decoder::OperandKind::Sgpr ||
						    reg < move.dst.reg) return false;
						const auto half = reg - move.dst.reg;
						if (!((move.opcode == Decoder::Opcode::S_MOV_B32 && half == 0u) ||
						      (move.opcode == Decoder::Opcode::S_MOV_B64 && half < 2u))) return false;
						return self(self, *definition, move.src0.reg + half, word, depth + 1u);
					};
					for (uint32_t reg = 0; reg < 105u; ++reg) {
						if (reg <= call.dst.reg + 1u && call.dst.reg <= reg + 1u) continue;
						if (!originates(originates, site.instruction, reg, 2u, 0u) ||
						    !originates(originates, site.instruction, reg + 1u, 3u, 0u)) continue;
						std::array<Value, 2> companion;
						for (uint32_t half = 0; half < 2u; ++half) {
							auto& root = query.value_storage.emplace_back(ValueOpcode::ReferenceU32);
							root.SetArg(0, Register(site.instruction, reg + half));
							companion[half] = Value(&root);
						}
						context_queries[site.instruction].emplace_back(reg, companion);
					}
				}
			}
		}
		for (uint32_t i = 0; i < 2u; ++i) {
			auto& root = query.value_storage.emplace_back(ValueOpcode::ReferenceU32);
			root.SetArg(0, values[i]);
			site.target[i] = Value(&root);
		}
	}
	RewriteToSsa(query.blocks);
	ConstantPropagationPass(query.blocks, query.wave_size);
	RemoveIdentities(query);
	EliminateDeadCode(query.blocks);
	for (auto& site: source.calls)
		for (auto& value: site.target) value = value.Instruction()->Arg(0).Resolve();
	std::vector<Value> targets;
	for (auto& site: source.calls) {
		if (site.kind == ShaderSource::Call::Kind::Table) {
			if (const auto* reason = MatchCallTable(query, site)) {
				unsupported(source.decoded.instructions[site.instruction].pc, reason);
				return;
			}
			for (auto& [reg, values]: context_queries[site.instruction]) {
				for (auto& value: values) value = value.Instruction()->Arg(0).Resolve();
				if (MatchCallContext(query, site, values)) site.context_registers.set(reg);
			}
			targets.insert(targets.end(), site.table_descriptor.begin(), site.table_descriptor.end());
		} else {
			targets.insert(targets.end(), site.target.begin(), site.target.end());
		}
	}
	query.value_storage.clear();
	for (auto& inst: block) {
		inst.SetParent(nullptr);
		if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 || inst.GetOpcode() == ValueOpcode::ReadConstBuffer)
			inst.SetFlags(MemoryFlags {.index = inst.Flags<MemoryFlags>().index});
	}
	query.value_storage.splice(query.value_storage.end(), block.Instructions());
	source.call_targets = std::move(static_cast<ResourcePlan&>(query));
	CompileSrtWalker(source.call_targets, targets);
}

namespace {

struct LeafRead {
	std::vector<Decoder::Instruction> body;
	std::string                       failure;
};

// Reads and decodes one leaf callee through the dispatch's ordered reader. Soft failures are
// reported for table targets; uniform targets keep their historical fatal diagnostics.
// decode_budget (non-null only for a refresh with a large table site) bounds the decoded bytes of
// all unique callees: charged once per decoded instruction before the body grows; an exhausted
// budget refuses softly, whatever the site kind, before any further read.
LeafRead ReadLeaf(uint64_t address, uint32_t call_pc, bool soft,
                  const std::function<bool(uint64_t, std::span<uint32_t>)>& read_code,
                  uint64_t* decode_budget = nullptr) {
	LeafRead result;
	const auto fail = [&](std::string reason) {
		if (!soft) EXIT("shader call at pc 0x%08x %s", call_pc, reason.c_str());
		result.failure = std::move(reason);
		result.body.clear();
		return result;
	};
	const auto refuse_budget = [&] {
		result.failure = fmt::format("exceeds the {}-byte decoded callee code budget", MaxCallDecodedBytes);
		result.body.clear();
		return result;
	};
	for (uint32_t index = 0;;) {
		std::array<uint32_t, Decoder::MaxInstructionRawWords> words {};
		EXIT_IF(index > UINT32_MAX / sizeof(uint32_t) - words.size());
		if (address > UINT64_MAX - (uint64_t {index} + words.size()) * sizeof(uint32_t))
			return fail("has a leaf that runs past the address space");
		if (index >= MaxLeafWords) return fail("has a leaf without a bounded S_SETPC_B64 return");
		if (decode_budget != nullptr && *decode_budget == 0u) return refuse_budget();
		uint32_t count = 1;
		Decoder::Instruction inst;
		for (;;) {
			if (!read_code(address + uint64_t {index} * sizeof(uint32_t), std::span(words).first(count)))
				return fail("cannot read leaf code");
			inst = {};
			if (soft) {
				// GPU-selected code is untrusted: a malformed word fails this dispatch, not the process.
				Decoder::SoftDecodeScope scope;
				Decoder::DecodeInstruction(words, 0u, inst);
				if (scope.Failed()) return fail("has a malformed leaf instruction (" + scope.Reason() + ")");
			} else {
				Decoder::DecodeInstruction(words, 0u, inst);
			}
			if (inst.word_count == 0u || inst.word_count > words.size())
				return fail("has a leaf instruction of invalid length");
			if (inst.word_count <= count) break;
			count = inst.word_count;
		}
		if (decode_budget != nullptr) {
			const uint64_t bytes = uint64_t {inst.word_count} * sizeof(uint32_t);
			if (bytes > *decode_budget) return refuse_budget();
			*decode_budget -= bytes;
		}
		inst.pc = index * sizeof(uint32_t);
		if (Decoder::IsDirectBranch(inst.opcode)) inst.branch_target += inst.pc;
		index += inst.word_count;
		if (inst.opcode == Decoder::Opcode::UNKNOWN || inst.opcode == Decoder::Opcode::UNSUPPORTED)
			return fail("has a leaf with an undecodable instruction");
		if (inst.opcode == Decoder::Opcode::S_SWAPPC_B64 ||
		    (!soft && inst.opcode == Decoder::Opcode::S_GETPC_B64) ||
		    inst.opcode == Decoder::Opcode::S_ENDPGM) {
			if (!soft) EXIT_NOT_IMPLEMENTED(true);
			static const bool diagnostic = [] {
				const char* value = std::getenv("KYTY_CALL_TABLE_DIAG");
				return value != nullptr && value[0] == '1';
			}();
			if (diagnostic) {
				const char* opcode = inst.opcode == Decoder::Opcode::S_SWAPPC_B64 ? "S_SWAPPC_B64" :
				                     inst.opcode == Decoder::Opcode::S_GETPC_B64 ? "S_GETPC_B64" : "S_ENDPGM";
				const auto raw = inst.word_count > 1u ? fmt::format("{:08x},{:08x}", words[0], words[1])
				                                    : fmt::format("{:08x}", words[0]);
				return fail(fmt::format(
				    "has a leaf with a nested call, PC read or program end "
				    "opcode={} leaf_pc=0x{:x} absolute=0x{:x} instruction_bytes={} scanned_leaf_bytes={} raw={}",
				    opcode, inst.pc, address + uint64_t {inst.pc}, uint64_t {inst.word_count} * sizeof(uint32_t),
				    uint64_t {index} * sizeof(uint32_t), raw));
			}
			return fail("has a leaf with a nested call, PC read or program end");
		}
		result.body.push_back(inst);
		if (inst.opcode == Decoder::Opcode::S_SETPC_B64) break;
	}
	return result;
}

// Each accepted load reads an incoming companion pair, either preserved by the
// entire true leaf or before any clobber in a non-reentered entry prefix.
std::vector<ShaderSource::Linked::ContextRead> FindContextReads(
    size_t site_index, uint32_t slot, const ShaderSource::Call& site,
    const LeafRead& leaf, uint32_t wave_size) {
	std::vector<ShaderSource::Linked::ContextRead> result;
	if (site.context_registers.none()) return result;
	uint32_t first_branch_target = UINT32_MAX;
	for (const auto& inst: leaf.body)
		if (Decoder::IsDirectBranch(inst.opcode)) first_branch_target = std::min(first_branch_target, inst.branch_target);
	auto stable = site.context_registers;
	for (const auto& inst: leaf.body) {
		if (inst.opcode == Decoder::Opcode::S_SETREG_B32 ||
		    inst.opcode == Decoder::Opcode::S_SWAPPC_B64 ||
		    inst.opcode == Decoder::Opcode::UNKNOWN || inst.opcode == Decoder::Opcode::UNSUPPORTED) return {};
		for (uint32_t reg = 0; reg < 105u; ++reg)
			if (stable[reg] && (CFG::MayWriteScalarRegister(inst, reg, wave_size) ||
			                    CFG::MayWriteScalarRegister(inst, reg + 1u, wave_size))) stable.reset(reg);
	}
	Decoder::Program flow;
	CFG::Graph flow_graph;
	bool flow_attempted = false, flow_usable = false;
	const auto reaches_entry = [&](uint32_t before, uint32_t reg) {
		if (!flow_attempted) {
			flow_attempted = true;
			if (leaf.body.size() > MaxContextFlowInstructions) return false;
			size_t branches = 0;
			for (const auto& inst: leaf.body) {
				if (!Decoder::IsDirectBranch(inst.opcode)) continue;
				// Each branch adds at most its target and fallthrough as leaders.
				if (++branches > (MaxContextFlowBlocks - 1u) / 2u) return false;
				const auto target = std::ranges::lower_bound(leaf.body, inst.branch_target,
				                                             {}, &Decoder::Instruction::pc);
				if (target == leaf.body.end() || target->pc != inst.branch_target) return false;
			}
			flow.instructions = leaf.body;
			// A true leaf's return is an exit for this analysis. Original instruction
			// PCs, raw slices and executable linked instructions are never changed.
			flow.instructions.back().opcode = Decoder::Opcode::S_ENDPGM;
			flow_graph = CFG::BuildGraph(flow);
			const auto* entry = flow_graph.FindBlock(flow_graph.entry_block);
			// FindScalarDefinition treats entry as a boundary. A backedge into it
			// must therefore disable this optional extension, not hide a clobber.
			flow_usable = !flow_graph.unsupported && entry != nullptr && entry->predecessors.empty();
		}
		if (!flow_usable || !std::ranges::any_of(flow_graph.blocks, [&](const auto& block) {
		        return block.inst_begin <= before && before < block.inst_end;
		    })) return false; // BuildGraph prunes unreachable instruction ranges.
		for (uint32_t half = 0; half < 2u; ++half) {
			std::vector<uint32_t> crossed;
			const auto definition = CFG::FindScalarDefinition(flow, flow_graph, before, reg + half,
			                                                  wave_size, crossed);
			if (!definition || *definition != UINT32_MAX || !crossed.empty()) return false;
		}
		return true;
	};
	auto live = site.context_registers;
	bool prefix = true;
	for (uint32_t index = 0; index < leaf.body.size(); ++index) {
		const auto& inst = leaf.body[index];
		const bool branch = Decoder::IsDirectBranch(inst.opcode) || inst.opcode == Decoder::Opcode::S_SETPC_B64;
		if (branch) prefix = false;
		const bool scalar_load = inst.opcode == Decoder::Opcode::S_LOAD_DWORD ||
		    inst.opcode == Decoder::Opcode::S_LOAD_DWORDX2 || inst.opcode == Decoder::Opcode::S_LOAD_DWORDX4 ||
		    inst.opcode == Decoder::Opcode::S_LOAD_DWORDX8 || inst.opcode == Decoder::Opcode::S_LOAD_DWORDX16;
		if (scalar_load && inst.src0.kind == Decoder::OperandKind::Sgpr && inst.src0.reg < 105u &&
		    inst.dst.kind == Decoder::OperandKind::Sgpr && inst.dst.reg + inst.data_dwords <= 106u &&
		    inst.data_dwords != 0u && inst.data_dwords <= 16u) {
			bool incoming = stable[inst.src0.reg];
			if (!incoming && prefix && live[inst.src0.reg]) {
				incoming = first_branch_target > inst.pc;
			}
			if (!incoming && site.context_registers[inst.src0.reg]) incoming = reaches_entry(index, inst.src0.reg);
			uint64_t dynamic = 0;
			bool fixed = inst.src1.kind == Decoder::OperandKind::Null;
			if (inst.src1.kind == Decoder::OperandKind::IntegerInlineConstant ||
			    inst.src1.kind == Decoder::OperandKind::LiteralConstant) {
				dynamic = inst.src1.value; fixed = true;
			}
			const int64_t offset = static_cast<int64_t>(dynamic) + static_cast<int32_t>(inst.offset);
			if (incoming && fixed && offset >= 0 && (dynamic & 3u) == 0u &&
			    (inst.offset & 3u) == 0u && (offset & 3) == 0 &&
			    static_cast<uint64_t>(offset) <= UINT32_MAX - uint64_t {inst.data_dwords} * 4u) {
				result.push_back({site_index, slot, inst.pc, static_cast<uint32_t>(offset), inst.data_dwords});
			}
		}
		for (uint32_t reg = 0; reg < 105u; ++reg)
			if (live[reg] && (CFG::MayWriteScalarRegister(inst, reg, wave_size) ||
			                  CFG::MayWriteScalarRegister(inst, reg + 1u, wave_size))) live.reset(reg);
	}
	return result;
}

struct ContextProof {
	std::vector<ShaderSource::Linked::ContextLoad> loads;
	std::vector<ShaderSource::CapturedData> data;
};
using ContextPayloadCache = std::map<std::pair<uint64_t, uint32_t>, std::optional<std::vector<uint32_t>>>;

ContextProof ProveContextLoads(std::span<const ShaderSource::Linked::ContextRead> candidates,
                              std::vector<CallTableContext>& tables,
                              const std::vector<std::vector<uint64_t>>& targets,
                              CallTableAttempts& attempts, ContextPayloadCache& cache,
                              const CallTableReader& reader) {
	ContextProof proof;
	if (candidates.size() > MaxCallContextRanges) return proof;
	// Optional companion probes run after all mandatory pointer/code reads.
	std::set<size_t> requested_tables;
	for (const auto& candidate: candidates) if (candidate.site < tables.size()) requested_tables.insert(candidate.site);
	for (const auto site: requested_tables) {
		auto& table = tables[site];
		if (!table.usable) continue;
		for (auto& pending: table.pending) {
			const uint64_t bytes = uint64_t {pending.available - 2u} * 4u;
			if (pending.address > UINT64_MAX - 16u || attempts.bytes > MaxCallTableAttemptBytes - bytes ||
			    !attempts.Charge(bytes) || !reader(pending.address + 8u,
			        std::span(pending.entry).subspan(2u, pending.available - 2u))) { table.usable = false; break; }
			table.Observe(pending.address, pending.entry, pending.available);
			if (!table.usable) break;
		}
		table.pending.clear();
	}
	std::set<size_t> used_tables;
	std::set<std::pair<uint64_t, uint32_t>> used_payloads;
	for (const auto& load: candidates) {
		if (load.site >= tables.size() || load.site >= targets.size() ||
		    load.slot >= targets[load.site].size() || !tables[load.site].usable) continue;
		const auto found = tables[load.site].contexts.find(targets[load.site][load.slot]);
		if (found == tables[load.site].contexts.end() || found->second.empty()) continue;
		bool invariant = true;
		std::vector<uint32_t> expected;
		std::vector<std::pair<uint64_t, uint32_t>> payloads;
		for (const uint64_t context: found->second) {
			const auto bytes = uint64_t {load.dwords} * 4u;
			constexpr uint64_t address_limit = uint64_t {1} << 48u;
			if (load.dwords == 0u || load.dwords > 16u || context >= address_limit ||
			    load.offset > address_limit - context || bytes > address_limit - context - load.offset) {
				invariant = false; break;
			}
			const std::pair<uint64_t, uint32_t> key {context + load.offset, load.dwords};
			auto [entry, inserted] = cache.try_emplace(key);
			if (inserted) {
				if (cache.size() > MaxCallContextRanges || attempts.bytes > MaxCallTableAttemptBytes - bytes ||
				    !attempts.Charge(bytes)) { invariant = false; break; }
				std::vector<uint32_t> words(load.dwords);
				if (reader(key.first, words)) entry->second = std::move(words);
			}
			if (!entry->second) { invariant = false; break; }
			if (expected.empty()) expected = *entry->second;
			else if (expected != *entry->second) { invariant = false; break; }
			payloads.push_back(key);
		}
		if (!invariant || expected.empty()) continue;
		proof.loads.push_back({load.site, load.slot, load.local_pc, expected});
		used_tables.insert(load.site);
		for (const auto& key: payloads) {
			if (used_payloads.insert(key).second) proof.data.push_back({key.first, *cache.at(key)});
		}
	}
	for (const auto site: used_tables)
		proof.data.insert(proof.data.end(), tables[site].records.begin(), tables[site].records.end());
	if (!NormalizeCaptured(proof.data)) return {};
	return proof;
}

void PublishContextData(ShaderSource& source, ContextProof proof) {
	source.call_data = std::move(proof.data);
	source.call_code_ranges.clear();
	const auto& linked = *source.linked;
	for (const auto& function: linked.functions) {
		if (source.calls[function.site].kind != ShaderSource::Call::Kind::Table) continue;
		const std::pair<uint64_t, uint64_t> range {linked.site_targets[function.site][function.slot],
		                                           uint64_t {function.size} * sizeof(uint32_t)};
		if (std::ranges::find(source.call_code_ranges, range) == source.call_code_ranges.end())
			source.call_code_ranges.push_back(range);
	}
	uint64_t bytes = 0;
	for (const auto& range: source.call_data) {
		const std::pair<uint64_t, uint64_t> span {range.address, uint64_t {range.words.size()} * 4u};
		source.call_code_ranges.push_back(span);
		source.reads.push_back(span);
		bytes += span.second;
	}
	if (!source.call_data.empty()) source.call_diagnostic += fmt::format(
	    "; context_loads={} context_snapshot_bytes={} context_snapshot_ranges={}",
	    linked.context_loads.size(), bytes, source.call_data.size());
}

} // namespace

const Decoder::Program* TryRefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime,
                                               std::string& failure) {
	EXIT_IF(source.unsupported_call_pc != UINT32_MAX);
	failure.clear();
	auto& reads = source.reads;
	reads.clear();
	source.call_data.clear();
	std::vector<uint32_t> flat;
	const bool walked = IR::WalkSrt(source.call_targets, runtime, flat, &reads);
	const std::function<bool(uint64_t, std::span<uint32_t>)> read_code =
	    [&](uint64_t address, std::span<uint32_t> words) {
		    if (runtime.read_specialization_memory == nullptr ||
		        !runtime.read_specialization_memory(runtime.userdata, address, words)) return false;
		    reads.emplace_back(address, words.size_bytes());
		    return true;
	    };
	const auto fail = [&](uint32_t pc, std::string reason) -> const Decoder::Program* {
		failure = fmt::format("pc 0x{:08x}: {}", pc, reason);
		return nullptr;
	};
	// The closed callee set of every site for THIS dispatch, in first-appearance order.
	std::vector<std::vector<uint64_t>> plan(source.calls.size());
	std::vector<CallTableContext> context_tables(source.calls.size());
	for (auto& context: context_tables) context.usable = false;
	uint64_t context_capture_budget = MaxCallContextBytes;
	ContextPayloadCache payload_cache;
	source.call_diagnostic.clear();
	source.call_table_ranges.clear();
	// Aggregate large-table work bounds for this refresh (see MaxCallTableScanBytes).
	uint64_t scan_budget     = MaxCallTableScanBytes;
	uint64_t range_budget    = MaxCallTableDependencyRanges;
	uint64_t fallback_budget = MaxCallTableFallbackDwords;
	// Detect a large site BEFORE any table work, so every request of this refresh is charged
	// under one policy. Unavailable descriptors still fail in the loop below.
	CallTableAttempts attempts;
	for (const auto& site: source.calls) {
		if (site.kind != ShaderSource::Call::Kind::Table || !walked) continue;
		std::array<uint32_t, 4> words {};
		bool available = true;
		for (uint32_t word = 0; available && word < 4u; ++word)
			available = IR::WalkedValue(source.call_targets, site.table_descriptor[word], words[word]);
		if (available && DescribeCallTable(words, site).slots > MaxCallTableSlots) attempts.enforced = true;
	}
	const bool large_refresh = attempts.enforced;
	// Large-table transport: the same strict ordered reader, but the caller journals dependencies.
	const CallTableReader read_transport = [&](uint64_t address, std::span<uint32_t> words) {
		return runtime.read_specialization_memory != nullptr &&
		       runtime.read_specialization_memory(runtime.userdata, address, words);
	};
	for (size_t i = 0; i < source.calls.size(); ++i) {
		auto&       site = source.calls[i];
		const auto& call = source.decoded.instructions[site.instruction];
		if (site.kind == ShaderSource::Call::Kind::Uniform) {
			uint32_t low = 0, high = 0;
			if (!walked || !IR::WalkedValue(source.call_targets, site.target[0], low) ||
			    !IR::WalkedValue(source.call_targets, site.target[1], high))
				EXIT("shader call at pc 0x%08x has an unavailable scalar target", call.pc);
			// SWAPPC reads the old aliased pair and ignores the target's low two bits.
			const auto address = ((uint64_t {high} << 32u) | low) & ~uint64_t {3};
			EXIT_IF(address == 0u);
			site.address = address;
			plan[i] = {address};
			continue;
		}
		std::array<uint32_t, 4> words {};
		for (uint32_t word = 0; word < 4u; ++word) {
			if (!walked || !IR::WalkedValue(source.call_targets, site.table_descriptor[word], words[word]))
				return fail(call.pc, "call table descriptor is unavailable");
		}
		const auto     table  = DescribeCallTable(words, site);
		const uint64_t base   = table.base;
		const uint64_t stride = table.raw_stride;
		const uint64_t size   = table.size;
		const uint64_t reach  = table.reach;
		const uint64_t first  = table.first;
		const uint64_t slots  = table.slots;
		if (slots == 0u) return fail(call.pc, fmt::format("call table at 0x{:x} has no entries", base));
		CallTableContext* context = nullptr;
		const uint64_t first_word = first + site.table_word_offset;
		const uint64_t limit = size & ~uint64_t {3};
		const uint64_t rows = first_word < limit ? std::min(slots, (limit - first_word - 1u) / site.table_stride + 1u) : 0u;
		const uint64_t end_word = rows != 0u ? std::min(first_word + (rows - 1u) * site.table_stride + 16u, limit) : first_word;
		const uint64_t record_bytes = rows != 0u ? (rows - 1u) * 16u + std::min<uint64_t>(16u, limit - first_word - (rows - 1u) * site.table_stride) : 0u;
		if (site.context_registers.any() && record_bytes <= context_capture_budget &&
		    end_word >= first_word) {
			context = &context_tables[i]; context->usable = true;
			context_capture_budget -= record_bytes;
		}
		source.call_table_ranges.emplace_back(base, std::min<uint64_t>(size, reach + site.table_word_offset + 8u));
		auto& targets = plan[i];
		uint64_t null_entries = 0;
		if (slots > MaxCallTableSlots) {
			CallTableScan scan;
			const auto reason = ScanLargeCallTable(
			    {.base = base, .size = size, .first = first, .slots = slots, .stride = site.table_stride,
			     .word_offset = site.table_word_offset},
			    targets, scan, scan_budget, range_budget, fallback_budget, attempts, read_transport, reads, context);
			if (!reason.empty())
				return fail(call.pc, fmt::format(
				    "call table at 0x{:x} {} "
				    "vsharp={:08x},{:08x},{:08x},{:08x} raw_stride={} records={} bytes={} "
				    "entry_stride={} addend=0x{:x} word_offset={} first={} reach={} entries={}",
				    base, reason, words[0], words[1], words[2], words[3],
				    stride, words[2], size, site.table_stride, site.table_addend,
				    site.table_word_offset, first, reach, slots));
			if (targets.empty()) return fail(call.pc, fmt::format("call table at 0x{:x} has only null entries", base));
			site.address = 0;
			source.call_diagnostic += fmt::format(
			    "{}pc=0x{:08x} table=0x{:x} bytes={} entry_stride={} entries={} null={} callees={} "
			    "scan_bytes={} chunks={} fallback_dwords={} dependency_ranges={}",
			    source.call_diagnostic.empty() ? "" : "; ", call.pc, base, size, site.table_stride, slots,
			    scan.null_entries, targets.size(), scan.span_bytes, scan.chunks, scan.fallback_dwords,
			    scan.ranges);
			continue;
		}
		// Small table: exact per-DWORD reads; its bytes and records count toward (never fail) the
		// large-table budgets.
		scan_budget -= std::min(scan_budget, slots * 8u);
		range_budget -= std::min(range_budget, slots * 2u);
		for (uint64_t offset = first; offset < reach; offset += site.table_stride) {
			std::array<uint32_t, 4> entry {};
			for (uint32_t half = 0; half < 2u; ++half) {
				const uint64_t at = offset + site.table_word_offset + half * sizeof(uint32_t);
				if (at + sizeof(uint32_t) > size) continue; // Out of range reads zero.
				if (!attempts.Charge(sizeof(uint32_t)))
					return fail(call.pc, fmt::format("call table at 0x{:x} {}", base, attempts.Refusal(sizeof(uint32_t))));
				if (base > UINT64_MAX - at || !read_code(base + at, std::span(entry).subspan(half, 1)))
					return fail(call.pc, fmt::format("call table entry at 0x{:x} is unreadable", base + at));
			}
			if (context != nullptr) {
				const uint64_t at = offset + site.table_word_offset;
				context->Defer(base + at, entry, at < limit ? static_cast<uint32_t>(std::min<uint64_t>(16u, limit - at) / 4u) : 0u);
			}
			const auto target = ((uint64_t {entry[1]} << 32u) | entry[0]) & ~uint64_t {3};
			if (target == 0u) {
				++null_entries;
				continue;
			}
			if (std::ranges::find(targets, target) != targets.end()) continue;
			if (targets.size() == MaxCallTableTargets)
				return fail(call.pc, fmt::format("call table at 0x{:x} names more than {} callees", base,
				                                 MaxCallTableTargets));
			targets.push_back(target);
		}
		// Null entries stay reachable: they take the observable miss path at run time.
		if (targets.empty()) return fail(call.pc, fmt::format("call table at 0x{:x} has only null entries", base));
		site.address = 0;
		source.call_diagnostic += fmt::format(
		    "{}pc=0x{:08x} table=0x{:x} bytes={} entry_stride={} entries={} null={} callees={}",
		    source.call_diagnostic.empty() ? "" : "; ", call.pc, base, size, site.table_stride, slots,
		    null_entries, targets.size());
	}
	if (large_refresh) source.call_diagnostic += fmt::format("; table_attempt_bytes={}", attempts.bytes);
	const auto query_reads = reads.size();
	bool verified_link_code = false;
	if (source.linked) {
		auto& linked = *source.linked;
		// Uniform sites may relocate identical code; table sites compile their callee addresses
		// into S_CALL_SELECT_B64, so any change of their closed set relinks.
		// A large refresh may reuse only a link within its linked-code budget (a pure-small refresh
		// may have built a bigger one); otherwise it falls through to the bounded rebuild below,
		// which refuses or, for smaller fresh code/membership, relinks.
		bool unchanged = linked.site_targets.size() == plan.size() &&
		                 !(large_refresh && linked.code.size() > MaxCallLinkedBytes / sizeof(uint32_t));
		for (size_t i = 0; unchanged && i < plan.size(); ++i) {
			if (source.calls[i].kind == ShaderSource::Call::Kind::Table) {
				unchanged = linked.site_targets[i] == plan[i];
			} else {
				unchanged = linked.site_targets[i].size() == 1u;
			}
		}
		for (size_t j = 0; unchanged && j < linked.functions.size(); ++j) {
			const auto& function = linked.functions[j];
			const auto  address  = plan[function.site][function.slot];
			const auto  expected = std::span(linked.code).subspan(function.offset, function.size);
			bool        repeated = false;
			for (size_t k = 0; k < j; ++k) {
				const auto& observed = linked.functions[k];
				if (plan[observed.site][observed.slot] != address) continue;
				repeated = true;
				unchanged = std::ranges::equal(expected, std::span(linked.code).subspan(observed.offset, observed.size));
				break;
			}
			if (repeated) continue;
			linked.observed_function.resize(function.size);
			unchanged = read_code(address, linked.observed_function) &&
			            std::ranges::equal(expected, linked.observed_function);
		}
		if (unchanged) {
			verified_link_code = true;
			auto proof = ProveContextLoads(linked.context_reads, context_tables, plan,
			                              attempts, payload_cache, read_transport);
			if (proof.loads == linked.context_loads) {
				for (size_t i = 0; i < plan.size(); ++i) linked.site_targets[i] = plan[i];
				// Data-only changes refresh the per-dispatch closure even when its
				// compiled constants and original function code can be reused.
				PublishContextData(source, std::move(proof));
				return &linked.decoded;
			}
		}
		reads.resize(query_reads);
	}
	// Large-refresh code bounds, checked before the caller is copied and before every insertion or
	// layout, so function_pc/stub/copy offsets stay below 8 MiB (no u32 overflow) and a refusal
	// publishes nothing: source.linked, its revision and earlier modules stay as they were.
	uint64_t decode_budget = MaxCallDecodedBytes;
	uint64_t linked_budget = MaxCallLinkedBytes;
	const auto charge_linked = [&](uint32_t pc, uint64_t bytes) -> bool {
		if (!large_refresh) return true;
		if (bytes > linked_budget) {
			failure = fmt::format("pc 0x{:08x}: linked code needs {} more bytes, {} of {} remain "
			                      "(unique decoded callee bytes {})",
			                      pc, bytes, linked_budget, MaxCallLinkedBytes, MaxCallDecodedBytes - decode_budget);
			return false;
		}
		linked_budget -= bytes;
		return true;
	};
	if (large_refresh && !charge_linked(source.decoded.instructions[source.calls.front().instruction].pc,
	                                    uint64_t {source.code.size()} * sizeof(uint32_t)))
		return nullptr;
	ShaderSource::Linked linked {.code = source.code, .decoded = source.decoded};
	const auto& last = source.decoded.instructions.back();
	EXIT_NOT_IMPLEMENTED(last.opcode != Decoder::Opcode::S_ENDPGM);
	// Keep the caller's data/footer at native offsets. Expand leaf CFGs per call site (and per
	// table callee) before SSA, so return edges neither require a dispatcher nor merge resource roots.
	uint32_t function_pc = last.pc + last.word_count * sizeof(uint32_t);
	std::map<uint64_t, LeafRead> functions;
	const auto load = [&](size_t site_index, uint64_t address) -> const LeafRead& {
		const auto& site = source.calls[site_index];
		auto [entry, inserted] = functions.try_emplace(address);
		if (inserted && verified_link_code) {
			// A data-only relink already read and compared every mandatory code
			// range. Decode its owned raw slice without spending guest reads again.
			const auto& old = *source.linked;
			const auto found = std::ranges::find_if(old.functions, [&](const auto& function) {
				return plan[function.site][function.slot] == address;
			});
			EXIT_IF(found == old.functions.end());
			const auto raw = std::span(old.code).subspan(found->offset, found->size);
			const auto read_owned = [&](uint64_t at, std::span<uint32_t> words) {
				if (at < address || ((at - address) & 3u) != 0u) return false;
				const auto offset = (at - address) / sizeof(uint32_t);
				if (offset > raw.size() || words.size() > raw.size() - offset) return false;
				std::copy_n(raw.begin() + offset, words.size(), words.begin());
				return true;
			};
			// Leaf slices terminate at S_SETPC_B64, not the S_ENDPGM contract
			// of DecodeProgram. Reuse the leaf decoder and its bounds/policy.
			entry->second = ReadLeaf(address, source.decoded.instructions[site.instruction].pc,
			                         site.kind == ShaderSource::Call::Kind::Table, read_owned,
			                         large_refresh ? &decode_budget : nullptr);
		} else if (inserted) {
			entry->second = ReadLeaf(address, source.decoded.instructions[site.instruction].pc,
			                         site.kind == ShaderSource::Call::Kind::Table, read_code,
			                         large_refresh ? &decode_budget : nullptr);
		}
		return entry->second;
	};
	bool context_candidate_overflow = false;
	// Appends one copy of a leaf whose S_SETPC_B64 returns after the call.
	const auto append = [&](size_t site_index, uint32_t slot, const LeafRead& leaf) -> std::string {
		const auto& site  = source.calls[site_index];
		const auto& call  = source.decoded.instructions[site.instruction];
		const bool  soft  = site.kind == ShaderSource::Call::Kind::Table;
		const auto& body  = leaf.body;
		const uint32_t function_size = body.back().pc + body.back().word_count * sizeof(uint32_t);
		EXIT_IF(function_pc > UINT32_MAX - function_size);
		const auto function_offset = static_cast<uint32_t>(linked.code.size());
		for (auto inst: body) {
			// A cached decoded body may have first been loaded by a TABLE site.
			if (!soft && inst.opcode == Decoder::Opcode::S_GETPC_B64) EXIT_NOT_IMPLEMENTED(true);
			const auto clobbers_target = CFG::MayWriteScalarRegister(inst, call.dst.reg, source.wave_size) ||
			                             CFG::MayWriteScalarRegister(inst, call.dst.reg + 1u, source.wave_size);
			if (clobbers_target) {
				if (!soft) EXIT_IF(true);
				return "callee writes its return address pair";
			}
			for (uint32_t code = 0; code < site.preserved_registers.size(); ++code) {
				if (site.preserved_registers[code] &&
				    CFG::MayWriteScalarRegister(inst, code, source.wave_size)) {
					if (!soft) EXIT("shader call at pc 0x%08x clobbers scalar target dependency s%u", call.pc, code);
					return fmt::format("callee clobbers scalar target dependency s{}", code);
				}
			}
			if (soft && inst.opcode == Decoder::Opcode::S_GETPC_B64) {
				// Preserve the next ORIGINAL guest PC before assigning this copy a synthetic PC.
				// ReadLeaf checked this address range; TABLE select marker words already key
				// the target address. Keep the exact original raw slice for refresh equality.
				inst.getpc_guest_next_pc = plan[site_index][slot] + uint64_t {inst.pc} + 4u;
			}
			linked.code.insert(linked.code.end(), inst.raw, inst.raw + inst.word_count);
			if (inst.opcode == Decoder::Opcode::S_SETPC_B64) {
				if (inst.src0.kind != Decoder::OperandKind::Sgpr || inst.src0.reg != call.dst.reg) {
					if (!soft) EXIT_NOT_IMPLEMENTED(true);
					return "callee does not return through its return address pair";
				}
				inst.opcode = Decoder::Opcode::S_BRANCH;
				inst.branch_target = call.pc + call.word_count * sizeof(uint32_t);
			} else if (Decoder::IsDirectBranch(inst.opcode)) {
				if (inst.branch_target >= function_size) {
					if (!soft) EXIT_IF(true);
					return "callee branches outside its body";
				}
				inst.branch_target += function_pc;
			}
			inst.pc += function_pc;
			linked.decoded.instructions.push_back(std::move(inst));
		}
		if (!context_candidate_overflow && site.kind == ShaderSource::Call::Kind::Table) {
			auto candidates = FindContextReads(site_index, slot, site, leaf, source.wave_size);
			if (candidates.size() > MaxCallContextRanges - linked.context_reads.size()) {
				context_candidate_overflow = true;
				linked.context_reads.clear();
			} else {
				linked.context_reads.insert(linked.context_reads.end(), candidates.begin(), candidates.end());
			}
		}
		linked.functions.push_back({function_offset, function_size / 4u, site_index, slot, function_pc});
		function_pc += function_size;
		return {};
	};
	for (size_t i = 0; i < source.calls.size(); ++i) {
		const auto& site = source.calls[i];
		const auto& call = source.decoded.instructions[site.instruction];
		if (site.kind == ShaderSource::Call::Kind::Uniform) {
			const auto& leaf = load(i, plan[i][0]);
			// Only a large refresh's decode budget can fail a uniform leaf softly.
			if (!leaf.failure.empty())
				return fail(call.pc, fmt::format("callee 0x{:x} {}", plan[i][0], leaf.failure));
			if (!charge_linked(call.pc, leaf.body.back().pc + leaf.body.back().word_count * sizeof(uint32_t)))
				return nullptr;
			const auto  start = function_pc;
			(void)append(i, 0, leaf);
			linked.decoded.instructions[site.instruction].branch_target = start;
			continue;
		}
		// Table site: the call becomes a branch to a compare chain evaluated on the pair the GPU
		// actually loaded (never a host snapshot of the pointer). Each candidate writes the return
		// address and enters its own leaf copy only when the pair names it. Any other pair (null,
		// out-of-range zero, or a pointer written after enumeration) executes S_CALL_TABLE_MISS_B64,
		// which records an observable GPU callable fault, then continues after the call.
		const auto& targets = plan[i];
		std::vector<const LeafRead*> leaves;
		for (const auto target: targets) {
			const auto& leaf = load(i, target);
			if (!leaf.failure.empty())
				return fail(call.pc, fmt::format("callee 0x{:x} {}", target, leaf.failure));
			leaves.push_back(&leaf);
		}
		// Stub (selects, miss, skip) plus one {S_CALL_ENTER_B64, leaf copy} per callee, charged
		// in 64-bit arithmetic before any of it is laid out or inserted.
		uint64_t site_bytes = (uint64_t {targets.size()} * 3u + 3u + 1u) * sizeof(uint32_t);
		for (const auto* leaf: leaves)
			site_bytes += 3u * sizeof(uint32_t) + leaf->body.back().pc + leaf->body.back().word_count * sizeof(uint32_t);
		if (!charge_linked(call.pc, site_bytes)) return nullptr;
		const auto return_pc = call.pc + call.word_count * sizeof(uint32_t);
		const auto stub_pc   = function_pc;
		const auto stub_size = static_cast<uint32_t>(targets.size() * 3u + 3u + 1u) * sizeof(uint32_t);
		EXIT_IF(function_pc > UINT32_MAX - stub_size);
		// Lay out {S_CALL_ENTER_B64, leaf copy} after the stub to learn their entry PCs.
		constexpr uint32_t enter_size = 3u * sizeof(uint32_t);
		uint32_t copy_pc = stub_pc + stub_size;
		std::vector<uint32_t> entries;
		for (const auto* leaf: leaves) {
			entries.push_back(copy_pc);
			const auto& body = leaf->body;
			copy_pc += enter_size + body.back().pc + body.back().word_count * sizeof(uint32_t);
		}
		for (size_t t = 0; t < targets.size(); ++t) {
			Decoder::Instruction select;
			select.pc            = function_pc;
			select.word_count    = 3;
			select.raw[0]        = CallSelectMarker;
			select.raw[1]        = static_cast<uint32_t>(targets[t]);
			select.raw[2]        = static_cast<uint32_t>(targets[t] >> 32u);
			select.family        = Decoder::Family::SOP1;
			select.opcode        = Decoder::Opcode::S_CALL_SELECT_B64;
			select.src0          = call.src0; // compared only; the pair is written on entry
			select.src_count     = 1;
			select.data_dwords   = 0;
			select.offset        = call.pc;
			select.branch_target = entries[t];
			linked.code.insert(linked.code.end(), select.raw, select.raw + select.word_count);
			linked.decoded.instructions.push_back(select);
			function_pc += select.word_count * sizeof(uint32_t);
		}
		Decoder::Instruction miss;
		miss.pc          = function_pc;
		miss.word_count  = 3;
		miss.raw[0]      = CallMissMarker;
		miss.raw[1]      = call.pc;
		miss.family      = Decoder::Family::SOP1;
		miss.opcode      = Decoder::Opcode::S_CALL_TABLE_MISS_B64;
		miss.src0        = call.src0;
		miss.src_count   = 1;
		miss.data_dwords = 0;
		linked.code.insert(linked.code.end(), miss.raw, miss.raw + miss.word_count);
		linked.decoded.instructions.push_back(miss);
		function_pc += miss.word_count * sizeof(uint32_t);
		Decoder::Instruction skip;
		skip.pc            = function_pc;
		skip.word_count    = 1;
		skip.family        = Decoder::Family::SOPP;
		skip.opcode        = Decoder::Opcode::S_BRANCH;
		skip.branch_target = return_pc;
		skip.raw[0] = 0xbf820000u | (static_cast<uint32_t>((static_cast<int64_t>(return_pc) -
		                                                    (int64_t {function_pc} + 4)) / 4) & 0xffffu);
		linked.code.push_back(skip.raw[0]);
		linked.decoded.instructions.push_back(skip);
		function_pc += sizeof(uint32_t);
		for (size_t t = 0; t < targets.size(); ++t) {
			EXIT_IF(function_pc != entries[t]);
			// Only the taken edge writes the return pair, so no selected return address flows into
			// later candidates' compares (or anywhere on a no-match path).
			Decoder::Instruction enter;
			enter.pc          = function_pc;
			enter.word_count  = 3;
			enter.raw[0]      = CallEnterMarker;
			enter.raw[1]      = call.pc;
			enter.family      = Decoder::Family::SOP1;
			enter.opcode      = Decoder::Opcode::S_CALL_ENTER_B64;
			enter.dst         = call.dst;
			enter.data_dwords = 2;
			enter.offset      = call.pc;
			linked.code.insert(linked.code.end(), enter.raw, enter.raw + enter.word_count);
			linked.decoded.instructions.push_back(enter);
			function_pc += enter_size;
			if (auto reason = append(i, static_cast<uint32_t>(t), *leaves[t]); !reason.empty())
				return fail(call.pc, fmt::format("callee 0x{:x} {}", targets[t], reason));
		}
		// The original SWAPPC must not write the pair before the compare chain reads it.
		auto& replaced         = linked.decoded.instructions[site.instruction];
		replaced.opcode        = Decoder::Opcode::S_BRANCH;
		replaced.family        = Decoder::Family::SOPP;
		replaced.dst           = {};
		replaced.src0          = {};
		replaced.src_count     = 0;
		replaced.data_dwords   = 1;
		replaced.branch_target = stub_pc;
	}
	reads.resize(query_reads);
	for (size_t j = 0; j < linked.functions.size(); ++j) {
		const auto& function = linked.functions[j];
		const auto  address  = plan[function.site][function.slot];
		bool        repeated = false;
		for (size_t k = 0; k < j && !repeated; ++k) {
			const auto& observed = linked.functions[k];
			repeated = plan[observed.site][observed.slot] == address;
		}
		if (!repeated) reads.emplace_back(address, function.size * sizeof(uint32_t));
	}
	// All required leaf reads, validation and native linked-code charges precede
	// optional probes. Lack of proof budget never refuses an otherwise valid call.
	auto context_proof = ProveContextLoads(linked.context_reads, context_tables, plan,
	                                      attempts, payload_cache, read_transport);
	uint64_t projection_bytes = 12u;
	for (const auto& projection: context_proof.loads) projection_bytes += 16u + projection.words.size() * 4u;
	if (large_refresh && !context_proof.loads.empty() && projection_bytes > linked_budget) context_proof = {};
	linked.context_loads = std::move(context_proof.loads);
	std::map<std::pair<size_t, uint32_t>, uint32_t> function_pcs;
	for (const auto& function: linked.functions) function_pcs.emplace(std::pair {function.site, function.slot}, function.synthetic_pc);
	std::map<uint32_t, const std::vector<uint32_t>*> projections;
	for (const auto& projection: linked.context_loads)
		projections.emplace(function_pcs.at({projection.site, projection.slot}) + projection.local_pc, &projection.words);
	for (auto& inst: linked.decoded.instructions)
		if (const auto found = projections.find(inst.pc); found != projections.end()) inst.call_context_words = *found->second;
	// Cache identity includes every semantic projection, outside untouched raw
	// Function slices. These data words are never decoded as guest instructions.
	if (!linked.context_loads.empty()) {
		uint64_t footer_words = 3u;
		for (const auto& projection: linked.context_loads) footer_words += 4u + projection.words.size();
		if (!charge_linked(source.decoded.instructions[source.calls.front().instruction].pc,
		                   footer_words * sizeof(uint32_t))) return nullptr;
		linked.code.insert(linked.code.end(), {ContextLoadMarker, 1u, static_cast<uint32_t>(linked.context_loads.size())});
		for (const auto& projection: linked.context_loads) {
			linked.code.insert(linked.code.end(), {static_cast<uint32_t>(projection.site), projection.slot,
			    projection.local_pc, static_cast<uint32_t>(projection.words.size())});
			linked.code.insert(linked.code.end(), projection.words.begin(), projection.words.end());
		}
	}
	linked.decoded.code = linked.code;
	linked.observed_function.resize(std::ranges::max_element(linked.functions, {},
	    &ShaderSource::Linked::Function::size)->size);
	linked.site_targets = std::move(plan);

	source.linked = std::move(linked);
	PublishContextData(source, std::move(context_proof));
	++source.revision;
	return &source.linked->decoded;
}

LinkedPcOrigin DescribeLinkedPc(const ShaderSource& source, uint32_t linked_pc) {
	LinkedPcOrigin origin;
	if (!source.linked || source.decoded.instructions.empty()) return origin;
	const auto&    last       = source.decoded.instructions.back();
	const uint32_t caller_end = last.pc + last.word_count * sizeof(uint32_t);
	if (linked_pc < caller_end) {
		origin.kind     = LinkedPcOrigin::Kind::Caller;
		origin.local_pc = linked_pc;
		return origin;
	}
	const auto& linked = *source.linked;
	// Beyond the last linked instruction there is nothing to attribute.
	if (linked.decoded.instructions.empty()) return origin;
	const auto&    linked_last = linked.decoded.instructions.back();
	const uint64_t linked_end  = uint64_t {linked_last.pc} + linked_last.word_count * sizeof(uint32_t);
	if (linked_pc >= linked_end) return origin;
	for (const auto& function: linked.functions) {
		if (function.synthetic_pc == UINT32_MAX || linked_pc < function.synthetic_pc ||
		    uint64_t {linked_pc - function.synthetic_pc} >= uint64_t {function.size} * sizeof(uint32_t))
			continue;
		if (function.site >= linked.site_targets.size() ||
		    function.slot >= linked.site_targets[function.site].size())
			return origin;
		origin.kind     = LinkedPcOrigin::Kind::Callee;
		origin.site     = function.site;
		origin.slot     = function.slot;
		origin.callee   = linked.site_targets[function.site][function.slot];
		origin.local_pc = linked_pc - function.synthetic_pc;
		return origin;
	}
	origin.kind = LinkedPcOrigin::Kind::Stub; // compare/miss/skip/enter stubs inside the linked span
	return origin;
}

const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime) {
	std::string failure;
	const auto* program = TryRefreshShaderSource(source, runtime, failure);
	if (program == nullptr) EXIT("shader call refresh failed: %s", failure.c_str());
	return *program;
}

} // namespace Libs::Graphics::ShaderRecompiler

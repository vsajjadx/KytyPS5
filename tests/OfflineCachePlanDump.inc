// Offline-only shader-store inspector. Uses the owning source's typed Io schema.
#include <xxhash.h>
#include <zstd.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <cstring>

namespace OfflineCachePlan {
using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler::IR;
struct Frame { uint32_t compressed_size, raw_size; uint64_t checksum; };
static_assert(sizeof(Frame) == 16 && sizeof(bool) == 1 && sizeof(ShaderType) == 4);
static_assert(std::endian::native == std::endian::little);
constexpr uint64_t WantedHash = 0xb6c04c59ad4fac06ull;
constexpr uint32_t ObservedCounterIndex = 4;
using Ids = std::unordered_map<const Inst*, size_t>;

void ValueText(FILE* out, Value value, const Ids& ids) {
  const auto* inst = value.TryInstruction();
  if (inst) { std::fprintf(out, "%%%zu", ids.at(inst)); return; }
  const auto type = value.GetType();
  switch (type) {
    case Type::Void: std::fputs("void", out); break;
    case Type::ScalarReg: std::fprintf(out, "sgpr%u", RegIndex(value.ScalarRegister())); break;
    case Type::VectorReg: std::fprintf(out, "vgpr%u", RegIndex(value.VectorRegister())); break;
    case Type::U1: std::fprintf(out, "%s", value.U1() ? "true" : "false"); break;
    case Type::U8: std::fprintf(out, "u8:%u", value.U8()); break;
    case Type::U16: std::fprintf(out, "u16:%u", value.U16()); break;
    case Type::U32: std::fprintf(out, "u32:0x%x", value.U32()); break;
    case Type::U64: std::fprintf(out, "u64:0x%llx", static_cast<unsigned long long>(value.U64())); break;
    default: std::fprintf(out, "type:%u", static_cast<unsigned>(type)); break;
  }
}
void Dependencies(Value value, std::set<uint32_t>& sgprs, bool& counter,
                  std::unordered_set<const Inst*>& seen) {
  const auto* inst = value.TryInstruction();
  if (!inst || !seen.insert(inst).second) return;
  if (inst->GetOpcode() == ValueOpcode::GetUserData && inst->NumArgs() == 1 &&
      inst->Arg(0).GetType() == Type::ScalarReg)
    sgprs.insert(RegIndex(inst->Arg(0).ScalarRegister()));
  if (inst->GetOpcode() == ValueOpcode::ReadConstBuffer &&
      inst->Flags<MemoryFlags>().index == ObservedCounterIndex) counter = true;
  for (size_t i=0; i<inst->NumArgs(); ++i) Dependencies(inst->Arg(i), sgprs, counter, seen);
}
void DependencyText(FILE* out, Value value) {
  std::set<uint32_t> sgprs; bool counter = false;
  std::unordered_set<const Inst*> seen;
  Dependencies(value, sgprs, counter, seen);
  std::fprintf(out, " counter_taint=%u sgprs=", counter ? 1u : 0u);
  for (const auto reg : sgprs) std::fprintf(out, "%u,", reg);
}
void Buffers(FILE* out, const char* label, const ShaderInfo& info) {
  for (size_t i=0; i<info.buffers.size(); ++i) {
    const auto& b=info.buffers[i];
    std::fprintf(out, "%s.buffer[%zu] source=%u first_use_pc=0x%x descriptor_index=%u read=%u written=%u scalar=%u live=%u formatted=%u atomic=%u max_byte_extent=%u\n",
        label, i, b.source, b.first_use_pc, b.descriptor_index, b.read, b.written,
        b.scalar, b.live, b.formatted, b.atomic, b.max_byte_extent);
  }
}
void Dump(FILE* out, const ResourcePlan& plan, const CompiledShaderInfo& module,
          uint64_t module_id, uint64_t file_offset, uint64_t stamp) {
  std::fprintf(out, "hash=%016llx module=%016llx frame_offset=%llu build_stamp=%llu plan_stage=%u module_stage=%u observed_counter_index=%u\n",
      static_cast<unsigned long long>(plan.shader_hash), static_cast<unsigned long long>(module_id),
      static_cast<unsigned long long>(file_offset), static_cast<unsigned long long>(stamp),
      static_cast<unsigned>(plan.stage), static_cast<unsigned>(module.stage), ObservedCounterIndex);
  std::fprintf(out, "NOTE plan metadata precedes emission; module metadata holds actual live/native descriptor indices. Counter taint is transitive static dependency, not an evaluated value.\n");
  Buffers(out, "plan", plan.info); Buffers(out, "module", module.info);
  Ids ids; size_t n=0; for (const auto& inst:plan.value_storage) ids.emplace(&inst,n++);
  for (size_t i=0; i<plan.memory_info.size(); ++i) {
    const auto& m=plan.memory_info[i];
    std::fprintf(out, "memory[%zu] kind=%u resource=%u sampler=%u offset=0x%x secondary=0x%x bits=%u dwords=%u planning=%u gpu_only=%u\n",
        i, static_cast<unsigned>(m.kind), m.resource, m.sampler, m.offset, m.secondary_offset,
        m.data_bits, m.data_dwords, m.planning_only, m.gpu_runtime_only);
  }
  for (size_t i=0; i<plan.descriptor_sources.size(); ++i) {
    const auto& s=plan.descriptor_sources[i];
    std::fprintf(out, "source[%zu] dwords=%u indirect=%u\n", i, s.dword_count, s.indirect_descriptor.has_value());
    for (uint32_t w=0; w<s.dword_count; ++w) {
      std::fprintf(out, "  word[%u]=", w); ValueText(out,s.dwords[w],ids);
      DependencyText(out,s.dwords[w]); std::fputc('\n',out);
    }
  }
  for (size_t i=0; i<plan.srt_reads.size(); ++i) {
    std::fprintf(out,"srt[%zu] flat=%u value=",i,plan.srt_reads[i].flat_offset);
    ValueText(out,plan.srt_reads[i].value,ids); std::fputc('\n',out);
  }
  for (size_t i=0; i<plan.control_flow.size(); ++i) {
    const auto& b=plan.control_flow[i];
    std::fprintf(out,"block[%zu] condition=",i); ValueText(out,b.condition,ids);
    DependencyText(out,b.condition);
    std::fprintf(out," uses_source5=%u successors(true,false)=",std::ranges::find(b.sources,5u)!=b.sources.end());
    for(auto x:b.successors)std::fprintf(out,"%u,",x);
    std::fputs(" sources=",out);for(auto x:b.sources)std::fprintf(out,"%u,",x);
    std::fputs(" images=",out);for(auto x:b.images)std::fprintf(out,"%u,",x);
    std::fputs(" srt_reads=",out);for(auto x:b.srt_reads)std::fprintf(out,"%u,",x);
    std::fputc('\n',out);
  }
  for (const auto& inst:plan.value_storage) {
    const auto name=ValueOpcodeName(inst.GetOpcode());
    std::fprintf(out,"%%%zu = %.*s(",ids.at(&inst),static_cast<int>(name.size()),name.data());
    for(size_t a=0;a<inst.NumArgs();++a) { if(a)std::fputc(',',out);ValueText(out,inst.Arg(a),ids); }
    std::fputc(')',out);
    if(inst.GetOpcode()==ValueOpcode::ReadConstBuffer || inst.GetOpcode()==ValueOpcode::LoadAddressU32 || inst.GetOpcode()==ValueOpcode::LoadBufferU32) {
      const auto flags=inst.Flags<MemoryFlags>(); std::fprintf(out," memory_index=%u cloned_pc=0x%x",flags.index,flags.pc);
    }
    DependencyText(out,Value(const_cast<Inst*>(&inst))); std::fputc('\n',out);
  }
}
int Run(const char* input, const char* directory) {
  try {
    std::ifstream stream(input,std::ios::binary|std::ios::ate);
    if(!stream)throw std::runtime_error("cannot open frozen input");
    const auto size=stream.tellg();
    if(size<=0 || size>256*1024*1024)throw std::runtime_error("input size outside bounded store limit");
    std::vector<uint8_t> bytes(static_cast<size_t>(size)); stream.seekg(0);
    if(!stream.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size())))throw std::runtime_error("short input read");
    Reader header{bytes}; uint64_t stamp=0; bool ring=false;
    Libs::Graphics::ShaderRecompiler::Spirv::Profile profile; uint32_t shared=0;
    IoAll(header,stamp,ring,profile,shared);
    if(!header.ok || stamp==0)throw std::runtime_error("invalid typed StoreHeader");
    const auto header_size=bytes.size()-header.bytes.size();
    std::printf("OFFLINE_HEADER bytes=%zu stamp=%llu ring=%u compute_shared=%u\n",header_size,static_cast<unsigned long long>(stamp),ring,shared);
    auto records=header.bytes; size_t frame_number=0, matches=0;
    std::filesystem::create_directories(directory);
    while(!records.empty()) {
      const auto offset=bytes.size()-records.size();
      if(records.size()<sizeof(Frame))throw std::runtime_error("truncated frame header");
      Frame frame{};std::memcpy(&frame,records.data(),sizeof(frame));
      if(frame.compressed_size==0 || frame.raw_size==0 || frame.raw_size>64*1024*1024 || frame.compressed_size>records.size()-sizeof(frame))throw std::runtime_error("invalid/truncated frame bounds");
      const auto compressed=records.subspan(sizeof(frame),frame.compressed_size);
      if(XXH3_64bits(compressed.data(),compressed.size())!=frame.checksum || ZSTD_getFrameContentSize(compressed.data(),compressed.size())!=frame.raw_size)throw std::runtime_error("frame checksum/content-size mismatch");
      std::vector<uint8_t> raw(frame.raw_size);
      const auto decoded=ZSTD_decompress(raw.data(),raw.size(),compressed.data(),compressed.size());
      if(ZSTD_isError(decoded)||decoded!=raw.size())throw std::runtime_error("zstd decode mismatch");
      Reader reader{raw}; uint8_t type=255;IoAll(reader,type);
      if(type>2)throw std::runtime_error("unknown Record enum");
      if(type==0) {
        ShaderType stage=ShaderType::Unknown;uint64_t hash=0;uint32_t userdata=0;std::vector<uint32_t> statics;
        IoAll(reader,stage,hash,userdata,statics);
        if(!reader.ok)throw std::runtime_error("invalid typed ProgramKey");
        if(hash==WantedHash) {
          std::vector<uint8_t> plan_bytes;uint64_t module_id=0;CompiledShaderInfo info;ResourceSpecialization specialization;
          IoAll(reader,plan_bytes,module_id,info,specialization);
          // Compute records have neither VS vertex formats nor rectangle PS parameters.
          uint32_t vertex_count=0,rect_count=0;IoAll(reader,vertex_count);
          if(!reader.ok || vertex_count!=0)throw std::runtime_error("target compute has vertex-format payload");
          IoAll(reader,rect_count);if(!reader.ok||rect_count!=0)throw std::runtime_error("target compute has rectangle parameters");
          bool mesh=false;std::vector<uint32_t> spirv;IoAll(reader,mesh,spirv);
          if(!reader.ok||!reader.bytes.empty()||stage!=ShaderType::Compute||info.stage!=stage||info.shader_hash!=hash||mesh)throw std::runtime_error("target module tail/type mismatch");
          ResourcePlan plan;
          if(!LoadResourcePlan(plan_bytes,plan)||SaveResourcePlan(plan)!=plan_bytes||plan.shader_hash!=hash||plan.stage!=stage)throw std::runtime_error("typed ResourcePlan load/roundtrip mismatch");
          const auto stem=std::string(directory)+"/b6c_frame"+std::to_string(frame_number)+"_module"+std::to_string(module_id);
          std::ofstream output(stem+".plan",std::ios::binary);output.write(reinterpret_cast<const char*>(plan_bytes.data()),static_cast<std::streamsize>(plan_bytes.size()));
          if(!output)throw std::runtime_error("plan output failed");output.close();
          FILE* dump=std::fopen((stem+".txt").c_str(),"w");if(!dump)throw std::runtime_error("dump output failed");
          Dump(dump,plan,info,module_id,offset,stamp);std::fclose(dump);
          std::printf("OFFLINE_MATCH frame=%zu offset=%zu userdata=%u static_dwords=%zu module=%016llx plan_bytes=%zu blocks=%zu sources=%zu output=%s\n",frame_number,offset,userdata,statics.size(),static_cast<unsigned long long>(module_id),plan_bytes.size(),plan.control_flow.size(),plan.descriptor_sources.size(),stem.c_str());
          ++matches;
        }
      }
      records=records.subspan(sizeof(frame)+frame.compressed_size);++frame_number;
    }
    std::printf("OFFLINE_DONE frames=%zu matches=%zu validated_all_frames=1\n",frame_number,matches);
    return matches?0:2;
  } catch(const std::exception& error) {std::fprintf(stderr,"OFFLINE_ERROR %s\n",error.what());return 1;}
}
} // namespace OfflineCachePlan

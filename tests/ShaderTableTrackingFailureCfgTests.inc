// Focused CPU cases for TABLE-linked resource-tracking soft failure. Include in shaderCfgTests.cpp next
// to ShaderTableLeafGetpcCfgTests.inc; call TestTableTrackingSoftFailure() from main. The two Probe*
// functions run ONLY as dedicated child processes (see README for CLI flags/CTest); a normal return
// from a probe is a test failure.
//
// Bad leaf (decoder-valid): its V# base word is v_readfirstlane of the caller's lane-varying v2 (GPU
// data, not a host-walkable value and not a memory descriptor table), then buffer_store_dword v6
// through it. A store through a non-walkable descriptor is the observed tracker refusal ("buffer
// descriptor is not a valid runtime value ..."). Support for that operation is NOT added.

namespace table_tracking_failure {
constexpr uint64_t table = 0xfe00e0000ull;
constexpr uint64_t leaf_a = table + 0x1000, leaf_b = table + 0x1100, leaf_bad = table + 0x1200;
constexpr uint32_t bad_store_pc = 0x18; // local PC of buffer_store_dword in the bad leaf

struct Memory { std::unordered_map<uint64_t, uint32_t> words; };

inline bool Read(void* cookie, uint64_t address, std::span<uint32_t> words) {
  auto& m = *static_cast<Memory*>(cookie);
  for (size_t i = 0; i < words.size(); ++i) {
    const auto at = m.words.find(address + i * 4u);
    if (at == m.words.end()) return false;
    words[i] = at->second;
  }
  return true;
}

inline std::vector<uint32_t> GoodLeaf(uint32_t value) {
  return {EncodeVop1(0x01, 6, 255), value, EncodeSop1(0x21, 125, 14)};
}

inline std::vector<uint32_t> BadLeaf() {
  return {EncodeVop1(0x02, 24, 258u),                       // 0x00 v_readfirstlane_b32 s24, v2 (GPU data)
          EncodeSMovB32(25, 128),                               // 0x04 s_mov_b32 s25, 0
          EncodeSMovB32(26, 255), 0x100u,                       // 0x08 s_mov_b32 s26, 0x100 (records)
          EncodeSMovB32(27, 255), 0x00014facu,                  // 0x10 s_mov_b32 s27, format word
          EncodeMubuf0(0x1c, 0, false, false), EncodeMubuf1(6, 6, 0), // 0x18 buffer_store_dword v6, s[24:27]
          EncodeSop1(0x21, 125, 14)};                           // 0x20 return through s[14:15]
}

inline void Place(Memory& memory, uint64_t address, const std::vector<uint32_t>& code) {
  for (size_t i = 0; i < code.size(); ++i) memory.words[address + i * 4u] = code[i];
}

constexpr std::array<uint32_t, 3> kFooter{0x4f524253u, 0x0000beefu, 0xfeedfaceu};

// 64-lane waterfall over (lane & 1), EXEC in s[40:41]/s[42:43], s5 = row * 16 (GPU-selected).
inline std::vector<uint32_t> Caller() {
  std::vector<uint32_t> shader{EncodeSop1(0x04, 40, 126), EncodeVop2(0x1b, 1, 129, 0),
                               EncodeVop2(0x1a, 2, 130, 0), EncodeVop1(0x01, 6, 128)};
  const auto loop = shader.size();
  shader.insert(shader.end(), {EncodeVop1(0x02, 4, 257), EncodeSop1(0x04, 42, 126), EncodeVopc(0xd2, 4, 1),
                               EncodeSop2(0x1e, 5, 4, 132), EncodeSmem0(0x09, 14, 4), 5u << 25u,
                               EncodeSopp(0x0c, 0), EncodeSop1(0x21, 14, 14), EncodeSop2(0x15, 42, 42, 126),
                               EncodeSop1(0x04, 126, 42)});
  shader.push_back(EncodeSopp(0x09, static_cast<uint32_t>(static_cast<int32_t>(loop) -
                                                          static_cast<int32_t>(shader.size() + 1u)) & 0xffffu));
  shader.insert(shader.end(), {EncodeSop1(0x04, 126, 40), EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 2),
                               EncodeSopp(0x01)});
  // Harmless data footer after S_ENDPGM: never decoded, but retained in source.code, so linked raw
  // DWORD offsets no longer coincide with synthetic PCs.
  shader.insert(shader.end(), kFooter.begin(), kFooter.end());
  return shader;
}
} // namespace table_tracking_failure

void TestTableTrackingSoftFailure() {
  using namespace ShaderRecompiler;
  namespace t = table_tracking_failure;
  t::Memory memory;
  FakeGuestMemory fake(&t::Read, &memory);
  t::Place(memory, t::leaf_a, t::GoodLeaf(0x11u));
  t::Place(memory, t::leaf_b, t::GoodLeaf(0x22u));
  t::Place(memory, t::leaf_bad, t::BadLeaf());
  const auto entry = [&](uint32_t row, uint64_t address) {
    memory.words[t::table + row * 16u] = static_cast<uint32_t>(address);
    memory.words[t::table + row * 16u + 4] = static_cast<uint32_t>(address >> 32u);
    memory.words[t::table + row * 16u + 8] = 0;
    memory.words[t::table + row * 16u + 12] = 0;
  };
  entry(0, t::leaf_a);
  entry(1, t::leaf_b);
  const auto shader = t::Caller();
  constexpr uint64_t output = t::table + 0x4000;
  std::array<uint32_t, 64> user_data{};
  user_data[0] = static_cast<uint32_t>(output); user_data[1] = output >> 32u;
  user_data[2] = 256; user_data[3] = 0x00014facu;
  user_data[8] = static_cast<uint32_t>(t::table); user_data[9] = t::table >> 32u;
  user_data[10] = 32; user_data[11] = 0x00014facu;
  auto options = MakeCompileOptions(ShaderType::Compute);
  ShaderComputeInputInfo compute = *options.input_info.compute;
  compute.thread_ids_num = 1; // v0 = lane id, so the row selection is really GPU data
  compute.threads_num[0] = 64; compute.threads_num[1] = compute.threads_num[2] = 1;
  options.input_info.compute = &compute;
  options.wave_size = 64;
  options.user_data = user_data;
  IR::SrtRuntime runtime{.user_data = user_data, .userdata = &memory, .read_specialization_memory = &t::Read};
  auto source = PrepareShaderSource(shader, options);
  Check(source.calls.size() == 1 && source.calls[0].kind == ShaderSource::Call::Kind::Table,
        "tracking-failure caller is not a TABLE site");
  std::string reason;
  const auto translate = [&](bool soft, std::string* failure_reason, uint32_t* failure_pc) {
    const auto* linked = TryRefreshShaderSource(source, runtime, reason);
    Check(linked != nullptr && reason.empty(), "tracking-failure table did not link");
    try {
      (void)TranslateProgram(*linked, options, soft);
      return true;
    } catch (const IR::ResourceTrackingFailure& failure) {
      if (failure_reason != nullptr) *failure_reason = failure.reason;
      if (failure_pc != nullptr) *failure_pc = failure.pc;
      Check(failure.message.find(failure.reason) != std::string::npos &&
                failure.message.find("shader resource tracking: hash=") == 0,
            "the typed failure lost the historical fatal message");
      return false;
    }
  };
  // Mapped positive first (both soft and default translation).
  Check(translate(true, nullptr, nullptr) && translate(false, nullptr, nullptr),
        "a valid TABLE link failed resource tracking");
  // Row 1 -> bad leaf: the soft translation throws the typed refusal with exact provenance.
  entry(1, t::leaf_bad);
  std::string failure;
  uint32_t failure_pc = UINT32_MAX;
  Check(!translate(true, &failure, &failure_pc), "unsupported TABLE leaf descriptor flow was accepted");
  Check(failure.find("buffer descriptor is not a valid runtime value") != std::string::npos &&
            failure.find("op=") != std::string::npos && failure.find("memory_kind=") != std::string::npos &&
            failure.find("dwords=1") != std::string::npos,
        "the refusal lost its reason or bounded op/memory details");
  const auto origin = DescribeLinkedPc(source, failure_pc);
  Check(origin.kind == LinkedPcOrigin::Kind::Callee && origin.site == 0 && origin.slot == 1 &&
            origin.callee == t::leaf_bad && origin.local_pc == t::bad_store_pc,
        "the refusal PC does not map back to the bad callee's store");
  {
    const auto& instructions = source.linked->decoded.instructions;
    const auto at = std::ranges::find(instructions, failure_pc, &Decoder::Instruction::pc);
    Check(at != instructions.end() && at->raw[0] == t::BadLeaf()[6] && at->raw[1] == t::BadLeaf()[7] &&
              failure_pc >= source.decoded.instructions.back().pc + 4u,
          "the refusal PC is not the synthetic PC of the bad store's linked copy");
    const auto caller_origin = DescribeLinkedPc(source, 0);
    Check(caller_origin.kind == LinkedPcOrigin::Kind::Caller && caller_origin.local_pc == 0,
          "caller PCs no longer map to themselves");
    // The footer is retained raw but not decoded; raw offsets and synthetic PCs therefore differ.
    const auto caller_words = source.decoded.instructions.back().pc / 4u + 1u;
    Check(source.code.size() == caller_words + t::kFooter.size() &&
              std::equal(t::kFooter.begin(), t::kFooter.end(), source.code.end() - t::kFooter.size()),
          "the post-END footer was not retained raw");
    const auto bad_function = std::ranges::find_if(source.linked->functions, [](const auto& function) {
      return function.site == 0 && function.slot == 1;
    });
    const auto bad = t::BadLeaf();
    Check(bad_function != source.linked->functions.end() && bad_function->size == bad.size() &&
              uint64_t{bad_function->offset} * 4u != bad_function->synthetic_pc &&
              failure_pc - bad_function->synthetic_pc == t::bad_store_pc &&
              std::equal(bad.begin(), bad.end(), source.linked->code.begin() + bad_function->offset),
          "raw function slices and synthetic PCs were conflated");
    // Unknown outside the linked span (exact end and UINT32_MAX); Stub for a real pseudo-op.
    const auto& linked_last = instructions.back();
    const auto linked_end = linked_last.pc + linked_last.word_count * 4u;
    Check(DescribeLinkedPc(source, linked_end).kind == LinkedPcOrigin::Kind::Unknown &&
              DescribeLinkedPc(source, UINT32_MAX).kind == LinkedPcOrigin::Kind::Unknown,
          "a PC outside the linked program was attributed");
    const auto select = std::ranges::find(instructions, Decoder::Opcode::S_CALL_SELECT_B64,
                                          &Decoder::Instruction::opcode);
    Check(select != instructions.end() && DescribeLinkedPc(source, select->pc).kind == LinkedPcOrigin::Kind::Stub,
          "a real compare-chain PC was not reported as a stub");
  }
  // Repeat: the same refusal, no state carried by the translator.
  std::string repeat;
  uint32_t repeat_pc = UINT32_MAX;
  Check(!translate(true, &repeat, &repeat_pc) && repeat == failure && repeat_pc == failure_pc,
        "a repeated refusal was not identical");
  // Recovery: the valid table translates again.
  entry(1, t::leaf_b);
  Check(translate(true, nullptr, nullptr), "a valid table did not recover after the refusal");
  std::printf("[host]    TableTrackingSoftFailure         ok\n");
}

// Child process only: a call-free program with the same unsupported flow keeps its fatal exit.
void ProbeTableTrackingNormalFatal() {
  using namespace ShaderRecompiler;
  std::vector<uint32_t> shader{EncodeVop2(0x1a, 2, 130, 0)}; // v_lshlrev_b32 v2, 2, v0 (lane-varying)
  const auto bad = table_tracking_failure::BadLeaf();
  shader.insert(shader.end(), bad.begin(), bad.end() - 1);
  shader.push_back(EncodeSopp(0x01));
  std::array<uint32_t, 64> user_data{};
  auto options = MakeCompileOptions(ShaderType::Compute);
  ShaderComputeInputInfo compute = *options.input_info.compute;
  compute.thread_ids_num = 1;
  compute.threads_num[0] = 64; compute.threads_num[1] = compute.threads_num[2] = 1;
  options.input_info.compute = &compute;
  options.wave_size = 64;
  options.user_data = user_data;
  std::fprintf(stderr, "TableTrackingNormalFatal: call-free unsupported descriptor flow\n");
  std::fflush(stderr);
  (void)TranslateProgram(shader, options);
  Check(false, "a call-free resource-tracking refusal returned instead of its historical fatal exit");
}

// Child process only: a pure Uniform link to the same bad leaf keeps its fatal exit.
void ProbeTableTrackingUniformFatal() {
  using namespace ShaderRecompiler;
  namespace t = table_tracking_failure;
  t::Memory memory;
  FakeGuestMemory fake(&t::Read, &memory);
  t::Place(memory, t::leaf_bad, t::BadLeaf());
  std::array<uint32_t, 64> user_data{};
  user_data[12] = static_cast<uint32_t>(t::leaf_bad); user_data[13] = t::leaf_bad >> 32u;
  const uint32_t caller[] = {EncodeVop2(0x1a, 2, 130, 0), EncodeSMovB32(14, 12), EncodeSMovB32(15, 13),
                             EncodeSop1(0x21, 14, 14), EncodeSopp(0x01)};
  auto options = MakeCompileOptions(ShaderType::Compute);
  ShaderComputeInputInfo compute = *options.input_info.compute;
  compute.thread_ids_num = 1;
  compute.threads_num[0] = 64; compute.threads_num[1] = compute.threads_num[2] = 1;
  options.input_info.compute = &compute;
  options.wave_size = 64;
  options.user_data = user_data;
  auto source = PrepareShaderSource(caller, options);
  Check(source.calls.size() == 1 && source.calls[0].kind == ShaderSource::Call::Kind::Uniform,
        "uniform tracking probe was not uniform");
  IR::SrtRuntime runtime{.user_data = user_data, .userdata = &memory, .read_specialization_memory = &t::Read};
  std::string reason;
  const auto* linked = TryRefreshShaderSource(source, runtime, reason);
  Check(linked != nullptr, "uniform tracking probe did not link");
  std::fprintf(stderr, "TableTrackingUniformFatal: pure Uniform link with unsupported descriptor flow\n");
  std::fflush(stderr);
  (void)TranslateProgram(*linked, options); // default: not TABLE-linked -> historical fatal
  Check(false, "a pure Uniform resource-tracking refusal returned instead of its historical fatal exit");
}

// Standalone focused CPU cases; included next to ShaderTableCallCfgTests.inc.
void TestTableLeafOriginalGetpc() {
  using namespace ShaderRecompiler;
  constexpr uint64_t table = 0xfe0090000ull, output = table + 0x1000;
  constexpr uint64_t first = 0x11d28da000ull, second = first + 0x4000;
  constexpr uint32_t getpc_pc = 0xe34, data_pc = 0x1000;
  constexpr uint64_t actual_next = 0x11d28dae38ull;
  struct Memory { std::unordered_map<uint64_t, uint32_t> words; } memory;
  const auto read = +[](void* cookie, uint64_t address, std::span<uint32_t> words) {
    auto& m = *static_cast<Memory*>(cookie);
    for (size_t i = 0; i < words.size(); ++i) {
      const auto at = m.words.find(address + i * 4u);
      if (at == m.words.end()) return false;
      words[i] = at->second;
    }
    return true;
  };
  FakeGuestMemory fake(read, &memory);
  std::vector<uint32_t> leaf(getpc_pc / 4u, EncodeSopp(0x00));
  leaf.insert(leaf.end(), {
      EncodeSop1(0x1f, 20, 0),             // actual shape: s_getpc_b64 s20:21
      EncodeVop1(0x01, 7, 20), EncodeVop1(0x01, 8, 21),
      EncodeSop2(0x00, 20, 20, 255), data_pc - getpc_pc - 4u,
      EncodeSop2(0x04, 21, 21, 128),       // carry into original PC high word
      EncodeSmem0(0x00, 24, 10), 125u << 25u,
      EncodeSopp(0x0c, 0), EncodeVop1(0x01, 6, 24), EncodeSop1(0x21, 125, 14)});
  for (const auto address : {first, second})
    for (size_t i = 0; i < leaf.size(); ++i) memory.words[address + i * 4u] = leaf[i];
  memory.words[first + data_pc] = 0x1234beefu;
  memory.words[second + data_pc] = 0x5a5acaceu;
  const auto entry = [&](uint32_t row, uint64_t address) {
    memory.words[table + row * 16u] = static_cast<uint32_t>(address);
    memory.words[table + row * 16u + 4] = static_cast<uint32_t>(address >> 32u);
  };
  entry(0, first | 3u); entry(1, second);
  std::vector<uint32_t> shader {
      EncodeSop1(0x04, 40, 126), EncodeVop2(0x1b, 1, 129, 0),
      EncodeVop2(0x1a, 2, 130, 0), EncodeVop1(0x01, 6, 128)};
  const auto loop = shader.size();
  shader.insert(shader.end(), {EncodeVop1(0x02, 4, 257), EncodeSop1(0x04, 42, 126),
      EncodeVopc(0xd2, 4, 1), EncodeSop2(0x1e, 5, 4, 132),
      EncodeSmem0(0x09, 14, 4), 5u << 25u, EncodeSopp(0x0c, 0), EncodeSop1(0x21, 14, 14),
      EncodeSop2(0x15, 42, 42, 126), EncodeSop1(0x04, 126, 42)});
  shader.push_back(EncodeSopp(0x09, static_cast<uint32_t>(static_cast<int32_t>(loop) -
      static_cast<int32_t>(shader.size() + 1u)) & 0xffffu));
  shader.insert(shader.end(), {EncodeSop1(0x04, 126, 40),
      EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 2),
      EncodeMubuf0(0x1c, 256), EncodeMubuf1(7, 0, 2),
      EncodeMubuf0(0x1c, 512), EncodeMubuf1(8, 0, 2), EncodeSopp(0x01)});
  std::array<uint32_t, 64> user_data {};
  user_data[0] = static_cast<uint32_t>(output); user_data[1] = output >> 32u;
  user_data[2] = 12; user_data[3] = 0x00014facu;
  user_data[8] = static_cast<uint32_t>(table); user_data[9] = table >> 32u;
  user_data[10] = 32; user_data[11] = 0x00014facu;
  auto options = MakeCompileOptions(ShaderType::Compute);
  ShaderComputeInputInfo compute = *options.input_info.compute;
  compute.thread_ids_num = 1;
  compute.threads_num[0] = 64; compute.threads_num[1] = compute.threads_num[2] = 1;
  options.input_info.compute = &compute;
  options.wave_size = 64;
  options.user_data = user_data;
  IR::SrtRuntime runtime {.user_data = user_data, .shader_base = 0x300000000ull,
      .userdata = &memory, .read_specialization_memory = read};
  auto source = PrepareShaderSource(shader, options);
  std::string reason;
  const auto* linked = TryRefreshShaderSource(source, runtime, reason);
  Check(linked != nullptr && reason.empty(), "TABLE leaf GETPC did not link");
  Check(source.calls[0].kind == ShaderSource::Call::Kind::Table, "GETPC fixture lost table selection");
  bool found_first = false, found_second = false;
  for (const auto& inst : linked->instructions) {
    if (inst.opcode != Decoder::Opcode::S_GETPC_B64) continue;
    Check(inst.pc != getpc_pc && inst.word_count == 1 && inst.raw[0] == 0xbe941f00u &&
              inst.dst.kind == Decoder::OperandKind::Sgpr && inst.dst.reg == 20,
          "GETPC provenance changed its original word, pair, width or synthetic layout");
    found_first |= inst.getpc_guest_next_pc == actual_next;
    found_second |= inst.getpc_guest_next_pc == second + getpc_pc + 4u;
  }
  Check(found_first && found_second, "shifted linked GETPC lost its exact original next guest PC");
  for (const auto& function : source.linked->functions) {
    const auto address = source.linked->site_targets[function.site][function.slot];
    for (uint32_t word = 0; word < function.size; ++word)
      Check(source.linked->code[function.offset + word] == memory.words.at(address + word * 4u),
            "GETPC metadata corrupted a raw function slice used by refresh");
  }
  auto translated = TranslateProgram(*linked, options);
  for (const auto* block : translated.program.blocks)
    for (const auto& inst : *block)
      Check(inst.GetOpcode() != IR::ValueOpcode::GetShaderBase,
            "TABLE leaf GETPC still uses the caller shader base");
  translated.program.source_reads = source.reads;
  const auto plan = IR::ExtractResourcePlan(translated.program);
  IR::ResourceSnapshot snapshot;
  IR::ResourceSpecialization specialization;
  Check(IR::MaterializeResources(plan, runtime, snapshot, specialization),
        "absolute64 PC-relative data could not be resource-materialized");
  const auto observed = [&](uint64_t address) {
    return std::ranges::any_of(snapshot.specialization_reads, [&](const auto& range) {
      return range.first <= address && address - range.first < range.second;
    });
  };
  Check(observed(first + data_pc) && observed(second + data_pc) &&
            std::ranges::find(snapshot.flattened_srt, 0x1234beefu) != snapshot.flattened_srt.end() &&
            std::ranges::find(snapshot.flattened_srt, 0x5a5acaceu) != snapshot.flattened_srt.end(),
        "original-PC data dependencies or absolute pointer values were lost");
  memory.words[first + data_pc] = 0x9876feefu;
  Check(IR::MaterializeResources(plan, runtime, snapshot, specialization) &&
            std::ranges::find(snapshot.flattened_srt, 0x9876feefu) != snapshot.flattened_srt.end(),
        "PC-relative literal data was cached as function code");
  const auto revision = source.revision;
  Check(TryRefreshShaderSource(source, runtime, reason) != nullptr && source.revision == revision,
        "unchanged raw code with metadata could not reuse its link");
  const auto original_code = source.linked->code;
  entry(0, second);
  Check(TryRefreshShaderSource(source, runtime, reason) != nullptr && source.revision == revision + 1 &&
            source.linked->code != original_code,
        "same-byte callee relocation did not enter the compiled function key");
  entry(0, first | 3u);
  Check(TryRefreshShaderSource(source, runtime, reason) != nullptr && source.linked->code == original_code,
        "restored original GETPC origin did not restore its exact function key");

  // CFG local-PC recognizers must not invent a synthetic target from tagged absolute GETPC.
  const uint32_t jump[] = {EncodeSop1(0x1f, 20, 0), EncodeSop2(0x00, 20, 20, 136),
      EncodeSop1(0x20, 125, 20), EncodeSopp(0x01)};
  Decoder::Program decoded;
  Decoder::DecodeProgram(jump, decoded);
  Check(!CFG::BuildGraph(decoded).unsupported, "ordinary caller PC-relative branch regressed");
  // Tagged absolute-PC CFG rejection is exercised in its dedicated fatal child probe.
  std::printf("[host]    TableLeafOriginalGetpc          ok\n");
}

// Run ONLY as a dedicated child process. A normal return is a test failure; the
// old pure-uniform fatal contract must not be widened by TABLE leaf support.
void ProbeUniformLeafGetpcFatal() {
  using namespace ShaderRecompiler;
  constexpr uint64_t address = 0xfe00c0000ull;
  const std::array<uint32_t, 2> leaf {EncodeSop1(0x1f, 20, 0), EncodeSop1(0x21, 125, 14)};
  auto read = +[](void* cookie, uint64_t at, std::span<uint32_t> words) {
    const auto& data = *static_cast<const std::array<uint32_t, 2>*>(cookie);
    if (at < address || at - address > sizeof(data) || words.size_bytes() > sizeof(data) - (at - address))
      return false;
    std::memcpy(words.data(), reinterpret_cast<const uint8_t*>(data.data()) + at - address, words.size_bytes());
    return true;
  };
  std::array<uint32_t, 64> user_data {};
  user_data[0] = static_cast<uint32_t>(address); user_data[1] = address >> 32u;
  const uint32_t caller[] = {EncodeSMovB32(14, 0), EncodeSMovB32(15, 1),
      EncodeSop1(0x21, 14, 14), EncodeSopp(0x01)};
  auto options = MakeCompileOptions(ShaderType::Compute); options.user_data = user_data;
  auto source = PrepareShaderSource(caller, options);
  Check(source.calls.size() == 1 && source.calls[0].kind == ShaderSource::Call::Kind::Uniform,
        "uniform fatal probe was not uniform");
  IR::SrtRuntime runtime {.user_data = user_data, .userdata = const_cast<void*>(static_cast<const void*>(&leaf)),
      .read_specialization_memory = read};
  std::fprintf(stderr, "UniformLeafGetpcFatal: invoking unchanged uniform GETPC rejection\n");
  std::fflush(stderr);
  std::string reason;
  (void)TryRefreshShaderSource(source, runtime, reason);
  Check(false, "uniform GETPC unexpectedly returned instead of its historical fatal rejection");
}

void ProbeTaggedLeafGetpcCfgFatal() {
  using namespace ShaderRecompiler;
  const uint32_t jump[] = {EncodeSop1(0x1f, 20, 0), EncodeSop2(0x00, 20, 20, 136),
      EncodeSop1(0x20, 125, 20), EncodeSopp(0x01)};
  Decoder::Program decoded;
  Decoder::DecodeProgram(jump, decoded);
  decoded.instructions[0].getpc_guest_next_pc = 0x11d28dae38ull;
  std::fprintf(stderr, "TaggedLeafGetpcCfgFatal: refusing fake local target for original absolute GETPC\n");
  std::fflush(stderr);
  (void)CFG::BuildGraph(decoded);
  Check(false, "tagged absolute GETPC was resolved as synthetic local control flow");
}

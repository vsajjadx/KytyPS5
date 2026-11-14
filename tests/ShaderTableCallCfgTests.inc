// === Paste into tests/shaderCfgTests.cpp right after TestMemoryFedScalarLeafCall() ===
// and add `TestTableSelectedLeafCall();` right after `TestMemoryFedScalarLeafCall();` in main.

// A GPU-selected call: per-lane index -> v_readfirstlane -> << 4 -> 16-byte {fnptr, data} entry of a
// user-data V# -> SWAPPC (the waterfall shape of a real title's material call). The closed callee
// set is enumerated per refresh through the ordered reader and lowered to S_CALL_SELECT_B64.
void TestTableSelectedLeafCall() {
  using namespace ShaderRecompiler;
  constexpr uint64_t base = 0xfe0080000ull;
  constexpr uint64_t leaf_a = base + 0x100, leaf_b = base + 0x140, leaf_c = base + 0x180,
                     leaf_bad = base + 0x1c0, leaf_clobber = base + 0x200;
  const uint32_t shader[] = {
      EncodeVop1(0x02, 4, 256),            // 0x00 v_readfirstlane_b32 s4, v0
      EncodeSop2(0x1e, 4, 4, 132),         // 0x04 s_lshl_b32 s4, s4, 4
      EncodeSmem0(0x0a, 8, 0), 4u << 25u,  // 0x08 s_buffer_load_dwordx4 s[8:11], s[0:3], soffset s4
      EncodeSopp(0x0c, 0),                 // 0x10 s_waitcnt
      EncodeSMovB32(14, 8),                // 0x14 s_mov_b32 s14, s8
      EncodeSMovB32(15, 9),                // 0x18 s_mov_b32 s15, s9
      EncodeSop1(0x21, 14, 14),            // 0x1c s_swappc_b64 s[14:15], s[14:15]
      EncodeSopp(0x01)};                   // 0x20 s_endpgm
  constexpr uint32_t call_pc = 0x1c, return_pc = 0x20, call_index = 6;
  // 4 KiB of small-table/leaf words, an 8 MiB + 4 KiB large-table region at base + 0x1000, then a
  // 10 MiB long-leaf region at base + 0x802000.
  constexpr uint64_t memory_bytes = 0x1000 + 0x800000 + 0x1000 + 0xa00000;
  struct Memory {
    std::vector<uint32_t> words = std::vector<uint32_t>(memory_bytes / 4u);
    bool readable = true;
    std::vector<std::pair<uint64_t, uint64_t>> holes; // unreadable [begin, end)
    uint64_t attempts = 0, largest = 0;
    uint64_t table_requested = 0; // bytes REQUESTED (incl. failed) inside the large-table region
  } memory;
  const auto at = [&](uint64_t address) -> uint32_t& { return memory.words[(address - base) / 4u]; };
  const auto leaf = [&](uint64_t address, uint32_t value) {
    at(address) = EncodeSMovB32(20, 255);
    at(address + 4) = value;
    at(address + 8) = EncodeSop1(0x21, 125, 14); // s_setpc_b64 s[14:15]
  };
  leaf(leaf_a, 0x11111111u);
  leaf(leaf_b, 0x22222222u);
  leaf(leaf_c, 0x33333333u);
  at(leaf_bad) = EncodeSopp(0x01);               // s_endpgm inside a callee
  at(leaf_clobber) = EncodeSMovB32(14, 129);     // writes its own return pair
  at(leaf_clobber + 4) = EncodeSop1(0x21, 125, 14);
  constexpr uint64_t leaf_malformed = base + 0x240;
  at(leaf_malformed) = EncodeSMovB32(20, 249);   // scalar source 249: fatal in the normal decoder
  at(leaf_malformed + 4) = EncodeSop1(0x21, 125, 14);
  const auto entry = [&](uint32_t index, uint64_t target) {
    at(base + index * 16u) = static_cast<uint32_t>(target);
    at(base + index * 16u + 4) = static_cast<uint32_t>(target >> 32u);
    at(base + index * 16u + 8) = 0xd00d0000u + index; // data words passed in s[16:17]
  };
  entry(0, leaf_a | 3u); // SWAPPC ignores the low two bits.
  entry(1, leaf_b);
  entry(2, leaf_a);      // duplicate callee
  const auto read = [](void *data, uint64_t address, std::span<uint32_t> out) {
    auto &m = *static_cast<Memory *>(data);
    ++m.attempts;
    if (address >= base + 0x1000 && address < base + 0x801000) m.table_requested += out.size_bytes();
    const uint64_t end = base + m.words.size() * 4u;
    if (!m.readable || address < base || address > end || out.size_bytes() > end - address)
      return false;
    for (const auto &[hole_begin, hole_end] : m.holes)
      if (address < hole_end && hole_begin < address + out.size_bytes()) return false;
    m.largest = std::max<uint64_t>(m.largest, out.size_bytes());
    std::memcpy(out.data(), reinterpret_cast<const uint8_t *>(m.words.data()) + address - base,
                out.size_bytes());
    return true;
  };
  FakeGuestMemory fake(read, &memory);
  std::array<uint32_t, 64> user_data{};
  user_data[0] = static_cast<uint32_t>(base);
  user_data[1] = static_cast<uint32_t>(base >> 32u);
  user_data[2] = 48;        // three 16-byte entries
  user_data[3] = 3u << 28u; // raw offset bounds
  auto options = MakeCompileOptions(ShaderType::Compute);
  options.user_data = user_data;
  IR::SrtRuntime runtime{.user_data = user_data, .userdata = &memory, .read_specialization_memory = read};

  auto source = PrepareShaderSource(shader, options);
  Check(source.unsupported_call_pc == UINT32_MAX && source.calls.size() == 1 &&
            source.calls[0].kind == ShaderSource::Call::Kind::Table &&
            source.calls[0].table_stride == 16 && source.calls[0].table_addend == 0 &&
            source.calls[0].table_word_offset == 0,
        "waterfall table call was not recognized as a closed table site");
  const auto selects = [](const Decoder::Program &program) {
    std::vector<const Decoder::Instruction *> result;
    for (const auto &inst : program.instructions)
      if (inst.opcode == Decoder::Opcode::S_CALL_SELECT_B64) result.push_back(&inst);
    return result;
  };
  const auto expect_link = [&](uint64_t first, uint64_t second, const char *text) {
    std::string failure;
    const auto *linked = TryRefreshShaderSource(source, runtime, failure);
    Check(linked != nullptr && failure.empty(), text);
    const auto chain = selects(*linked);
    Check(chain.size() == 2 && chain[0]->raw[1] == static_cast<uint32_t>(first) &&
              chain[0]->raw[2] == static_cast<uint32_t>(first >> 32u) &&
              chain[1]->raw[1] == static_cast<uint32_t>(second) &&
              chain[0]->offset == call_pc && chain[0]->src0.reg == 14,
          text);
    const auto &call = linked->instructions[call_index];
    Check(call.opcode == Decoder::Opcode::S_BRANCH && call.branch_target == chain[0]->pc,
          "table call still writes its pair before the compare chain");
    const auto *miss = chain[1] + 1;
    const auto *skip = miss + 1;
    Check(miss->opcode == Decoder::Opcode::S_CALL_TABLE_MISS_B64 && miss->raw[1] == call_pc &&
              miss->src0.reg == 14 && skip->opcode == Decoder::Opcode::S_BRANCH &&
              skip->branch_target == return_pc,
          "no-match path does not record a callable miss before continuing after the call");
    for (const auto *select : chain) {
      const auto target = std::ranges::find(linked->instructions, select->branch_target,
                                            &Decoder::Instruction::pc);
      Check(target != linked->instructions.end() && target->opcode == Decoder::Opcode::S_CALL_ENTER_B64 &&
                target->offset == call_pc && target->dst.reg == 14 && select->data_dwords == 0 &&
                (target + 1)->opcode == Decoder::Opcode::S_MOV_B32,
            "candidate does not enter its own leaf copy through the return-pair write");
    }
    return linked;
  };
  const auto *linked = expect_link(leaf_a, leaf_b, "closed table did not link both distinct callees");
  Check(source.revision == 1 &&
            std::ranges::find(source.reads, std::pair<uint64_t, uint64_t>{base + 16, 4}) != source.reads.end() &&
            std::ranges::find(source.reads, std::pair<uint64_t, uint64_t>{leaf_a, 12}) != source.reads.end() &&
            std::ranges::find(source.reads, std::pair<uint64_t, uint64_t>{leaf_b, 12}) != source.reads.end(),
        "table entries and callee code are not dispatch dependencies");
  auto translated = TranslateProgram(*linked, options);
  bool records_miss = false, escapes_return = false;
  for (const auto *block : translated.program.blocks)
    for (const auto &inst : *block) {
      records_miss |= inst.GetOpcode() == IR::ValueOpcode::CallTableMiss;
      escapes_return |= inst.GetOpcode() == IR::ValueOpcode::CallReturnAddress;
    }
  Check(records_miss && !escapes_return,
        "the miss path was dropped or a return address survived linking");
  translated.program.source_reads = source.reads;
  const auto plan = IR::ExtractResourcePlan(translated.program);
  Check(plan.source_reads == source.reads && plan.capture_specialization_reads,
        "table call dependencies were omitted from resource alias validation");

  std::string failure;
  (void)TryRefreshShaderSource(source, runtime, failure);
  Check(source.revision == 1 && failure.empty(), "unchanged table relinked");
  entry(1, leaf_c);
  (void)expect_link(leaf_a, leaf_c, "changed table entry did not relink");
  Check(source.revision == 2, "changed table entry reused the old link");
  entry(1, leaf_b);
  (void)expect_link(leaf_a, leaf_b, "restored table entry did not relink");
  Check(source.revision == 3, "A->B->A reused a stale link");
  leaf(leaf_b, 0x44444444u);
  (void)TryRefreshShaderSource(source, runtime, failure);
  Check(source.revision == 4 && source.linked->code.end() !=
            std::ranges::find(source.linked->code, 0x44444444u),
        "changed callee bytes retained stale code");

  #include "snippets/table-call-large-cfg.inc"

  // Every unsupported table state fails the refresh with a reason, and a later valid table links.
  struct FailureCase {
    const char *expected;
    std::function<void()> apply, restore;
  };
  const std::array cases{
      // Former 256-entry limit: 600000 entries now exceed the aggregate scan-byte budget instead.
      FailureCase{"scan bytes", [&] { user_data[2] = 16u * 600000u; }, [&] { user_data[2] = 48; }},
      FailureCase{"only null entries",
                  [&] { entry(0, 0); entry(1, 0); entry(2, 0); },
                  [&] { entry(0, leaf_a | 3u); entry(1, leaf_b); entry(2, leaf_a); }},
      FailureCase{"program end", [&] { entry(1, leaf_bad); }, [&] { entry(1, leaf_b); }},
      FailureCase{"return address pair", [&] { entry(1, leaf_clobber); }, [&] { entry(1, leaf_b); }},
      FailureCase{"malformed leaf instruction", [&] { entry(1, leaf_malformed); }, [&] { entry(1, leaf_b); }},
      FailureCase{"unreadable", [&] { memory.readable = false; }, [&] { memory.readable = true; }},
  };
  for (const auto &test : cases) {
    test.apply();
    Check(TryRefreshShaderSource(source, runtime, failure) == nullptr &&
              failure.find(test.expected) != std::string::npos && failure.find("0x0000001c") != std::string::npos,
          "invalid call table did not fail explicitly with its reason");
    test.restore();
    Check(TryRefreshShaderSource(source, runtime, failure) != nullptr && failure.empty(),
          "a failed table poisoned the next valid dispatch");
  }
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
  user_data[2] = 16u * 600000u;
  ExpectFatal([&] { (void)RefreshShaderSource(source, runtime); },
              "the fatal wrapper accepted an over-budget table");
  user_data[2] = 48;
#endif

  // A caller that also builds a data pointer with S_GETPC_B64 (the real shader's PC-relative
  // descriptors) still translates: only saved return addresses are forbidden from escaping.
  std::vector<uint32_t> with_getpc{EncodeSop1(0x1f, 28, 0)}; // s_getpc_b64 s[28:29]
  with_getpc.insert(with_getpc.end(), std::begin(shader), std::end(shader) - 1);
  with_getpc.insert(with_getpc.end(), {EncodeVop1(0x01, 1, 28), EncodeMubuf0(0x1c, 0, false),
                                       EncodeMubuf1(1, 0, 0), EncodeSopp(0x01)});
  auto getpc_source = PrepareShaderSource(with_getpc, options);
  Check(getpc_source.unsupported_call_pc == UINT32_MAX && getpc_source.calls.size() == 1,
        "S_GETPC_B64 caller lost its table call");
  const auto *getpc_linked = TryRefreshShaderSource(getpc_source, runtime, failure);
  Check(getpc_linked != nullptr, "S_GETPC_B64 caller did not link");
  // Previously a fatal GetShaderBase guard. The caller's data pointer keeps its ORIGINAL PC
  // (getpc at pc 0 -> shader base + 4); linked callees cannot use S_GETPC_B64 at all.
  const auto getpc_program = TranslateProgram(*getpc_linked, options);
  bool original_pc = false;
  for (const auto *block : getpc_program.program.blocks)
    for (const auto &inst : *block)
      if (inst.GetOpcode() == IR::ValueOpcode::IAdd64 && inst.Arg(0).TryInstruction() != nullptr &&
          inst.Arg(0).TryInstruction()->GetOpcode() == IR::ValueOpcode::GetShaderBase &&
          inst.Arg(1).Resolve().IsImmediate() && inst.Arg(1).Resolve().U64() == 4u)
        original_pc = true;
  Check(original_pc, "caller S_GETPC_B64 lost its original PC provenance");

  // SMEM immediates are signed: (X << 4) + 8 with displacement -8 is rejected explicitly.
  uint32_t negative[std::size(shader) + 1];
  std::copy(std::begin(shader), std::begin(shader) + 2, negative);
  negative[2] = EncodeSop2(0x00, 4, 4, 136);          // s_add_u32 s4, s4, 8
  negative[3] = shader[2];
  negative[4] = (4u << 25u) | 0x1ffff8u;              // soffset s4, offset -8
  std::copy(std::begin(shader) + 4, std::end(shader), negative + 5);
  const auto negative_source = PrepareShaderSource(negative, options);
  Check(negative_source.unsupported_call_pc != UINT32_MAX && negative_source.calls.empty() &&
            std::string(negative_source.unsupported_call_reason).find("negative") != std::string::npos,
        "negative SMEM table displacement was not rejected explicitly");

  // A 4-byte entry stride cannot hold a 64-bit pointer table: explicit unsupported call.
  uint32_t narrow[std::size(shader)];
  std::copy(std::begin(shader), std::end(shader), narrow);
  narrow[1] = EncodeSop2(0x1e, 4, 4, 130); // << 2
  const auto rejected = PrepareShaderSource(narrow, options);
  Check(rejected.unsupported_call_pc == call_pc && rejected.calls.empty() &&
            rejected.unsupported_call_reason != nullptr &&
            std::string(rejected.unsupported_call_reason).find("8 bytes") != std::string::npos,
        "narrow-stride table call was not rejected explicitly");
}

// Native GPU fixture: a TABLE-selected leaf reads PC-relative data through S_GETPC_B64.
// Paste/include inside VulkanHarness in tests/ShaderRecompilerComputeTests.cpp (next to
// CheckTableSelectedCall). Glue/CLI/CTest are owned by the integration agent (see README).
//
// Leaves A and B have IDENTICAL instruction bytes at different guest VAs. Each runs
//   909 x s_nop; 0xe34: s_getpc_b64 s[20:21]; v7=s20; v8=s21 (original next PC, before the add);
//   s_add_u32 s20, s20, 0x1c8; s_addc_u32 s21, s21, 0; s_load_dword s24, s[20:21], null;
//   v6=s24; return through s[14:15].
// so data = leaf + 0xe38 + 0x1c8 = leaf + 0x1000, outside the callee code (ends at +0xe60).
// Leaf A sits at 0x2fffff000: its original next PC 0x2fffffe38 + 0x1c8 carries into the high word
// (data 0x300000000). Addresses are synthetic; the CPU test covers the captured 0x11d28dae38.
void CheckTableLeafOriginalGetpc() {
  constexpr const char *name = "TableLeafOriginalGetpc";
  constexpr uintptr_t base = 0x00000002ffff0000ull;
  constexpr uint64_t size = 0x20000, alignment = 0x10000;
  constexpr uint64_t table = base + 0x100;
  constexpr uint64_t leaf_a = base + 0xf000, leaf_b = base + 0x3000, leaf_bad = base + 0x6000;
  constexpr uint32_t getpc_pc = 0xe34, data_pc = 0x1000, leaf_words = getpc_pc / 4u + 11u;
  constexpr uint64_t data_a = leaf_a + data_pc, data_b = leaf_b + data_pc;
  constexpr uint64_t output = base + 0x11000;
  constexpr u32 lanes = 64, columns = 3, tail_words = 16;
  constexpr u32 sentinel = 0xa5a5a5a5u;
  constexpr u32 initial_a = 0xa1a1a1a1u, initial_b = 0xb2b2b2b2u, cpu_a = 0x0c0ffee1u, gpu_a = 0x6e6e6e6eu;
  static_assert(leaf_a + getpc_pc + 4u + (data_pc - getpc_pc - 4u) == 0x300000000ull,
                "leaf A's ADD/ADDC must carry across the low 32 bits");
  static_assert(((leaf_a + getpc_pc + 4u) >> 32u) == 2u && (data_a >> 32u) == 3u);
  static_assert(leaf_a + leaf_words * 4u <= data_a && leaf_b + leaf_words * 4u <= data_b &&
                leaf_bad + leaf_words * 4u <= base + 0x7000 && data_b < base + 0x6000);
  static_assert(output + (columns * lanes + tail_words) * 4u <= base + size && data_a + 4u <= output);

  // Identical callable bytes for A and B; leaf_bad differs only in its GETPC destination (the
  // return pair s[14:15]), which the linker must reject softly.
  const auto make_leaf = [](u32 getpc_destination) {
    std::vector<u32> leaf(getpc_pc / 4u, EncodeSopp(0x00));      // s_nop 0 (909 words)
    leaf.insert(leaf.end(), {
        EncodeSop1(0x1f, getpc_destination, 0),                  // 0xe34 s_getpc_b64 (0xbe941f00 for s20)
        EncodeVop1(0x01, 7, 20), EncodeVop1(0x01, 8, 21),        // capture the original pair first
        EncodeSop2(0x00, 20, 20, 255), data_pc - getpc_pc - 4u,  // s_add_u32 s20, s20, 0x1c8
        EncodeSop2(0x04, 21, 21, InlineU32(0)),                  // s_addc_u32 s21, s21, 0
        EncodeSmem0(0x00, 24, 10), 125u << 25u,                  // s_load_dword s24, s[20:21], null
        EncodeSopp(0x0c, 0),                                     // s_waitcnt
        EncodeVop1(0x01, 6, 24),                                 // v6 = data
        EncodeSop1(0x21, 125, 14)});                             // return through s[14:15]
    return leaf;
  };
  const auto leaf_code = make_leaf(20);
  const auto bad_code = make_leaf(14);
  Require(name, "leaf shape", leaf_code.size() == leaf_words && leaf_code[getpc_pc / 4u] == 0xbe941f00u &&
              leaf_code[getpc_pc / 4u + 4u] == 0x1c8u,
          "GETPC leaf fixture does not have the specified layout");

  // Caller: per-lane index (lane & 1) waterfall; EXEC saved in s[40:41] and the remaining lanes in
  // s[42:43] (the leaf's GETPC writes s[20:21]). Three output columns of 64 DWORDs: data, PC lo, PC hi.
  std::vector<u32> code{
      EncodeSop1(0x04, 40, 126),                    // s_mov_b64 s[40:41], exec
      EncodeVop2(0x1b, 1, InlineU32(1), 0),         // v_and_b32 v1, 1, v0   (per-lane index)
      EncodeVop2(0x1a, 2, InlineU32(2), 0),         // v_lshlrev_b32 v2, 2, v0
      EncodeVop1(0x01, 6, InlineU32(0)),            // v6 = 0
      EncodeVop1(0x01, 7, InlineU32(0)),            // v7 = 0
      EncodeVop1(0x01, 8, InlineU32(0))};           // v8 = 0
  const auto loop = code.size();
  code.insert(code.end(), {
      EncodeVop1(0x02, 4, Vgpr(1)),                 // v_readfirstlane_b32 s4, v1
      EncodeSop1(0x04, 42, 126),                    // s_mov_b64 s[42:43], exec
      EncodeVopc(0xd2, 4, 1),                       // v_cmpx_eq_u32 s4, v1
      EncodeSop2(0x1e, 5, 4, InlineU32(4)),         // s_lshl_b32 s5, s4, 4
      EncodeSmem0(0x09, 14, 4), 5u << 25u,          // s_buffer_load_dwordx2 s[14:15], s[8:11], soffset s5
      EncodeSopp(0x0c, 0),                          // s_waitcnt
      EncodeSop1(0x21, 14, 14),                     // s_swappc_b64 s[14:15], s[14:15]
      EncodeSop2(0x15, 42, 42, 126),                // s_andn2_b64 s[42:43], s[42:43], exec
      EncodeSop1(0x04, 126, 42)});                  // s_mov_b64 exec, s[42:43]
  code.push_back(EncodeSopp(0x09, static_cast<u32>(static_cast<int32_t>(loop) -
                                                   static_cast<int32_t>(code.size() + 1u)) & 0xffffu));
  code.insert(code.end(), {
      EncodeSop1(0x04, 126, 40),                    // s_mov_b64 exec, s[40:41]
      EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 2),        // column 0: data
      EncodeMubuf0(0x1c, 256), EncodeMubuf1(7, 0, 2),   // column 1: original next PC low
      EncodeMubuf0(0x1c, 512), EncodeMubuf1(8, 0, 2)}); // column 2: original next PC high
  AppendEnd(&code);
  ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
      {.type = Prospero::ShaderBinaryType::kCs,
       .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
  EnsureRuntimeContext();
  int64_t physical = -1;
  Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
      0, LibKernel::Memory::KernelGetDirectMemorySize(), size, alignment, 0, &physical) == 0,
      "GETPC fixture allocation failed");
  void *mapped = reinterpret_cast<void *>(base);
  Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
      &mapped, size, 0x3, 0x10, physical, alignment) == 0 && mapped == reinterpret_cast<void *>(base),
      "GETPC fixture mapping at the carry-stress base failed");
  std::memset(mapped, 0, size);
  for (const auto [address, words] : {std::pair{leaf_a, &leaf_code}, std::pair{leaf_b, &leaf_code},
                                      std::pair{leaf_bad, &bad_code}})
    LibKernel::Memory::WriteBacking(address, words->data(), words->size() * sizeof(u32));
  LibKernel::Memory::WriteBacking(data_a, &initial_a, sizeof(initial_a));
  LibKernel::Memory::WriteBacking(data_b, &initial_b, sizeof(initial_b));
  const auto backing = [&](uint64_t address) {
    u32 value = 0;
    Require(name, "backing", LibKernel::Memory::TryReadBacking(address, &value, sizeof(value)),
            "GETPC fixture backing is unreadable");
    return value;
  };
  {
    std::vector<u32> a(leaf_words), b(leaf_words);
    Require(name, "identical callee bytes",
            LibKernel::Memory::TryReadBacking(leaf_a, a.data(), a.size() * sizeof(u32)) &&
                LibKernel::Memory::TryReadBacking(leaf_b, b.data(), b.size() * sizeof(u32)) && a == b &&
                a == leaf_code,
            "leaves A and B are not byte-identical");
    Require(name, "initial data", backing(data_a) == initial_a && backing(data_b) == initial_b &&
                initial_a != 0u && initial_b != 0u && initial_a != initial_b,
            "mapped CPU data is not nonzero and distinct");
  }

  RenderContext context(m_runtime_context);
  context.InitializeGpu(nullptr);
  LibKernel::Memory::InstallGpuResources(&context);
  context.GetGpu().SendCommandSync([&] {
    CommandProcessor processor(context, 0);
    processor.BufferInit();
    context.MapMemory(base, size);
    auto &cache = context.GetBufferCache();
    auto &shaders = processor.GetShCtx();
    const auto &faults = BufferCacheTestAccess::Faults(cache);
    shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                         .num_thread_x = lanes, .num_thread_y = 1, .num_thread_z = 1,
                         .wave_size = 64, .user_sgpr = 12});
    const auto raw_descriptor = [](uint64_t address, u32 bytes) {
      ShaderBufferResource v{};
      v.UpdateAddress48(address);
      v.fields[2] = bytes;
      v.fields[3] = DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      return v;
    };
    const auto set_vsharp = [&](u32 sgpr, const ShaderBufferResource &v) {
      for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(sgpr + i, v.fields[i]);
    };
    set_vsharp(0, raw_descriptor(output, columns * lanes * sizeof(u32)));
    set_vsharp(8, raw_descriptor(table, 2u * 16u));
    const auto write_entry = [&](u32 index, uint64_t target) {
      Require(name, "refresh table entry", context.InvalidateMemory(table + index * 16u, 8),
              "GETPC call table escaped mapped memory");
      LibKernel::Memory::WriteBacking(table + index * 16u, &target, sizeof(target));
    };
    const auto write_data = [&](uint64_t address, u32 value) {
      Require(name, "refresh data", context.InvalidateMemory(address, sizeof(value)),
              "GETPC data escaped mapped memory");
      LibKernel::Memory::WriteBacking(address, &value, sizeof(value));
    };
    struct Expect { uint64_t leaf; u32 data; };
    std::vector<std::pair<uint64_t, uint64_t>> reads;
    std::vector<uint32_t> flattened;
    // One dispatch; expect_module=false must bind nothing and leave every output/tail word alone.
    const auto run = [&](const std::string &round, bool expect_module, Expect even, Expect odd) {
      constexpr u32 words = columns * lanes + tail_words;
      Require(name, "reset output", context.InvalidateMemory(output, words * 4u), "output escaped mapped memory");
      std::array<u32, words> actual{};
      actual.fill(sentinel);
      LibKernel::Memory::WriteBacking(output, actual.data(), sizeof(actual));
      const auto misses_before = faults.CallTableMisses();
      ShaderComputeInputInfo input{};
      const auto *program = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
      Require(name, round.c_str(), (program != nullptr) == expect_module, round + ": unexpected module binding");
      reads.clear();
      flattened.clear();
      if (program != nullptr && input.stage.resources != nullptr) {
        reads = input.stage.resources->specialization_reads;
        flattened = input.stage.resources->flattened_srt;
      }
      processor.DispatchDirect(1, 1, 1, 0x41u);
      context.RunGarbageCollector();
      context.GetCommandScheduler().Finish();
      cache.ReadMemory(output, sizeof(actual));
      Require(name, "read output", LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)),
              "output was not readable");
      for (u32 lane = 0; lane < lanes; ++lane) {
        const auto &e = (lane & 1u) == 0u ? even : odd;
        const uint64_t next_pc = e.leaf + getpc_pc + 4u;
        const std::array<u32, columns> expected{e.data, static_cast<u32>(next_pc), static_cast<u32>(next_pc >> 32u)};
        for (u32 column = 0; column < columns; ++column) {
          const u32 value = expect_module ? expected[column] : sentinel;
          Require(name, round.c_str(), actual[column * lanes + lane] == value,
                  round + " column " + std::to_string(column) + " lane " + std::to_string(lane) + " expected " +
                      Hex(value) + ", got " + Hex(actual[column * lanes + lane]));
        }
      }
      for (u32 i = columns * lanes; i < words; ++i)
        Require(name, round.c_str(), actual[i] == sentinel, round + ": tail canary " + std::to_string(i) + " changed");
      Require(name, round.c_str(), faults.CallTableMisses() == misses_before, round + ": unexpected callable miss");
      return program != nullptr ? program->id : 0;
    };
    const auto depends_on = [&](uint64_t address) {
      return std::ranges::any_of(reads, [&](const auto &range) {
        return range.first <= address && address - range.first < range.second;
      });
    };
    const auto holds = [&](u32 value) { return std::ranges::find(flattened, value) != flattened.end(); };

    // G1: row0 = A|3 (SWAPPC ignores the low bits), row1 = B. Exact original next PC and data
    // per parity; A's original PC + 0x1c8 carries into the high word.
    write_entry(0, leaf_a | 3u);
    write_entry(1, leaf_b);
    const Expect a{leaf_a, initial_a}, b{leaf_b, initial_b};
    const auto id_ab = run("original GETPC A/B", true, a, b);
    Require(name, "data dependencies", depends_on(data_a) && depends_on(data_b) && holds(initial_a) &&
                holds(initial_b),
            "PC-relative data is not a current scalar specialization read");
    // G2: unchanged inputs reuse the module.
    Require(name, "unchanged reuse", run("unchanged A/B", true, a, b) == id_ab, "unchanged table relinked");
    // G3: row0 relocated to B (identical bytes, different origin) changes the module key and
    // returns B's original PC and data on even lanes too; restoring A gives the exact original module.
    write_entry(0, leaf_b);
    const auto id_bb = run("relocated row0 to B", true, b, b);
    Require(name, "relocation re-key", id_bb != 0 && id_bb != id_ab, "same-byte relocation reused A's module");
    write_entry(0, leaf_a | 3u);
    Require(name, "restore A", run("restored A/B", true, a, b) == id_ab, "restored origin lost the original module");
    // G4: CPU changes ONLY the literal data: fresh output from the same module (data is a current
    // specialization dependency, not baked code).
    write_data(data_a, cpu_a);
    const Expect a_cpu{leaf_a, cpu_a};
    Require(name, "cpu data reuse", run("cpu data change", true, a_cpu, b) == id_ab,
            "a data-only change produced new code");
    Require(name, "cpu data dependency", depends_on(data_a) && holds(cpu_a) && !holds(initial_a),
            "the materialized data did not follow the current CPU bytes");
    // G5: the GPU rewrites the data through a persistent owner while CPU backing stays old. The
    // dependency read must synchronize it before materialization (dirty before, clean after).
    {
      auto [buffer, offset] = cache.ObtainBuffer(data_a, sizeof(u32), true, false, {}, false);
      buffer->Fill(offset, sizeof(u32), gpu_a);
    }
    Require(name, "gpu dirty before", cache.HasGpuDirtyBytes(data_a, 4) && backing(data_a) == cpu_a,
            "GPU rewrite was not pending with stale CPU backing before the dispatch");
    const Expect a_gpu{leaf_a, gpu_a};
    Require(name, "gpu data reuse", run("gpu data rewrite", true, a_gpu, b) == id_ab,
            "a GPU data rewrite produced new code");
    Require(name, "gpu data synchronized", !cache.HasGpuDirtyBytes(data_a, 4) && backing(data_a) == gpu_a &&
                depends_on(data_a) && holds(gpu_a),
            "the GPU-written data was not read in order");
    // G6: a GETPC writing the return pair s[14:15] is rejected softly (nothing bound, outputs
    // untouched); restoring B recovers the exact original module.
    write_entry(1, leaf_bad);
    (void)run("GETPC into return pair fails", false, a_gpu, b);
    write_entry(1, leaf_b);
    Require(name, "return-pair recovery", run("recovered A/B", true, a_gpu, b) == id_ab,
            "recovery after the return-pair rejection lost the original module");
    context.UnmapMemory(base, size);
    context.GetCommandScheduler().Finish();
  });
  LibKernel::Memory::InstallGpuResources(nullptr);
  context.ShutdownGpu();
  Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
          "GETPC fixture release failed");
  std::printf("[gpu]     %-32s ok\n", name);
}

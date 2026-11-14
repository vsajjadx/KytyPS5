// === rev2. Paste inside VulkanHarness in tests/ShaderRecompilerComputeTests.cpp, after CheckScalarLeafCall() ===
// Also add inside the existing `struct BufferCacheTestAccess { ... };`:
//   static const FaultManager &Faults(const BufferCache &cache) { return cache.m_fault_manager; }
// CLI branch (main, next to --scalar-leaf-call-only):
//   if (argc == 2 && std::strcmp(argv[1], "--table-call-only") == 0) {
//     VulkanHarness vulkan;
//     vulkan.CheckTableSelectedCall();
//     return 0;
//   }
// CMakeLists.txt: add_test(NAME table_selected_call
//                   COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --table-call-only)

  // A real 64-lane waterfall over a 16-byte {fnptr, data} table (index = lane & 1). Before the
  // waterfall every lane stores the same pointer pair into entry 1 through a writable V# (s[12:15]);
  // pointing that V# at a scratch area disables the mutation. The compare chain must act on the
  // pointer the GPU loads (dynamic), and every null/out-of-range/unknown pointer must be recorded
  // as a GPU callable miss; never a silent success.
  void CheckTableSelectedCall() {
    constexpr const char *name = "TableSelectedCall";
    constexpr uintptr_t base = 0x0000000208a00000ull;
    constexpr uintptr_t alias_va = 0x0000000208b00000ull;
    constexpr uint64_t size = 0x40000; // the large-table rounds use base + 0x18000..0x38010
    constexpr auto table = base + 0x100;
    constexpr auto scratch = base + 0x3000;
    constexpr auto leaf_a = base + 0x1000, leaf_b = base + 0x1100, leaf_c = base + 0x1200,
                   leaf_bad = base + 0x1300;
    constexpr auto output = base + 0x4000;
    constexpr u32 lanes = 64, sentinel = 0xa5a5a5a5u, value_a = 0x11u, value_b = 0x22u, value_c = 0x33u;
    std::vector<u32> code{
        EncodeSop1(0x04, 20, 126),                    // s_mov_b64 s[20:21], exec
        EncodeVop2(0x1b, 1, InlineU32(1), 0),         // v_and_b32 v1, 1, v0   (per-lane index)
        EncodeVop2(0x1a, 2, InlineU32(2), 0),         // v_lshlrev_b32 v2, 2, v0
        EncodeVop1(0x01, 6, InlineU32(0)),            // v_mov_b32 v6, 0
        EncodeVop1(0x01, 3, 6),                       // v_mov_b32 v3, s6 (new entry-1 lo)
        EncodeVop1(0x01, 4, 7),                       // v_mov_b32 v4, s7 (new entry-1 hi)
        EncodeMubuf0(0x1c, 16, false, false), EncodeMubuf1(3, 3, 0),  // store v3 -> s[12:15]+16
        EncodeMubuf0(0x1c, 20, false, false), EncodeMubuf1(4, 3, 0),  // store v4 -> s[12:15]+20
        EncodeVop1(0x01, 8, 16),                      // v_mov_b32 v8, s16 (new entry-0 lo)
        EncodeVop1(0x01, 9, 17),                      // v_mov_b32 v9, s17 (new entry-0 hi)
        EncodeMubuf0(0x1c, 0, false, false), EncodeMubuf1(8, 3, 0),   // store v8 -> s[12:15]+0
        EncodeMubuf0(0x1c, 4, false, false), EncodeMubuf1(9, 3, 0),   // store v9 -> s[12:15]+4
        EncodeSopp(0x0c, 0),                          // s_waitcnt
        0xf4800000u, 0xfa000000u};                    // s_dcache_inv (captured f5edf70dbb1f6057 encoding)
    const auto loop = code.size();
    code.insert(code.end(), {
        EncodeVop1(0x02, 4, Vgpr(1)),                 // v_readfirstlane_b32 s4, v1
        EncodeSop1(0x04, 18, 126),                    // s_mov_b64 s[18:19], exec
        EncodeVopc(0xd2, 4, 1),                       // v_cmpx_eq_u32 s4, v1
        EncodeSop2(0x1e, 5, 4, InlineU32(4)),         // s_lshl_b32 s5, s4, 4
        EncodeSmem0(0x09, 14, 4), 5u << 25u,          // s_buffer_load_dwordx2 s[14:15], s[8:11], soffset s5
        EncodeSopp(0x0c, 0),                          // s_waitcnt
        EncodeSop1(0x21, 14, 14),                     // s_swappc_b64 s[14:15], s[14:15]
        EncodeSop2(0x15, 18, 18, 126),                // s_andn2_b64 s[18:19], s[18:19], exec
        EncodeSop1(0x04, 126, 18)});                  // s_mov_b64 exec, s[18:19]
    code.push_back(EncodeSopp(0x09, static_cast<u32>(static_cast<int32_t>(loop) -
                                                     static_cast<int32_t>(code.size() + 1u)) & 0xffffu));
    code.insert(code.end(), {
        EncodeSop1(0x04, 126, 20),                    // s_mov_b64 exec, s[20:21]
        EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 2)});  // buffer_store_dword v6, v2, s[0:3]
    // Same program plus one global store: unbounded writes -> table dispatch must fail closed.
    std::vector<u32> flat_code(code);
    flat_code.insert(flat_code.end(), {EncodeFlat0(0x1c, 2), EncodeFlat1(0, 125, 6, 2)});
    // Same program plus one IMAGE_STORE through a T# in s[24:31]: storage-image extent is
    // unproven, so the table dispatch must fail closed.
    std::vector<u32> image_code(code);
    image_code.insert(image_code.end(), {EncodeMimg0(0x08, 0x1), EncodeMimg1(6, 0, 6)});
    AppendEnd(&code);
    AppendEnd(&flat_code);
    #include "snippets/table-call-flatlocal-setup.inc"
    #include "snippets/table-call-large-setup.inc"
    AppendEnd(&image_code);
    for (const auto *program_code : {&code, &flat_code, &image_code})
      ShaderMapUserData(reinterpret_cast<uint64_t>(program_code->data()),
          {.type = Prospero::ShaderBinaryType::kCs,
           .code_size_bytes = static_cast<u32>(program_code->size() * sizeof(u32))});
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
        "table call allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, size, 0x3, 0x10, physical, size) == 0 && mapped == reinterpret_cast<void *>(base),
        "table call mapping failed");
    std::memset(mapped, 0, size);
    const auto write_leaf = [](uint64_t address, u32 value) {
      const std::array leaf{EncodeVop1(0x01, 6, 255), value, EncodeSop1(0x21, 125, 14)};
      LibKernel::Memory::WriteBacking(address, leaf.data(), sizeof(leaf));
    };
    write_leaf(leaf_a, value_a);
    write_leaf(leaf_b, value_b);
    write_leaf(leaf_c, value_c);
    const std::array bad_leaf{EncodeSopp(0x01)};
    LibKernel::Memory::WriteBacking(leaf_bad, bad_leaf.data(), sizeof(bad_leaf));

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
                           .wave_size = 64, .user_sgpr = 18});
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
      set_vsharp(0, raw_descriptor(output, lanes * sizeof(u32)));
      const auto set_table_records = [&](u32 entries) { set_vsharp(8, raw_descriptor(table, entries * 16u)); };
      // The writable V# covers `writable` (scratch disables both stores); entry 0 receives
      // `pointer0`, entry 1 `pointer`.
      const auto set_mutation = [&](uint64_t writable, uint64_t pointer, uint64_t pointer0 = leaf_a) {
        set_vsharp(12, raw_descriptor(writable, 32));
        shaders.SetCsUserSgpr(6, static_cast<u32>(pointer));
        shaders.SetCsUserSgpr(7, static_cast<u32>(pointer >> 32u));
        shaders.SetCsUserSgpr(16, static_cast<u32>(pointer0));
        shaders.SetCsUserSgpr(17, static_cast<u32>(pointer0 >> 32u));
      };
      const auto write_entry = [&](u32 index, uint64_t target) {
        Require(name, "refresh table entry", context.InvalidateMemory(table + index * 16u, 8),
                "call table escaped mapped memory");
        LibKernel::Memory::WriteBacking(table + index * 16u, &target, sizeof(target));
      };
      // One dispatch. expect_module=false: the dispatch must fail explicitly (no module bound).
      // expect_miss: pointers a recorded GPU callable miss may report (empty: no miss allowed).
      // The record must equal exactly one of them (a coherent tuple), with this shader and pc.
      const auto run = [&](const std::string &round, bool expect_module, u32 even, u32 odd,
                           std::initializer_list<uint64_t> expect_miss) {
        Require(name, "reset output", context.InvalidateMemory(output, lanes * 4), "output escaped mapped memory");
        std::array<u32, lanes> actual{};
        actual.fill(sentinel);
        LibKernel::Memory::WriteBacking(output, actual.data(), sizeof(actual));
        const auto misses_before = faults.CallTableMisses();
        ShaderComputeInputInfo input{};
        const auto *program = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
        Require(name, round.c_str(), (program != nullptr) == expect_module, round + ": unexpected module binding");
        processor.DispatchDirect(1, 1, 1, 0x41u);
        context.RunGarbageCollector();               // fault-buffer pass (BDA + call misses)
        context.GetCommandScheduler().Finish();      // its deferred readback
        cache.ReadMemory(output, lanes * 4);
        Require(name, "read output", LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)),
                "output was not readable");
        for (u32 lane = 0; lane < lanes; ++lane) {
          const u32 expected = (lane & 1u) == 0 ? even : odd;
          Require(name, round.c_str(), actual[lane] == expected,
                  round + " lane " + std::to_string(lane) + " expected " + Hex(expected) + ", got " +
                      Hex(actual[lane]));
        }
        const auto missed = faults.CallTableMisses() - misses_before;
        if (expect_miss.size() == 0) {
          Require(name, round.c_str(), missed == 0, round + ": unexpected callable miss");
        } else {
          const auto last = faults.LastCallTableMiss();
          Require(name, round.c_str(), missed != 0 && std::ranges::find(expect_miss, last.target) != expect_miss.end() &&
                      last.pc != 0 && last.shader_hash != 0,
                  round + ": the callable miss was not recorded with its exact pointer (got " +
                      std::to_string(missed) + " misses, target " + Hex(static_cast<u32>(last.target)) + ")");
        }
        return program != nullptr ? program->id : 0;
      };
      set_table_records(2);
      set_mutation(scratch, 0);
      // R1 CPU table {A, B}.
      write_entry(0, leaf_a | 3u);
      write_entry(1, leaf_b);
      const auto id_ab = run("cpu table A/B", true, value_a, value_b, {});
      // R2 swapped -> relinked module, swapped outputs.
      write_entry(0, leaf_b);
      write_entry(1, leaf_a);
      Require(name, "re-key", run("swapped B/A", true, value_b, value_a, {}) != id_ab,
              "a changed table reused the old link");
      // R3 GPU-written entry 0 := A, ordered download before enumeration.
      {
        auto [buffer, offset] = cache.ObtainBuffer(table, 8, true);
        buffer->Fill(offset, 4, static_cast<u32>(leaf_a));
        buffer->Fill(offset + 4, 4, static_cast<u32>(leaf_a >> 32u));
      }
      (void)run("gpu-written entry", true, value_a, value_a, {});
      Require(name, "ordered table read", !cache.HasGpuDirtyBytes(table, 8), "the table was not read in order");
      // R4 NULL entry 1: odd lanes take the observable miss path (pointer 0), even lanes run A.
      write_entry(0, leaf_a);
      write_entry(1, 0);
      (void)run("null entry", true, value_a, 0, {0});
      // R5 out-of-range index: one-entry table; index 1 loads zero -> miss.
      write_entry(1, leaf_b);
      set_table_records(1);
      (void)run("out-of-range entry", true, value_a, 0, {0});
      set_table_records(2);
      // R6 same-dispatch write of an UNKNOWN pointer (C) into entry 1 after host enumeration of {A,B}.
      set_mutation(table, leaf_c);
      (void)run("same-dispatch unknown pointer", true, value_a, 0, {leaf_c});
      // R7 same-dispatch write of a KNOWN pointer (A) into entry 1: dynamic, not a snapshot of B.
      write_entry(1, leaf_b);
      set_mutation(table, leaf_a);
      (void)run("same-dispatch known pointer", true, value_a, value_a, {});
      set_mutation(scratch, 0);
      // R12 two distinct unknown pointers race in one dispatch (different high words): the
      // record is ONE coherent tuple, never lo/hi from different misses; then no stale replay.
      constexpr uint64_t unknown_d = 0x0000dead00001000ull;
      write_entry(0, leaf_a);
      write_entry(1, leaf_b);
      {
        set_mutation(table, unknown_d, leaf_c);
        const auto before = faults.CallTableMisses();
        (void)run("racing unknown pointers", true, 0, 0, {leaf_c, unknown_d});
        Require(name, "both racing misses counted", faults.CallTableMisses() - before >= 2,
                "two distinct missing selections were not both counted");
      }
      set_mutation(scratch, 0);
      write_entry(0, leaf_a);
      write_entry(1, leaf_b);
      {
        const auto before = faults.CallTableMisses();
        context.RunGarbageCollector();
        context.GetCommandScheduler().Finish();
        Require(name, "no stale replay", faults.CallTableMisses() == before,
                "a harvested callable miss was reported again");
      }
      // R8 table page aliased at another VA: same-dispatch alias writes would be invisible -> explicit failure.
      write_entry(1, leaf_b);
      void *alias = reinterpret_cast<void *>(alias_va);
      Require(name, "alias map", LibKernel::Memory::KernelMapDirectMemory(
          &alias, 0x4000, 0x3, 0x10, physical, 0x4000) == 0, "table alias mapping failed");
      (void)run("aliased table fails", false, sentinel, sentinel, {});
      Require(name, "alias unmap", LibKernel::Memory::KernelMunmap(alias_va, 0x4000) == 0, "alias unmap failed");
      // R9 callee code writable through this dispatch's buffer binding -> explicit failure.
      set_mutation(leaf_a, 0);
      (void)run("writable callee code fails", false, sentinel, sentinel, {});
      set_mutation(scratch, 0);
      // R10 invalid callee (s_endpgm) -> explicit failure; R11 recovery binds the A/B module.
      write_entry(1, leaf_bad);
      (void)run("invalid callee fails", false, sentinel, sentinel, {});
      write_entry(1, leaf_b);
      Require(name, "recovery", run("recovered A/B", true, value_a, value_b, {}) == id_ab,
              "the restored table did not bind the A/B module");
      #include "snippets/table-call-flatlocal-rounds.inc"
      // R13 flat/global store in the table program: fail closed (callee code cannot be proven
      // unwritten), then the store-free program runs again.
      const auto bind_code = [&](const std::vector<u32> &program_code, u32 user_sgprs = 18) {
        shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(program_code.data()),
                             .num_thread_x = lanes, .num_thread_y = 1, .num_thread_z = 1,
                             .wave_size = 64, .user_sgpr = static_cast<uint8_t>(user_sgprs)});
      };
      bind_code(flat_code);
      (void)run("flat store fails closed", false, sentinel, sentinel, {});
      bind_code(code);
      (void)run("store-free program recovers", true, value_a, value_b, {});
      #include "snippets/R14-prepared-storage.inc"
      #include "snippets/table-call-large-rounds.inc"
      context.UnmapMemory(base, size);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "table call release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

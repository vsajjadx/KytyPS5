  // Standalone TABLE-resource rejection test. The live 16-byte table row is
  // {callee64, context64}; S_BUFFER_LOAD_DWORDX4 really puts context64 in s16:17.
  // Mapped nested-resource execution is required FIRST. Only then test that
  // an unreadable live context rejects this dispatch and allows exact recovery.
  void CheckTableCallResourceFailure() {
    constexpr const char* name = "TableCallResourceFailure";
    constexpr uintptr_t base = 0x0000000209c00000ull;
    constexpr uint64_t bytes = 0x10000;
    constexpr auto table = base + 0x100, first = base + 0x1000;
    constexpr auto second = base + 0x1100, output = base + 0x4000;
    constexpr auto context_a = base + 0x6000, context_b = base + 0x6100;
    constexpr auto unmapped_context = base + 0x40000;
    constexpr u32 lanes = 64, tail_words = 16, sentinel = 0xa5a5a5a5u;
    constexpr u32 value_a = 0x5511u, value_b = 0x9922u;
    // One GPU-selected uniform call, without an outer waterfall backedge: the
    // descriptor loads have a unique dominating memory instruction per component.
    // lane0 chooses s6&1; other lane values keep the selector GPU-derived/finite.
    std::vector<u32> code {
        EncodeVop2(0x1d, 1, 6, 0),                    // v1 = s6 XOR lane
        EncodeVop2(0x1b, 1, InlineU32(1), 1),         // v1 &= 1
        EncodeVop2(0x1a, 2, InlineU32(2), 0),         // v2 = lane * 4
        EncodeVop1(0x01, 6, InlineU32(0)),
        EncodeVop1(0x02, 4, Vgpr(1)),                 // s4 = first active lane's index
        EncodeSop2(0x1e, 5, 4, InlineU32(4)),         // s5 = index * 16
        EncodeSmem0(0x0a, 14, 4), 5u << 25u,          // s14:17 = table[index].{fn,data}
        EncodeSopp(0x0c, 0), EncodeSop1(0x21, 14, 14),
        EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 2)};   // output V# s0:3
    AppendEnd(&code);
    auto cold_code = code;
    cold_code[3] = EncodeVop1(0x01, 6, InlineU32(1)); // overwritten by every valid callee
    for (const auto* program : {&code, &cold_code})
      ShaderMapUserData(reinterpret_cast<uint64_t>(program->data()),
          {.type = Prospero::ShaderBinaryType::kCs,
           .code_size_bytes = static_cast<u32>(program->size() * sizeof(u32))});
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), bytes, bytes, 0, &physical) == 0,
        "resource-failure allocation failed");
    void* mapped = reinterpret_cast<void*>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, bytes, 0x3, 0x10, physical, bytes) == 0 && mapped == reinterpret_cast<void*>(base),
        "resource-failure mapping failed");
    std::memset(mapped, 0, bytes);
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      context.MapMemory(base, bytes);
      auto& cache = context.GetBufferCache();
      auto& shaders = processor.GetShCtx();
      const auto bind_caller = [&](const std::vector<u32>& caller) {
        shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(caller.data()),
            .num_thread_x = lanes, .num_thread_y = 1, .num_thread_z = 1,
            .wave_size = 64, .user_sgpr = 12});
      };
      bind_caller(code);
      const auto descriptor = [](uint64_t address, u32 size) {
        ShaderBufferResource result {};
        result.UpdateAddress48(address);
        result.fields[2] = size;
        result.fields[3] = DstSel(4, 5, 6, 7) |
            (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
        return result;
      };
      const auto set_vsharp = [&](u32 sgpr, const ShaderBufferResource& resource) {
        for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(sgpr + i, resource.fields[i]);
      };
      set_vsharp(0, descriptor(output, lanes * sizeof(u32)));
      set_vsharp(8, descriptor(table, 2u * 16u));
      const auto write = [&](uint64_t address, const void* data, size_t size) {
        Require(name, "invalidate", context.InvalidateMemory(address, size),
                "fixture write escaped mapped backing");
        LibKernel::Memory::WriteBacking(address, data, size);
      };
      const auto write_rows = [&](uint64_t data_a, uint64_t data_b) {
        const std::array<uint64_t, 4> rows {first, data_a, second, data_b};
        write(table, rows.data(), sizeof(rows));
      };
      const auto good_leaf = [&](uint64_t address, u32 value) {
        const std::array leaf {EncodeVop1(0x01, 6, 255), value,
            EncodeSop1(0x21, 125, 14)};
        write(address, leaf.data(), sizeof(leaf));
      };
      const auto resource_leaf = [&](uint64_t address, u32 descriptor_offset) {
        // A nested T# is host-materialized through the pointer table; a raw V#
        // load could instead become an IndirectBuffer and remain GPU-side.
        // s_load_dwordx8 s24:31,[s16:17]+offset; null SOFFSET125, NOT live s0.
        const std::array leaf {EncodeSmem0(0x03, 24, 8), EncodeSmem1(descriptor_offset, 125),
            EncodeSopp(0x0c, 0), EncodeMimg0(0x00, 0x1), EncodeMimg1(6, 0, 6),
            EncodeSopp(0x0c, 0), EncodeSop1(0x21, 125, 14)};
        write(address, leaf.data(), sizeof(leaf));
      };
      // The other row's context is mapped and contains null T#s. The failure
      // under test is the first row's unreadable descriptor, not an invalid opcode.
      const std::array<u32, 12> null_descriptors {};
      write(context_a, null_descriptors.data(), sizeof(null_descriptors));
      good_leaf(second, value_b);
      ShaderComputeInputInfo last_successful {};
      const auto run = [&](const char* round, bool expected_module, u32 even, u32 odd) {
        std::array<u32, lanes + tail_words> actual;
        actual.fill(sentinel);
        // Persistent GPU reset, independent of a previously GPU-owned output.
        auto [buffer, offset] = cache.ObtainBuffer(output, sizeof(actual), true, false, {}, false);
        buffer->Fill(offset, sizeof(actual), sentinel);
        ShaderComputeInputInfo input {};
        input.stage = last_successful.stage; // A later rejection must clear an actual earlier stage.
        const auto* module = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
        Require(name, round, (module != nullptr) == expected_module,
                std::string(round) + ": incorrect explicit per-dispatch result");
        if (!expected_module) Require(name, round, !input.stage,
            "failed TABLE materialization published an earlier stage");
        if (expected_module) last_successful.stage = input.stage;
        processor.DispatchDirect(1, 1, 1, 0x41u);
        context.GetCommandScheduler().Finish();
        cache.ReadMemory(output, sizeof(actual));
        Require(name, round, LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)),
                "cannot read resource-failure output");
        for (u32 lane = 0; lane < lanes; ++lane)
          Require(name, round, actual[lane] == ((lane & 1u) ? odd : even),
                  std::string(round) + ": failed dispatch wrote output or recovery used stale code");
        for (u32 i = lanes; i < actual.size(); ++i)
          Require(name, round, actual[i] == sentinel, "output tail was overwritten");
        return module == nullptr ? uint64_t {0} : module->id;
      };
      resource_leaf(first, 0);
      write_rows(context_a, context_a);
      shaders.SetCsUserSgpr(6, 0u);
      const auto resource_id = run("mapped nested resource positive FIRST", true, 0u, 0u);
      Require(name, "mapped positive cache", run("mapped positive cached", true, 0u, 0u) == resource_id,
              "mapped nested resource did not reuse its exact module");
      shaders.SetCsUserSgpr(6, 1u);
      (void)run("GPU selector chooses second valid leaf", true, value_b, value_b);
      shaders.SetCsUserSgpr(6, 0u);
      write_rows(unmapped_context, context_a); // function bytes/target set remain unchanged
      (void)run("cached context-only resource failure", false, sentinel, sentinel);
      (void)run("repeat cached context-only failure", false, sentinel, sentinel);
      write_rows(context_a, context_a);
      Require(name, "mapped data recovery", run("same-code context recovery", true, 0u, 0u) == resource_id,
              "unreadable context poisoned the valid resource module");
      write_rows(unmapped_context, context_a);
      bind_caller(cold_code);
      (void)run("cold live-context resource failure", false, sentinel, sentinel);
      (void)run("repeated cold live-context failure", false, sentinel, sentinel);
      bind_caller(code);
      good_leaf(first, value_a);
      const auto original = run("recover changed callee", true, value_a, value_a);
      Require(name, "cached good", run("cached good callee", true, value_a, value_a) == original,
              "unchanged recovered module was not reused");
      resource_leaf(first, 0);
      (void)run("cached code change to resource failure", false, sentinel, sentinel);
      (void)run("repeat rejected changed callee", false, sentinel, sentinel);
      resource_leaf(first, 16);
      (void)run("another failing callee source revision", false, sentinel, sentinel);
      good_leaf(first, value_a);
      Require(name, "recover original module", run("restore exact original callee", true,
          value_a, value_a) == original, "failed resource revisions poisoned the original module");
      context.UnmapMemory(base, bytes);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, bytes) == 0,
            "resource-failure allocation release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Real table-selected call, fresh per-bind data and a finite writer whose footprint
  // exceeds its advertised NumRecords. All bytes below are fixture-owned.
  void CheckTableCallContextBinding() {
    constexpr const char *name = "TableCallContextBinding";
    constexpr uint64_t base = 0x000000020c000000ull, size = 0x40000;
    constexpr uint64_t table = base + 0x100, leaf = base + 0x1000;
    constexpr uint64_t context_a = base + 0x6000, context_b = base + 0x6100;
    constexpr uint64_t context_c = base + 0x6200, output = base + 0x20000;
    constexpr std::array<u32, 4> payload{0x13579bdfu, 0x2468ace0u, 0x789abcdeu, 0x01234567u};
    const std::vector<u32> caller{
        EncodeVop2(0x1b, 1, InlineU32(1), 0),
        EncodeVop1(0x02, 4, Vgpr(1)),
        EncodeSop2(0x1e, 5, 4, InlineU32(4)),
        EncodeSmem0(0x0a, 24, 4), EncodeSmem1(0, 5),
        EncodeSopp(0x0c, 0),
        EncodeSop1(0x03, 14, 24), EncodeSop1(0x03, 15, 25),
        EncodeSop1(0x03, 16, 26), EncodeSop1(0x03, 17, 27),
        EncodeSop1(0x21, 14, 14),
        EncodeMubuf0(0x1e, 0, false, false), EncodeMubuf1(6, 0, 0),
        EncodeSopp(0x01)};
    // Both branch paths reach the load before the context pair is clobbered.
    // The pair is not globally preserved, so this exercises per-load CFG proof.
    const std::array leaf_code{
        EncodeSopp(0x04, 1), EncodeSopp(0x00),
        EncodeSmem0(0x02, 24, 8), EncodeSmem1(0, 125), EncodeSopp(0x0c, 0),
        EncodeVop1(0x01, 6, 24), EncodeVop1(0x01, 7, 25),
        EncodeVop1(0x01, 8, 26), EncodeVop1(0x01, 9, 27),
        EncodeSop1(0x03, 16, InlineU32(0)),
        EncodeSop1(0x21, 125, 14)};
    ShaderMapUserData(reinterpret_cast<uint64_t>(caller.data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<u32>(caller.size() * sizeof(u32))});
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
        "context fixture allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, size, 0x3, 0x10, physical, size) == 0 &&
        mapped == reinterpret_cast<void *>(base), "context fixture mapping failed");
    std::memset(mapped, 0, size);
    LibKernel::Memory::WriteBacking(leaf, leaf_code.data(), sizeof(leaf_code));
    for (const auto context_address : {context_a, context_b, context_c})
      LibKernel::Memory::WriteBacking(context_address, payload.data(), sizeof(payload));
    const std::array<uint64_t, 4> records{leaf, context_a, leaf, context_b};
    LibKernel::Memory::WriteBacking(table, records.data(), sizeof(records));
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      context.MapMemory(base, size);
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      auto &shaders = processor.GetShCtx();
      auto &cache = context.GetBufferCache();
      auto &pipelines = context.GetPipelineCache();
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(caller.data()),
          .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
          .wave_size = 64, .user_sgpr = 12});
      const auto descriptor = [](uint64_t address, u32 bytes) {
        ShaderBufferResource v{};
        v.UpdateAddress48(address);
        v.fields[2] = bytes;
        v.fields[3] = DstSel(4, 5, 6, 7) |
            (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u) | (2u << 28u);
        return v;
      };
      const auto set_vsharp = [&](u32 first, uint64_t address, u32 bytes) {
        const auto v = descriptor(address, bytes);
        for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(first + i, v.fields[i]);
      };
      const auto bind = [&](ShaderComputeInputInfo &input) {
        for (auto &dimension : input.workgroup_counts) dimension = 1;
        return pipelines.GetComputeProgram(shaders.GetCs(), input);
      };
      const auto captured = [](const CallClosure &closure, uint64_t address, u32 value) {
        return std::ranges::any_of(closure.data, [&](const auto &range) {
          return address >= range.address && (address - range.address) % 4u == 0 &&
              (address - range.address) / 4u < range.words.size() &&
              range.words[(address - range.address) / 4u] == value;
        });
      };
      set_vsharp(0, output, sizeof(payload));
      set_vsharp(8, table, sizeof(records));
      ShaderComputeInputInfo valid{};
      const auto *first = bind(valid);
      Require(name, "real invariant context bind", first && valid.stage.call_closure &&
          captured(*valid.stage.call_closure, context_a, payload[0]) &&
          captured(*valid.stage.call_closure, context_b, payload[0]),
          "real table/context source did not publish both invariant payload spans");
      Require(name, "real fixed writer footprint",
          std::ranges::any_of(first->info.info.buffers, [](const auto &resource) {
            return resource.written && resource.max_byte_extent >= 16;
          }), "fixture did not decode a fixed 16-byte store");
      const auto first_id = first->id;
      processor.DispatchDirect(1, 1, 1, 0x41u);
      cache.ReadMemory(output, sizeof(payload));
      std::array<u32, 4> actual{};
      Require(name, "native context payload",
          LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)) &&
          actual == payload, "native call did not consume the invariant captured payload");
      Require(name, "mutate companion", context.InvalidateMemory(table + 24u, 8),
              "companion table write escaped mapping");
      LibKernel::Memory::WriteBacking(table + 24u, &context_c, sizeof(context_c));
      ShaderComputeInputInfo refreshed{};
      const auto *reused = bind(refreshed);
      Require(name, "data-only module reuse",
          reused && reused->id == first_id && refreshed.stage.call_closure &&
          captured(*refreshed.stage.call_closure, table + 24u, static_cast<u32>(context_c)) &&
          captured(*refreshed.stage.call_closure, context_c, payload[0]),
          "unchanged module identity kept obsolete companion/payload snapshot bytes");
      // GetSize()==1 ends before context_a, but DWORDX4 writes overlap its first 12 bytes.
      set_vsharp(0, context_a - 4, 1);
      ShaderComputeInputInfo rejected{};
      Require(name, "finite raw writer refused at bind", bind(rejected) == nullptr &&
          !rejected.stage, "one advertised byte concealed a 16-byte write into call data");
      set_vsharp(0, output, sizeof(payload));
      ShaderComputeInputInfo recovered{};
      const auto *restored = bind(recovered);
      Require(name, "valid bind recovery", restored && restored->id == first_id,
              "restoring a disjoint writer did not recover the compiled module");
      context.UnmapMemory(base, size);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelMunmap(base, size) == 0 &&
        LibKernel::Memory::KernelReleaseDirectMemory(physical, size) == 0,
        "context fixture release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

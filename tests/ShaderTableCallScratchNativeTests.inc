  // Real table-call closure with bounded per-invocation Scratch.
  void CheckTableCallScratchBinding() {
    constexpr uint64_t base = 0x000000020e000000ull, size = 0x40000;
    constexpr uint64_t table = base + 0x100, leaf = base + 0x1000;
    constexpr uint64_t context_a = base + 0x6000, context_b = base + 0x6100;
    constexpr uint64_t output = base + 0x20000;
    constexpr u32 scratch_words = 64, sentinel = 0xa5a5a5a5u, poison = 0x9abcdef0u;
    constexpr std::array<u32, 4> payload{0x13579bdfu, 0x2468ace0u, 0x789abcdeu, 0x01234567u};
    std::vector<u32> leaf_code;
    leaf_code.push_back(EncodeSop1(0x03, 20, InlineU32(0)));
    leaf_code.push_back(EncodeVop1(0x01, 21, InlineU32(0)));
    const auto scratch_store = [&](u32 data, u32 offset = 0) {
      leaf_code.insert(leaf_code.end(), {
          EncodeFlat0(0x1c, 1, offset), EncodeFlat1(0, 20, data, 21)});
    };
    const auto scratch_load = [&](u32 dst, u32 offset = 0) {
      leaf_code.insert(leaf_code.end(), {
          EncodeFlat0(0x0c, 1, offset), EncodeFlat1(dst, 20, 0, 21)});
    };
    scratch_store(0);
    leaf_code.push_back(EncodeVop2(0x25, 20, InlineU32(17), 0));
    scratch_store(20, scratch_words * 4u - 4u);
    scratch_load(24, scratch_words * 4u - 4u);
    AppendVMovLiteral(&leaf_code, 20, poison);
    AppendSMovLiteral(&leaf_code, 50, 0x55555555u);
    AppendSMovLiteral(&leaf_code, 51, 0x55555555u);
    leaf_code.push_back(EncodeSop1(0x04, 126, 50)); // Even lanes in both halves.
    scratch_store(20);
    leaf_code.push_back(EncodeSop1(0x04, 126, 0xc1u)); // Restore the full workgroup.
    AppendSMovLiteral(&leaf_code, 20, scratch_words * 4u);
    scratch_store(20);
    scratch_load(23);
    AppendSMovLiteral(&leaf_code, 20, 0xfffffffcu);
    scratch_store(20);
    scratch_load(25);
    leaf_code.push_back(EncodeSop1(0x03, 20, InlineU32(0)));
    scratch_load(22);
    leaf_code.insert(leaf_code.end(), {
        EncodeSopp(0x0c, 0),
        EncodeSmem0(0x02, 24, 8), EncodeSmem1(0, 125), EncodeSopp(0x0c, 0),
        EncodeVop1(0x01, 6, 24), EncodeVop1(0x01, 7, 25),
        EncodeVop1(0x01, 8, 26), EncodeVop1(0x01, 9, 27),
        EncodeSop1(0x21, 125, 14)});
    const auto make_caller = [&](int physical_segment) {
      std::vector<u32> code{
          EncodeVop2(0x1b, 1, InlineU32(1), 0),
          EncodeVop1(0x02, 4, Vgpr(1)),
          EncodeSop2(0x1e, 5, 4, InlineU32(4)),
          EncodeSmem0(0x0a, 24, 4), EncodeSmem1(0, 5), EncodeSopp(0x0c, 0),
          EncodeSop1(0x03, 14, 24), EncodeSop1(0x03, 15, 25),
          EncodeSop1(0x03, 16, 26), EncodeSop1(0x03, 17, 27),
          EncodeSop1(0x21, 14, 14)};
      if (physical_segment >= 0) {
        AppendVMovLiteral(&code, 30, static_cast<u32>(context_a));
        AppendVMovLiteral(&code, 31, static_cast<u32>(context_a >> 32u));
        code.insert(code.end(), {EncodeFlat0(0x1c, static_cast<u32>(physical_segment)),
                                EncodeFlat1(0, 0x7d, 20, 30)});
      }
      code.push_back(EncodeVop2(0x1a, 1, InlineU32(6), 0));
      const std::array<u32, 8> values{6, 7, 8, 9, 22, 24, 23, 25};
      for (u32 word = 0; word < values.size(); ++word) {
        code.push_back(EncodeMubuf0(0x1c, word * 4u, false, true));
        code.push_back(EncodeMubuf1(values[word], 0, 1));
      }
      code.push_back(EncodeSopp(0x01));
      return code;
    };
    std::array callers{make_caller(-1), make_caller(0), make_caller(2), make_caller(-1)};
    callers[3].insert(callers[3].end() - 1, EncodeSopp(0x00));
    for (size_t i = 0; i < callers.size(); ++i) {
      const auto &code = callers[i];
      ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
          {.type = Prospero::ShaderBinaryType::kCs,
           .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32)),
           .scratch_size_dwords = i == 3 ? 0u : scratch_words});
    }
    const auto descriptor = [](uint64_t address, u32 bytes) {
      ShaderBufferResource value{};
      value.UpdateAddress48(address);
      value.fields[2] = bytes;
      value.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u) | (2u << 28u);
      return value;
    };
    const auto captured = [](const CallClosure &closure, uint64_t address, u32 value) {
      return std::ranges::any_of(closure.data, [&](const auto &range) {
        return address >= range.address && (address - range.address) % 4u == 0 &&
            (address - range.address) / 4u < range.words.size() &&
            range.words[(address - range.address) / 4u] == value;
      });
    };
    EnsureRuntimeContext();
    for (const u32 wave : {32u, 64u}) {
      const char *name = wave == 32u ? "TableCallPrivateScratch32" : "TableCallPrivateScratch64";
      int64_t physical = -1;
      Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
          0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
          "scratch-call fixture allocation failed");
      void *mapped = reinterpret_cast<void *>(base);
      Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
          &mapped, size, 0x3, 0x10, physical, size) == 0 && mapped == reinterpret_cast<void *>(base),
          "scratch-call fixture mapping failed");
      std::memset(mapped, 0xa5, size);
      LibKernel::Memory::WriteBacking(leaf, leaf_code.data(), leaf_code.size() * sizeof(u32));
      for (const auto address : {context_a, context_b})
        LibKernel::Memory::WriteBacking(address, payload.data(), sizeof(payload));
      const std::array<uint64_t, 4> records{leaf, context_a, leaf, context_b};
      LibKernel::Memory::WriteBacking(table, records.data(), sizeof(records));
      struct ProtectedRange { uint64_t address; std::vector<u32> words; };
      std::vector<ProtectedRange> protected_ranges;
      for (const auto &[address, bytes] : std::array<std::pair<uint64_t, size_t>, 4>{{
          {table - 16u, sizeof(records) + 32u},
          {leaf - 16u, leaf_code.size() * sizeof(u32) + 32u},
          {context_a - 16u, sizeof(payload) + 32u},
          {context_b - 16u, sizeof(payload) + 32u}}}) {
        ProtectedRange range{address, std::vector<u32>(bytes / sizeof(u32))};
        Require(name, "capture protected bytes", LibKernel::Memory::TryReadBacking(
            address, range.words.data(), bytes), "fixture code/data capture failed");
        protected_ranges.push_back(std::move(range));
      }
      {
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
          const auto set_shader = [&](size_t index) {
            shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(callers[index].data()),
                .num_thread_x = wave, .num_thread_y = 1, .num_thread_z = 1,
                .wave_size = static_cast<uint8_t>(wave), .user_sgpr = 12});
          };
          const auto set_descriptor = [&](u32 first, uint64_t address, u32 bytes) {
            const auto value = descriptor(address, bytes);
            for (u32 word = 0; word < 4u; ++word)
              shaders.SetCsUserSgpr(first + word, value.fields[word]);
          };
          const auto bind = [&](ShaderComputeInputInfo &input) {
            for (auto &dimension : input.workgroup_counts) dimension = 1;
            return pipelines.GetComputeProgram(shaders.GetCs(), input);
          };
          set_shader(0);
          set_descriptor(0, output, wave * 64u);
          set_descriptor(8, table, sizeof(records));
          ShaderComputeInputInfo input{};
          const auto *module = bind(input);
          Require(name, "private scratch bind", module && input.stage.call_closure &&
              module->info.wave_size == wave && module->info.info.function_scratch &&
              input.scratch_size_dwords == scratch_words &&
              captured(*input.stage.call_closure, context_a, payload[0]) &&
              captured(*input.stage.call_closure, context_b, payload[0]),
              "private Scratch was rejected as an unbounded guest-memory writer");
          const auto module_id = module->id;
          processor.DispatchDirect(1, 1, 1, wave == 32u ? 0x8041u : 0x41u);
          std::vector<u32> actual(wave * 16u + 4u), expected(actual.size(), sentinel);
          cache.ReadMemory(output, actual.size() * sizeof(u32));
          for (u32 lane = 0; lane < wave; ++lane) {
            const auto at = lane * 16u;
            std::copy(payload.begin(), payload.end(), expected.begin() + at);
            expected[at + 4u] = (lane & 1u) == 0u ? poison : lane;
            expected[at + 5u] = lane + 17u;
            expected[at + 6u] = expected[at + 7u] = 0u;
          }
          Require(name, "native private storage and bounds", LibKernel::Memory::TryReadBacking(
              output, actual.data(), actual.size() * sizeof(u32)) && actual == expected,
              "Scratch crossed invocations, ignored EXEC, escaped its extent or changed output canaries");
          for (const size_t index : {1u, 2u, 3u}) {
            set_shader(index);
            ShaderComputeInputInfo rejected{};
            const char *stage = index == 1u ? "physical Flat writer refused" :
                index == 2u ? "physical Global writer refused" : "missing Scratch allocation refused";
            Require(name, stage, bind(rejected) == nullptr && !rejected.stage,
                "a nonlocal writer or unallocated Scratch bypassed closure validation");
          }
          set_shader(0);
          ShaderComputeInputInfo recovered{};
          const auto *restored = bind(recovered);
          Require(name, "valid bind recovery", restored && restored->id == module_id,
              "a refused writer poisoned the valid private-Scratch module");
          for (const auto &range : protected_ranges) {
            std::vector<u32> current(range.words.size());
            Require(name, "code data and surrounding canaries", LibKernel::Memory::TryReadBacking(
                range.address, current.data(), current.size() * sizeof(u32)) && current == range.words,
                "Scratch or a refused writer modified captured code/data bytes");
          }
          context.UnmapMemory(base, size);
          context.GetCommandScheduler().Finish();
        });
        LibKernel::Memory::InstallGpuResources(nullptr);
        context.ShutdownGpu();
      }
      Require(name, "release", LibKernel::Memory::KernelMunmap(base, size) == 0 &&
          LibKernel::Memory::KernelReleaseDirectMemory(physical, size) == 0,
          "scratch-call fixture release failed");
      std::printf("[gpu]     %-32s ok\n", name);
    }
  }

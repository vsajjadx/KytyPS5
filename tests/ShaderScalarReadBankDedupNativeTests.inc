  // Native SGPR-bank provenance is not an extra scalar-table identity after the
  // four V# words resolve to the same expressions. Exercise the production
  // planner, materializer, descriptor bindings and GPU dispatch, not a mock reader.
  void CheckScalarReadBankDedup() {
    using namespace ShaderRecompiler::IR;
    enum class Variant { Copied, OtherTable, OtherOffset, OtherBounds };
    constexpr uint64_t base = 0x000000020d000000ull, size = 0x40000;
    constexpr uint64_t table_a = base + 0x100, table_b = base + 0x200;
    constexpr uint64_t input_a = base + 0x1000, input_b = base + 0x1100;
    constexpr uint64_t output = base + 0x20000;
    constexpr u32 value_a = 0x13579bdfu, value_b = 0x2468ace0u, sentinel = 0xdeadbeefu;
    const auto descriptor = [](uint64_t address, u32 bytes) {
      ShaderBufferResource value{};
      value.UpdateAddress48(address);
      value.fields[2] = bytes;
      value.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u) | (2u << 28u);
      return value;
    };
    EnsureRuntimeContext();
    for (const u32 wave : {32u, 64u}) {
      for (const auto variant : {Variant::Copied, Variant::OtherTable,
                                 Variant::OtherOffset, Variant::OtherBounds}) {
        const char *suffix = variant == Variant::Copied ? "Copied" :
            variant == Variant::OtherTable ? "OtherTable" :
            variant == Variant::OtherOffset ? "OtherOffset" : "OtherBounds";
        const std::string name = std::string("ScalarReadBank") + suffix + std::to_string(wave);
        std::vector<u32> code;
        for (u32 word = 0; word < 4u; ++word)
          code.push_back(EncodeSop1(0x03, 4u + word, word)); // s0:3 -> s4:7
        if (variant == Variant::OtherTable)
          AppendSMovLiteral(&code, 4, static_cast<u32>(table_b));
        if (variant == Variant::OtherBounds)
          code.push_back(EncodeSop1(0x03, 6, InlineU32(16)));
        code.insert(code.end(), {
            EncodeSmem0(0x0a, 24, 0), EncodeSmem1(0, 125),
            EncodeSmem0(0x0a, 28, 2),
            EncodeSmem1(variant == Variant::OtherOffset ? 16u : 0u, 125),
            EncodeSopp(0x0c, 0),
            EncodeMubuf0(0x0c, 0, false, false), EncodeMubuf1(1, 6, 0),
            EncodeMubuf0(0x0c, 0, false, false), EncodeMubuf1(2, 7, 0),
            EncodeSopp(0x0c, 0),
            EncodeMubuf0(0x1c, 0, false, false), EncodeMubuf1(1, 2, 0),
            EncodeMubuf0(0x1c, 4, false, false), EncodeMubuf1(2, 2, 0),
            EncodeSopp(0x01)});
        ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
            {.type = Prospero::ShaderBinaryType::kCs,
             .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
        int64_t physical = -1;
        Require(name.c_str(), "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
            0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
            "scalar-bank fixture allocation failed");
        void *mapped = reinterpret_cast<void *>(base);
        Require(name.c_str(), "map", LibKernel::Memory::KernelMapDirectMemory(
            &mapped, size, 0x3, 0x10, physical, size) == 0 &&
            mapped == reinterpret_cast<void *>(base), "scalar-bank fixture mapping failed");
        std::memset(mapped, 0, size);
        const auto a = descriptor(input_a, 4), b = descriptor(input_b, 4);
        for (const auto table : {table_a, table_b}) {
          LibKernel::Memory::WriteBacking(table, a.fields, sizeof(a.fields));
          LibKernel::Memory::WriteBacking(table + 16u, b.fields, sizeof(b.fields));
        }
        LibKernel::Memory::WriteBacking(input_a, &value_a, sizeof(value_a));
        LibKernel::Memory::WriteBacking(input_b, &value_b, sizeof(value_b));
        const std::array<u32, 4> untouched{sentinel, sentinel, sentinel, sentinel};
        LibKernel::Memory::WriteBacking(output, untouched.data(), sizeof(untouched));
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
            shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                .wave_size = static_cast<uint8_t>(wave), .user_sgpr = 12});
            for (const auto &[first, value] :
                std::array<std::pair<u32, ShaderBufferResource>, 2>{{
                    {0, descriptor(table_a, 32)}, {8, descriptor(output, 8)}}})
              for (u32 word = 0; word < 4u; ++word)
                shaders.SetCsUserSgpr(first + word, value.fields[word]);
            const auto bind = [&](ShaderComputeInputInfo &input) {
              for (auto &dimension : input.workgroup_counts) dimension = 1;
              return pipelines.GetComputeProgram(shaders.GetCs(), input);
            };
            ShaderComputeInputInfo initial{};
            const auto *module = bind(initial);
            Require(name.c_str(), "native bind", module && initial.stage.resources,
                    "scalar-bank shader could not bind");
            const auto module_id = module->id;
            const auto *plan = PipelineCacheTestAccess::ActivityPlan(pipelines, initial.stage.resources);
            Require(name.c_str(), "owned plan", plan != nullptr, "bound resource plan is missing");
            const auto slots = plan->srt_reads.size();
            const auto roots = module->info.info.buffers.size();
            const auto native_buffers = module->info.bindings.descriptor_counts[
                static_cast<size_t>(DescriptorBindingKind::Buffers)];
            const auto run = [&](u32 left, u32 right) {
              processor.DispatchDirect(1, 1, 1, wave == 32u ? 0x8041u : 0x41u);
              cache.ReadMemory(output, sizeof(untouched));
              std::array<u32, 4> actual{};
              const std::array<u32, 4> expected{left, right, sentinel, sentinel};
              Require(name.c_str(), "native values and bounds",
                  LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)) &&
                      actual == expected, "scalar-table bank identity changed GPU results or canaries");
            };
            run(value_a, variant == Variant::OtherOffset ? value_b : value_a);
            std::printf("[gpu]     %s slots=%zu roots=%zu native=%u\n",
                        name.c_str(), slots, roots, native_buffers);
            const bool shared = variant == Variant::Copied;
            Require(name.c_str(), "resolved scalar-read identity",
                slots == (shared ? 4u : 8u) && roots == (shared ? 2u : 3u) &&
                    native_buffers == roots,
                shared ? "copied native V# bank created duplicate scalar slots or buffer roots" :
                         "different table expression, offset or bounds were incorrectly merged");
            // Equal current payloads in separate expressions must remain separate on a later bind.
            const auto changed_table = variant == Variant::OtherTable ? table_b : table_a;
            Require(name.c_str(), "invalidate table", context.InvalidateMemory(changed_table, sizeof(b.fields)),
                    "table mutation escaped the mapped fixture");
            LibKernel::Memory::WriteBacking(changed_table, b.fields, sizeof(b.fields));
            Require(name.c_str(), "invalidate output", context.InvalidateMemory(output, sizeof(untouched)),
                    "output reset escaped the mapped fixture");
            LibKernel::Memory::WriteBacking(output, untouched.data(), sizeof(untouched));
            ShaderComputeInputInfo refreshed{};
            const auto *reused = bind(refreshed);
            Require(name.c_str(), "current materialization with module reuse",
                reused && reused->id == module_id && refreshed.stage.resources,
                "table content mutation lost the stable specialization");
            const u32 left = variant == Variant::OtherTable ? value_a : value_b;
            run(left, value_b);
            context.UnmapMemory(base, size);
            context.GetCommandScheduler().Finish();
          });
          LibKernel::Memory::InstallGpuResources(nullptr);
          context.ShutdownGpu();
        }
        Require(name.c_str(), "release", LibKernel::Memory::KernelMunmap(base, size) == 0 &&
            LibKernel::Memory::KernelReleaseDirectMemory(physical, size) == 0,
            "scalar-bank fixture release failed");
        std::printf("[gpu]     %-32s ok\n", name.c_str());
      }
    }
  }

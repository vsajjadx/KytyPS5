  // An owned call-data snapshot must be what every GPU alias reads, even when
  // CPU backing returns to A after B was uploaded into the canonical owner.
  void CheckCallDataSnapshot() {
    constexpr const char *name = "CallDataSnapshot";
    constexpr uint64_t base = 0x000000020bc00000ull, size = 0xc0000;
    constexpr uint64_t table = base + 0x100, payload = base + 0x40100;
    constexpr uint64_t output = base + 0x80000;
    constexpr u32 table_a = 0x11223344u, payload_a = 0x55667788u;
    constexpr u32 table_b = 0xa1b2c3d4u, payload_b = 0xe5f60718u;
    constexpr std::array<u32, 4> untouched{0xcafef00du, 0xcafef00du, 0xcafef00du, 0xcafef00du};
    std::vector<u32> code{
        EncodeVop1(0x01, 0, InlineU32(0)),
        EncodeMubuf0(0x0c, 0, false, false), EncodeMubuf1(1, 0, 0),
        EncodeMubuf0(0x0c, 0, false, false), EncodeMubuf1(3, 2, 0)};
    AppendVMovLiteral(&code, 20, static_cast<u32>(payload));
    AppendVMovLiteral(&code, 21, static_cast<u32>(payload >> 32u));
    code.insert(code.end(), {EncodeFlat0(0x0c, 2), EncodeFlat1(2, 125, 0, 20)});
    AppendVMovLiteral(&code, 20, static_cast<u32>(table));
    code.insert(code.end(), {EncodeFlat0(0x0c, 2), EncodeFlat1(4, 125, 0, 20),
        EncodeSopp(0x0c, 0),
        EncodeMubuf0(0x1c, 0, false, false), EncodeMubuf1(1, 1, 0),
        EncodeMubuf0(0x1c, 4, false, false), EncodeMubuf1(2, 1, 0),
        EncodeMubuf0(0x1c, 8, false, false), EncodeMubuf1(3, 1, 0),
        EncodeMubuf0(0x1c, 12, false, false), EncodeMubuf1(4, 1, 0)});
    AppendEnd(&code);
    ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), size, 0x40000, 0, &physical) == 0,
        "fixture allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, size, 0x3, 0x10, physical, 0x40000) == 0 &&
        mapped == reinterpret_cast<void *>(base), "fixture mapping failed");
    std::memset(mapped, 0, size);
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      context.MapMemory(base, size);
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      auto &scheduler = context.GetCommandScheduler();
      auto &cache = context.GetBufferCache();
      auto &executor = context.GetRenderExecutor();
      auto &shaders = processor.GetShCtx();
      const auto write = [&](uint64_t address, const void *data, uint64_t bytes) {
        Require(name, "invalidate", context.InvalidateMemory(address, bytes),
                "fixture write escaped mapped memory");
        LibKernel::Memory::WriteBacking(address, data, bytes);
      };
      const auto descriptor = [](uint64_t address, u32 bytes) {
        ShaderBufferResource v{};
        v.UpdateAddress48(address);
        v.fields[2] = bytes;
        v.fields[3] = DstSel(4, 5, 6, 7) |
            (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u) | (2u << 28u); // Raw zero-stride access.
        return v;
      };
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
          .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
          .wave_size = 64, .user_sgpr = 12});
      for (const auto &[reg, address, bytes] :
           std::array<std::tuple<u32, uint64_t, u32>, 3>{{
             {0, table, 4}, {4, output, 16}, {8, payload, 4}}}) {
        const auto v = descriptor(address, bytes);
        for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(reg + i, v.fields[i]);
      }
      write(table, &table_a, 4);
      write(payload, &payload_a, 4);
      write(output, untouched.data(), sizeof(untouched));
      ShaderComputeInputInfo input{};
      for (auto &dimension : input.workgroup_counts) dimension = 1;
      const auto *module = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
      Require(name, "native reader", module != nullptr && module->info.info.uses_dma,
              "fixture did not compile real SSBO and BDA reads");
      auto *pipeline = context.GetPipelineCache().GetComputePipeline(input);
      Require(name, "pipeline", pipeline != nullptr, "fixture pipeline was not created");
      auto closure = std::make_shared<CallClosure>();
      closure->shader_hash = module->info.shader_hash;
      closure->mapping_generation = LibKernel::Memory::MappingGeneration();
      closure->protection_generation = LibKernel::Memory::ProtectionGeneration();
      closure->data = {{table, {table_a}}, {payload, {payload_a}}};
      input.stage.call_closure = closure;
      PreparedBindings prepared;
      executor.PrepareBindings(input.stage, prepared);
      auto &ray_dword = const_cast<ShaderModule *>(module)->info.bindings.ray_query_dword;
      const auto saved_ray_dword = ray_dword;
      Require(name, "fixture has ordinary raw reads",
          saved_ray_dword == ShaderRecompiler::IR::PushData::NoStart,
          "fixture unexpectedly already used native ray queries");
      ray_dword = 0; // Model the accepted native-AS binding marker; no AS build is reached.
      Require(name, "native AS representation refused",
          !RenderExecutorTestAccess::PrepareCallData(executor, prepared),
          "captured data was admitted without proving native-AS input disjointness");
      ray_dword = saved_ray_dword;
      Require(name, "prepare captured data",
          RenderExecutorTestAccess::PrepareCallData(executor, prepared),
          "captured data could not be prepared");
      auto *stage = &prepared;
      const auto stages = std::span{&stage, 1u};
      executor.FindBuffers(stages);
      cache.PrepareBdaRange(table, 4);
      cache.PrepareBdaRange(payload, 4);
      executor.RebindBuffers(prepared);
      const auto validate = [&](bool after_commit, auto &proof) {
        return RenderExecutorTestAccess::ValidateCallClosure(
            executor, input.stage, prepared, proof, after_commit);
      };
      auto initial_check = RenderExecutorTestAccess::NewCallClosureCheck();
      Require(name, "initial complete proof", validate(false, initial_check),
              "ordinary disjoint bindings did not prove the captured data");
      const auto table_binding = std::ranges::find_if(prepared.buffer_sources,
          [&](const auto &source) { return source.address == table; });
      Require(name, "actual table SSBO", table_binding != prepared.buffer_sources.end(),
              "fixture lost its real table SSBO");
      const auto table_index = static_cast<size_t>(table_binding - prepared.buffer_sources.begin());
      const auto native_table = prepared.buffers[table_index];
      prepared.buffers[table_index].offset += 4;
      Require(name, "native read alias mismatch rejected", !validate(false, initial_check),
              "a shifted SSBO view escaped canonical data validation");
      prepared.buffers[table_index] = native_table;
      Require(name, "native view recovery", validate(false, initial_check),
              "restoring the canonical SSBO did not restore the proof");

      // A second virtual mapping of the table's physical page is an actual alias, even
      // with a freshly sampled mapping generation.
      constexpr uint64_t alias_address = base + 0x200000;
      void *alias = reinterpret_cast<void *>(alias_address);
      Require(name, "physical alias map", LibKernel::Memory::KernelMapDirectMemory(
          &alias, 0x4000, 0x3, 0x10, physical, 0x4000) == 0,
          "could not construct the physical alias negative case");
      closure->mapping_generation = LibKernel::Memory::MappingGeneration();
      closure->protection_generation = LibKernel::Memory::ProtectionGeneration();
      Require(name, "physical alias rejected", !validate(false, initial_check),
              "same physical bytes at another guest address were accepted");
      Require(name, "physical alias unmap",
          LibKernel::Memory::KernelMunmap(alias_address, 0x4000) == 0,
          "physical alias cleanup failed");
      closure->mapping_generation = LibKernel::Memory::MappingGeneration();
      closure->protection_generation = LibKernel::Memory::ProtectionGeneration();
      Require(name, "alias recovery", validate(false, initial_check),
              "unique table backing did not recover");

      // Commit real descriptors, then replace their native owners through a legitimate
      // cache merge. The stale handles must not be blessed by a fresh CPU byte comparison.
      executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute, *pipeline, stages);
      (void)cache.FindBuffer(base, size);
      vk::DescriptorBufferInfo merged_table;
      Require(name, "actual owner replacement", cache.CanonicalBinding(table, 4, merged_table) &&
          merged_table.buffer != native_table.buffer, "fixture did not replace the native owner");
      Require(name, "owner replacement rejected", !validate(true, initial_check),
              "post-commit owner replacement was accepted");
      executor.RebindBuffers(prepared);
      Require(name, "merged disjoint owner recovery", validate(false, initial_check),
              "one canonical owner containing disjoint inputs and output was rejected");

      // A real preceding GPU transfer owns newer table bytes. No CPU snapshot may replace it.
      auto [written_table, written_offset] = cache.ObtainBuffer(table, 4, true, false, {}, false);
      written_table->Fill(written_offset, 4, table_b);
      Require(name, "preceding GPU writer rejected", !validate(false, initial_check),
              "GPU-newer data was accepted as a current CPU snapshot");
      cache.ReadMemory(table, 4);
      write(table, &table_a, 4);
      write(table, &table_b, 4);
      write(payload, &payload_b, 4);
      context.PrepareBda(); // Canonical native owners now contain B.
      // A separate small read-only SSBO would also retain B in a private stream copy.
      write(table, &table_b, 4);
      write(payload, &payload_b, 4);
      executor.RebindBuffers(prepared);
      write(table, &table_a, 4);
      write(payload, &payload_a, 4);
      auto check = RenderExecutorTestAccess::NewCallClosureCheck();
      Require(name, "pre-commit CPU proof",
          RenderExecutorTestAccess::ValidateCallClosure(executor, input.stage, prepared, check, false),
          "CPU A did not pass the original immutable closure checks");
      const auto preparation_tick = scheduler.CurrentTick();
      scheduler.Wait(preparation_tick); // Retire the allocation prefix, before the consumer exists.
      Require(name, "preparation tick retired", scheduler.CurrentTick() > preparation_tick,
              "fixture did not cross a real command-buffer retirement before commit");
      executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute, *pipeline, stages);
      Require(name, "post-commit CPU proof",
          RenderExecutorTestAccess::ValidateCallClosure(executor, input.stage, prepared, check, true),
          "CPU A did not pass post-commit closure checks");
      RenderExecutorTestAccess::WriteCallData(executor, prepared, check, scheduler.Current());
      const auto command = scheduler.Current().Handle();
      command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
      command.dispatch(1, 1, 1);
      scheduler.NoteRecordedWork();
      // Drop every local preparation lease before submission. A new full chunk would
      // reuse and overwrite this source unless CommitBindings retained the consumer tick.
      const auto consumer_tick = scheduler.CurrentTick();
      const auto staged_table = prepared.call_data_sources[0];
      prepared.uploads = {};
      std::vector<uint8_t> pressure_bytes(1024u * 1024u, 0x3c);
      BufferCache::PreparedUpload pressure;
      const auto [pressure_buffer, pressure_offset] = cache.UploadPrepared(
          pressure, pressure_bytes.data(), pressure_bytes.size(), 256);
      (void)pressure_offset;
      Require(name, "consumer lease survives early reset",
          scheduler.CurrentTick() == consumer_tick && pressure_buffer->Handle() != staged_table.buffer,
          "the pending consumer's immutable source was reused after preparation reset");
      cache.ReadMemory(output, sizeof(untouched));
      std::array<u32, 4> actual{};
      Require(name, "read output", LibKernel::Memory::TryReadBacking(
          output, actual.data(), sizeof(actual)), "GPU output readback failed");
      std::fprintf(stderr, "CALL_DATA_ABA ssbo_table=%08x bda_payload=%08x ssbo_payload=%08x bda_table=%08x\n",
          actual[0], actual[1], actual[2], actual[3]);
      Require(name, "captured bytes reach every GPU alias",
          actual == std::array<u32, 4>{table_a, payload_a, payload_a, table_a},
          "CPU A/B/A passed byte checks but GPU consumed B from an SSBO/BDA alias");
      RenderExecutorTestAccess::ResetBindings(executor);
      scheduler.Finish();
      cache.RunGarbageCollector();
      Require(name, "consumer leases retired", BufferCacheTestAccess::PreparedRetirements(cache) == 0,
              "completed call-data snapshots remained pinned");

      // A finite raw descriptor advertises one byte, but the four real fixed stores
      // require sixteen prepared bytes. Capture output+4 to exercise that actual extent.
      const auto short_output = descriptor(output, 1);
      for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(4 + i, short_output.fields[i]);
      ShaderComputeInputInfo narrow_input{};
      for (auto &dimension : narrow_input.workgroup_counts) dimension = 1;
      Require(name, "finite raw writer module",
          context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), narrow_input) != nullptr,
          "raw writer metadata fixture could not compile");
      auto write_closure = std::make_shared<CallClosure>();
      write_closure->shader_hash = narrow_input.stage.module->info.shader_hash;
      write_closure->mapping_generation = LibKernel::Memory::MappingGeneration();
      write_closure->protection_generation = LibKernel::Memory::ProtectionGeneration();
      write_closure->data = {{output + 4, {payload_a}}};
      narrow_input.stage.call_closure = write_closure;
      PreparedBindings narrow;
      executor.PrepareBindings(narrow_input.stage, narrow);
      Require(name, "prepare write extent negative",
          RenderExecutorTestAccess::PrepareCallData(executor, narrow),
          "write extent negative could not stage the captured word");
      auto *narrow_stage = &narrow;
      executor.FindBuffers(std::span{&narrow_stage, 1u});
      executor.RebindBuffers(narrow);
      const auto actual_write = std::ranges::find_if(narrow.buffer_sources,
          [&](const auto &source) { return source.address == output; });
      Require(name, "actual finite prepared write extent",
          short_output.GetSize() == 1 && actual_write != narrow.buffer_sources.end() &&
          actual_write->size >= 16, "fixture did not expand the finite raw writer to its real payload");
      auto write_check = RenderExecutorTestAccess::NewCallClosureCheck();
      Require(name, "actual prepared write overlap rejected",
          !RenderExecutorTestAccess::ValidateCallClosure(executor, narrow_input.stage, narrow,
              write_check, false),
          "a write beyond advertised NumRecords overlapped immutable data without refusal");
      narrow.uploads = {};
      RenderExecutorTestAccess::ResetBindings(executor);
      context.UnmapMemory(base, size);
      scheduler.Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    Require(name, "release", LibKernel::Memory::KernelMunmap(base, size) == 0 &&
        LibKernel::Memory::KernelReleaseDirectMemory(physical, size) == 0,
        "fixture allocation release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

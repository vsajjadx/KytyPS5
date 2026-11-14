  void CheckActivityGuardedNative(bool proof) {
    const char *name = proof ? "ActivityGuardedNative" : "ActivityGuardedNativeOriginal";
    Require(name, "activity proof environment",
            ShaderRecompiler::IR::HostActivityProofEnabled() == proof,
            proof ? "run with KYTY_HOST_ACTIVITY_PROOF=1" : "run without KYTY_HOST_ACTIVITY_PROOF");
    constexpr uintptr_t base = 0x0000000209600000ull;
    constexpr uint64_t allocation_size = 0x40000;
    constexpr uint64_t table_size = BufferCache::CACHING_PAGESIZE;
    constexpr uint64_t table = base;
    constexpr uint64_t count_address = base + 1u * BufferCache::CACHING_PAGESIZE;
    constexpr uint64_t output1 = base + 2u * BufferCache::CACHING_PAGESIZE;
    constexpr uint64_t output2 = base + 3u * BufferCache::CACHING_PAGESIZE;
    constexpr uint64_t flag_address = base + 4u * BufferCache::CACHING_PAGESIZE;
    constexpr uint64_t unmapped_target = base + 0x80000;  // outside the mapped allocation
    constexpr uint64_t alias_va = base + 0x100000;        // second VA for the table page
    constexpr uint32_t lanes = 64;
    constexpr uint32_t output_bytes = lanes * sizeof(u32);
    constexpr uint32_t sentinel = 0xa5a5a5a5u;
    constexpr u32 user_sgprs = 2;  // s[0:1] table pointer; s2 WorkGroupID.x (no SGPR flag)

    std::vector<u32> consumer;
    AppendVop3(&consumer, 0x346u, 0, user_sgprs, InlineU32(6), Vgpr(0));
    consumer.insert(consumer.end(), {
        EncodeSmem0(0x02u, 4, 0), EncodeSmem1(0xd0u, 125),   // s_load_dwordx4 s[4:7], s[0:1], 0xd0
        EncodeSopp(0x0c, 0),
        EncodeSmem0(0x08u, 12, 2), EncodeSmem1(0, 125),      // s_buffer_load_dword s12, s[4:7], 0
        EncodeSopp(0x0c, 0),
        EncodeVopc(0xc3u, 12, 0),                       // v_cmp_le_u32 vcc, s12, v0
        EncodeSop1(0x08u, 16, 106),                     // s_not_b64 s[16:17], vcc
        EncodeSop1(0x04u, 126, 16)});                   // s_mov_b64 exec, s[16:17]
    const auto execz = consumer.size();
    consumer.push_back(0);
    consumer.insert(consumer.end(), {
        EncodeSmem0(0x02u, 8, 0), EncodeSmem1(0x00u, 125),   // output V#, only under EXEC
        EncodeSopp(0x0c, 0),
        EncodeVop2(0x25u, 1, InlineU32(1), 0),          // v1 = v0 + 1
        EncodeVop2(0x1au, 2, InlineU32(2), 0),          // v2 = v0 << 2
        EncodeMubuf0(0x1cu), EncodeMubuf1(1, 2, 2),     // buffer_store_dword v1, v2, s[8:11]
        EncodeSmem0(0x02u, 20, 0), EncodeSmem1(0xe0u, 125), // auxiliary V# inside main count path
        EncodeSopp(0x0c, 0),
        EncodeSmem0(0x08u, 13, 10), EncodeSmem1(0, 125), // s_buffer_load s13, s[20:23]
        EncodeSopp(0x0c, 0),
        EncodeSopc(0x06u, 13, InlineU32(0))});          // auxiliary-memory flag, not SGPR flag
    const auto scc1 = consumer.size();
    consumer.push_back(0);
    consumer.insert(consumer.end(), {
        EncodeSmem0(0x02u, 8, 0), EncodeSmem1(0x10u, 125),   // second output V#, flag-guarded
        EncodeSopp(0x0c, 0),
        EncodeMubuf0(0x1cu), EncodeMubuf1(1, 2, 2)});
    consumer[execz] = EncodeSopp(0x08, static_cast<u32>(consumer.size() - execz - 1u));
    consumer[scc1] = EncodeSopp(0x05, static_cast<u32>(consumer.size() - scc1 - 1u));
    AppendEnd(&consumer);
    // Producer: the NativeIndirectDispatch writer (stores SGPR12..14 through three V#s).
    std::vector<u32> writer;
    for (u32 i = 0; i < 3; i++) {
      writer.push_back(EncodeVop1(0x01u, 1, 12u + i));
      writer.push_back(EncodeMubuf0(0x1cu, 0, false, false));
      writer.push_back(EncodeMubuf1(1, i, 0));
    }
    AppendEnd(&writer);
    for (const auto *code : {&consumer, &writer}) {
      ShaderMapUserData(reinterpret_cast<uint64_t>(code->data()),
          {.type = Prospero::ShaderBinaryType::kCs,
           .code_size_bytes = static_cast<uint32_t>(code->size() * sizeof(u32))});
    }

    EnsureRuntimeContext();
    int64_t direct_offset = -1;
    Require(name, "direct allocation",
            Libs::LibKernel::Memory::KernelAllocateDirectMemory(
                0, Libs::LibKernel::Memory::KernelGetDirectMemorySize(),
                allocation_size, 0x10000, 0, &direct_offset) == 0,
            "activity fixture allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "direct mapping",
            Libs::LibKernel::Memory::KernelMapDirectMemory(
                &mapped, allocation_size, 0x3, 0x10, direct_offset, 0x10000) == 0 &&
                mapped == reinterpret_cast<void *>(base),
            "activity fixture mapping failed");
    std::memset(mapped, 0xa5, allocation_size);
    std::memset(reinterpret_cast<void *>(count_address), 0, 16);
    std::memset(reinterpret_cast<void *>(flag_address), 0, 16);
    const auto descriptor = [](uint64_t address, u32 bytes) {
      ShaderBufferResource v{};
      v.UpdateAddress48(address);
      v.fields[2] = bytes;
      v.fields[3] = DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      return v;
    };
    const auto write_table_vsharp = [&](uint64_t offset, uint64_t address, u32 bytes) {
      const auto v = descriptor(address, bytes);
      Require(name, "table write",
              LibKernel::Memory::TryWriteBacking(table + offset, v.fields, sizeof(v.fields)),
              "the CPU descriptor table could not be written");
    };

    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      // The table has its own mapping so it can be unmapped/remapped alone.
      context.MapMemory(table, table_size);
      context.MapMemory(base + table_size, allocation_size - table_size);
      write_table_vsharp(0x00, unmapped_target, output_bytes);  // cold round: unmapped output
      write_table_vsharp(0x10, unmapped_target, output_bytes); // stays untouched while aux flag0
      write_table_vsharp(0xd0, count_address, 16);
      write_table_vsharp(0xe0, flag_address, 16);
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      auto &scheduler = context.GetCommandScheduler();
      auto &cache = context.GetBufferCache();
      auto &shaders = processor.GetShCtx();
      const auto set_sgpr_buffer = [&](u32 sgpr, uint64_t address, u32 bytes) {
        const auto v = descriptor(address, bytes);
        for (u32 i = 0; i < 4; i++) shaders.SetCsUserSgpr(sgpr + i, v.fields[i]);
      };
      // Producer writes {values[0..2]} to the three DWORDs at address (always on the GPU).
      const auto produce = [&](uint64_t address, std::array<u32, 3> values) {
        shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(writer.data()),
                             .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                             .wave_size = 64, .user_sgpr = 15});
        for (u32 i = 0; i < 3; i++) {
          set_sgpr_buffer(i * 4u, address + i * 4u, 4);
          shaders.SetCsUserSgpr(12u + i, values[i]);
        }
        processor.DispatchDirect(1, 1, 1, 0x41u);
      };
      const auto bind_consumer = [&] {
        shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(consumer.data()),
                             .num_thread_x = lanes, .num_thread_y = 1, .num_thread_z = 1,
                             .wave_size = 64, .user_sgpr = user_sgprs, .tgid_x_en = true});
        shaders.SetCsUserSgpr(0, static_cast<u32>(table));
        shaders.SetCsUserSgpr(1, static_cast<u32>(table >> 32u));
      };
      const ShaderRecompiler::IR::ResourceSnapshot *snapshot = nullptr;
      const auto reuse = [&] {
        return snapshot ? ShaderRecompiler::IR::ActivitySnapshotReuseCount(*snapshot) : 0u;
      };
      uint32_t hit_rounds = 0, counter_reads = 0, round_index = 0;
      const auto round = [&](u32 count, bool auxiliary_nonzero, bool expect_hit) {
        const auto label = "round " + std::to_string(round_index++) + " count " +
                           std::to_string(count) + " aux " + std::to_string(auxiliary_nonzero);
        for (const auto address : {output1, output2}) {
          auto [buffer, offset] = cache.ObtainBuffer(address, output_bytes, true, false, {}, false);
          buffer->Fill(offset, output_bytes, sentinel);
        }
        produce(count_address, {count, 0u, 0u});
        u32 stale = 0;
        Require(name, "fresh GPU counter",
                cache.HasGpuDirtyBytes(count_address, 4) &&
                    LibKernel::Memory::TryReadBacking(count_address, &stale, sizeof(stale)),
                label + ": main counter is not a fresh GPU-owned value");
        const bool flag_was_dirty = cache.HasGpuDirtyBytes(flag_address, 4);
        bind_consumer();
        const auto reuse_before = reuse();
        const auto tick = scheduler.CurrentTick();
        processor.DispatchDirect(1, 1, 1, 0x41u);
        const bool host_read = !cache.HasGpuDirtyBytes(count_address, 4);
        const auto hits = reuse() - reuse_before;
        counter_reads += host_read ? 1u : 0u;
        hit_rounds += hits != 0u ? 1u : 0u;
        if (expect_hit && proof) {
          u32 backing = 0;
          Require(name, "guarded proof reuse",
                  hits == 1u && !host_read && !flag_was_dirty && scheduler.CurrentTick() == tick &&
                      LibKernel::Memory::TryReadBacking(count_address, &backing, sizeof(backing)) && backing == stale,
                  label + ": expected one guarded reuse without host counter read/submission");
        } else {
          Require(name, "original fresh host path", hits == 0u && host_read && scheduler.CurrentTick() > tick,
                  label + ": expected a fresh original counter read and zero proof hits");
        }
        if (count == 0u && flag_was_dirty)
          Require(name, "cold0 preserves unread auxiliary", cache.HasGpuDirtyBytes(flag_address, 4),
                  label + ": empty traversal speculated into the auxiliary buffer");
        for (const auto [address, enabled] : {std::pair{output1, true}, std::pair{output2, auxiliary_nonzero}}) {
          cache.ReadMemory(address, output_bytes);
          std::array<u32, lanes> actual{};
          Require(name, "output readback",
                  LibKernel::Memory::TryReadBacking(address, actual.data(), sizeof(actual)), label + ": readback failed");
          for (u32 i = 0; i < lanes; ++i) {
            const u32 expected = enabled && i < count ? i + 1u : sentinel;
            Require(name, "exact main/aux output", actual[i] == expected,
                    label + " output " + Hex(static_cast<u32>(address)) + " word " + std::to_string(i) +
                        " expected " + Hex(expected) + " got " + Hex(actual[i]));
          }
        }
      };
      // Cold0: both V# targets are unmapped and the GPU-written auxiliary stays unread.
      produce(flag_address, {0u, 0u, 0u});
      round(0u, false, false);
      ShaderComputeInputInfo input{};
      const auto *program = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
      Require(name, "consumer snapshot", program && input.stage.resources,
              "no program-owned resource snapshot after the first cold0 round");
      snapshot = input.stage.resources;
      round(0u, false, false);
      // Positive seed: common target valid, optional target remains unmapped/untouched.
      write_table_vsharp(0x00, output1, output_bytes);
      round(37u, false, false);
      const auto reads_before_warm = counter_reads;
      for (const u32 count : {0u, 1u, 37u, 0u, 37u}) round(count, false, true); // 5 hits
      Require(name, "guarded warm CPU-read reduction", !proof || counter_reads == reads_before_warm,
              "guarded warm counts still synchronized the main counter to the host");
      // Same-value GPU flag rewrite must reject stale CPU-current proof, then reseed.
      produce(flag_address, {0u, 0u, 0u});
      round(1u, false, false);
      round(37u, false, true); // 6
      // An inactive optional V# was not observed/required under strict auxiliary0. Changing
      // only those bytes must preserve the SAME certified snapshot and legal reuse. This
      // round's existing per-word GPU checks still require every optional output sentinel.
      const auto *activity_plan = PipelineCacheTestAccess::ActivityPlan(context.GetPipelineCache(), snapshot);
      Require(name, "guarded snapshot metadata", activity_plan && snapshot &&
                  activity_plan->info.buffers.size() == snapshot->buffers.size(),
              "the guarded snapshot has no matching resource plan");
      uint32_t disabled_outputs = 0;
      for (uint32_t i = 0; i < activity_plan->info.buffers.size(); ++i)
        if (activity_plan->info.buffers[i].written &&
            std::ranges::all_of(snapshot->buffers[i].dwords, [](u32 word) { return word == 0u; }))
          ++disabled_outputs;
      Require(name, "inactive optional snapshot remains zero", disabled_outputs == 1u,
              "the guarded snapshot did not retain exactly one zero optional output V#");
      const auto prior_bindings = snapshot->buffers;
      const auto prior_counter_reads = counter_reads;
      write_table_vsharp(0x10, output2, output_bytes);
      round(1u, false, true); // 7: unrelated inactive-word mutation must be a strict proof hit
      Require(name, "inactive mutation preserves certified bindings",
              snapshot->buffers == prior_bindings && (!proof || counter_reads == prior_counter_reads),
              "an inactive V# mutation changed the certified snapshot or synchronized the count");
      round(0u, false, true); // 8
      // Only the independent GPU flag now changes: fresh fallback must discover the NEW V#.
      produce(flag_address, {1u, 0u, 0u});
      round(37u, true, false); // fresh fallback must bind and write the optional output
      round(1u, true, true);  // 9: ordinary full-joint proof
      produce(flag_address, {0u, 0u, 0u});
      round(37u, false, false); // auxiliary1-to0 must not reuse the old full snapshot
      round(0u, false, true);   // 10: guarded subset again
      // GPU-dirty source metadata with identical bytes must also fall back before reuse.
      std::array<u32, 3> words{};
      Require(name, "descriptor rewrite capture",
              !cache.HasGpuDirtyBytes(table, sizeof(words)) &&
                  LibKernel::Memory::TryReadBacking(table, words.data(), sizeof(words)),
              "common V# words are not CPU-current");
      produce(table, words);
      round(37u, false, false);
      round(1u, false, true); // 11
      // Missing observation is deliberately unsupported: scalar buffer size0 synthesizes0.
      write_table_vsharp(0xe0, flag_address, 0u);
      context.UnmapMemory(table, table_size); context.MapMemory(table, table_size);
      round(37u, false, false);
      round(1u, false, false);
      write_table_vsharp(0xe0, flag_address, 16u);
      context.UnmapMemory(table, table_size); context.MapMemory(table, table_size);
      round(37u, false, false);
      round(0u, false, true); // 12
      // A mapping transition with identical table bytes cannot reuse its former certificate.
      context.UnmapMemory(table, table_size); context.MapMemory(table, table_size);
      round(0u, false, false);
      round(37u, false, false);
      round(1u, false, true); // 13
      // Actual physical alias of the auxiliary page: reject both reuse and NEW guarded seed.
      Require(name, "unique auxiliary page before alias",
              LibKernel::Memory::IsBackingUniquelyMapped(flag_address, table_size) &&
                  !cache.HasGpuDirtyBytes(flag_address, 4), "auxiliary page is not CPU-current/unique");
      void *alias = reinterpret_cast<void *>(alias_va);
      Require(name, "physical auxiliary alias",
              LibKernel::Memory::KernelMapDirectMemory(&alias, table_size, 0x3, 0x10,
                  direct_offset + (flag_address - base), table_size) == 0 &&
                  alias == reinterpret_cast<void *>(alias_va) &&
                  !LibKernel::Memory::IsBackingUniquelyMapped(flag_address, 4),
              "second VA did not create a real alias of the auxiliary backing");
      round(37u, false, false);
      round(1u, false, false);
      Require(name, "physical auxiliary alias removal",
              LibKernel::Memory::KernelMunmap(alias_va, table_size) == 0 &&
                  LibKernel::Memory::IsBackingUniquelyMapped(flag_address, table_size),
              "auxiliary uniqueness did not return after unmap");
      round(37u, false, false);
      round(0u, false, true); // 14
      Require(name, "exact guarded native reuse count", hit_rounds == (proof ? 14u : 0u),
              "guarded proof hits " + std::to_string(hit_rounds));
      context.UnmapMemory(table, table_size);
      context.UnmapMemory(base + table_size, allocation_size - table_size);
      scheduler.Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "unmap direct backing", LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
            "guarded activity fixture mapping release failed");
    Require(name, "release direct backing", LibKernel::Memory::KernelReleaseDirectMemory(direct_offset, allocation_size) == 0,
            "guarded activity fixture allocation release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

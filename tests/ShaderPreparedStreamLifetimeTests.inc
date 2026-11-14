  // An actual flattened-SRT upload must outlive a later shared-ring wrap until the pending
  // compute consumer is recorded. This exercises data lifetime after the command-handle fix.
  void CheckPreparedStreamLifetime() {
    constexpr const char *name = "PreparedStreamLifetime";
    namespace IR = ShaderRecompiler::IR;
    constexpr uintptr_t base = 0x0000000208a00000ull;
    constexpr uint64_t allocation_size = 0x100000;
    constexpr uint64_t table_size = 0x10000;
    constexpr uint64_t source_address = base + 0x40000;
    constexpr uint64_t output_address = base + 0x80000;
    constexpr u32 value = 0x72d0e319u;
    constexpr std::array<u32, 3> initial{0u, 0x13579bdfu, 0x2468ace0u};
    // S_LOAD becomes the real CPU-flattened SRT binding. Its value is shader data, not an
    // address: a bad lifetime can only change output[0], not direct an invalid memory access.
    const std::vector<u32> code{
        EncodeSmem0(0x00u, 12, 0), EncodeSmem1(0, 125), EncodeSopp(0x0c, 0),
        EncodeVop1(0x01u, 1, 12), EncodeMubuf0(0x1cu, 0, false, false),
        EncodeMubuf1(1, 1, 0), EncodeSopp(0x01, 0)};
    ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<uint32_t>(code.size() * sizeof(u32))});
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size, allocation_size,
        0, &physical) == 0, "guest fixture allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, allocation_size, 0x3, 0x10, physical, allocation_size) == 0 &&
        mapped == reinterpret_cast<void *>(base), "guest fixture mapping failed");
    std::memset(mapped, 0, allocation_size); // Includes the real all-null guest image table.
    Require(name, "source and tail initialization",
        LibKernel::Memory::TryWriteBacking(source_address, &value, sizeof(value)) &&
        LibKernel::Memory::TryWriteBacking(output_address, initial.data(), sizeof(initial)),
        "guest fixture values could not be initialized");

    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      context.MapMemory(base, allocation_size);
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      auto &scheduler = context.GetCommandScheduler();
      auto &executor = context.GetRenderExecutor();
      auto &cache = context.GetBufferCache();
      auto &stream = cache.GetUtilityBuffer(MemoryUsage::Stream);
      auto &heap = context.GetImageHeap();
      auto &shaders = processor.GetShCtx();

      // Normalize the next stream reservation to zero without fabricating watcher state.
      scheduler.Finish();
      const auto [normalizing, normalizing_offset] = stream.Map(stream.Size(), 1);
      Require(name, "normalize ring", normalizing != nullptr && normalizing_offset == 0,
              "the full-size normalization reservation did not start at zero");
      stream.Commit();
      scheduler.Wait(scheduler.CurrentTick());

      heap.BeginDraw();
      IR::ImageResource sampled{};
      sampled.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
      IR::ImageHeapView heap_view{};
      Require(name, "real dirty heap table", heap.Acquire(
          {base, table_size, 32, 0, table_size / 32, &sampled}, heap_view),
          "the all-null 64KiB table was not acquired");
      heap.Rebind(executor);

      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                          .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                          .wave_size = 64, .user_sgpr = 8});
      shaders.SetCsUserSgpr(0, static_cast<u32>(source_address));
      shaders.SetCsUserSgpr(1, static_cast<u32>(source_address >> 32u));
      ShaderBufferResource output{};
      output.UpdateAddress48(output_address);
      output.fields[2] = sizeof(initial);
      output.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(4 + i, output.fields[i]);
      ShaderComputeInputInfo input{};
      for (auto &dimension : input.workgroup_counts) dimension = 1;
      const auto *module = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
      Require(name, "real flattened SRT", module != nullptr && input.stage.resources != nullptr &&
          module->info.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::FlattenedSrt)] == 1u &&
          input.stage.resources->flattened_srt == std::vector<u32>{value},
          "the shader did not retain exactly the intended CPU-flattened value");
      // Use the existing restart-fixture technique: select a heap-capable layout before its
      // native pipeline is created. The SPIR-V may legally leave that extra set unused.
      const_cast<ShaderModule *>(module)->info.info.uses_image_heap = true;
      auto *pipeline = context.GetPipelineCache().GetComputePipeline(input);
      Require(name, "heap-capable compute pipeline", pipeline != nullptr && pipeline->uses_image_heap,
              "the compute pipeline did not include the acquired heap table");

      PreparedBindings prepared;
      executor.PrepareBindings(input.stage, prepared);
      auto *stage = &prepared;
      const auto stages = std::span{&stage, 1u};
      executor.FindBuffers(stages);
      executor.RebindBuffers(prepared); // NativeUpload allocates the real pending SRT bytes here.
      const auto retained = prepared.flattened_srt;
      // Keep the exact RED consumer assertion below, while accepting retained backing.
      // The dependent probe also leaves this setup valid against the original ring backend.
      const auto *retained_bytes = [&]<typename Prepared>(const Prepared &binding) -> const uint8_t * {
        if (retained.buffer == stream.Handle() && retained.offset <= stream.Size() &&
            retained.range <= stream.Size() - retained.offset)
          return stream.Mapped().data() + retained.offset;
        if constexpr (requires { binding.uploads.buffers; }) {
          for (const auto &owner : binding.uploads.buffers)
            if (owner->Handle() == retained.buffer && retained.offset <= owner->Size() &&
                retained.range <= owner->Size() - retained.offset)
              return owner->Mapped().data() + retained.offset;
        }
        return nullptr;
      }(prepared);
      Require(name, "prepared SRT reservation", retained_bytes != nullptr && retained.range == sizeof(value),
          "the actual prepared SRT has no retained backing for its exact source DWORD");
      u32 before = 0;
      std::memcpy(&before, retained_bytes, sizeof(before));
      Require(name, "prepared bytes initialized", before == value,
              "the pending flattened SRT did not initially contain the CPU value");

      const auto cursor = StreamBufferTestAccess::Offset(stream);
      Require(name, "prepared reservation cursor", cursor > 0 && cursor <= stream.Size(),
              "the shared ring has no bounded tail padding reservation");
      const auto [padding, padding_offset] = stream.Map(
          cursor == stream.Size() ? stream.Size() : stream.Size() - cursor, 1);
      Require(name, "fill remaining ring", padding != nullptr &&
          padding_offset == (cursor == stream.Size() ? 0 : cursor),
          "tail padding unexpectedly wrapped or could not be reserved");
      stream.Commit(); // No GPU consumer of the prepared SRT has been recorded at this tick.
      const auto prepared_tick = scheduler.CurrentTick();
      executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute, *pipeline, stages);
      Require(name, "late wrap reached", scheduler.CurrentTick() == prepared_tick + 1,
              "the real image-heap slot copy did not submit/restart the prepared tick");
      u32 after = 0;
      std::memcpy(&after, retained_bytes, sizeof(after));
      std::fprintf(stderr,
          "PREPARED_STREAM before=%08x after_heap_wrap=%08x offset=%llu range=%llu old_tick=%llu consumer_tick=%llu\n",
          before, after, static_cast<unsigned long long>(retained.offset),
          static_cast<unsigned long long>(retained.range),
          static_cast<unsigned long long>(prepared_tick),
          static_cast<unsigned long long>(scheduler.CurrentTick()));

      // A second full-ring wrap must not turn allocation+1 into a guessed lifetime.
      const auto second_tick = scheduler.CurrentTick();
      const auto [second_padding, second_offset] = stream.Map(stream.Size(), 1);
      Require(name, "second wrap padding", second_padding != nullptr && second_offset == 0,
              "the second full-ring reservation could not be made");
      stream.Commit();
      const std::array<u32, 4> zeros{};
      (void)stream.Copy(zeros.data(), sizeof(zeros), 16);
      Require(name, "two actual restarts", scheduler.CurrentTick() > second_tick,
              "the second stream wrap did not submit its current prefix");
      std::memcpy(&after, retained_bytes, sizeof(after));
      // The second submission also retired descriptor commands; bind the prepared data
      // again on the current command buffer before the native consumer.
      executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute, *pipeline, stages);

      // Fetch the current handle AFTER CommitBindings; command restart correctness is separate.
      const auto command = scheduler.Current().Handle();
      command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
      command.dispatch(1, 1, 1);
      scheduler.NoteRecordedWork();
      cache.ReadMemory(output_address, sizeof(initial));
      std::array<u32, 3> actual{};
      Require(name, "native output readback", LibKernel::Memory::TryReadBacking(
          output_address, actual.data(), sizeof(actual)), "native output could not be read back");
      Require(name, "prepared bytes survive until native consumer",
          after == value && actual == std::array<u32, 3>{value, initial[1], initial[2]},
          "pending SRT upload was reused before its consumer: before=" + Hex(before) +
          " after_heap_wrap=" + Hex(after) + " GPU=" + Hex(actual[0]) +
          " expected=" + Hex(value) + " tails=" + Hex(actual[1]) + "," + Hex(actual[2]));
      // Native copies verify the other deferred input classes share retained backing.
      // The pressure branch submits only the recorded prefix; these consumers deliberately
      // do not exist yet while the first guest/host inputs remain pinned.
      const std::array<u32, 3> host_input{0x91b53d27u, 0x8428a6d1u, 0x376af0c5u};
      std::vector<uint8_t> chunk(1024u * 1024u, 0x5a);
      const auto pressure_tick = scheduler.CurrentTick();
      {
        BufferCache::PreparedUploadScope scope(cache);
        const auto [guest_input, guest_offset] = cache.ObtainBuffer(source_address, sizeof(value), false);
        const auto [host_buffer, host_offset] = cache.UploadPrepared(
            host_input.data(), sizeof(host_input), 256);
        Require(name, "read-only prepared owner", guest_input != &stream &&
            guest_input->Usage() == MemoryUsage::Stream && !guest_input->Mapped().empty(),
                "small CPU-owned input escaped to the shared allocation-tick ring");
        // More than the cached budget is pinned by this one unrecorded preparation.
        // Overflow is contiguous, owned and limited by these actual live input bytes.
        for (unsigned i = 0; i < 65; ++i)
          (void)cache.UploadPrepared(chunk.data(), chunk.size(), 256);
        Require(name, "bounded pressure path", scheduler.CurrentTick() > pressure_tick &&
            BufferCacheTestAccess::PreparedPoolBytes(cache) <= 64u * 1024u * 1024u,
            "upload pressure did not drain its recorded prefix or exceeded the cached budget");
        u32 guest_word = 0;
        std::memcpy(&guest_word, guest_input->Mapped().data() + guest_offset, sizeof(guest_word));
        Require(name, "unrecorded inputs survive pressure", guest_word == value &&
            std::memcmp(host_buffer->Mapped().data() + host_offset,
                        host_input.data(), sizeof(host_input)) == 0,
            "pressure recycled a preparation's still-unconsumed input");
        const auto [destination, destination_offset] = cache.ObtainBuffer(
            output_address, sizeof(initial), true, false, {}, false);
        destination->CopyFrom(scheduler.Current(), *guest_input, guest_offset, destination_offset, 4);
        destination->CopyFrom(scheduler.Current(), *host_buffer, host_offset, destination_offset + 4, 4);
      } // The copy commands now own their actual retirement tick.
      {
        BufferCache::PreparedUploadScope later(cache);
        // A later preparation may recycle chunks only after the earlier native copies finish.
        (void)cache.UploadPrepared(chunk.data(), chunk.size(), 256);
      }
      cache.ReadMemory(output_address, sizeof(initial));
      Require(name, "retained inputs native copies", LibKernel::Memory::TryReadBacking(
          output_address, actual.data(), sizeof(actual)) &&
          actual == std::array<u32, 3>{value, host_input[0], initial[2]},
          "prepared guest/host inputs or exact destination tail changed across pressure/reuse");
      scheduler.Finish();
      cache.RunGarbageCollector();
      Require(name, "completed upload leases collected",
          BufferCacheTestAccess::PreparedRetirements(cache) == 0,
          "completed upload leases remained pinned after an idle GPU collection boundary");
      const auto cached_bytes = BufferCacheTestAccess::PreparedPoolBytes(cache);
      for (unsigned i = 0; i < 128; ++i) {
        BufferCache::PreparedUploadScope skipped(cache);
        (void)cache.UploadPrepared(host_input.data(), sizeof(host_input), 256);
      } // No consumer: skipped preparations must still be reclaimed after their prefix retires.
      {
        BufferCache::PreparedUploadScope oversized(cache);
        std::vector<uint8_t> large(2u * 1024u * 1024u + 32u, 0x3c);
        const auto [buffer, offset] = cache.UploadPrepared(large.data(), large.size(), 256);
        Require(name, "contiguous oversized preparation",
            offset <= buffer->Size() && large.size() <= buffer->Size() - offset &&
            std::memcmp(buffer->Mapped().data() + offset, large.data(), large.size()) == 0,
            "a large preparation was split, truncated or overwritten");
      }
      scheduler.Finish();
      cache.RunGarbageCollector();
      Require(name, "skipped and oversized leases retired",
          BufferCacheTestAccess::PreparedRetirements(cache) == 0 &&
          BufferCacheTestAccess::PreparedPoolBytes(cache) == cached_bytes,
          "skipped/large preparations leaked leases or grew the completed upload pool");
      context.UnmapMemory(base, allocation_size);
      scheduler.Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "unmap", LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
            "guest fixture mapping release failed");
    Require(name, "release", LibKernel::Memory::KernelReleaseDirectMemory(physical, allocation_size) == 0,
            "guest fixture allocation release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

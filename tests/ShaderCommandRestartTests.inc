  // Included inside VulkanHarness. The guard rejects an ended native command buffer before
  // forwarding it to Vulkan, so the old product fails with an assertion, not undefined recording.
  void CheckCommandRestart(const char *selector) {
    constexpr const char *name = "ImageHeapCommandRestart";
    constexpr uintptr_t base = 0x0000000206e00000ull;
    constexpr uint64_t allocation_size = 0x100000;
    constexpr uint64_t table_size = 0x10000;
    constexpr uint64_t output_address = base + 0x80000;
    constexpr uint64_t color_address = base + 0xa0000;
    namespace IR = ShaderRecompiler::IR;
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size,
        allocation_size, 0, &physical) == 0, "guest fixture allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, allocation_size, 0x3, 0x10, physical, allocation_size) == 0 &&
        mapped == reinterpret_cast<void *>(base), "guest fixture mapping failed");
    std::memset(mapped, 0, allocation_size);

    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      context.MapMemory(base, allocation_size);
      auto &scheduler = context.GetCommandScheduler();
      auto &executor = context.GetRenderExecutor();
      auto &heap = context.GetImageHeap();
      auto &shaders = processor.GetShCtx();
      auto &registers = scheduler.Current().GetRegisters();
      auto &user_config = scheduler.Current().GetUserConfig();
      auto &device = context.GetGraphics().device;
      auto &stream = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
      static CommandScheduler *guard_scheduler = nullptr;
      static PFN_vkCmdPushConstants push_constants = nullptr;
      static PFN_vkCmdPushDescriptorSetKHR push_descriptors = nullptr;
      static PFN_vkCmdBindPipeline bind_pipeline = nullptr;
      static PFN_vkCmdSetViewportWithCount set_viewport = nullptr;
      static PFN_vkCmdPipelineBarrier pipeline_barrier = nullptr;
      static const char *guard_case = nullptr;
      guard_scheduler = &scheduler;
      auto &dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
      push_constants = std::exchange(dispatch.vkCmdPushConstants,
          [](VkCommandBuffer cmd, VkPipelineLayout layout, VkShaderStageFlags stages,
             uint32_t offset, uint32_t size, const void *values) {
            Require("ImageHeapCommandRestart", guard_case,
                cmd == static_cast<VkCommandBuffer>(guard_scheduler->Current().Handle()),
                "push constants targeted the retired native command buffer");
            push_constants(cmd, layout, stages, offset, size, values);
          });
      push_descriptors = std::exchange(dispatch.vkCmdPushDescriptorSetKHR,
          [](VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout,
             uint32_t set, uint32_t count, const VkWriteDescriptorSet *writes) {
            Require("ImageHeapCommandRestart", guard_case,
                cmd == static_cast<VkCommandBuffer>(guard_scheduler->Current().Handle()),
                "descriptors targeted the retired native command buffer");
            push_descriptors(cmd, point, layout, set, count, writes);
          });
      bind_pipeline = std::exchange(dispatch.vkCmdBindPipeline,
          [](VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipeline pipeline) {
            Require("ImageHeapCommandRestart", guard_case,
                cmd == static_cast<VkCommandBuffer>(guard_scheduler->Current().Handle()),
                "pipeline binding targeted the retired native command buffer");
            bind_pipeline(cmd, point, pipeline);
          });
      set_viewport = std::exchange(dispatch.vkCmdSetViewportWithCount,
          [](VkCommandBuffer cmd, uint32_t count, const VkViewport *viewports) {
            Require("ImageHeapCommandRestart", guard_case,
                cmd == static_cast<VkCommandBuffer>(guard_scheduler->Current().Handle()),
                "draw dynamic state targeted the retired native command buffer");
            set_viewport(cmd, count, viewports);
          });
      pipeline_barrier = std::exchange(dispatch.vkCmdPipelineBarrier,
          [](VkCommandBuffer cmd, VkPipelineStageFlags source, VkPipelineStageFlags dest,
             VkDependencyFlags flags, uint32_t memory_count, const VkMemoryBarrier *memory,
             uint32_t buffer_count, const VkBufferMemoryBarrier *buffers,
             uint32_t image_count, const VkImageMemoryBarrier *images) {
            Require("ImageHeapCommandRestart", guard_case,
                cmd == static_cast<VkCommandBuffer>(guard_scheduler->Current().Handle()),
                "barrier targeted the retired native command buffer");
            pipeline_barrier(cmd, source, dest, flags, memory_count, memory,
                             buffer_count, buffers, image_count, images);
          });
      struct Restore {
        ~Restore() {
          auto &d = VULKAN_HPP_DEFAULT_DISPATCHER;
          d.vkCmdPushConstants = push_constants;
          d.vkCmdPushDescriptorSetKHR = push_descriptors;
          d.vkCmdBindPipeline = bind_pipeline;
          d.vkCmdSetViewportWithCount = set_viewport;
          d.vkCmdPipelineBarrier = pipeline_barrier;
          guard_scheduler = nullptr;
        }
      } restore;

      uint64_t next_table = base;
      const auto force_wrap = [&] {
        scheduler.Finish();
        heap.BeginDraw();
        IR::ImageResource sampled{};
        sampled.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
        IR::ImageHeapView view{};
        Require(name, "resident dirty table", heap.Acquire(
            {next_table, table_size, 32, 0, table_size / 32, &sampled}, view),
            "the all-null 64KiB table was not resident");
        next_table += table_size;
        heap.Rebind(executor);
        // No GPU command consumes this reservation. Its CurrentTick watch forces the next
        // actual slot upload to submit and begin a different native command buffer.
        const auto [data, offset] = stream.Map(stream.Size(), 1);
        Require(name, "fill current stream epoch", data != nullptr && offset == 0,
                "could not fill the cache-owned stream ring");
        stream.Commit();
        return scheduler.CurrentTick();
      };
      const auto is = [&](const char *part) {
        return std::strcmp(selector, "--command-restart-only") == 0 ||
               std::strcmp(selector, part) == 0;
      };

      if (is("--command-restart-bindings-only")) {
        guard_case = "CommitBindings after slot upload";
        ShaderModule module;
        module.info.stage = ShaderType::Compute;
        module.info.bindings.push_data_start_dword = 0;
        module.info.info.user_data_registers = {0, 1};
        module.info.bindings.memory_offset_dword = 2;
        IR::ResourceSnapshot snapshot;
        snapshot.user_data = {0x12345678, 0x90abcdef};
        ShaderStageRuntime runtime{&module, &snapshot};
        PreparedBindings bindings;
        executor.PrepareBindings(runtime, bindings);
        const vk::DescriptorSetLayoutCreateInfo empty_info{};
        vk::DescriptorSetLayout empty = nullptr;
        RequireVk(name, "empty layout", device.createDescriptorSetLayout(
            &empty_info, nullptr, &empty), "createDescriptorSetLayout");
        const std::array layouts{empty, context.GetGraphics().image_heap_layout};
        const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute, 0,
                                        IR::NativePushConstantSize};
        vk::PipelineLayoutCreateInfo info{};
        info.setLayoutCount = layouts.size();
        info.pSetLayouts = layouts.data();
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &push;
        PipelineCache::Pipeline pipeline{};
        pipeline.push_stages = vk::ShaderStageFlagBits::eCompute;
        pipeline.uses_image_heap = true;
        RequireVk(name, "binding layout", device.createPipelineLayout(
            &info, nullptr, &pipeline.pipeline_layout), "createPipelineLayout");
        auto *stage = &bindings;
        const auto tick = force_wrap();
        executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute,
                                pipeline, std::span{&stage, 1u});
        Require(name, "binding wrap reached", scheduler.CurrentTick() == tick + 1,
                "the heap upload did not restart the current command recording");
        scheduler.Finish();
        device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
        device.destroyDescriptorSetLayout(empty, nullptr);
      }

      if (is("--command-restart-draw-only")) {
        guard_case = "ExecutePreparedDraw after slot upload";
        static const auto vertex = [] {
          std::vector<u32> code;
          AppendVMovLiteral(&code, 1, 0xbf800000u);
          AppendVMovLiteral(&code, 2, 0x40400000u);
          code.push_back(EncodeVopc(0xc2, InlineU32(1), 5));
          code.push_back(EncodeVop2(0x01, 3, Vgpr(1), 2));
          code.push_back(EncodeVopc(0xc2, InlineU32(2), 5));
          code.push_back(EncodeVop2(0x01, 4, Vgpr(1), 2));
          AppendVMovU32(&code, 0, 0);
          AppendVMovLiteral(&code, 6, 0x3f800000u);
          code.insert(code.end(), {EncodeExp0(0x0c, 0xf), EncodeExp1(3, 4, 0, 6)});
          AppendEnd(&code);
          return code;
        }();
        static const auto pixel = [] {
          std::vector<u32> code;
          AppendVMovLiteral(&code, 0, 0x3e800000u);
          AppendVMovLiteral(&code, 1, 0x3f800000u);
          code.insert(code.end(), {EncodeExp0(0, 0xf), EncodeExp1(0, 0, 0, 1)});
          AppendEnd(&code);
          return code;
        }();
        static ShaderUserData user_data{};
        const auto vs = reinterpret_cast<uint64_t>(vertex.data());
        const auto ps = reinterpret_cast<uint64_t>(pixel.data());
        ShaderMapUserData(vs, {.type = Prospero::ShaderBinaryType::kGs,
            .user_data = &user_data, .code_size_bytes = static_cast<u32>(vertex.size() * 4)});
        ShaderMapUserData(ps, {.type = Prospero::ShaderBinaryType::kPs,
            .code_size_bytes = static_cast<u32>(pixel.size() * 4)});
        shaders.SetEsShaderBase(vs);
        shaders.SetPsShaderBase(ps);
        constexpr uint32_t extent = 8;
        registers.SetViewportTransformControl(0x300);
        registers.SetViewportScaleOffset(0, 4, 4, 4, 4, 1, 0);
        registers.SetViewportZMax(0, 1);
        registers.SetScreenScissor(0, 0, extent, extent);
        registers.SetWindowScissor(0, 0, extent, extent, false);
        registers.SetGenericScissor(0, 0, extent, extent, false);
        registers.SetRenderTargetMask(0xf);
        registers.SetShaderMask(0xf);
        registers.SetPsInControl(0x8000);
        registers.SetTargetOutputMode(0, 4);
        registers.SetColorBase(0, {.addr = color_address});
        registers.SetColorInfo(0, {.format = Prospero::ChannelLayout::k32_32_32_32,
            .channel_type = Prospero::ChannelType::kFloat,
            .channel_order = Prospero::ChannelOrder::kStandard});
        registers.SetColorAttrib2(0, {.height = extent - 1, .width = extent - 1});
        registers.SetColorAttrib3(0, {.tile_mode = Prospero::TileMode::kLinear,
            .dimension = 1});
        user_config.SetPrimitiveType(Prospero::PrimitiveType::kTriList);
        DrawRenderState state{};
        context.GetPipelineCache().GetGraphicsPrograms(scheduler.Current(), state);
        Require(name, "native draw programs", state.vertex_info[0].stage && state.ps_active,
                "the native draw fixture did not compile");
        // Select the existing heap-capable pipeline-layout path without requiring a texture
        // lookup in the shader. The real dirty table above exercises its production upload.
        const_cast<ShaderModule *>(state.vertex_info[0].stage.module)->info.info.uses_image_heap = true;
        RenderExecutorTestAccess::DrawAuto(executor, scheduler.Current(),
                                         {.vertex_count = 3, .instance_count = 1});
        RenderColorInfo color{};
        RenderExecutorTestAccess::ResolveRenderColorTarget(executor, scheduler.Current(), color, 0);
        TextureCacheTestAccess::ClearImage(context.GetTextureCache(), scheduler.Current(),
            color.image_id, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}, {});
        const auto tick = force_wrap();
        RenderExecutorTestAccess::DrawAuto(executor, scheduler.Current(),
                                         {.vertex_count = 3, .instance_count = 1});
        Require(name, "draw wrap reached", scheduler.CurrentTick() == tick + 1,
                "the draw did not wrap inside its heap-slot upload");
        const auto pixels = ReadCachedTexel(name, context, color.image_id, {}, {extent, extent, 1});
        Require(name, "draw output", pixels.size() == extent * extent * 4,
                "the draw output has an unexpected extent");
        for (size_t i = 0; i < pixels.size(); ++i)
          Require(name, "draw output", pixels[i] == (i % 4 == 3 ? 0x3f800000u : 0x3e800000u),
                  "the restarted draw lost its pipeline, bindings or dynamic state");
      }

      if (is("--command-restart-compute-only")) {
        guard_case = "DispatchDirect after slot upload";
        static const auto code = [] {
          std::vector<u32> words;
          words.push_back(EncodeVop2(0x25u, 1, InlineU32(1), 0));
          words.push_back(EncodeVop2(0x1au, 2, InlineU32(2), 0));
          words.push_back(EncodeMubuf0(0x1cu));
          words.push_back(EncodeMubuf1(1, 0, 2));
          AppendEnd(&words);
          return words;
        }();
        const auto address = reinterpret_cast<uint64_t>(code.data());
        ShaderMapUserData(address, {.type = Prospero::ShaderBinaryType::kCs,
            .code_size_bytes = static_cast<u32>(code.size() * 4)});
        shaders.SetCsShader({.data_addr = address, .num_thread_x = 4,
            .num_thread_y = 1, .num_thread_z = 1, .wave_size = 64, .user_sgpr = 4});
        ShaderBufferResource output{};
        output.UpdateAddress48(output_address);
        output.fields[2] = 64;
        output.fields[3] = DstSel(4, 5, 6, 7) |
            (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
        for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(i, output.fields[i]);
        ShaderComputeInputInfo input{};
        Require(name, "native compute program", context.GetPipelineCache().GetComputeProgram(
            shaders.GetCs(), input), "the native compute fixture did not compile");
        const_cast<ShaderModule *>(input.stage.module)->info.info.uses_image_heap = true;
        Require(name, "compute pipeline", context.GetPipelineCache().GetComputePipeline(input) != nullptr,
                "the native compute pipeline is pending");
        std::memset(reinterpret_cast<void *>(output_address), 0xa5, 64);
        context.InvalidateMemory(output_address, 64);
        const auto tick = force_wrap();
        executor.DispatchDirect(scheduler.Current(), 1, 1, 1, 0x41u, false);
        Require(name, "compute wrap reached", scheduler.CurrentTick() == tick + 1,
                "the dispatch did not wrap inside its heap-slot upload");
        std::array<u32, 16> actual{};
        context.GetBufferCache().ReadMemory(output_address, sizeof(actual));
        Require(name, "compute readback", LibKernel::Memory::TryReadBacking(
            output_address, actual.data(), sizeof(actual)), "the GPU output was unavailable");
        for (u32 i = 0; i < actual.size(); ++i)
          Require(name, "compute output", actual[i] == (i < 4 ? i + 1 : 0xa5a5a5a5u),
                  "the restarted dispatch lost its descriptors or wrote outside its range");
      }
      scheduler.Finish();
      context.UnmapMemory(base, allocation_size);
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    Require(name, "unmap", LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
            "guest fixture unmapping failed");
    Require(name, "release", LibKernel::Memory::KernelReleaseDirectMemory(
        physical, allocation_size) == 0, "guest fixture release failed");
    std::printf("[host]    %-32s ok (%s)\n", name, selector);
  }

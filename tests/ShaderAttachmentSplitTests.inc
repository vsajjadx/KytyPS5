  struct AttachmentSplitGuard {
    static inline CommandScheduler *scheduler = nullptr;
    static inline std::array<VkImage, 3> images{}; // Two MRTs, then depth/stencil.
    static inline VkImageView depth_view = VK_NULL_HANDLE;
    static inline std::array<uint64_t, 3> dependency_ticks{};
    static inline std::array<VkImageAspectFlags, 3> dependency_aspects{};
    static inline std::array<uint64_t, 3> dependency_serials{};
    static inline uint64_t barrier_serial = 0, previous_scope_serial = 0;
    static inline uint64_t previous_tick = UINT64_MAX;
    static inline uint32_t scopes = 0;
    static inline PFN_vkCmdPipelineBarrier2 old_barrier = nullptr;
    static inline PFN_vkCmdBeginRendering old_begin = nullptr;

    static bool Covers(VkPipelineStageFlags2 source_stage, VkAccessFlags2 source,
                       VkPipelineStageFlags2 destination_stage, VkAccessFlags2 destination,
                       bool depth) {
      constexpr auto all = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
      const VkPipelineStageFlags2 source_required = depth ? VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                                                         : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      const VkPipelineStageFlags2 destination_required = depth ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                                                              : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      const auto stage_covers = [=](VkPipelineStageFlags2 mask, VkPipelineStageFlags2 required) {
        return (mask & all) != 0 || (mask & required) == required;
      };
      const VkAccessFlags2 write = depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                        : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      const VkAccessFlags2 read = depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                       : VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
      return stage_covers(source_stage, source_required) && stage_covers(destination_stage, destination_required) &&
             (source & (write | VK_ACCESS_2_MEMORY_WRITE_BIT)) != 0 &&
             (destination & (read | VK_ACCESS_2_MEMORY_READ_BIT)) != 0;
    }

    static void Record(size_t target, VkImageAspectFlags aspects, uint64_t tick) {
      if (dependency_ticks[target] != tick || dependency_serials[target] <= previous_scope_serial)
        dependency_aspects[target] = 0;
      dependency_ticks[target] = tick;
      dependency_aspects[target] |= aspects;
      dependency_serials[target] = barrier_serial;
    }

    static void Observe(const VkDependencyInfo &dependency) {
      ++barrier_serial;
      const auto tick = scheduler->CurrentTick();
      for (uint32_t i = 0; i < dependency.memoryBarrierCount; ++i) {
        const auto &barrier = dependency.pMemoryBarriers[i];
        for (size_t target = 0; target < images.size(); ++target)
          if (Covers(barrier.srcStageMask, barrier.srcAccessMask,
                     barrier.dstStageMask, barrier.dstAccessMask, target == 2))
            Record(target, target == 2 ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                       : VK_IMAGE_ASPECT_COLOR_BIT, tick);
      }
      for (uint32_t i = 0; i < dependency.imageMemoryBarrierCount; ++i) {
        const auto &barrier = dependency.pImageMemoryBarriers[i];
        for (size_t target = 0; target < images.size(); ++target) {
          const auto aspects = target == 2 ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
          const auto &range = barrier.subresourceRange;
          if (barrier.image == images[target] && (range.aspectMask & aspects) != 0 &&
              range.baseMipLevel == 0 && range.levelCount != 0 &&
              range.baseArrayLayer == 0 && range.layerCount != 0 &&
              Covers(barrier.srcStageMask, barrier.srcAccessMask,
                     barrier.dstStageMask, barrier.dstAccessMask, target == 2))
            Record(target, range.aspectMask & aspects, tick);
        }
      }
    }

    AttachmentSplitGuard(CommandScheduler &current, std::array<VkImage, 3> watched,
                         VkImageView depth) {
      scheduler = &current;
      images = watched;
      depth_view = depth;
      dependency_ticks.fill(UINT64_MAX);
      dependency_aspects.fill(0);
      dependency_serials.fill(0);
      barrier_serial = previous_scope_serial = 0;
      previous_tick = UINT64_MAX;
      scopes = 0;
      auto &dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
      old_barrier = std::exchange(dispatch.vkCmdPipelineBarrier2,
          +[](VkCommandBuffer command, const VkDependencyInfo *dependency) {
            Observe(*dependency);
            old_barrier(command, dependency);
          });
      old_begin = std::exchange(dispatch.vkCmdBeginRendering,
          +[](VkCommandBuffer command, const VkRenderingInfo *rendering) {
            if (rendering->pDepthAttachment && rendering->pDepthAttachment->imageView == depth_view) {
              const auto tick = scheduler->CurrentTick();
              if (previous_tick != UINT64_MAX) {
                for (size_t target = 0; target < images.size(); ++target) {
                  const auto aspects = target == 2 ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                                   : VK_IMAGE_ASPECT_COLOR_BIT;
                  Require("AttachmentSplit", "external store-to-load dependency",
                          dependency_ticks[target] == tick && dependency_serials[target] > previous_scope_serial &&
                              (dependency_aspects[target] & aspects) == aspects,
                          "new rendering scope loaded attachment " + std::to_string(target) +
                              " without a color/depth-stencil write-to-read dependency");
                }
              }
              previous_tick = tick;
              previous_scope_serial = barrier_serial;
              ++scopes;
            }
            old_begin(command, rendering);
          });
    }
    ~AttachmentSplitGuard() {
      auto &dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
      dispatch.vkCmdPipelineBarrier2 = old_barrier;
      dispatch.vkCmdBeginRendering = old_begin;
      scheduler = nullptr;
    }
  };

  // Real CountDraw splits, a preserved depth plane, incrementing stencil and two additive MRTs.
  void CheckAttachmentSplit() {
    constexpr const char *name = "AttachmentSplit";
    constexpr uintptr_t base = 0x0000000208c00000ull;
    constexpr uint64_t allocation_size = 0xc0000, mode_stride = 0x40000;
    constexpr uint32_t extent = 16, draws = 17;
    constexpr std::array<std::array<float, 4>, 2> sources{{
        {1.0f, 2.0f, 0.5f, 0.25f}, {0.125f, -1.0f, 4.0f, 1.0f}}};
    constexpr std::array<std::array<float, 4>, 2> clears{{
        {0.5f, -1.0f, 3.0f, 0.0f}, {2.0f, 4.0f, -2.0f, 1.0f}}};
    static const auto vertex = [] {
      std::vector<u32> code;
      AppendVMovLiteral(&code, 1, 0xbf800000u);
      AppendVMovLiteral(&code, 2, 0x40400000u);
      code.push_back(EncodeVopc(0xc2, InlineU32(1), 5));
      code.push_back(EncodeVop2(0x01, 3, Vgpr(1), 2));
      code.push_back(EncodeVopc(0xc2, InlineU32(2), 5));
      code.push_back(EncodeVop2(0x01, 4, Vgpr(1), 2));
      AppendVMovLiteral(&code, 0, 0x3e800000u); // z = 0.25.
      AppendVMovLiteral(&code, 6, 0x3f800000u);
      code.insert(code.end(), {EncodeExp0(0x0c, 0xf), EncodeExp1(3, 4, 0, 6)});
      for (u32 parameter = 0; parameter < 8; ++parameter)
        code.insert(code.end(), {EncodeExp0(0x20 + parameter, 0xf), EncodeExp1(0, 0, 0, 0)});
      AppendEnd(&code);
      return code;
    }();
    std::vector<u32> pixel;
    for (u32 target = 0; target < 2; ++target) {
      for (u32 component = 0; component < 4; ++component)
        AppendVMovLiteral(&pixel, component, std::bit_cast<u32>(sources[target][component]));
      pixel.insert(pixel.end(), {EncodeExp0(target, 0xf, target == 1), EncodeExp1(0, 1, 2, 3)});
    }
    AppendEnd(&pixel);
    static ShaderUserData user_data{};
    const auto vs = reinterpret_cast<uint64_t>(vertex.data());
    const auto ps = reinterpret_cast<uint64_t>(pixel.data());
    ShaderMapUserData(vs, {.type = Prospero::ShaderBinaryType::kGs, .user_data = &user_data,
                          .code_size_bytes = static_cast<u32>(vertex.size() * 4)});
    ShaderMapUserData(ps, {.type = Prospero::ShaderBinaryType::kPs,
                          .code_size_bytes = static_cast<u32>(pixel.size() * 4)});
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size, 0x10000, 0,
        &physical) == 0, "direct allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, allocation_size, 0x3, 0x10, physical, 0x10000) == 0 &&
        mapped == reinterpret_cast<void *>(base), "direct mapping failed");
    std::memset(mapped, 0, allocation_size);
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] { context.MapMemory(base, allocation_size); });
    std::array<std::array<std::vector<u32>, 4>, 3> results;
    // Final FD0 case closes rendering at the same draw boundaries without any submission.
    constexpr std::array<u32, 3> modes{0, 8, 0};
    for (u32 mode = 0; mode < modes.size(); ++mode) {
      Config::Load({.flush_draws = modes[mode]});
      context.GetGpu().SendCommandSync([&] {
        CommandProcessor processor(context, 0);
        processor.BufferInit();
        auto &scheduler = context.GetCommandScheduler();
        auto &executor = context.GetRenderExecutor();
        auto &cache = context.GetTextureCache();
        auto &registers = processor.GetCtx();
        registers.Reset();
        registers.SetViewportTransformControl(0x300);
        registers.SetViewportScaleOffset(0, 8, 8, 8, 8, 1, 0);
        registers.SetViewportZMax(0, 1);
        registers.SetScreenScissor(0, 0, extent, extent);
        registers.SetWindowScissor(0, 0, extent, extent, false);
        registers.SetGenericScissor(0, 0, extent, extent, false);
        registers.SetRenderTargetMask(0xff);
        registers.SetShaderMask(0xff);
        std::array<RenderColorInfo, 2> colors;
        for (u32 target = 0; target < 2; ++target) {
          registers.SetTargetOutputMode(target, 4);
          registers.SetColorBase(target, {.addr = base + mode * mode_stride + 0x20000 + target * 0x10000});
          registers.SetColorInfo(target, {.format = Prospero::ChannelLayout::k32_32_32_32,
              .channel_type = Prospero::ChannelType::kFloat,
              .channel_order = Prospero::ChannelOrder::kStandard});
          registers.SetColorAttrib2(target, {.height = extent - 1, .width = extent - 1});
          registers.SetColorAttrib3(target, {.tile_mode = Prospero::TileMode::kLinear, .dimension = 1});
          HW::BlendControl blend{};
          blend.enable = true;
          blend.color_srcblend = blend.color_destblend = blend.alpha_srcblend = blend.alpha_destblend =
              static_cast<uint8_t>(Prospero::BlendFactor::kOne);
          registers.SetBlendControl(target, blend);
          RenderExecutorTestAccess::ResolveRenderColorTarget(executor, scheduler.Current(), colors[target], target);
          vk::ClearValue clear{};
          clear.color.float32 = clears[target];
          TextureCacheTestAccess::ClearImage(cache, scheduler.Current(), colors[target].image_id,
                                            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}, clear);
        }
        HW::DepthRenderTarget target{};
        target.z_info.format = Prospero::DepthFormat::kZ32F;
        target.z_info.texture_compatibility = Prospero::TextureCompatiblePlaneCompression::kEnable;
        target.stencil_info.format = Prospero::StencilFormat::k8UInt;
        target.stencil_info.texture_compatibility = Prospero::TextureCompatibleStencil::kEnable;
        target.z_read_base_addr = target.z_write_base_addr = base + mode * mode_stride;
        target.stencil_read_base_addr = target.stencil_write_base_addr = base + mode * mode_stride + 0x10000;
        target.size = {extent - 1, extent - 1, true};
        registers.SetDepthRenderTarget(target);
        HW::DepthControl depth_control{};
        depth_control.z_enable = depth_control.z_write_enable = depth_control.stencil_enable = true;
        depth_control.zfunc = static_cast<uint8_t>(vk::CompareOp::eLessOrEqual);
        depth_control.stencilfunc = static_cast<uint8_t>(vk::CompareOp::eAlways);
        registers.SetDepthControl(depth_control);
        HW::StencilControl stencil_control{};
        stencil_control.stencil_zpass = static_cast<uint8_t>(Prospero::StencilOp::kAddClamp);
        registers.SetStencilControl(stencil_control);
        HW::StencilMask stencil_mask{};
        stencil_mask.stencil_mask = stencil_mask.stencil_writemask = 0xff;
        registers.SetStencilMask(stencil_mask);
        registers.SetRenderControl({});
        RenderDepthInfo depth;
        RenderExecutorTestAccess::ResolveRenderDepthTarget(executor, scheduler.Current(), depth);
        vk::ClearValue clear{};
        clear.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
        TextureCacheTestAccess::ClearImage(cache, scheduler.Current(), depth.image_id,
            {vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1}, clear);
        auto &shaders = processor.GetShCtx();
        shaders.SetEsShaderBase(vs);
        shaders.SetGsShaderResource2({});
        shaders.SetPsShaderBase(ps);
        processor.GetUcfg().SetPrimitiveType(Prospero::PrimitiveType::kTriList);
        const auto depth_view = cache.FindDepthTarget(depth.image_id, depth.desc);
        {
          AttachmentSplitGuard guard(scheduler,
              {static_cast<VkImage>(cache.GetImage(colors[0].image_id).backing.image),
               static_cast<VkImage>(cache.GetImage(colors[1].image_id).backing.image),
               static_cast<VkImage>(cache.GetImage(depth.image_id).backing.image)},
              static_cast<VkImageView>(depth_view));
          const auto tick = scheduler.CurrentTick();
          for (u32 draw = 0; draw < draws; ++draw) {
            if (mode == 2 && (draw == 8 || draw == 16)) scheduler.Current().EndRendering();
            processor.DrawIndexAuto({.vertex_count = 3, .instance_count = 1});
          }
          Require(name, "real split count", scheduler.CurrentTick() - tick == (mode == 1 ? 2u : 0u),
                  "17 draws did not cross exactly the expected CountDraw boundaries");
          Require(name, "same-scope continuation", AttachmentSplitGuard::scopes == (mode == 0 ? 1u : 3u),
                  "unchanged active rendering was restarted between identical draws");
        }
        for (u32 attachment = 0; attachment < 2; ++attachment)
          results[mode][attachment] = ReadCachedTexel(name, context, colors[attachment].image_id, {}, {extent, extent, 1});
        results[mode][2] = ReadCachedTexel(name, context, depth.image_id, {}, {extent, extent, 1});
        auto stencil = CreateHostBuffer(name, extent * extent, vk::BufferUsageFlagBits::eTransferDst, {});
        auto &image = cache.GetImage(depth.image_id);
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, scheduler.Current().Handle());
        vk::BufferImageCopy copy{};
        copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
        copy.imageExtent = {extent, extent, 1};
        scheduler.Current().Handle().copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, stencil.buffer, 1, &copy);
        vk::BufferMemoryBarrier host{};
        host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        host.dstAccessMask = vk::AccessFlagBits::eHostRead;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = stencil.buffer;
        host.size = stencil.size;
        scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                                                    {}, 0, nullptr, 1, &host, 0, nullptr);
        scheduler.Finish();
        results[mode][3] = ReadBuffer(name, stencil, extent * extent / 4);
        DestroyBuffer(&stencil);
        RenderExecutorTestAccess::ResetBindings(executor);
      });
    }
    Config::Load({});
    context.GetGpu().SendCommandSync([&] { context.UnmapMemory(base, allocation_size); context.GetCommandScheduler().Finish(); });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "unmap", LibKernel::Memory::KernelMunmap(base, allocation_size) == 0, "fixture unmap failed");
    Require(name, "release", LibKernel::Memory::KernelReleaseDirectMemory(physical, allocation_size) == 0, "fixture release failed");
    for (u32 mode = 0; mode < modes.size(); ++mode) {
      for (u32 attachment = 0; attachment < 2; ++attachment) {
        Require(name, "color size", results[mode][attachment].size() == extent * extent * 4, "MRT extent mismatch");
        for (size_t i = 0; i < results[mode][attachment].size(); ++i)
          Require(name, "accumulated MRT", results[mode][attachment][i] ==
              std::bit_cast<u32>(clears[attachment][i % 4] + draws * sources[attachment][i % 4]), "MRT sum lost across render scopes");
      }
      Require(name, "depth size", results[mode][2].size() == extent * extent, "depth extent mismatch");
      for (const auto value : results[mode][2])
        Require(name, "depth preservation", value == 0x3e800000u, "depth did not retain 0.25 across scopes");
      Require(name, "stencil size", results[mode][3].size() == extent * extent / 4, "stencil extent mismatch");
      for (const auto value : results[mode][3])
        Require(name, "stencil accumulation", value == 0x11111111u, "stencil did not increment exactly 17 times");
    }
    Require(name, "scope agreement", results[0] == results[1] && results[0] == results[2],
            "submitted, unsubmitted or continuous attachment results differ");
    std::printf("[gpu]     %-32s ok\n", name);
  }

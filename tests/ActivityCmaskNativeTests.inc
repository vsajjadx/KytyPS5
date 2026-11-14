  // Included within VulkanHarness; real GPU count writes and sampled color output.
  void CheckActivityCmaskNative(bool proof) {
    const char *name = proof ? "ActivityCmaskNative" : "ActivityCmaskNativeOriginal";
    Require(name, "activity proof environment", ShaderRecompiler::IR::HostActivityProofEnabled() == proof,
            proof ? "run with KYTY_HOST_ACTIVITY_PROOF=1" : "run with KYTY_HOST_ACTIVITY_PROOF=0");
    constexpr uintptr_t base = 0x0000000209a00000ull;
    constexpr uint64_t allocation_size = 0x80000;
    constexpr uint64_t table = base, count_address = base + 0x10000, output = base + 0x20000;
    constexpr uint64_t data_address = base + 0x40000, metadata_address = base + 0x60000;
    constexpr uint64_t metadata_size = 0x1000, alias_va = base + 0x200000;
    constexpr uint64_t alias_size = BufferCache::CACHING_PAGESIZE;
    constexpr u32 lanes = 64, output_bytes = lanes * sizeof(u32), sentinel = 0xa5a5a5a5u;
    constexpr u32 user_sgprs = 2, width = 256, height = 128;
    std::vector<u32> consumer;
    AppendVop3(&consumer, 0x346u, 0, user_sgprs, InlineU32(6), Vgpr(0));
    consumer.insert(consumer.end(), {
        EncodeSmem0(0x02u, 4, 0), EncodeSmem1(0xd0u, 125), EncodeSopp(0x0c, 0),
        EncodeSmem0(0x08u, 12, 2), EncodeSmem1(0, 125), EncodeSopp(0x0c, 0),
        EncodeVopc(0xc3u, 12, 0), EncodeSop1(0x08u, 16, 106), EncodeSop1(0x04u, 126, 16)});
    const auto execz = consumer.size(); consumer.push_back(0);
    consumer.insert(consumer.end(), {
        EncodeSmem0(0x02u, 8, 0), EncodeSmem1(0x00u, 125),
        EncodeSmem0(0x03u, 24, 0), EncodeSmem1(0x20u, 125), // flat T# s[24:31]
        EncodeSmem0(0x02u, 32, 0), EncodeSmem1(0x40u, 125), // sampler s[32:35]
        EncodeSopp(0x0c, 0)});
    AppendVMovLiteral(&consumer, 20, 0x3f000000u);
    AppendVMovLiteral(&consumer, 21, 0x3f000000u); // normalized (0.5,0.5), uniform native clear
    consumer.insert(consumer.end(), {
        EncodeMimg0(0x20u, 1u), EncodeMimg1(3, 20, 6, 8),
        EncodeVop2(0x1au, 2, InlineU32(2), 0), EncodeMubuf0(0x1cu), EncodeMubuf1(3, 2, 2)});
    consumer[execz] = EncodeSopp(0x08, static_cast<u32>(consumer.size() - execz - 1u));
    AppendEnd(&consumer);
    std::vector<u32> writer;
    for (u32 i = 0; i < 3; ++i)
      writer.insert(writer.end(), {EncodeVop1(0x01u, 1, 12u + i),
          EncodeMubuf0(0x1cu, 0, false, false), EncodeMubuf1(1, i, 0)});
    AppendEnd(&writer);
    for (const auto *code : {&consumer, &writer})
      ShaderMapUserData(reinterpret_cast<uint64_t>(code->data()),
          {.type = Prospero::ShaderBinaryType::kCs,
           .code_size_bytes = static_cast<u32>(code->size() * sizeof(u32))});
    EnsureRuntimeContext();
    int64_t direct_offset = -1;
    Require(name, "direct allocation", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size, 0x10000, 0,
        &direct_offset) == 0, "CMASK activity allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "direct mapping", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, allocation_size, 0x3, 0x10, direct_offset, 0x10000) == 0 &&
        mapped == reinterpret_cast<void *>(base), "CMASK activity mapping failed");
    std::memset(mapped, 0, allocation_size);
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      context.MapMemory(base, allocation_size);
      CommandProcessor processor(context, 0); processor.BufferInit();
      auto &scheduler = context.GetCommandScheduler(); auto &cache = context.GetBufferCache();
      auto &textures = context.GetTextureCache(); auto &shaders = processor.GetShCtx();
      auto &executor = context.GetRenderExecutor();
      const auto buffer_descriptor = [](uint64_t address, u32 size) {
        ShaderBufferResource d{}; d.UpdateAddress48(address); d.fields[2] = size;
        d.fields[3] = DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
        return d;
      };
      const auto output_vsharp = buffer_descriptor(output, output_bytes);
      const auto count_vsharp = buffer_descriptor(count_address, 16);
      const ShaderTextureResource tsharp{{static_cast<u32>(data_address >> 8u),
          (static_cast<u32>(Prospero::BufferFormat::k32Float) << 20u) | (((width - 1u) & 3u) << 30u),
          ((width - 1u) >> 2u) | ((height - 1u) << 14u),
          DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::TileMode::kRenderTarget) << 20u) |
              (static_cast<u32>(Prospero::ImageType::kColor2D) << 28u), 0, 0x00700000u, 0, 0}};
      ShaderSamplerResource sampler{};
      LibKernel::Memory::WriteBacking(table, output_vsharp.fields, sizeof(output_vsharp.fields));
      LibKernel::Memory::WriteBacking(table + 0x20, tsharp.fields, sizeof(tsharp.fields));
      LibKernel::Memory::WriteBacking(table + 0x40, sampler.fields, sizeof(sampler.fields));
      LibKernel::Memory::WriteBacking(table + 0xd0, count_vsharp.fields, sizeof(count_vsharp.fields));
      TextureCache::ImageDesc color{};
      color.type = TextureCache::BindingType::RenderTarget; color.info.data = {data_address, 0};
      color.info.pixel_format = vk::Format::eR32Sfloat; color.info.guest_format = Prospero::BufferFormat::k32Float;
      color.info.type = Prospero::ImageType::kColor2D; color.info.extent = {width, height, 1};
      color.info.resources = {1, 1}; color.info.bytes_per_block = 4;
      color.info.samples = 1; color.info.tile_mode = Prospero::TileMode::kRenderTarget;
      color.info.UpdateSize();
      TileSizeAlign cmask_size{};
      Require(name, "CMASK owner size", TileGetCmaskSize(width, height, 1, 1, cmask_size) &&
          cmask_size.size == metadata_size && color.info.data.size == 0x20000,
          "fixture does not match the observed 128KiB data/4KiB CMASK shape");
      color.info.metadata.kind = ImageMetadataKind::Cmask;
      color.info.metadata.range = {metadata_address, metadata_size};
      color.info.metadata.clear_register_valid = true;
      color.view_info.format = color.info.pixel_format; color.view_info.type = vk::ImageViewType::e2D;
      color.view_info.aspect = vk::ImageAspectFlagBits::eColor;
      color.view_info.usage = vk::ImageUsageFlagBits::eColorAttachment;
      ImageId owner{};
      const auto acquire_target = [&](TextureCache::ImageDesc desc) {
        RenderExecutorTestAccess::ResetBindings(executor);
        const auto id = textures.FindImage(desc);
        Require(name, "normal native target preparation", id &&
            textures.FindRenderTarget(id, desc) != nullptr, "normal CMASK target acquisition failed");
        return id;
      };
      color.info.metadata.clear_word = 0x3f800000u;
      owner = acquire_target(color); // normal CMASK expansion records a real native clear
      const auto check_owner = [&] {
        const auto *image = TextureCacheTestAccess::Owner(textures, owner);
        Require(name, "registered native CMASK owner", image && image->registered &&
            image->usage.render_target && image->info.data == color.info.data &&
            image->info.metadata.kind == ImageMetadataKind::Cmask &&
            image->info.metadata.range == color.info.metadata.range,
            "plain sampling lost its exact live CMASK owner");
      };
      check_owner();
      const auto produce_count = [&](u32 count) {
        shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(writer.data()),
            .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1, .wave_size = 64, .user_sgpr = 15});
        for (u32 i = 0; i < 3; ++i) {
          const auto d = buffer_descriptor(count_address + 4u * i, 4);
          for (u32 j = 0; j < 4; ++j) shaders.SetCsUserSgpr(4u * i + j, d.fields[j]);
          shaders.SetCsUserSgpr(12u + i, i == 0 ? count : 0u);
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
      uint32_t rounds = 0, hit_rounds = 0, counter_reads = 0;
      const auto round = [&](u32 count, u32 expected_pixel, bool expect_hit) {
        const auto label = "round " + std::to_string(rounds++) + " count " + std::to_string(count);
        auto [buffer, offset] = cache.ObtainBuffer(output, output_bytes, true, false, {}, false);
        buffer->Fill(offset, output_bytes, sentinel);
        produce_count(count);
        u32 old_backing = 0;
        Require(name, "fresh GPU count", cache.HasGpuDirtyBytes(count_address, 4) &&
            LibKernel::Memory::TryReadBacking(count_address, &old_backing, 4), label);
        bind_consumer();
        const auto before_hits = reuse(), tick = scheduler.CurrentTick();
        processor.DispatchDirect(1, 1, 1, 0x41u);
        const auto hits = reuse() - before_hits;
        const bool host_read = !cache.HasGpuDirtyBytes(count_address, 4);
        counter_reads += host_read ? 1u : 0u; hit_rounds += hits != 0u ? 1u : 0u;
        if (proof && expect_hit) {
          u32 after_backing = 0;
          Require(name, "plain CMASK-owner activity reuse", hits == 1u && !host_read &&
              scheduler.CurrentTick() == tick &&
              LibKernel::Memory::TryReadBacking(count_address, &after_backing, 4) &&
              after_backing == old_backing,
              label + ": expected certified reuse without counter download/submission");
        } else {
          Require(name, "fresh original CMASK path", hits == 0u && host_read &&
              scheduler.CurrentTick() > tick, label + ": expected fresh host count and no reuse");
        }
        cache.ReadMemory(output, output_bytes);
        std::array<u32, lanes> actual{};
        Require(name, "GPU sampled output readback",
            LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)), label);
        for (u32 i = 0; i < lanes; ++i)
          Require(name, "fresh sampled texel and exact tails", actual[i] == (i < count ? expected_pixel : sentinel),
              label + " word " + std::to_string(i) + " expected " + Hex(i < count ? expected_pixel : sentinel) +
              " got " + Hex(actual[i]));
      };
      // Cold empty traversals cannot seed an unobserved resource set.
      round(0, 0x3f800000u, false);
      ShaderComputeInputInfo input{};
      const auto *module = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
      Require(name, "native resource snapshot", module && input.stage.resources,
              "consumer has no resource snapshot");
      snapshot = input.stage.resources;
      round(0, 0x3f800000u, false);
      round(37, 0x3f800000u, false);
      const auto warm_reads = counter_reads;
      for (u32 count : {0u, 1u, 37u, 0u, 37u}) round(count, 0x3f800000u, true); // 5
      Require(name, "warm counter-read reduction", !proof || warm_reads == counter_reads,
              "warm plain CMASK sampling still downloaded GPU counts");
      check_owner();
      // Real GPU CMASK mutation followed by normal target materialization must give NEW color.
      auto [metadata_buffer, metadata_offset] = cache.ObtainBuffer(
          metadata_address, metadata_size, true, false, {}, false);
      metadata_buffer->Fill(metadata_offset, metadata_size, 0u);
      Require(name, "actual GPU CMASK write", cache.HasGpuDirtyBytes(metadata_address, metadata_size),
              "metadata mutation did not produce GPU-owned bytes");
      const auto old_image_epoch = Libs::Graphics::Image::activity_epoch.load();
      color.info.metadata.clear_word = 0x40000000u; // 2.0
      Require(name, "same owner after new CMASK clear", acquire_target(color) == owner &&
          (!proof || Libs::Graphics::Image::activity_epoch.load() > old_image_epoch), "CMASK clear lost owner/invalidation");
      round(37, 0x40000000u, false); round(1, 0x40000000u, true); // 6
      // CPU non-clear CMASK bytes are not DCC keys; native color must remain 2.0.
      WriteMetadata(context, metadata_address, metadata_size, 0x40404040u);
      color.info.metadata.clear_word = 0x40400000u; // 3.0, must not clear at CMASK code0x40
      Require(name, "non-clear CMASK owner", acquire_target(color) == owner, "CMASK owner changed");
      round(37, 0x40000000u, false); round(1, 0x40000000u, true); // 7
      WriteMetadata(context, metadata_address, metadata_size, 0u);
      Require(name, "CPU CMASK re-clear owner", acquire_target(color) == owner, "CMASK owner changed");
      round(37, 0x40400000u, false); round(0, 0x40400000u, true); // 8
      // A second VA for CMASK backing rejects BOTH reuse and replacement seeding.
      Require(name, "unique metadata setup", LibKernel::Memory::IsBackingUniquelyMapped(metadata_address, metadata_size),
              "CMASK backing was not unique");
      void *alias = reinterpret_cast<void *>(alias_va);
      Require(name, "physical CMASK alias", LibKernel::Memory::KernelMapDirectMemory(
          &alias, alias_size, 0x3, 0x10, direct_offset + (metadata_address - base), alias_size) == 0 &&
          alias == reinterpret_cast<void *>(alias_va) &&
          !LibKernel::Memory::IsBackingUniquelyMapped(metadata_address, metadata_size),
          "CMASK physical alias was not installed");
      round(37, 0x40400000u, false); round(1, 0x40400000u, false);
      Require(name, "remove metadata alias", LibKernel::Memory::KernelMunmap(alias_va, alias_size) == 0 &&
          LibKernel::Memory::IsBackingUniquelyMapped(metadata_address, metadata_size),
          "CMASK uniqueness did not return");
      round(37, 0x40400000u, false); round(1, 0x40400000u, true); // 9
      // Remove ONLY the metadata GPU mapping: data owner remains valid for normal sampling.
      context.UnmapMemory(metadata_address, metadata_size);
      Require(name, "unmapped metadata owner survives", !context.IsMapped(metadata_address, metadata_size) &&
          TextureCacheTestAccess::Contains(textures, owner), "metadata unmap also removed color owner");
      round(37, 0x40400000u, false); round(1, 0x40400000u, false);
      context.MapMemory(metadata_address, metadata_size);
      round(37, 0x40400000u, false); round(0, 0x40400000u, true); // 10
      // Zero-sized native metadata is accepted by the ordinary cache, never by this proof.
      auto empty_metadata = color; empty_metadata.info.metadata.range.size = 0;
      Require(name, "empty native metadata setup", acquire_target(empty_metadata) == owner,
              "empty native metadata replaced the color owner");
      round(37, 0x40400000u, false); round(1, 0x40400000u, false);
      WriteMetadata(context, metadata_address, metadata_size, 0u);
      Require(name, "restore CMASK owner", acquire_target(color) == owner, "CMASK owner changed");
      round(37, 0x40400000u, false); round(1, 0x40400000u, true); // 11
      // Native DCC remains excluded even for identical live color data and view.
      auto dcc_owner = color; dcc_owner.info.metadata.kind = ImageMetadataKind::Dcc;
      WriteMetadata(context, metadata_address, metadata_size, 0x5a5a5a5au);
      Require(name, "native DCC negative setup", acquire_target(dcc_owner) == owner,
              "DCC negative did not retain the same native owner");
      round(37, 0x40400000u, false); round(1, 0x40400000u, false);
      WriteMetadata(context, metadata_address, metadata_size, 0u);
      Require(name, "restore plain CMASK native owner", acquire_target(color) == owner,
              "CMASK restoration changed the data owner");
      round(37, 0x40400000u, false); round(1, 0x40400000u, true); // 12
      // Deleting/recreating a same-address owner must reject its previous certificate.
      scheduler.Finish();
      const auto old_serial = TextureCacheTestAccess::Owner(textures, owner)->serial;
      TextureCacheTestAccess::DeleteImage(textures, owner);
      WriteMetadata(context, metadata_address, metadata_size, 0u);
      color.info.metadata.clear_word = 0x40800000u; // 4.0 in a new native image
      owner = acquire_target(color);
      Require(name, "same-address new owner identity", TextureCacheTestAccess::Owner(textures, owner)->serial != old_serial,
              "owner recreation reused the old identity");
      round(37, 0x40800000u, false);
      round(0, 0x40800000u, true); round(37, 0x40800000u, true); // 14
      check_owner();
      Require(name, "exact native CMASK reuse count", hit_rounds == (proof ? 14u : 0u),
              "CMASK proof hits " + std::to_string(hit_rounds));
      context.UnmapMemory(base, allocation_size);
      scheduler.Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "unmap direct allocation", LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
            "CMASK fixture unmap failed");
    Require(name, "release direct allocation", LibKernel::Memory::KernelReleaseDirectMemory(direct_offset, allocation_size) == 0,
            "CMASK fixture release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

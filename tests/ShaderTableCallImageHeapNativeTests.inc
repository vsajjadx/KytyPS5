  // Real sampled heap selected through a table call and captured context.
  void CheckTableCallImageHeapBinding() {
    constexpr uint64_t base = 0x000000020f000000ull, size = 0x200000;
    constexpr uint64_t call_table = base + 0x100, leaf = base + 0x1000;
    constexpr uint64_t context_a = base + 0x6000, context_b = base + 0x18000;
    constexpr uint64_t texture_a = base + 0x20000, texture_b = base + 0x21000;
    constexpr uint64_t output = base + 0x40000, separate_table = base + 0x100000;
    constexpr u32 table_bytes = 0x10000, sentinel = 0xa5a5a5a5u;
    enum class Case { Disjoint, OddFirst, SourceOverlap, SourceAba, TextureOverlap,
                      RetiredView, NullTexture, NullToLive, DccDisjoint, MetadataOverlap,
                      PlainDepth, DepthStencilAssociation, AssociatedChildOverlap,
                      AssociatedHtileDisjoint, AssociatedHtileOverlap, AssociationReparent,
                      StandaloneStencilDisjoint, StandaloneStencilOverlap };
    const auto descriptor = [](uint64_t address, u32 bytes) {
      ShaderBufferResource result{};
      result.UpdateAddress48(address);
      result.fields[2] = bytes;
      result.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u) | (2u << 28u);
      return result;
    };
    const auto image_descriptor = [](uint64_t address) {
      std::array<u32, 8> result{};
      result[0] = static_cast<u32>(address >> 8u);
      result[1] = static_cast<u32>(address >> 40u) |
          (static_cast<u32>(Prospero::BufferFormat::k32Float) << 20u) | (3u << 30u);
      result[2] = 3u << 14u;
      result[3] = DstSel(4, 0, 0, 1) |
          (static_cast<u32>(Prospero::ImageType::kColor2D) << 28u);
      ShaderTextureResource decoded{};
      std::copy(result.begin(), result.end(), decoded.fields);
      Require("TableCallImageHeap", "texture geometry",
          decoded.Width5() == 3u && decoded.Height5() == 3u && decoded.Base40() == address,
          "native descriptor must describe the initialized 4x4 texture");
      return result;
    };
    const auto make_caller = [&](bool odd_first) {
      std::vector<u32> code;
      if (odd_first) code.push_back(EncodeVopc(0xd5, InlineU32(0), 0));
      code.insert(code.end(), {
          EncodeVop2(0x1b, 1, InlineU32(1), 0), EncodeVop1(0x02, 4, Vgpr(1)),
          EncodeSop2(0x1e, 5, 4, InlineU32(4)),
          EncodeSmem0(0x0a, 24, 4), EncodeSmem1(0, 5), EncodeSopp(0x0c, 0),
          EncodeSop1(0x03, 14, 24), EncodeSop1(0x03, 15, 25),
          EncodeSop1(0x03, 16, 26), EncodeSop1(0x03, 17, 27),
          EncodeSop1(0x21, 14, 14), EncodeVop2(0x1a, 30, InlineU32(4), 0),
          EncodeMubuf0(0x1c, 0), EncodeMubuf1(6, 0, 30),
          EncodeMubuf0(0x1c, 4), EncodeMubuf1(7, 0, 30), EncodeSopp(0x01)});
      return code;
    };
    const u32 texture_pitch = TileGetTexturePitch(
        Prospero::BufferFormat::k32Float, 4u, Prospero::TileMode::kLinear);
    Require("TableCallImageHeap", "linear texture pitch", texture_pitch >= 4u,
        "native row footprint cannot contain the four initialized texels");
    std::array callers{make_caller(false), make_caller(true)};
    std::vector<u32> leaf_code{
        EncodeSmem0(0x02, 24, 8), EncodeSmem1(0, 125),
        EncodeSmem0(0x03, 64, 8), EncodeSmem1(16, 125), EncodeSopp(0x0c, 0),
        EncodeVop1(0x02, 22, Vgpr(1)), EncodeSop2(0x1e, 23, 22, InlineU32(5)),
        EncodeSmem0(0x0b, 32, 12), EncodeSmem1(0, 23), EncodeSopp(0x0c, 0),
        EncodeSop1(0x04, 40, InlineU32(0)), EncodeSop1(0x04, 42, InlineU32(0))};
    AppendSMovLiteral(&leaf_code, 41, 0x00fff000u);
    AppendVMovLiteral(&leaf_code, 10, std::bit_cast<u32>(0.5f));
    AppendVMovLiteral(&leaf_code, 11, std::bit_cast<u32>(0.5f));
    leaf_code.insert(leaf_code.end(), {
        EncodeMimg0(0x27, 1), EncodeMimg1(6, 10, 8, 10),
        EncodeVop1(0x01, 7, 64), EncodeSop1(0x21, 125, 14)});
    for (const auto &code : callers)
      ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
          {.type = Prospero::ShaderBinaryType::kCs,
           .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
    EnsureRuntimeContext();
    for (const u32 wave : {32u, 64u}) {
      for (const auto which : {Case::Disjoint, Case::OddFirst, Case::SourceOverlap,
                               Case::SourceAba, Case::TextureOverlap, Case::RetiredView,
                               Case::NullTexture, Case::NullToLive, Case::DccDisjoint,
                               Case::MetadataOverlap, Case::PlainDepth, Case::DepthStencilAssociation,
                               Case::AssociatedChildOverlap, Case::AssociatedHtileDisjoint,
                               Case::AssociatedHtileOverlap, Case::AssociationReparent,
                               Case::StandaloneStencilDisjoint, Case::StandaloneStencilOverlap}) {
        constexpr std::array labels{"Disjoint", "OddFirst", "SourceOverlap", "SourceAba",
                                    "TextureOverlap", "RetiredView", "Null", "NullToLive",
                                    "DccDisjoint", "MetadataOverlap", "PlainDepth", "DepthStencilAssociation",
                                    "AssociatedChildOverlap", "AssociatedHtileDisjoint", "AssociatedHtileOverlap",
                                    "AssociationReparent", "StandaloneStencilDisjoint", "StandaloneStencilOverlap"};
        const std::string name = std::string("TableCallHeap") + std::to_string(wave) +
            labels[static_cast<unsigned>(which)];
        const bool source_overlap = which == Case::SourceOverlap || which == Case::SourceAba;
        const bool sampled_overlap = which == Case::TextureOverlap;
        const bool null_texture = which == Case::NullTexture || which == Case::NullToLive;
        const bool odd_first = which == Case::OddFirst;
        const bool metadata_overlap = which == Case::MetadataOverlap;
        const bool dcc = which == Case::DccDisjoint || metadata_overlap;
        const bool child_overlap = which == Case::AssociatedChildOverlap;
        const bool htile_overlap = which == Case::AssociatedHtileOverlap;
        const bool htile = which == Case::AssociatedHtileDisjoint || htile_overlap;
        const bool reparent = which == Case::AssociationReparent;
        const bool standalone_overlap = which == Case::StandaloneStencilOverlap;
        const bool standalone_stencil = which == Case::StandaloneStencilDisjoint || standalone_overlap;
        const bool associated_depth = which == Case::DepthStencilAssociation || child_overlap || htile || reparent;
        const bool has_stencil = associated_depth || standalone_stencil;
        const bool depth = which == Case::PlainDepth || has_stencil;
        const bool positive = which == Case::Disjoint || odd_first || which == Case::NullTexture ||
            which == Case::DccDisjoint || which == Case::PlainDepth ||
            which == Case::DepthStencilAssociation || which == Case::AssociatedHtileDisjoint ||
            which == Case::StandaloneStencilDisjoint;
        const uint64_t dcc_texture = base + 0x80000;
        const uint64_t metadata_address = metadata_overlap ? context_a : base + 0xa0000;
        const uint64_t stencil_address = standalone_overlap ? context_a : base + 0x60000;
        const uint64_t depth_address = htile ? base + 0xc0000 : texture_a;
        const uint64_t htile_address = htile_overlap ? base : base + 0xe0000;
        const uint64_t image_table = source_overlap ? context_a + 16u : separate_table;
        auto image_a = image_descriptor(sampled_overlap ? context_a :
            null_texture ? base + size + 0x40000u : depth_address);
        if (dcc) {
          image_a = {static_cast<u32>(dcc_texture >> 8u),
              static_cast<u32>(dcc_texture >> 40u) | 0xc1600000u, 0x000fc00fu,
              0x91b00204u, 0, 0x00700000u,
              0x002b0000u | (static_cast<u32>(metadata_address >> 8u) << 24u),
              static_cast<u32>(metadata_address >> 16u)};
          ShaderTextureResource decoded{};
          std::copy(image_a.begin(), image_a.end(), decoded.fields);
          Require(name.c_str(), "DCC descriptor geometry", decoded.Width5() == 63u &&
              decoded.Height5() == 63u && decoded.Format() == Prospero::BufferFormat::k32Float &&
              decoded.TileMode() == Prospero::TileMode::kRenderTarget && decoded.MetaCompress() &&
              (decoded.MetaAddr() << 8u) == metadata_address && decoded.Base40() == dcc_texture,
              "DCC fixture lost its real texture or metadata layout");
        }
        TileSizeAlign depth_layout{}, stencil_layout{}, htile_layout{};
        if (htile) {
          Require(name.c_str(), "native HTile layout",
              TileGetDepthSize(4u, 4u, Prospero::DepthFormat::kZ32F,
                  Prospero::StencilFormat::k8UInt, true, stencil_layout, htile_layout, depth_layout) &&
              depth_layout.size != 0u && depth_layout.size <= 0x10000u &&
              stencil_layout.size != 0u && stencil_layout.size <= 0x10000u &&
              htile_layout.size != 0u && htile_layout.size <= 0x8000u &&
              htile_layout.align != 0u && (htile_address % htile_layout.align) == 0u,
              "HTile fixture exceeds its real aligned depth/stencil/metadata allocations");
          image_a[3] |= static_cast<u32>(Prospero::TileMode::kDepth) << 20u;
          image_a[6] = 0x00280000u | (static_cast<u32>(htile_address >> 8u) << 24u);
          image_a[7] = static_cast<u32>(htile_address >> 16u);
          ShaderTextureResource decoded{};
          std::copy(image_a.begin(), image_a.end(), decoded.fields);
          Require(name.c_str(), "native HTile descriptor",
              IsSupportedDepthTextureEncoding(decoded, false) &&
              (decoded.MetaAddr() << 8u) == htile_address && decoded.Base40() == depth_address,
              "sampled HTile descriptor is not a supported native depth encoding");
        }
        const auto image_b = image_descriptor(texture_b);
        std::array<u32, 12> payload{};
        const auto table_vsharp = descriptor(image_table, table_bytes);
        std::copy_n(table_vsharp.fields, 4u, payload.begin());
        std::copy(image_a.begin(), image_a.end(), payload.begin() + 4u);
        int64_t physical = -1;
        Require(name.c_str(), "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
            0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
            "heap-call fixture allocation failed");
        void *mapped = reinterpret_cast<void *>(base);
        Require(name.c_str(), "map", LibKernel::Memory::KernelMapDirectMemory(
            &mapped, size, 0x3, 0x10, physical, size) == 0 &&
            mapped == reinterpret_cast<void *>(base), "heap-call fixture mapping failed");
        std::memset(mapped, 0, size);
        std::memset(reinterpret_cast<void *>(output - 16u), 0xa5, wave * 16u + 32u);
        LibKernel::Memory::WriteBacking(leaf, leaf_code.data(), leaf_code.size() * sizeof(u32));
        for (const auto address : {context_a, context_b})
          LibKernel::Memory::WriteBacking(address, payload.data(), sizeof(payload));
        LibKernel::Memory::WriteBacking(image_table, image_a.data(), sizeof(image_a));
        LibKernel::Memory::WriteBacking(image_table + 32u, image_b.data(), sizeof(image_b));
        const std::array<uint64_t, 4> records{leaf, context_a, leaf, context_b};
        LibKernel::Memory::WriteBacking(call_table, records.data(), sizeof(records));
        std::vector<u32> pixels(texture_pitch * 4u);
        for (const auto &[address, value] : std::array<std::pair<uint64_t, float>, 2>{{
            {texture_a, depth ? 0.25f : 2.0f}, {texture_b, 9.0f}}}) {
          std::ranges::fill(pixels, std::bit_cast<u32>(value));
          LibKernel::Memory::WriteBacking(
              address, pixels.data(), pixels.size() * sizeof(u32));
        }
        if (dcc) {
          TileSizeAlign layout{};
          TileGetTextureTotalSize(Prospero::BufferFormat::k32Float, 64u, 64u, 1u, 1u,
              Prospero::TileMode::kRenderTarget, false, layout);
          Require(name.c_str(), "DCC native footprint", layout.size != 0 &&
              layout.size <= 0x10000u && (layout.size % sizeof(u32)) == 0u,
              "DCC fixture data exceeds its isolated allocation");
          std::vector<u32> tiled_pixels(layout.size / sizeof(u32), std::bit_cast<u32>(2.0f));
          LibKernel::Memory::WriteBacking(dcc_texture, tiled_pixels.data(), layout.size);
          if (!metadata_overlap) {
            const std::vector<u32> uncompressed_metadata(0x1000u / sizeof(u32), UINT32_MAX);
            LibKernel::Memory::WriteBacking(metadata_address, uncompressed_metadata.data(), 0x1000u);
          }
        }
        if (htile) {
          const std::vector<u32> depth_pixels(depth_layout.size / sizeof(u32), std::bit_cast<u32>(0.25f));
          LibKernel::Memory::WriteBacking(depth_address, depth_pixels.data(), depth_layout.size);
          if (!htile_overlap) {
            const std::vector<u32> metadata(htile_layout.size / sizeof(u32), UINT32_MAX);
            LibKernel::Memory::WriteBacking(htile_address, metadata.data(), htile_layout.size);
          }
        }
        struct ProtectedRange { uint64_t address; std::vector<u32> words; };
        std::vector<ProtectedRange> protected_ranges;
        size_t protected_bytes = 0;
        std::vector<std::pair<uint64_t, size_t>> guard_spans{
            {call_table - 16u, sizeof(records) + 32u},
            {leaf - 16u, leaf_code.size() * sizeof(u32) + 32u},
            {context_a - 16u, sizeof(payload) + 32u},
            {context_b - 16u, sizeof(payload) + 32u},
            {output - 16u, 16u}, {output + wave * 16u, 16u}};
        if (dcc) guard_spans.emplace_back(metadata_address, 0x1000u);
        if (htile) guard_spans.emplace_back(htile_address, htile_layout.size);
        for (const auto &[address, bytes] : guard_spans) {
          ProtectedRange range{address, std::vector<u32>(bytes / sizeof(u32))};
          Require(name.c_str(), "capture guest canaries", LibKernel::Memory::TryReadBacking(
              address, range.words.data(), bytes), "fixture capture failed");
          protected_ranges.push_back(std::move(range));
          protected_bytes += bytes;
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
            auto &heap = context.GetImageHeap();
            auto &scheduler = context.GetCommandScheduler();
            auto &executor = context.GetRenderExecutor();
            auto &pipelines = context.GetPipelineCache();
            Libs::Graphics::Buffer gpu_canaries(context.GetGraphics(), scheduler,
                MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, protected_bytes);
            for (const auto &range : protected_ranges)
              cache.PrepareBdaRange(range.address, range.words.size() * sizeof(u32));
            context.PrepareBda();
            const auto verify_canaries = [&] {
              const auto command = scheduler.Current().Handle();
              command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                  vk::PipelineStageFlagBits::eTransfer, {},
                  vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                                    .dstAccessMask = vk::AccessFlagBits::eTransferRead}, {}, {});
              uint64_t cursor = 0;
              for (const auto &range : protected_ranges) {
                const auto bytes = range.words.size() * sizeof(u32);
                vk::DescriptorBufferInfo native;
                Require(name.c_str(), "canonical GPU canary source",
                    cache.CanonicalBinding(range.address, bytes, native),
                    "protected bytes lost their canonical native owner");
                command.copyBuffer(native.buffer, gpu_canaries.Handle(),
                    vk::BufferCopy{native.offset, cursor, bytes});
                cursor += bytes;
              }
              command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                  vk::PipelineStageFlagBits::eHost, {},
                  vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                    .dstAccessMask = vk::AccessFlagBits::eHostRead}, {}, {});
              scheduler.NoteRecordedWork();
              scheduler.Finish();
              gpu_canaries.Invalidate(0, protected_bytes);
              cursor = 0;
              for (const auto &range : protected_ranges) {
                const auto bytes = range.words.size() * sizeof(u32);
                std::vector<u32> guest(range.words.size());
                Require(name.c_str(), "GPU closure and output canaries",
                    std::memcmp(gpu_canaries.Mapped().data() + cursor,
                                range.words.data(), bytes) == 0,
                    "native GPU code/context/table/output guards changed");
                Require(name.c_str(), "guest closure and output canaries",
                    LibKernel::Memory::TryReadBacking(range.address, guest.data(), bytes) &&
                        guest == range.words, "guest backing guards changed");
                cursor += bytes;
              }
            };
            const auto write_image = [&](const std::array<u32, 8> &words) {
              Require(name.c_str(), "source descriptor invalidation",
                  context.InvalidateMemory(image_table, sizeof(words)),
                  "fixture descriptor write escaped mapping");
              LibKernel::Memory::WriteBacking(image_table, words.data(), sizeof(words));
            };
            shaders.SetCsShader({
                .data_addr = reinterpret_cast<uint64_t>(callers[odd_first ? 1u : 0u].data()),
                .num_thread_x = wave, .num_thread_y = 1, .num_thread_z = 1,
                .wave_size = static_cast<uint8_t>(wave), .user_sgpr = 12});
            for (const auto &[reg, address, bytes] :
                 std::array<std::tuple<u32, uint64_t, u32>, 2>{{
                   {0, output, wave * 16u}, {8, call_table, sizeof(records)}}}) {
              const auto value = descriptor(address, bytes);
              for (u32 word = 0; word < 4u; ++word)
                shaders.SetCsUserSgpr(reg + word, value.fields[word]);
            }
            ImageId depth_owner{}, alternate_depth_owner{};
            if (depth) {
              auto native_depth = MakeLinearDesc(depth_address, 0,
                  has_stencil ? vk::Format::eD32SfloatS8Uint : vk::Format::eD32Sfloat,
                  Prospero::BufferFormat::k32Float, Prospero::ImageType::kColor2D,
                  {4, 4, 1}, 1, 4, 1);
              native_depth.info.pitch = 0;
              if (htile) native_depth.info.tile_mode = Prospero::TileMode::kDepth;
              native_depth.info.UpdateSize();
              native_depth.view_info.aspect = vk::ImageAspectFlagBits::eDepth;
              if (has_stencil)
                native_depth.info.stencil = {stencil_address, htile ? stencil_layout.size : 0x1000u};
              if (htile) {
                native_depth.type = TextureCache::BindingType::DepthTarget;
                native_depth.view_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
                native_depth.info.metadata.kind = ImageMetadataKind::Htile;
                native_depth.info.metadata.range = {htile_address, htile_layout.size};
                Require(name.c_str(), "native parent HTile footprint",
                    native_depth.info.data.size == depth_layout.size,
                    "HTile parent and real target layout disagree");
              }
              depth_owner = context.GetTextureCache().FindImage(native_depth);
              Require(name.c_str(), "native depth owner", static_cast<bool>(depth_owner),
                  "depth fixture failed to register its real native owner");
              if (associated_depth && !child_overlap)
                TextureCacheTestAccess::AssociateStencil(context.GetTextureCache(), depth_owner,
                    native_depth.info.stencil);
              if (reparent) {
                auto alternate = native_depth;
                alternate.info.data.address = base + 0xb0000;
                alternate_depth_owner = context.GetTextureCache().FindImage(alternate);
                Require(name.c_str(), "second native parent", alternate_depth_owner &&
                    alternate_depth_owner != depth_owner,
                    "reparent fixture did not create a separate registered depth owner");
              }
            }
            heap.BeginDraw();
            ShaderComputeInputInfo input{};
            for (auto &dimension : input.workgroup_counts) dimension = 1;
            const auto *module = pipelines.GetComputeProgram(shaders.GetCs(), input);
            if (module == nullptr) {
              Require(name.c_str(), "early image backing overlap refusal",
                  sampled_overlap && ImageHeapTestAccess::MemoryCount(heap) != 0,
                  "valid native heap call failed before prepared closure validation");
              verify_canaries();
              context.UnmapMemory(base, size);
              scheduler.Finish();
              return;
            }
            if (child_overlap)
              // A retained historical association differs from the parent's current stencil range.
              TextureCacheTestAccess::AssociateStencil(context.GetTextureCache(), depth_owner,
                  GuestRange{context_a, 0x1000u});
            Require(name.c_str(), "genuine captured heap call",
                input.stage.call_closure && !input.stage.call_closure->data.empty() &&
                    module->info.info.uses_image_heap && module->info.wave_size == wave,
                "native instructions did not materialize a captured image heap");
            const auto root = std::ranges::find_if(module->info.info.images,
                [](const auto &image) { return image.heap; });
            Require(name.c_str(), "read-only sampled heap",
                root != module->info.info.images.end() && root->read && !root->written &&
                    root->resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled,
                "fixture did not compile the sampled heap access class");
            auto *pipeline = pipelines.GetComputePipeline(input);
            Require(name.c_str(), "native compute pipeline", pipeline != nullptr,
                "heap-call compute pipeline was not created");
            PreparedBindings prepared;
            executor.PrepareBindings(input.stage, prepared);
            Require(name.c_str(), "prepare immutable data",
                RenderExecutorTestAccess::PrepareCallData(executor, prepared),
                "fixture call data was not prepared");
            auto *stage = &prepared;
            const auto stages = std::span{&stage, 1u};
            executor.FindBuffers(stages);
            if (module->info.info.uses_dma) context.PrepareBda();
            heap.Rebind(executor);
            executor.RebindImages(prepared);
            executor.RebindBuffers(prepared);
            const auto &native_view = *ImageHeapTestAccess::Views(heap).front();
            const auto heap_step = native_view.step, heap_first = native_view.first;
            const auto heap_count = native_view.count;
            Require(name.c_str(), "GPU row domain", heap_first == 0u && heap_step == 32u &&
                heap_count >= 2u && native_view.slots.size() > 8u,
                "native view does not contain both GPU-selected descriptor rows");
            for (u32 row = 0; row < 2u; ++row) {
              const auto &element = ImageHeapTestAccess::Element(heap, native_view.slots[row * 8u]);
              const auto *owner = TextureCacheTestAccess::Owner(
                  context.GetTextureCache(), element.binding.image_id);
              if (row == 0u && null_texture) {
                Require(name.c_str(), "actual unmapped null texture", !element.binding.image_id &&
                    element.binding.image_view == nullptr && element.serial == 0u &&
                    element.value.dwords == image_a,
                    "unmapped native descriptor did not resolve to a stable null binding");
              } else {
                const u32 dimension = row == 0u && dcc ? 64u : 4u;
                Require(name.c_str(), "native sampled texture layout", owner && owner->registered &&
                    element.binding.image_view != nullptr &&
                    element.binding.desc.info.extent == vk::Extent3D{dimension, dimension, 1} &&
                    owner->info.extent == vk::Extent3D{dimension, dimension, 1} &&
                    (row == 0u && htile ? owner->info.pitch == TileGetTexturePitch(
                        Prospero::BufferFormat::k32Float, 4u, Prospero::TileMode::kDepth) :
                        dimension != 4u || owner->info.pitch == texture_pitch),
                    "sampled descriptor did not retain its initialized native texture footprint");
              }
            }
            const auto closure_overlaps = [&](GuestRange range) {
              if (range.Empty()) return false;
              const auto overlaps = [&](const auto &block) {
                return ImageRangeOverlaps(range,
                    GuestRange{block.address, block.words.size() * sizeof(u32)});
              };
              const auto &closure = *input.stage.call_closure;
              return std::ranges::any_of(closure.code, overlaps) ||
                  std::ranges::any_of(closure.data, overlaps) ||
                  std::ranges::any_of(closure.table_ranges, [&](const auto &span) {
                    return ImageRangeOverlaps(range, GuestRange{span.first, span.second});
                  });
            };
            if (dcc || depth) {
              const auto &element = ImageHeapTestAccess::Element(heap,
                  native_view.slots[native_view.first / sizeof(u32)]);
              const auto *owner = TextureCacheTestAccess::Owner(
                  context.GetTextureCache(), element.binding.image_id);
              Require(name.c_str(), "materialized protected heap view", owner && owner->registered &&
                  element.binding.image_view != nullptr &&
                  !closure_overlaps(element.binding.desc.info.data) &&
                  !closure_overlaps(owner->info.data),
                  "metadata/association fixture needs a real disjoint sampled image backing");
              if (dcc) {
                const GuestRange metadata{metadata_address, 0x1000u};
                Require(name.c_str(), "actual DCC metadata relation",
                    element.binding.desc.info.metadata.kind == ImageMetadataKind::Dcc &&
                    element.binding.desc.info.metadata.range == metadata &&
                    owner->info.metadata.range == metadata &&
                    closure_overlaps(metadata) == metadata_overlap,
                    "DCC negative must overlap metadata only; its control must be disjoint");
              } else {
                Require(name.c_str(), "actual native depth association",
                    element.binding.image_id == depth_owner && owner->info.IsDepth() &&
                    !owner->depth_id && owner->info.HasStencil() == has_stencil &&
                    (!owner->stencil_ids.empty()) == associated_depth,
                    "heap element did not retain the intended depth owner and association");
                if (has_stencil) {
                  Require(name.c_str(), "native parent standalone stencil input",
                      owner->info.stencil == GuestRange{stencil_address, htile ? stencil_layout.size : 0x1000u} &&
                      closure_overlaps(owner->info.stencil) == standalone_overlap,
                      "native stencil input range lost its independently checked overlap relation");
                }
                if (htile) {
                  Require(name.c_str(), "real native parent HTile input",
                      owner->info.tile_mode == Prospero::TileMode::kDepth &&
                      owner->info.metadata.kind == ImageMetadataKind::Htile &&
                      owner->info.metadata.range == GuestRange{htile_address, htile_layout.size} &&
                      closure_overlaps(owner->info.metadata.range) == htile_overlap,
                      "associated parent metadata must have the real target footprint and intended overlap");
                }
                if (child_overlap) {
                  Require(name.c_str(), "historical child is the only overlapping image input",
                      !closure_overlaps(element.binding.desc.info.metadata.range) &&
                      !closure_overlaps(element.binding.desc.info.stencil) &&
                      !closure_overlaps(owner->info.metadata.range) &&
                      !closure_overlaps(owner->info.stencil),
                      "child-range negative was preempted by a requested or parent input overlap");
                }
                if (associated_depth) {
                  const auto child = owner->stencil_ids.front();
                  Require(name.c_str(), "registered stencil child",
                      context.GetTextureCache().GetImage(child).depth_id == depth_owner &&
                      closure_overlaps(context.GetTextureCache().GetImage(child).info.data) == child_overlap,
                      "associated child must be real with the intended immutable-input overlap");
                }
              }
            }
            const auto acquire_again = [&] {
              ShaderRecompiler::IR::ImageHeapView view{};
              Require(name.c_str(), "reacquire actual heap view",
                  heap.Acquire({image_table, table_bytes, heap_step, heap_first, heap_count, &*root}, view),
                  "native heap view was not available");
            };
            if (which == Case::SourceAba) {
              heap.BeginDraw();
              write_image(image_b);
              acquire_again();
              heap.Rebind(executor);
              executor.RebindImages(prepared);
              executor.RebindBuffers(prepared);
              executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute,
                  *pipeline, stages);
              scheduler.Finish();
              write_image(image_a);
              context.PrepareBda(); // Restore canonical GPU A without refreshing the heap view.
              // The private slot view still represents B after CPU bytes return to A.
              acquire_again();
              const auto &view = *ImageHeapTestAccess::Views(heap).front();
              std::array<u32, 8> current{};
              Require(name.c_str(), "real source-table ABA",
                  LibKernel::Memory::TryReadBacking(image_table, current.data(), sizeof(current)) &&
                      current == image_a &&
                      ImageHeapTestAccess::Element(heap, view.slots[0]).value.dwords == image_b,
                  "fixture did not retain B in the derived view after CPU backing returned to A");
              verify_canaries(); // Establish exact GPU A before the refused consumer.
            }
            ImageHeap::CallClosureProof graph_proof;
            std::string graph_reason;
            if (child_overlap || htile_overlap || standalone_overlap || reparent) {
              std::vector<GuestRange> graph_guards;
              for (const auto *ranges : {&input.stage.call_closure->code, &input.stage.call_closure->data})
                for (const auto &range : *ranges)
                  graph_guards.push_back({range.address, range.words.size() * sizeof(u32)});
              for (const auto &[address, bytes] : input.stage.call_closure->table_ranges)
                graph_guards.push_back({address, bytes});
              std::ranges::sort(graph_guards, {}, &GuestRange::address);
              size_t unique = 0;
              for (const auto range : graph_guards) {
                if (unique && range.address <= graph_guards[unique - 1].End()) {
                  auto &previous = graph_guards[unique - 1];
                  previous.size = std::max(previous.End(), range.End()) - previous.address;
                } else graph_guards[unique++] = range;
              }
              graph_guards.resize(unique);
              const bool captured = heap.CaptureCallClosure(module->info.info.images,
                  input.stage.resources->flattened_srt, graph_guards, graph_proof, graph_reason);
              Require(name.c_str(), "real association graph capture", captured == reparent &&
                  (reparent || graph_reason.find("overlaps protected call inputs") != std::string::npos),
                  "graph proof did not isolate the actual associated input range");
            }
            auto proof = RenderExecutorTestAccess::NewCallClosureCheck();
            const auto validate = [&](bool committed) {
              return RenderExecutorTestAccess::ValidateCallClosure(
                  executor, input.stage, prepared, proof, committed);
            };
            const bool before = validate(false);
            if (source_overlap || sampled_overlap || metadata_overlap || child_overlap ||
                htile_overlap || standalone_overlap) {
              Require(name.c_str(), "overlap refused before consumer", !before,
                  "heap source, image metadata, or association bypassed closure validation");
            } else {
              Require(name.c_str(), "disjoint sampled heap admitted", before,
                  "genuine disjoint sampled heap was rejected before native sampling");
              executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute,
                  *pipeline, stages);
              Require(name.c_str(), "proof survives Commit active-view clearing", validate(true),
                  "committed heap identities lost their immutable-closure proof");
              if (reparent) {
                Require(name.c_str(), "committed association graph proof",
                    heap.ValidateCallClosure(module->info.info.images,
                        input.stage.resources->flattened_srt, graph_proof, graph_reason),
                    "association graph proof did not survive Commit");
                auto &textures = context.GetTextureCache();
                const auto child = textures.GetImage(depth_owner).stencil_ids.front();
                const auto child_serial = textures.GetImage(child).serial;
                const auto owner_serial = textures.GetImage(depth_owner).serial;
                const auto view_serial = native_view.serial;
                const auto draw = ImageHeapTestAccess::Draw(heap);
                const auto mapping = LibKernel::Memory::MappingGeneration();
                const auto protection = LibKernel::Memory::ProtectionGeneration();
                TextureCacheTestAccess::AssociateStencil(textures, alternate_depth_owner,
                    textures.GetImage(child).info.data);
                const auto &old_parent = textures.GetImage(depth_owner);
                const auto &new_parent = textures.GetImage(alternate_depth_owner);
                std::array<u32, 8> source{};
                Require(name.c_str(), "same-draw real reassociation",
                    old_parent.serial == owner_serial && old_parent.stencil_ids.empty() &&
                    textures.GetImage(child).serial == child_serial &&
                    textures.GetImage(child).depth_id == alternate_depth_owner &&
                    std::ranges::find(new_parent.stencil_ids, child) != new_parent.stencil_ids.end() &&
                    native_view.serial == view_serial && ImageHeapTestAccess::Draw(heap) == draw &&
                    LibKernel::Memory::MappingGeneration() == mapping &&
                    LibKernel::Memory::ProtectionGeneration() == protection &&
                    LibKernel::Memory::TryReadBacking(image_table, source.data(), sizeof(source)) &&
                    source == image_a,
                    "fixture changed an unrelated draw, mapping, source, or native-view identity");
                Require(name.c_str(), "reassociated child invalidates exact graph proof",
                    !heap.ValidateCallClosure(module->info.info.images,
                        input.stage.resources->flattened_srt, graph_proof, graph_reason) &&
                    graph_reason.find("association graph changed") != std::string::npos && !validate(true),
                    "a changed parent-child edge reused the committed proof");
              } else if (which == Case::RetiredView) {
                heap.UnmapMemory(image_table, table_bytes);
                Require(name.c_str(), "retired committed heap view refused", !validate(true),
                    "stale committed heap view was accepted after its owner was removed");
              } else if (which == Case::NullToLive) {
                heap.BeginDraw();
                write_image(image_descriptor(texture_a));
                acquire_again();
                heap.Rebind(executor);
                const auto &view = *ImageHeapTestAccess::Views(heap).front();
                Require(name.c_str(), "null became a materialized image",
                    static_cast<bool>(ImageHeapTestAccess::ImageOf(heap, view.slots[0])),
                    "null-to-live fixture did not materialize its native texture");
                Require(name.c_str(), "changed committed null identity refused", !validate(true),
                    "committed null binding silently became a different live texture");
              } else {
                RenderExecutorTestAccess::WriteCallData(
                    executor, prepared, proof, scheduler.Current());
                const auto command = scheduler.Current().Handle();
                command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
                command.dispatch(1, 1, 1);
                scheduler.NoteRecordedWork();
              }
            }
            verify_canaries();
            std::vector<u32> actual(wave * 4u + 4u), expected(actual.size(), sentinel);
            cache.ReadMemory(output, actual.size() * sizeof(u32));
            if (positive) {
              for (u32 lane = odd_first ? 1u : 0u; lane < wave; ++lane) {
                expected[lane * 4u] = null_texture ? 0u :
                    std::bit_cast<u32>(odd_first ? 9.0f : depth ? 0.25f : 2.0f);
                expected[lane * 4u + 1u] = image_a[0];
              }
            }
            Require(name.c_str(), positive ? "native selected texture and inactive canaries" :
                                             "refused consumer leaves output unchanged",
                LibKernel::Memory::TryReadBacking(
                    output, actual.data(), actual.size() * sizeof(u32)) && actual == expected,
                "native image sample selected the wrong texture or a refused dispatch executed");
            RenderExecutorTestAccess::ResetBindings(executor);
            scheduler.Finish();
            context.UnmapMemory(base, size);
            scheduler.Finish();
          });
          LibKernel::Memory::InstallGpuResources(nullptr);
          context.ShutdownGpu();
        }
        Require(name.c_str(), "release", LibKernel::Memory::KernelMunmap(base, size) == 0 &&
            LibKernel::Memory::KernelReleaseDirectMemory(physical, size) == 0,
            "heap-call fixture release failed");
        std::printf("[gpu]     %-32s ok\n", name.c_str());
      }
    }
  }

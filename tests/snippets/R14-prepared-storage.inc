// === Delta for multicall-shadow tests/ShaderTableCallNativeTests.inc (base 91c32883...) ===
// 0) Add inside the existing `struct RenderExecutorTestAccess { ... };`:
//      static const PreparedBindings &ComputeBindings(const RenderExecutor &executor) {
//        return executor.m_compute_bindings;
//      }
// 1) Add inside the existing `struct PipelineCacheTestAccess { ... };`:
//      static uint64_t CallFailures(const PipelineCache &cache) {
//        uint64_t total = 0;
//        for (const auto &[hash, count] : cache.m_call_failures) total += count;
//        return total;
//      }
// 2) Replace the whole current "R14 written storage image ... fails closed" block (the `{ ... }`
//    after "store-free program recovers") with the block below. image_code, bind_code(.., 32) and the
//    T# in s[24:31] stay exactly as in the base (Menu's correction).

      // Prepared-storage rounds: written storage images are validated AFTER native preparation against the exact
      // requested range and the final native owner's full backing. A rejected dispatch keeps its
      // bound module but records no guest dispatch (outputs stay sentinel) and counts a failure.
      {
        auto &pipelines = context.GetPipelineCache();
        auto &textures = context.GetTextureCache();
        const auto set_sharp = [&](uint64_t address) {
          ShaderTextureResource sharp{};
          sharp.fields[0] = static_cast<u32>(address >> 8u);
          sharp.fields[1] = static_cast<u32>(Prospero::BufferFormat::k32UInt) << 20u; // 1x1
          sharp.fields[3] = DstSel(4, 5, 6, 7) |
                            (static_cast<u32>(Prospero::ImageType::kColor2D) << 28u);
          for (u32 i = 0; i < 8; ++i) shaders.SetCsUserSgpr(24 + i, sharp.fields[i]);
        };
        const auto rejected = [&](const char *round) {
          const auto failures = PipelineCacheTestAccess::CallFailures(pipelines);
          bind_code(image_code, 32);
          (void)run(round, true, sentinel, sentinel, {});
          Require(name, round, PipelineCacheTestAccess::CallFailures(pipelines) == failures + 1,
                  std::string(round) + ": the prepared closure did not reject exactly this dispatch");
        };
        // Code-overlap owners left by a rejected round are retired through the texture cache's
        // normal unmap path after GPU completion (the strict currentness check rightly refuses any
        // registered image over callee code). Guest memory and the buffer cache stay mapped.
        const auto retire_code_owners = [&](const char *round, uint64_t first, uint64_t bytes) {
          context.GetCommandScheduler().Finish();
          textures.UnmapMemory(first, bytes);
          context.GetCommandScheduler().Finish();
          Require(name, round, !textures.IsRegionRegistered(leaf_a, 12),
                  std::string(round) + ": an image owner still covers the callee code");
        };
        const auto recover = [&](const char *round) {
          const auto failures = PipelineCacheTestAccess::CallFailures(pipelines);
          bind_code(code);
          (void)run(round, true, value_a, value_b, {});
          Require(name, round, PipelineCacheTestAccess::CallFailures(pipelines) == failures,
                  std::string(round) + ": recovery recorded a failure (callee code changed?)");
        };
        // prepared-storage disjoint image: runs; exact call outputs and exact written texel.
        constexpr auto image_address = base + 0x5000;
        set_sharp(image_address);
        bind_code(image_code, 32);
        (void)run("prepared-storage disjoint written image runs", true, value_a, value_b, {});
        const auto image = textures.FindImageFromRange(image_address, sizeof(u32), false);
        Require(name, "written texel", image && ReadCachedTexel(name, context, image) == std::vector<u32>{value_a},
                "lane 0's IMAGE_STORE of its callee value was not written exactly");
        recover("prepared-storage recovery after disjoint image");
        // prepared-storage code overlap: the requested image range covers callee code -> rejected
        // after preparation. The callee's original bytes must be untouched (no host preparation
        // writeback may hide the reason), and recovery must link the same code.
        const auto read_leaf = [&] {
          std::array<u32, 3> words{};
          Require(name, "read callee code", LibKernel::Memory::TryReadBacking(leaf_a, words.data(), sizeof(words)),
                  "callee code was not readable");
          return words;
        };
        const auto leaf_before = read_leaf();
        set_sharp(leaf_a);
        rejected("prepared-storage image over callee code rejected");
        Require(name, "callee code untouched", read_leaf() == leaf_before,
                "preparation of the rejected dispatch changed the callee's code bytes");
        retire_code_owners("prepared-storage retire code-overlap owner", leaf_a, 0x1000);
        Require(name, "callee code untouched after cleanup", read_leaf() == leaf_before,
                "retiring the code-overlap owner changed the callee's code bytes");
        recover("prepared-storage recovery after code overlap");
        Require(name, "callee code untouched after recovery", read_leaf() == leaf_before,
                "callee code changed across rejection and recovery");
        // prepared-storage code alias: callee code page aliased at another VA (image disjoint) -> rejected.
        set_sharp(image_address);
        {
          void *alias = reinterpret_cast<void *>(alias_va);
          Require(name, "code alias map", LibKernel::Memory::KernelMapDirectMemory(
              &alias, 0x4000, 0x3, 0x10, physical, 0x4000) == 0, "code alias mapping failed");
          const auto failures = PipelineCacheTestAccess::CallFailures(pipelines);
          bind_code(image_code, 32);
          (void)run("prepared-storage aliased callee code rejected", false, sentinel, sentinel, {});
          Require(name, "prepared-storage aliased callee code rejected",
                  PipelineCacheTestAccess::CallFailures(pipelines) == failures + 2,
                  "the early alias guard did not reject both program lookups (run + DispatchDirect)");
          Require(name, "code alias unmap", LibKernel::Memory::KernelMunmap(alias_va, 0x4000) == 0,
                  "code alias unmap failed");
        }
        recover("prepared-storage recovery after code alias");
        // prepared-storage larger owner (resolver-supported mip-subset view). Owner and request are
        // both decoded from real T#s (same address, extent 256x4, linear R32, kColor2D); only the
        // level count differs. Guest linear layout (TileGetTextureSize): pitch 64-element aligned,
        // smaller mips first, mip 0 last, total 256-aligned -> owner (2 levels) = 1024 + 4096 =
        // 5120 bytes covers leaf_a (base+0x1000); request (1 level) = 4096 bytes ends at leaf_a.
        // TextureCache::ResolveOverlap's equal-address/equal-block-extent branch then returns the
        // cached owner for the smaller request. Every premise is asserted from actual state.
        const auto set_sharp_levels = [&](uint64_t address, u32 levels) {
          ShaderTextureResource sharp{};
          sharp.fields[0] = static_cast<u32>(address >> 8u);
          sharp.fields[1] = (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 20u) |
                            ((255u & 3u) << 30u);                       // width-1 = 255 (low bits)
          sharp.fields[2] = (255u >> 2u) | (3u << 14u);                 // width-1 high bits, height-1 = 3
          sharp.fields[3] = DstSel(4, 5, 6, 7) | ((levels - 1u) << 16u) |  // base level 0, last level
                            (static_cast<u32>(Prospero::ImageType::kColor2D) << 28u);  // tile 0 = linear
          sharp.fields[5] = (levels - 1u) << 4u;                        // max mip
          for (u32 i = 0; i < 8; ++i) shaders.SetCsUserSgpr(24 + i, sharp.fields[i]);  // image_code srsrc 6
        };
        const auto &prepared = RenderExecutorTestAccess::ComputeBindings(context.GetRenderExecutor());
        const auto leaf_before_owner = read_leaf();
        // Seed: the 2-level request creates the native owner through normal preparation; it covers
        // the callee code, so this dispatch is rejected too.
        set_sharp_levels(base, 2);
        rejected("prepared-storage seed larger owner rejected");
        Require(name, "seeded owner prepared", !prepared.images.empty() && prepared.images[0].image_id,
                "the 2-level storage request was not prepared");
        const auto big_id = prepared.images[0].image_id;
        {
          const auto &big_owner = textures.GetImage(big_id);
          Require(name, "larger owner covers code",
                  big_owner.registered && big_owner.info.resources.levels == 2 &&
                      big_owner.info.data.address == base &&
                      leaf_a + 12u <= big_owner.info.data.address + big_owner.info.data.size,
                  "the seeded 2-level owner does not extend over the callee code");
        }
        Require(name, "callee code untouched by seeding", read_leaf() == leaf_before_owner,
                "seeding the larger owner changed the callee's code bytes");
        // Request: the 1-level view must resolve to THAT owner while its own requested range ends
        // before the callee code; the owner range must reject it.
        set_sharp_levels(base, 1);
        rejected("prepared-storage larger owner over callee code rejected");
        Require(name, "request resolved to the larger owner",
                !prepared.images.empty() && prepared.images[0].image_id == big_id,
                "the 1-level storage request did not resolve to the cached 2-level owner");
        {
          const auto &requested = prepared.images[0].desc.info.data;
          Require(name, "request alone is disjoint",
                  requested.address == base && requested.size != 0 && requested.address + requested.size <= leaf_a,
                  "the requested storage range itself overlaps the callee code");
          const auto &owner = textures.GetImage(big_id).info.data;
          Require(name, "selected owner overlaps code",
                  owner.address <= leaf_a && leaf_a < owner.address + owner.size,
                  "the selected owner no longer covers the callee code");
        }
        Require(name, "callee code untouched by larger owner", read_leaf() == leaf_before_owner,
                "preparation against the larger owner changed the callee's code bytes");
        retire_code_owners("prepared-storage retire larger owner", base, 0x2000);
        Require(name, "callee code untouched after larger-owner cleanup", read_leaf() == leaf_before_owner,
                "retiring the larger owner changed the callee's code bytes");
        set_sharp(image_address);
        recover("prepared-storage recovery after larger owner");
      }

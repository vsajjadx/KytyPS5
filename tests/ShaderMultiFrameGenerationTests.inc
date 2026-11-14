  // Included inside VulkanHarness; the image oracle itself has no renderer dependency.
  void CheckMultiFrameGeneration() {
    constexpr const char *name = "MultiFrameGeneration";
    constexpr auto usage = vk::ImageUsageFlagBits::eStorage |
                           vk::ImageUsageFlagBits::eSampled;
    const char *dump_env = std::getenv("KYTY_MULTIFG_ORACLE_DUMP_DIR");
    const bool dump_requested = dump_env && *dump_env;
    auto &graphics = RuntimeContext();
    CommandScheduler scheduler(Renderer(), graphics);
    const auto general = [&](Image &image) {
      if (image.layout != vk::ImageLayout::eGeneral)
        TransitionImage(name, &image, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    };
    const auto submit = [&] {
      const auto tick = scheduler.Submit();
      Require(name, "tick", tick != 0 && scheduler.CurrentTick() > tick,
              "recording tick did not advance on the explicit submission");
      scheduler.Wait(tick); // Already submitted tick; never Wait(CurrentTick()).
      Require(name, "completion", scheduler.IsFree(tick), "submission not complete after Wait");
      return tick;
    };
    // Also exercise row pitch/bufferOffset and untouched host-buffer gaps. The source
    // image is larger than the logical output, so its own unwritten border is checked.
    const auto read_strided = [&](Image &image) {
      constexpr u32 prefix = 16, tail = 23;
      const u32 stride = image.width + 7;
      const size_t words = prefix + size_t(stride) * (image.height + 3) + tail;
      std::vector<u32> initial(words, MultiFgOracle::Canary);
      auto buffer = CreateHostBuffer(name, words * sizeof(u32),
          vk::BufferUsageFlagBits::eTransferDst, initial);
      auto command = BeginCommands(name, "strided readback");
      AddImageBarrier(command, image.image, image.layout,
          vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eAllCommands,
          vk::PipelineStageFlagBits::eTransfer,
          vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
          vk::AccessFlagBits::eTransferRead);
      vk::BufferImageCopy copy {};
      copy.bufferOffset = prefix * sizeof(u32);
      copy.bufferRowLength = stride;
      copy.bufferImageHeight = image.height + 3;
      copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
      copy.imageExtent = vk::Extent3D {image.width, image.height, 1};
      command.copyImageToBuffer(image.image, vk::ImageLayout::eTransferSrcOptimal,
          buffer.buffer, 1, &copy);
      EndSubmitAndFree(name, "strided readback", command);
      image.layout = vk::ImageLayout::eTransferSrcOptimal;
      auto raw = ReadBuffer(name, buffer, words);
      DestroyBuffer(&buffer);
      for (size_t index = 0; index < words; ++index) {
        const bool copied = index >= prefix && index < prefix + size_t(stride) * image.height &&
            (index - prefix) % stride < image.width;
        Require(name, "copy bounds", copied || raw[index] == MultiFgOracle::Canary,
                "image copy overwrote row padding/prefix/tail");
      }
      std::vector<u32> packed(size_t(image.width) * image.height);
      for (u32 y = 0; y < image.height; ++y)
        std::copy_n(raw.data() + prefix + size_t(y) * stride, image.width,
                    packed.data() + size_t(y) * image.width);
      return packed;
    };

    for (u32 factor = 2; factor <= 6; ++factor) {
      for (u32 variant = 0; variant < 2; ++variant) {
        // Non-multiple dimensions and opposite two-dimensional motion are independent
        // controls against aperture-only matching or a hardcoded one-direction warp.
        MultiFgOracle::Scene scene {variant ? 257u : 256u, variant ? 131u : 128u,
            variant, variant ? -12.0 : 12.0, variant ? 6.0 : -6.0, 11};
        std::vector<Image> inputs;
        std::array<Image, 5> outputs {};
        std::array<vk::Image, 5> handles {};
        for (u32 frame = 0; frame <= 12; ++frame)
          inputs.push_back(CreateImage2D(name, scene.width, scene.height,
              vk::Format::eR8G8B8A8Unorm, usage,
              MultiFgOracle::Frame(scene, frame), 1, vk::ImageLayout::eGeneral));
        auto resized_scene = scene;
        ++resized_scene.width; ++resized_scene.height;
        auto resized_input = CreateImage2D(name, resized_scene.width, resized_scene.height,
            vk::Format::eR8G8B8A8Unorm, usage, MultiFgOracle::Frame(resized_scene, 13),
            1, vk::ImageLayout::eGeneral);
        const u32 physical_width = scene.width + 19, physical_height = scene.height + 7;
        const std::vector<u32> canaries(size_t(physical_width) * physical_height,
                                      MultiFgOracle::Canary);
        for (u32 index = 0; index < factor - 1; ++index) {
          outputs[index] = CreateImage2D(name, physical_width, physical_height,
              vk::Format::eR8G8B8A8Unorm, usage, canaries, 1, vk::ImageLayout::eGeneral);
          handles[index] = outputs[index].image;
        }
        const uint64_t flow_bytes = uint64_t((scene.width + 7) / 8) * ((scene.height + 7) / 8) * 4;
        const uint64_t image_bytes = uint64_t(scene.width) * scene.height * 4;
        std::array<Buffer, 4> diagnostic_buffers {};
        MultiFrameDiagnostics diagnostics {};
        if (dump_requested) {
          const std::array<uint64_t, 4> sizes {flow_bytes, 8, image_bytes, image_bytes};
          for (u32 index = 0; index < diagnostic_buffers.size(); ++index)
            diagnostic_buffers[index] = CreateHostBuffer(name, sizes[index],
                vk::BufferUsageFlagBits::eTransferDst, {});
          diagnostics.optical_flow = {diagnostic_buffers[0].buffer, flow_bytes};
          diagnostics.scene_change = {diagnostic_buffers[1].buffer, 8};
          diagnostics.previous = {diagnostic_buffers[2].buffer, image_bytes};
          diagnostics.current = {diagnostic_buffers[3].buffer, image_bytes};
        }
        // Inputs/outputs outlive the generator and its imported SDK views.
        {
          FrameGenerator generator(graphics, scheduler);
          const auto run = [&](vk::Image input, std::span<const vk::Image> targets,
                               u32 width, u32 height, u32 requested, bool pace = true,
                               const MultiFrameDiagnostics *capture = nullptr) {
            if (pace) std::this_thread::sleep_for(std::chrono::milliseconds(17));
            auto command = scheduler.BeginCommand().Handle();
            const auto before_tick = scheduler.CurrentTick();
            const auto result = generator.GenerateMultiple(command, input, targets,
                                                            width, height, requested, capture);
            Require(name, "record only", scheduler.CurrentTick() == before_tick,
                    "GenerateMultiple submitted or advanced the scheduler tick");
            submit();
            return result;
          };
          const auto span = std::span<const vk::Image>(handles.data(), factor - 1);
          for (u32 frame = 0; frame <= 12; ++frame) {
            for (u32 index = 0; index < factor - 1; ++index) general(outputs[index]);
            // Exercise validation before history advance immediately before the
            // pair used by the numerical oracle. Failed calls must not consume B.
            if (frame == 12) {
              std::array<std::vector<u32>, 5> before;
              for (u32 index = 0; index < factor - 1; ++index) {
                before[index] = ReadImage(name, &outputs[index]); general(outputs[index]);
              }
              const auto reject = [&](vk::Image input, std::span<const vk::Image> targets,
                                      u32 requested) {
                const auto r = run(input, targets, scene.width, scene.height, requested, false);
                Require(name, "rejected input", r.status == MultiFrameStatus::Failed &&
                    r.count == 0 && r.written_mask == 0,
                    "invalid input published output or failed to reject");
              };
              reject(inputs[frame].image, span, 1);
              reject(inputs[frame].image, span, 7);
              reject(inputs[frame].image, span.first(span.size() - 1), factor);
              reject(nullptr, span, factor);
              auto invalid = handles;
              invalid[0] = nullptr;
              reject(inputs[frame].image, {invalid.data(), factor - 1}, factor);
              invalid[0] = inputs[frame].image;
              reject(inputs[frame].image, {invalid.data(), factor - 1}, factor);
              if (factor > 2) {
                invalid = handles; invalid[1] = invalid[0];
                reject(inputs[frame].image, {invalid.data(), factor - 1}, factor);
              }
              for (u32 index = 0; index < factor - 1; ++index) {
                Require(name, "failed output preservation", ReadImage(name, &outputs[index]) == before[index],
                        "invalid call modified an external output");
                general(outputs[index]);
              }
            }
            const auto result = run(inputs[frame].image, span,
                                    scene.width, scene.height, factor, true,
                                    frame == 12 && dump_requested ? &diagnostics : nullptr);
            if (frame < 6) {
              Require(name, "seed/warmup", result.status == MultiFrameStatus::Seeded &&
                  result.count == 0 && result.written_mask == 0,
                  "first input did not seed without publication");
              for (u32 index = 0; index < factor - 1; ++index) {
                Require(name, "seed output preservation", ReadImage(name, &outputs[index]) == canaries,
                        "seed modified an external output");
                general(outputs[index]);
              }
            } else {
              Require(name, "generated batch", result.status == MultiFrameStatus::Generated &&
                  result.count == factor - 1 && result.written_mask == (1u << (factor - 1)) - 1u,
                  "not every requested intermediate was published");
              for (u32 index = 0; index < factor - 1; ++index)
                outputs[index].layout = vk::ImageLayout::eTransferSrcOptimal;
            }
          }
          Require(name, "scene status", !generator.MultipleSceneChanged(),
                  "last batch is a scene fallback or its GPU scene status is unknown");
          std::array<std::vector<u32>, 4> diagnostic_words;
          if (dump_requested) {
            diagnostic_words[0] = ReadBuffer(name, diagnostic_buffers[0], flow_bytes / 4);
            diagnostic_words[1] = ReadBuffer(name, diagnostic_buffers[1], 2);
            diagnostic_words[2] = ReadBuffer(name, diagnostic_buffers[2], image_bytes / 4);
            diagnostic_words[3] = ReadBuffer(name, diagnostic_buffers[3], image_bytes / 4);
          }
          for (u32 index = 0; index < factor - 1; ++index) {
            const auto pixels = read_strided(outputs[index]);
            const MultiFgOracle::View physical {pixels, physical_width, physical_height, physical_width};
            Require(name, "output bounds", MultiFgOracle::OutsideLogicalUntouched(
                physical, scene.width, scene.height), "warp wrote beyond logical output extent");
            const MultiFgOracle::View logical {pixels, scene.width, scene.height, physical_width};
            const auto quality = MultiFgOracle::Position(logical, scene, factor, index + 1);
            std::printf("MULTIFG_ORACLE factor=%u variant=%u ordinal=%u mae=%.4f rmse=%.4f alt_mse=%.4f blend_mse=%.4f okay=%u\n",
                factor, variant, index + 1, quality.mae_bytes, quality.rmse_bytes,
                quality.alternate_mse, quality.blend_mse, quality.okay ? 1u : 0u);
            const auto dump_directory = MultiFgDiagnostic::EventDirectory(factor, variant, index + 1);
            if (!dump_directory.empty()) {
              const auto previous_pixels = ReadImage(name, &inputs[scene.previous_frame]);
              const auto current_pixels = ReadImage(name, &inputs[scene.previous_frame + 1]);
              MultiFgDiagnostic::Save(dump_directory,
                  {previous_pixels, scene.width, scene.height, scene.width},
                  {current_pixels, scene.width, scene.height, scene.width},
                  logical, scene, factor, index + 1, quality);
              MultiFgDiagnostic::SaveFlow(dump_directory, diagnostic_words[0], diagnostic_words[1],
                  {diagnostic_words[2], scene.width, scene.height, scene.width},
                  {diagnostic_words[3], scene.width, scene.height, scene.width},
                  {previous_pixels, scene.width, scene.height, scene.width},
                  {current_pixels, scene.width, scene.height, scene.width}, scene);
              general(inputs[scene.previous_frame]);
              general(inputs[scene.previous_frame + 1]);
            }
            Require(name, "temporal oracle", quality.okay, quality.reason.c_str());
            general(outputs[index]);
          }
          Require(name, "input preservation", ReadImage(name, &inputs.back()) ==
              MultiFgOracle::Frame(scene, 12), "generation modified a real input");
          general(inputs.back());
          generator.Reset();
          std::array<std::vector<u32>, 5> before_reset;
          for (u32 index = 0; index < factor - 1; ++index) {
            before_reset[index] = ReadImage(name, &outputs[index]); general(outputs[index]);
          }
          const auto reset = run(inputs.back().image, span, scene.width, scene.height, factor);
          Require(name, "reset seed", reset.status == MultiFrameStatus::Seeded &&
              reset.count == 0 && reset.written_mask == 0,
              "Reset did not seed/preserve external output");
          for (u32 index = 0; index < factor - 1; ++index) {
            Require(name, "reset preservation", ReadImage(name, &outputs[index]) == before_reset[index],
                    "Reset seeding modified external output");
            general(outputs[index]);
          }
          const auto resize = run(resized_input.image, span,
                                  resized_scene.width, resized_scene.height, factor);
          Require(name, "resize seed", resize.status == MultiFrameStatus::Seeded &&
              resize.count == 0 && resize.written_mask == 0, "resize did not reseed");
          for (u32 index = 0; index < factor - 1; ++index)
            Require(name, "resize preservation", ReadImage(name, &outputs[index]) == before_reset[index],
                    "resize seeding modified external output");
        }
        for (auto &buffer : diagnostic_buffers) DestroyBuffer(&buffer);
        for (auto &output : outputs) DestroyImage(&output);
        DestroyImage(&resized_input);
        for (auto &input : inputs) DestroyImage(&input);
      }
    }
  }

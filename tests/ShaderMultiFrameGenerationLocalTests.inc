  // Included in VulkanHarness. This supplements, rather than edits, the original oracle.
  void CheckMultiFrameGenerationLocal() {
    namespace Local = MultiFgLocalOracle;
    constexpr const char *name = "MultiFrameGenerationLocal";
    const char *dump_env = std::getenv("KYTY_MULTIFG_LOCAL_DUMP_DIR");
    const bool dump_requested = dump_env && *dump_env;
    const char *postfailure_env = std::getenv("KYTY_MULTIFG_SCD_FAILURE_DIR");
    const bool postfailure_requested = postfailure_env && *postfailure_env;
    auto &graphics = RuntimeContext();
    CommandScheduler scheduler(Renderer(), graphics);
    constexpr auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
    const auto general = [&](Image &image) {
      if (image.layout != vk::ImageLayout::eGeneral)
        TransitionImage(name, &image, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    };
    for (u32 factor = 2; factor <= 6; ++factor) {
      for (const auto kind : {Local::Kind::OppositePanels, Local::Kind::Occlusion}) {
        const Local::Scene scene {kind};
        std::vector<Image> inputs;
        std::array<Image, 5> outputs {};
        std::array<vk::Image, 5> handles {};
        for (u32 frame = 0; frame <= 12; ++frame)
          inputs.push_back(CreateImage2D(name, scene.Width(), scene.Height(),
              vk::Format::eR8G8B8A8Unorm, usage, Local::Frame(scene, frame),
              1, vk::ImageLayout::eGeneral));
        const std::vector<u32> initial(size_t(scene.Width()) * scene.Height(), MultiFgOracle::Canary);
        for (u32 index = 0; index < factor - 1; ++index) {
          outputs[index] = CreateImage2D(name, scene.Width(), scene.Height(),
              vk::Format::eR8G8B8A8Unorm, usage, initial, 1, vk::ImageLayout::eGeneral);
          handles[index] = outputs[index].image;
        }
        const uint64_t flow_bytes = uint64_t((scene.Width() + 7) / 8) * ((scene.Height() + 7) / 8) * 4;
        const uint64_t image_bytes = uint64_t(scene.Width()) * scene.Height() * 4;
        std::array<Buffer, 5> capture_buffers {};
        MultiFrameDiagnostics diagnostics {};
        if (dump_requested) {
          const std::array<uint64_t, 5> capacities {flow_bytes, 8, image_bytes, image_bytes, flow_bytes * 4};
          for (u32 index = 0; index < capture_buffers.size(); ++index)
            capture_buffers[index] = CreateHostBuffer(name, capacities[index],
                vk::BufferUsageFlagBits::eTransferDst, {});
          diagnostics.optical_flow = {capture_buffers[0].buffer, flow_bytes};
          diagnostics.scene_change = {capture_buffers[1].buffer, 8};
          diagnostics.previous = {capture_buffers[2].buffer, image_bytes};
          diagnostics.current = {capture_buffers[3].buffer, image_bytes};
          diagnostics.refined_flow = {capture_buffers[4].buffer, flow_bytes * 4};
        }
        // SDK views and every real/output image remain alive until this scope ends.
        {
          FrameGenerator generator(graphics, scheduler);
          const auto targets = std::span<const vk::Image>(handles.data(), factor - 1);
          for (u32 frame = 0; frame <= 12; ++frame) {
            for (u32 index = 0; index < factor - 1; ++index) general(outputs[index]);
            std::this_thread::sleep_for(std::chrono::milliseconds(17));
            auto command = scheduler.BeginCommand().Handle();
            const auto before_tick = scheduler.CurrentTick();
            const auto result = generator.GenerateMultiple(command, inputs[frame].image,
                targets, scene.Width(), scene.Height(), factor,
                frame == 12 && dump_requested ? &diagnostics : nullptr);
            Require(name, "record only", scheduler.CurrentTick() == before_tick,
                    "generation advanced the scheduler tick");
            const auto tick = scheduler.Submit();
            scheduler.Wait(tick);
            Require(name, "completed", tick == before_tick && scheduler.IsFree(tick),
                    "explicit generation submit did not complete");
            if (frame < 6) {
              Require(name, "seed", result.status == MultiFrameStatus::Seeded &&
                  result.count == 0 && result.written_mask == 0, "initial OF warmup published output");
            } else {
              Require(name, "full output batch", result.status == MultiFrameStatus::Generated &&
                  result.count == factor - 1 && result.written_mask == (1u << (factor - 1)) - 1u,
                  "local scene did not record every requested temporal position");
              for (u32 index = 0; index < factor - 1; ++index)
                outputs[index].layout = vk::ImageLayout::eTransferSrcOptimal;
            }
          }
          const auto scene_status = generator.GetMultipleSceneStatus();
          std::printf("MULTIFG_LOCAL_SCENE_STATUS scene=%u factor=%u state=%u flags=0x%08x recorded_tick=%llu completed_tick=%llu dump_requested=%u\n",
              static_cast<unsigned>(kind), factor, static_cast<unsigned>(scene_status.state),
              scene_status.flags, static_cast<unsigned long long>(scene_status.recorded_tick),
              static_cast<unsigned long long>(scene_status.completed_tick), dump_requested ? 1u : 0u);
          std::fflush(stdout);
          // This optional observation begins only AFTER the original completed batch
          // already reported Changed. The original GenerateMultiple had nullptr diagnostics.
          if (postfailure_requested && !dump_requested &&
              scene_status.state == MultiFrameSceneState::Changed && scene_status.flags != 0) {
            auto scd_buffer = CreateHostBuffer(name, 8, vk::BufferUsageFlagBits::eTransferDst, {});
            auto diagnostic_command = scheduler.BeginCommand().Handle();
            const bool copied = generator.CopyMultipleSceneForDiagnostics(diagnostic_command, scd_buffer.buffer, 8);
            const auto copy_tick = scheduler.Submit();
            scheduler.Wait(copy_tick);
            const auto after_copy_status = generator.GetMultipleSceneStatus();
            std::vector<u32> scd_words;
            if (copied) scd_words = ReadBuffer(name, scd_buffer, 2);
            if (copied) {
              const auto scd0 = std::bit_cast<float>(scd_words[0]);
              std::printf("MULTIFG_POSTFAIL_SCD scene=%u factor=%u word0=0x%08x word0_f32=%.9g word1=0x%08x historybits1=%u original_flags=0x%08x original_recorded=%llu original_completed=%llu copy_tick=%llu after_state=%u after_flags=0x%08x after_recorded=%llu after_completed=%llu\n",
                  static_cast<unsigned>(kind), factor, scd_words[0], static_cast<double>(scd0),
                  scd_words[1], scd_words[1] & 15u, scene_status.flags,
                  static_cast<unsigned long long>(scene_status.recorded_tick),
                  static_cast<unsigned long long>(scene_status.completed_tick),
                  static_cast<unsigned long long>(copy_tick), static_cast<unsigned>(after_copy_status.state),
                  after_copy_status.flags, static_cast<unsigned long long>(after_copy_status.recorded_tick),
                  static_cast<unsigned long long>(after_copy_status.completed_tick));
            } else {
              std::printf("MULTIFG_POSTFAIL_SCD scene=%u factor=%u copy_failed=1 scd_words=unknown original_flags=0x%08x original_recorded=%llu original_completed=%llu copy_tick=%llu after_state=%u after_flags=0x%08x\n",
                  static_cast<unsigned>(kind), factor, scene_status.flags,
                  static_cast<unsigned long long>(scene_status.recorded_tick),
                  static_cast<unsigned long long>(scene_status.completed_tick),
                  static_cast<unsigned long long>(copy_tick), static_cast<unsigned>(after_copy_status.state), after_copy_status.flags);
            }
            std::fflush(stdout);
            MultiFgLocalDiagnostic::SavePostFailureScd(scene, factor, scene_status,
                after_copy_status, copy_tick, copied, scd_words);
            DestroyBuffer(&scd_buffer); // Separate copy submission has completed.
            std::fflush(stdout);
            std::fflush(stderr);
          }
          std::array<std::vector<u32>, 5> capture_words;
          if (dump_requested) {
            const std::array<uint64_t, 5> counts {flow_bytes / 4, 2, image_bytes / 4, image_bytes / 4, flow_bytes};
            for (u32 index = 0; index < capture_buffers.size(); ++index)
              capture_words[index] = ReadBuffer(name, capture_buffers[index], counts[index]);
          }
          std::array<std::vector<u32>, 5> captured_outputs;
          std::array<std::array<Local::Result, 2>, 5> captured_metrics {};
          if (dump_requested) {
            std::printf("MULTIFG_LOCAL_RAW_SCD scene=%u factor=%u word0=0x%08x word1=0x%08x slot1_low4=%u\n",
                static_cast<unsigned>(kind), factor, capture_words[1][0], capture_words[1][1], capture_words[1][1] & 15u);
            std::fflush(stdout);
            const auto previous = ReadImage(name, &inputs[11]);
            const auto current = ReadImage(name, &inputs[12]);
            for (u32 index = 0; index < factor - 1; ++index) {
              captured_outputs[index] = ReadImage(name, &outputs[index]);
              captured_metrics[index] = Local::Position(Local::Packed(captured_outputs[index], scene), scene, factor, index + 1);
              const auto directory = MultiFgLocalDiagnostic::Directory(static_cast<unsigned>(kind), factor, index + 1);
              if (!directory.empty()) {
                MultiFgLocalDiagnostic::Save(directory, scene, factor, index + 1,
                    Local::Packed(previous, scene), Local::Packed(current, scene),
                    Local::Packed(captured_outputs[index], scene), capture_words, captured_metrics[index]);
                MultiFgLocalDiagnostic::SaveSceneStatus(directory, scene, factor, scene_status);
              }
            }
            std::fflush(stdout);
            std::fflush(stderr);
          }
          Require(name, "scene status", !generator.MultipleSceneChanged(),
                  "the final local batch has a scene fallback or unknown GPU status");
          Require(name, "original scene status", scene_status.state == MultiFrameSceneState::Clear,
                  "original completed batch was not clear before optional post-failure observation");
          for (u32 index = 0; index < factor - 1; ++index) {
            const auto pixels = dump_requested ? std::move(captured_outputs[index]) : ReadImage(name, &outputs[index]);
            const auto results = dump_requested ? captured_metrics[index] :
                Local::Position(Local::Packed(pixels, scene), scene, factor, index + 1);
            uint64_t graded = 0;
            for (const auto& result : results) {
              graded += result.pixels;
              std::printf("MULTIFG_LOCAL scene=%u factor=%u ordinal=%u region=%s valid=%llu mae=%.6f rmse=%.6f alt_mse=%.6f blend_mse=%.6f okay=%u\n",
                  static_cast<unsigned>(kind), factor, index + 1, Local::Name(result.region),
                  static_cast<unsigned long long>(result.pixels), result.mae, result.rmse,
                  result.alternate_mse, result.blend_mse, result.okay ? 1u : 0u);
              Require(name, Local::Name(result.region), result.okay,
                      "known local temporal position/static overlap failed the unchanged error envelope");
            }
            std::printf("MULTIFG_LOCAL_MASK scene=%u factor=%u ordinal=%u graded=%llu excluded_boundary_or_unobserved=%llu\n",
                static_cast<unsigned>(kind), factor, index + 1,
                static_cast<unsigned long long>(graded),
                static_cast<unsigned long long>(uint64_t(scene.Width()) * scene.Height() - graded));
          }
          Require(name, "input preserved", ReadImage(name, &inputs.back()) == Local::Frame(scene, 12),
                  "generation changed the local scene real input");
        }
        for (auto &buffer : capture_buffers) DestroyBuffer(&buffer);
        for (auto &output : outputs) DestroyImage(&output);
        for (auto &input : inputs) DestroyImage(&input);
      }
    }
  }

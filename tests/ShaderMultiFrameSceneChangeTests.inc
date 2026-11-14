  // Included inside VulkanHarness. Observations are copied only after the original
  // generation batch has completed; no pre-warp diagnostic commands are injected.
  void CheckMultiFrameSceneChange() {
    namespace Local = MultiFgLocalOracle;
    constexpr const char *name = "MultiFrameSceneChange";
    constexpr auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
    const Local::Scene scene {Local::Kind::Occlusion};
    const u32 width = scene.Width(), height = scene.Height();
    auto &graphics = RuntimeContext();
    CommandScheduler scheduler(Renderer(), graphics);
    const auto general = [&](Image &image) {
      if (image.layout != vk::ImageLayout::eGeneral)
        TransitionImage(name, &image, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    };
    const auto complete = [&](uint64_t expected_tick) {
      const auto tick = scheduler.Submit();
      scheduler.Wait(tick); // Submitted tick only; never implicitly submit through Wait.
      Require(name, "explicit completion", tick == expected_tick && scheduler.IsFree(tick),
              "generation/copy changed its recording tick or did not complete");
      return tick;
    };
    const std::vector<u32> canaries(size_t(width) * height, MultiFgOracle::Canary);
    const std::vector<u32> black(size_t(width) * height, 0xff000000u);
    const std::vector<u32> white(size_t(width) * height, 0xffffffffu);
    const auto previous = Local::Frame(scene, 11);
    const auto current = Local::Frame(scene, 12);
    const auto normal_cpu = MultiFgScdReference::Evaluate(previous, current, width, height);
    const float normal_reference = normal_cpu.score;
    const auto cut_cpu = MultiFgScdReference::Evaluate(black, white, width, height);
    const float cut_reference = cut_cpu.score;
    constexpr float reference_tolerance = 2e-4f;
    Require(name, "independent references", normal_cpu.valid && cut_cpu.valid && std::isfinite(normal_reference) &&
        std::abs(normal_reference - .0014f) <= 1e-5f && cut_reference > .99f,
        "the independent endpoint reference no longer describes the intended test scenes");
    for (u32 factor = 2; factor <= 6; ++factor) {
      for (u32 cut = 0; cut < 2; ++cut) {
        std::array<u32, 2> first_words {};
        std::vector<Image> inputs;
        for (u32 frame = 0; frame <= 12; ++frame)
          inputs.push_back(CreateImage2D(name, width, height, vk::Format::eR8G8B8A8Unorm,
              usage, cut ? (frame == 12 ? white : black) : Local::Frame(scene, frame),
              1, vk::ImageLayout::eGeneral));
        for (u32 repeat = 0; repeat < 2; ++repeat) {
          std::array<Image, 5> outputs {};
          std::array<vk::Image, 5> handles {};
          for (u32 index = 0; index < factor - 1; ++index) {
            outputs[index] = CreateImage2D(name, width, height, vk::Format::eR8G8B8A8Unorm,
                usage, canaries, 1, vk::ImageLayout::eGeneral);
            handles[index] = outputs[index].image;
          }
          // Each repetition owns a fresh SDK/OF/history context. All imported
          // images remain alive until after its final submitted readback completes.
          {
            FrameGenerator generator(graphics, scheduler);
            const auto targets = std::span<const vk::Image>(handles.data(), factor - 1);
            MultiFrameResult final_result {};
            for (u32 frame = 0; frame <= 12; ++frame) {
              for (u32 index = 0; index < factor - 1; ++index) general(outputs[index]);
              std::this_thread::sleep_for(std::chrono::milliseconds(17));
              auto command = scheduler.BeginCommand().Handle();
              const auto recording_tick = scheduler.CurrentTick();
              const auto result = generator.GenerateMultiple(command, inputs[frame].image,
                  targets, width, height, factor); // nullptr diagnostics: original batch unchanged.
              Require(name, "record only", scheduler.CurrentTick() == recording_tick,
                      "GenerateMultiple implicitly submitted or advanced the tick");
              complete(recording_tick);
              if (frame < 6) {
                Require(name, "six seed frames", result.status == MultiFrameStatus::Seeded &&
                    result.count == 0 && result.written_mask == 0,
                    "initial OF warmup published intermediate outputs");
                for (u32 index = 0; index < factor - 1; ++index) {
                  Require(name, "seed canaries", ReadImage(name, &outputs[index]) == canaries,
                          "a seed frame changed an external output");
                  general(outputs[index]);
                }
              } else {
                Require(name, "full recorded batch", result.status == MultiFrameStatus::Generated &&
                    result.count == factor - 1 && result.written_mask == (1u << (factor - 1)) - 1u,
                    "warmed batch did not record every requested intermediate");
                for (u32 index = 0; index < factor - 1; ++index)
                  outputs[index].layout = vk::ImageLayout::eTransferSrcOptimal;
              }
              final_result = result;
            }
            const auto original_status = generator.GetMultipleSceneStatus();
            const bool accepted = final_result.status == MultiFrameStatus::Generated &&
                                  !generator.MultipleSceneChanged();
            // Allocate/read only AFTER completing the final unchanged GPU batch.
            std::vector<u32> guard_words(6, MultiFgOracle::Canary);
            auto capture = CreateHostBuffer(name, guard_words.size() * sizeof(u32),
                vk::BufferUsageFlagBits::eTransferDst, guard_words);
            auto copy_command = scheduler.BeginCommand().Handle();
            const auto copy_recording_tick = scheduler.CurrentTick();
            const bool copied = generator.CopyMultipleSceneForDiagnostics(copy_command, capture.buffer, 8);
            const auto copy_tick = complete(copy_recording_tick);
            const auto raw = ReadBuffer(name, capture, guard_words.size());
            DestroyBuffer(&capture);
            const auto after_status = generator.GetMultipleSceneStatus();
            const float score = std::bit_cast<float>(raw[0]);
            std::printf("MULTIFG_SCD scene=%s factor=%u repeat=%u word0=0x%08x scalar=%.9g word1=0x%08x low4=%u reference=%.9g tolerance=%.9g state=%u flags=0x%08x recorded_tick=%llu completed_tick=%llu copy_tick=%llu accepted=%u first_word0=0x%08x first_word1=0x%08x\n",
                cut ? "cut" : "occlusion", factor, repeat, raw[0], static_cast<double>(score),
                raw[1], raw[1] & 15u, static_cast<double>(cut ? cut_reference : normal_reference),
                static_cast<double>(reference_tolerance), static_cast<unsigned>(original_status.state),
                original_status.flags, static_cast<unsigned long long>(original_status.recorded_tick),
                static_cast<unsigned long long>(original_status.completed_tick),
                static_cast<unsigned long long>(copy_tick), accepted ? 1u : 0u, first_words[0], first_words[1]);
            std::fflush(stdout);
            Require(name, "completed raw copy", copied && std::isfinite(score) &&
                original_status.recorded_tick != 0 &&
                original_status.completed_tick >= original_status.recorded_tick,
                "completed batch did not expose a finite SCD scalar");
            Require(name, "eight byte copy bounds", std::equal(raw.begin() + 2, raw.end(), guard_words.begin() + 2),
                    "the eight-byte SCD copy modified its host-buffer tail");
            Require(name, "observation preserves state", after_status.state == original_status.state &&
                after_status.flags == original_status.flags &&
                after_status.recorded_tick == original_status.recorded_tick,
                "post-completion readback changed the original scene status");
            if (cut) {
              Require(name, "genuine cut", original_status.state == MultiFrameSceneState::Changed &&
                  original_status.flags != 0 && (raw[1] & 15u) != 0 && score > .99f &&
                  std::abs(score - cut_reference) <= reference_tolerance && !accepted,
                  "a real black-to-white cut was not rejected by the completed scene gate");
            } else {
              Require(name, "normal occlusion", original_status.state == MultiFrameSceneState::Clear &&
                  original_status.flags == 0 && (raw[1] & 15u) == 0 && accepted &&
                  std::abs(score - normal_reference) <= reference_tolerance,
                  "rigid occlusion diverged from the independent CPU SCD or falsely cut");
            }
            if (repeat == 0) first_words = {raw[0], raw[1]};
            else Require(name, "fresh context determinism", first_words == std::array<u32, 2> {raw[0], raw[1]},
                         "identical input sequences produced different SCD/history bits in fresh contexts");
            Require(name, "real input preserved", ReadImage(name, &inputs.back()) == (cut ? white : current),
                    "generation/copy changed the final real input");
            general(inputs.back());
          }
          for (auto &output : outputs) DestroyImage(&output);
        }
        for (auto &input : inputs) DestroyImage(&input);
      }
    }
    CheckMultiFrameContextModeSwitch();
  }

  void CheckMultiFrameContextModeSwitch() {
    constexpr const char *name = "MultiFrameContextModeSwitch";
    constexpr u32 width = 384, height = 256, factor = 3;
    constexpr auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
    auto &graphics = RuntimeContext();
    CommandScheduler scheduler(Renderer(), graphics);
    const std::vector<u32> black(size_t(width) * height, 0xff000000u);
    const std::vector<u32> canaries(black.size(), MultiFgOracle::Canary);
    auto input = CreateImage2D(name, width, height, vk::Format::eR8G8B8A8Unorm,
        usage, black, 1, vk::ImageLayout::eGeneral);
    auto legacy_output = CreateImage2D(name, width, height, vk::Format::eR8G8B8A8Unorm,
        usage, canaries, 1, vk::ImageLayout::eGeneral);
    std::array<Image, 2> outputs {};
    std::array<vk::Image, 2> handles {};
    for (u32 index = 0; index < outputs.size(); ++index) {
      outputs[index] = CreateImage2D(name, width, height, vk::Format::eR8G8B8A8Unorm,
          usage, canaries, 1, vk::ImageLayout::eGeneral);
      handles[index] = outputs[index].image;
    }
    const auto general = [&](Image &image) {
      if (image.layout != vk::ImageLayout::eGeneral)
        TransitionImage(name, &image, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    };
    const auto complete = [&](uint64_t recording_tick) {
      const auto tick = scheduler.Submit(); scheduler.Wait(tick);
      Require(name, "submitted tick", tick == recording_tick && scheduler.IsFree(tick),
              "mixed-mode generation/copy did not complete its explicit submission");
    };
    {
      FrameGenerator generator(graphics, scheduler);
      std::vector<u32> original_legacy;
      const auto legacy = [&](u32 phase) {
        for (u32 frame = 0; frame < 8; ++frame) {
          general(legacy_output);
          std::this_thread::sleep_for(std::chrono::milliseconds(17));
          auto command = scheduler.BeginCommand().Handle();
          const auto tick = scheduler.CurrentTick();
          const bool generated = generator.Generate(command, input.image, legacy_output.image, width, height);
          const bool written = generator.OutputWritten();
          Require(name, "legacy record only", scheduler.CurrentTick() == tick,
                  "legacy Generate advanced the scheduler tick");
          complete(tick);
          Require(name, "legacy reset and warm", written && generated == (frame != 0),
                  "legacy mode did not reseed once then resume after a context transition");
          if (written) legacy_output.layout = vk::ImageLayout::eTransferSrcOptimal;
        }
        const auto pixels = ReadImage(name, &legacy_output);
        if (phase == 0) original_legacy = pixels;
        else Require(name, "legacy restoration", pixels == original_legacy,
                     "returning from Multi changed warmed static legacy output");
      };
      legacy(0);
      // Switching a context cannot silently submit an already-recording legacy batch.
      general(legacy_output);
      std::this_thread::sleep_for(std::chrono::milliseconds(17));
      auto command = scheduler.BeginCommand().Handle();
      const auto pending_tick = scheduler.CurrentTick();
      const bool legacy_recorded = generator.Generate(command, input.image, legacy_output.image, width, height);
      const bool legacy_written = generator.OutputWritten();
      const auto rejected = generator.GenerateMultiple(command, input.image, handles, width, height, factor);
      Require(name, "unsubmitted switch rejected", legacy_recorded && legacy_written &&
          rejected.status == MultiFrameStatus::Failed && rejected.count == 0 && rejected.written_mask == 0 &&
          scheduler.CurrentTick() == pending_tick,
          "a context switch consumed/submitted an unsubmitted legacy batch");
      complete(pending_tick);
      legacy_output.layout = vk::ImageLayout::eTransferSrcOptimal;
      for (u32 index = 0; index < outputs.size(); ++index) {
        Require(name, "rejected switch outputs", ReadImage(name, &outputs[index]) == canaries,
                "rejected mode switch changed an external Multi output");
        general(outputs[index]);
      }
      for (u32 frame = 0; frame < 13; ++frame) {
        for (auto &output : outputs) general(output);
        std::this_thread::sleep_for(std::chrono::milliseconds(17));
        command = scheduler.BeginCommand().Handle();
        const auto tick = scheduler.CurrentTick();
        const auto result = generator.GenerateMultiple(command, input.image, handles, width, height, factor);
        Require(name, "multi record only", scheduler.CurrentTick() == tick,
                "Multi context transition advanced the scheduler tick");
        complete(tick);
        Require(name, "multi reset and warm", frame < 6 ?
            result.status == MultiFrameStatus::Seeded && result.count == 0 && result.written_mask == 0 :
            result.status == MultiFrameStatus::Generated && result.count == factor - 1 &&
                result.written_mask == (1u << (factor - 1)) - 1u,
            "switching to Multi did not start a fresh six-frame OF sequence");
        if (frame >= 6)
          for (auto &output : outputs) output.layout = vk::ImageLayout::eTransferSrcOptimal;
      }
      Require(name, "multi completed scene", !generator.MultipleSceneChanged(),
              "a warmed static Multi sequence failed its completed scene gate");
      legacy(1);
      Require(name, "input preserved", ReadImage(name, &input) == black,
              "switching SDK context modes changed the real input");
      std::printf("MULTIFG_CONTEXT_MODES sequence=fsr-multi-fsr factor=%u all_submitted=1 restored=1\n", factor);
      std::fflush(stdout);
    }
    for (auto &output : outputs) DestroyImage(&output);
    DestroyImage(&legacy_output); DestroyImage(&input);
  }

  // Included inside VulkanHarness. Private test layers/vectors are NOT passed to the GPU.
  void CheckMultiFrameGenerationHud() {
    namespace Hud=MultiFgHudOracle;
    constexpr const char *name="MultiFrameGenerationHud";
    constexpr auto usage=vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled;
    const char* dump_env=std::getenv("KYTY_MULTIFG_HUD_DUMP_DIR");
    const bool dump_requested=dump_env&&*dump_env;
    auto &graphics=RuntimeContext();CommandScheduler scheduler(Renderer(),graphics);
    const auto general=[&](Image &image){
      if(image.layout!=vk::ImageLayout::eGeneral)
        TransitionImage(name,&image,vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits::eAllCommands,vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eMemoryRead|vk::AccessFlagBits::eMemoryWrite,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
    };
    std::vector<Image> inputs;
    for(u32 frame=0;frame<=12;++frame)
      inputs.push_back(CreateImage2D(name,Hud::Width,Hud::Height,vk::Format::eR8G8B8A8Unorm,
          usage,Hud::Frame(frame),1,vk::ImageLayout::eGeneral));
    const std::vector<u32> canaries(size_t(Hud::Width)*Hud::Height,MultiFgOracle::Canary);
    bool all_quality=true;
    for(u32 factor=2;factor<=6;++factor){
      std::array<Image,5> outputs{};std::array<vk::Image,5> handles{};
      for(u32 index=0;index<factor-1;++index){
        outputs[index]=CreateImage2D(name,Hud::Width,Hud::Height,vk::Format::eR8G8B8A8Unorm,
            usage,canaries,1,vk::ImageLayout::eGeneral);handles[index]=outputs[index].image;
      }
      {
        FrameGenerator generator(graphics,scheduler);
        for(u32 frame=0;frame<=12;++frame){
          for(u32 index=0;index<factor-1;++index)general(outputs[index]);
          std::this_thread::sleep_for(std::chrono::milliseconds(17));
          auto command=scheduler.BeginCommand().Handle();const auto recording_tick=scheduler.CurrentTick();
          const auto result=generator.GenerateMultiple(command,inputs[frame].image,
              std::span<const vk::Image>(handles.data(),factor-1),Hud::Width,Hud::Height,factor);
          Require(name,"record only",scheduler.CurrentTick()==recording_tick,
                  "HUD generation submitted or advanced its recording tick");
          const auto tick=scheduler.Submit();scheduler.Wait(tick);
          Require(name,"completion",tick==recording_tick&&scheduler.IsFree(tick),
                  "HUD generation did not complete its explicit submission");
          if(frame<6){
            Require(name,"six seed frames",result.status==MultiFrameStatus::Seeded&&result.count==0&&result.written_mask==0,
                    "HUD warmup published an intermediate");
            for(u32 index=0;index<factor-1;++index){
              Require(name,"seed preservation",ReadImage(name,&outputs[index])==canaries,
                      "HUD seed changed an external output");general(outputs[index]);
            }
          }else{
            Require(name,"full output batch",result.status==MultiFrameStatus::Generated&&
                result.count==factor-1&&result.written_mask==(1u<<(factor-1))-1u,
                "warmed HUD pair did not record every requested temporal position");
            for(u32 index=0;index<factor-1;++index)outputs[index].layout=vk::ImageLayout::eTransferSrcOptimal;
          }
        }
        const auto status=generator.GetMultipleSceneStatus();
        std::printf("MULTIFG_HUD_SCENE factor=%u state=%u flags=0x%08x recorded=%llu completed=%llu\n",
            factor,static_cast<unsigned>(status.state),status.flags,
            static_cast<unsigned long long>(status.recorded_tick),static_cast<unsigned long long>(status.completed_tick));
        std::fflush(stdout);
        if(dump_requested){
          // Original GenerateMultiple(nullptr) and its completed submission are unchanged.
          // Allocate only now; copy existing fine fields in a separate, retained tick.
          constexpr uint64_t field_bytes=uint64_t(Hud::Width)*Hud::Height*16;
          std::array<Buffer,2> buffers{};
          for(auto &buffer:buffers)
            buffer=CreateHostBuffer(name,field_bytes,vk::BufferUsageFlagBits::eTransferDst,{});
          auto copy_command=scheduler.BeginCommand().Handle();
          const auto expected_copy_tick=scheduler.CurrentTick();
          const bool copied=generator.CopyMultipleFineForDiagnostics(copy_command,
              {buffers[0].buffer,field_bytes},{buffers[1].buffer,field_bytes});
          const auto copy_tick=scheduler.Submit();scheduler.Wait(copy_tick);
          const auto after=generator.GetMultipleSceneStatus();
          std::printf("MULTIFG_HUD_FINE_COPY factor=%u copied=%u original_state=%u original_flags=0x%08x original_recorded=%llu original_completed=%llu copy_tick=%llu after_state=%u after_flags=0x%08x after_recorded=%llu after_completed=%llu\n",
              factor,copied?1u:0u,static_cast<unsigned>(status.state),status.flags,
              static_cast<unsigned long long>(status.recorded_tick),static_cast<unsigned long long>(status.completed_tick),
              static_cast<unsigned long long>(copy_tick),static_cast<unsigned>(after.state),after.flags,
              static_cast<unsigned long long>(after.recorded_tick),static_cast<unsigned long long>(after.completed_tick));
          std::fflush(stdout);
          Require(name,"fine diagnostic copy",copied&&copy_tick==expected_copy_tick&&scheduler.IsFree(copy_tick),
                  "post-completion fine diagnostic copy was unavailable or did not complete");
          std::array<std::vector<u32>,2> fields;
          for(u32 index=0;index<buffers.size();++index)
            fields[index]=ReadBuffer(name,buffers[index],field_bytes/4);
          MultiFgHudDiagnostic::SavePair(factor,fields,{status.recorded_tick,status.completed_tick,copy_tick,
              after.recorded_tick,after.completed_tick,static_cast<u32>(status.state),status.flags,
              static_cast<u32>(after.state),after.flags});
          for(auto &buffer:buffers)DestroyBuffer(&buffer); // Copy submission has completed.
          Require(name,"fine observation preserves state",after.state==status.state&&after.flags==status.flags&&
              after.recorded_tick==status.recorded_tick,"fine readback changed the original completed scene status");
        }
        Require(name,"scene gate",status.state==MultiFrameSceneState::Clear&&!generator.MultipleSceneChanged(),
                "a fixed HUD over rigid moving background falsely rejected the completed scene");
        for(u32 index=0;index<factor-1;++index){
          const auto pixels=ReadImage(name,&outputs[index]);
          const auto metrics=Hud::Position(Hud::Packed(pixels),factor,index+1);
          const auto coverage=Hud::CoverageAt(double(index+1)/factor);
          std::printf("MULTIFG_HUD_COVERAGE factor=%u ordinal=%u hud=%llu aa=%llu bg_observed=%llu bg_unobservable=%llu bg_border_excluded=%llu ungraded_hud=%llu\n",
              factor,index+1,static_cast<unsigned long long>(coverage.hud),
              static_cast<unsigned long long>(coverage.antialias),
              static_cast<unsigned long long>(coverage.background_observed),
              static_cast<unsigned long long>(coverage.background_unobservable),
              static_cast<unsigned long long>(coverage.background_border_excluded),
              static_cast<unsigned long long>(coverage.ungraded_hud));
          Require(name,"HUD mask coverage",coverage.ungraded_hud==0&&coverage.hud>10'000&&coverage.antialias>1'000,
                  "a static HUD/letter-edge pixel was excluded from grading");
          // Save only completed synthetic pixels. No diagnostic commands/history changes
          // are added to the original generation batch, including on a failed ROI.
          MultiFgHudDiagnostic::Save(factor,index+1,pixels);
          for(const auto &metric:metrics){
            std::printf("MULTIFG_HUD factor=%u ordinal=%u region=%s pixels=%llu mae=%.6f rmse=%.6f alternate_mse=%.6f blend_mse=%.6f okay=%u\n",
                factor,index+1,Hud::Name(metric.region),static_cast<unsigned long long>(metric.pixels),
                metric.mae,metric.rmse,metric.alternate_mse,metric.blend_mse,metric.okay?1u:0u);
            all_quality&=metric.okay;
          }
          std::fflush(stdout);
        }
        Require(name,"input preservation",ReadImage(name,&inputs.back())==Hud::Frame(12),
                "HUD generation modified its real composited input");general(inputs.back());
      }
      for(auto &output:outputs)DestroyImage(&output);
    }
    for(auto &input:inputs)DestroyImage(&input);
    // Collect every factor/ordinal/ROI before reporting failure; no failed ROI is waived.
    Require(name,"static HUD and moving background",all_quality,
            "one or more opaque/AA-edge/panel/visible-background temporal-position ROIs failed");
  }

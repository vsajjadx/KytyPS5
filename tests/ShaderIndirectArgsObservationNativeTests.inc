// Included inside VulkanHarness. Real GPU arguments/output controls precede the
// typed-observation assertion. The external gates are test-only; runtime queue
// policy, wait selection and normal callback execution are not changed.
void CheckIndirectArgsObservation(bool enable_filter,
    Libs::Graphics::IndirectArgsObserverControlTestAccess::Mode control_mode =
        Libs::Graphics::IndirectArgsObserverControlTestAccess::Mode::RealArgumentCopy) {
  namespace Causal = Libs::Graphics::CausalDiagnostics;
  using Observer = Libs::Graphics::IndirectArgsDiagnostics;
  using Control = Libs::Graphics::IndirectArgsObserverControlTestAccess;
  constexpr std::array<u32, 3> scratch_payload{0x13579bdfu, 0x2468ace0u, 0x1badf00du};
  constexpr u32 scratch_canary = 0xd15ea5e5u;
  constexpr const char* name = "IndirectArgsObservation";
  constexpr uintptr_t base = 0x000000020ba00000ull;
  constexpr uint64_t size = 0x100000, case_size = 0x10000;
  constexpr u32 sentinel = 0xa5a5a5a5u, output_words = 16;
  constexpr std::array<u32, 10> groups_x{0, 1, 2, 3, 1, 2, 3, 1, 2, 2};
  constexpr size_t matching_cases = groups_x.size() - 1;
  constexpr uint64_t args_copy_bytes = 5 * sizeof(u32);
  constexpr uint64_t output_copy_bytes = (output_words + 2) * sizeof(u32);
  constexpr uint64_t case_copy_bytes = args_copy_bytes + output_copy_bytes;
  constexpr uint64_t readback_bytes = groups_x.size() * case_copy_bytes;
  const auto args_at = [&](size_t i) {
    return base + i * case_size + BufferCache::CACHING_PAGESIZE - sizeof(u32);
  };
  const auto output_at = [&](size_t i) {
    return base + i * case_size + BufferCache::CACHING_PAGESIZE + 0x100u;
  };

  std::vector<u32> writer;
  for (u32 word = 0; word < 3; ++word) {
    writer.push_back(EncodeVop1(0x01u, 1, 12u + word));
    writer.push_back(EncodeMubuf0(0x1cu, 0, false, false));
    writer.push_back(EncodeMubuf1(1, word, 0));
  }
  AppendEnd(&writer);
  std::vector<u32> consumer;
  AppendVop3(&consumer, 0x346u, 1, 8, InlineU32(2), Vgpr(0));
  consumer.insert(consumer.end(), {
      EncodeVop2(0x25u, 2, InlineU32(1), 1), // value = global lane + 1
      EncodeVop2(0x1au, 1, InlineU32(2), 1), // byte offset = global lane * 4
      EncodeMubuf0(0x1cu), EncodeMubuf1(2, 0, 1),
      EncodeSop1(0x04u, 20, 126),           // preserve EXEC
      EncodeVopc(0xd2u, InlineU32(0), 1),   // only global invocation zero
      EncodeVop1(0x01u, 3, InlineU32(7)),
      EncodeMubuf0(0x1cu, 0, false, false), EncodeMubuf1(3, 1, 0),
      EncodeSop1(0x04u, 126, 20)});         // restore EXEC
  // The target changes arguments after consuming them. A diagnostic read placed
  // after dispatch would see 7, not the original nonzero workgroup count.
  auto filtered_consumer = consumer;
  filtered_consumer.push_back(EncodeSopp(0x00u, 0)); // same work, different real code hash
  AppendEnd(&consumer);
  AppendEnd(&filtered_consumer);
  for (const auto* code : {&writer, &consumer, &filtered_consumer}) {
    ShaderMapUserData(reinterpret_cast<uint64_t>(code->data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<u32>(code->size() * sizeof(u32))});
  }
  HW::ComputeShaderInfo hash_regs{};
  hash_regs.cs_regs = {.data_addr = reinterpret_cast<uint64_t>(consumer.data()),
      .num_thread_x = 4, .num_thread_y = 1, .num_thread_z = 1,
      .wave_size = 64, .user_sgpr = 8, .tgid_x_en = true};
  ShaderComputeInputInfo hash_input{};
  const auto expected_hash = PrepareProgram(hash_regs, hash_input).hash;
  std::array<char, 17> filter{};
  Require(name, "exact filter formatting",
      std::snprintf(filter.data(), filter.size(), "%016llx",
                    static_cast<unsigned long long>(expected_hash)) == 16,
      "could not format the real mapped shader hash");
#if defined(_WIN32)
  Require(name, "child filter setup",
      _putenv_s("KYTY_INDIRECT_ARGS_DIAG", enable_filter ? filter.data() : "") == 0,
      "could not set this test child's diagnostic filter");
#else
  Require(name, "child filter setup",
      (enable_filter ? setenv("KYTY_INDIRECT_ARGS_DIAG", filter.data(), 1)
                     : unsetenv("KYTY_INDIRECT_ARGS_DIAG")) == 0,
      "could not set this test child's diagnostic filter");
#endif
  const bool diagnostic = enable_filter && FaultPeekEnabled();
  Require(name, "requested private control gates",
      control_mode == Control::Mode::RealArgumentCopy || diagnostic,
      "barrier/scratch controls require the exact causal and fault-peek child gates");
  const bool control_run = diagnostic && control_mode != Control::Mode::RealArgumentCopy;
  const char* control_name = !diagnostic ? "observer-off" :
      control_mode == Control::Mode::ComposedBarrierOnly ? "composed-broad-barrier" :
      control_mode == Control::Mode::ImmutableScratchCopy ? "immutable-scratch-copy" :
      "real-argument-copy";
  std::printf("[control] mode=%s; ordinary producer/consumer dependencies retained\n", control_name);
  Require(name, "exact child opt-in",
      IndirectArgsDiagnosticEnabled() == diagnostic &&
          (!diagnostic || IndirectArgsDiagnosticShader() == expected_hash),
      "indirect diagnostics ignored their shader filter or required child gates");
  Require(name, "external gate precondition", !GpuSyncDebugEnabled(),
      "external timeline gate requires KYTY_GPU_SYNC=0");
  EnsureRuntimeContext();

  int64_t physical = -1;
  Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
      0, LibKernel::Memory::KernelGetDirectMemorySize(), size, case_size, 0, &physical) == 0,
      "indirect diagnostic test allocation failed");
  void* mapped = reinterpret_cast<void*>(base);
  Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
      &mapped, size, 0x3, 0x10, physical, case_size) == 0 &&
      mapped == reinterpret_cast<void*>(base), "indirect diagnostic mapping failed");
  std::memset(mapped, 0xa5, size);
  const std::array<u32, 3> stale_args{17, 1, 1};
  for (size_t i = 0; i < groups_x.size(); ++i)
    LibKernel::Memory::WriteBacking(args_at(i), stale_args.data(), sizeof(stale_args));

  RenderContext context(m_runtime_context);
  context.InitializeGpu(nullptr);
  LibKernel::Memory::InstallGpuResources(&context);
  auto& gpu = context.GetGpu();
  auto& scheduler = context.GetCommandScheduler();
  auto& master = scheduler.GetMasterSemaphore();
  auto& cache = context.GetBufferCache();
  CommandProcessor processor(context, 0);
  std::unique_ptr<Libs::Graphics::Buffer> readback;
  vk::Semaphore gate = nullptr;
  uint64_t producer_tick = 0, host_wait_tick = 0;
  uint32_t recording_thread = 0;
  std::atomic_uint normal_callbacks{0};
  std::array<Causal::ParentRef, groups_x.size()> origins{};
  std::array<uint64_t, matching_cases> dispatch_sequences{}, dispatch_ticks{},
      native_handles{}, native_offsets{};
  Observer::Snapshot blocked{}, query_only{}, entered_only{}, observed{}, repeated{};
  Control::Report control_blocked{}, control_query{}, control_entered{},
      control_observed{}, control_repeated{}, control_foreign{};
  if (Causal::Enabled()) {
    for (size_t i = 0; i < origins.size(); ++i) {
      origins[i] = Causal::RecordEvent(Causal::Family::Guest,
          Causal::Make(Causal::Event::GuestEnqueue, Causal::Domain(&gpu), Causal::NewId())).parent;
    }
  }
  const auto idle = [&] {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool seen = false;
    do {
      const auto state = GuestGpuTestAccess::Snapshot(gpu);
      if (state.consistent && !state.processing && state.commands == 0 &&
          state.pending_commands == 0 && state.submissions == 0) { seen = true; break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    Require(name, "renderer idle", seen, "renderer did not return to its empty idle queue");
  };
  const auto release = [&](uint64_t value) {
    vk::SemaphoreSignalInfo signal{};
    signal.sType = vk::StructureType::eSemaphoreSignalInfo;
    signal.semaphore = gate;
    signal.value = value;
    Require(name, "release owned native gate",
        m_runtime_context.device.signalSemaphore(&signal) == vk::Result::eSuccess,
        "could not release the test's already-submitted native gate");
  };
  gpu.SendCommandSync([&] {
    if (diagnostic) {
      auto* observer = cache.IndirectArgsObserver();
      Require(name, "install private observer control before recording",
          observer != nullptr && Control::Install(*observer, control_mode, scratch_payload, scratch_canary),
          "test control was not installed before the first producer/consumer");
    }
    processor.BufferInit();
    context.MapMemory(base, size);
    cache.PrepareBdaRange(base, size);
    context.PrepareBda();
    readback = std::make_unique<Libs::Graphics::Buffer>(context.GetGraphics(), scheduler,
        MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, readback_bytes);
    vk::SemaphoreTypeCreateInfo type{};
    type.sType = vk::StructureType::eSemaphoreTypeCreateInfo;
    type.semaphoreType = vk::SemaphoreType::eTimeline;
    vk::SemaphoreCreateInfo create{};
    create.sType = vk::StructureType::eSemaphoreCreateInfo;
    create.pNext = &type;
    Require(name, "native gate creation",
        m_runtime_context.device.createSemaphore(&create, nullptr, &gate) == vk::Result::eSuccess,
        "could not create the external timeline gate");
    auto& shaders = processor.GetShCtx();
    const auto set_buffer = [&](u32 reg, uint64_t address, u32 bytes) {
      ShaderBufferResource value{};
      value.UpdateAddress48(address);
      value.fields[2] = bytes;
      value.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      for (u32 word = 0; word < 4; ++word) shaders.SetCsUserSgpr(reg + word, value.fields[word]);
    };
    recording_thread = static_cast<uint32_t>(Common::Thread::GetThreadIdUnique());
    for (size_t i = 0; i < groups_x.size(); ++i) {
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(writer.data()),
          .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
          .wave_size = 64, .user_sgpr = 15});
      for (u32 word = 0; word < 3; ++word) {
        set_buffer(word * 4u, args_at(i) + word * sizeof(u32), sizeof(u32));
        shaders.SetCsUserSgpr(12u + word, word == 0 ? groups_x[i] : 1u);
      }
      processor.DispatchDirect(1, 1, 1, 0x41u);
      std::array<u32, 3> backing{};
      Require(name, "GPU-written arguments",
          cache.HasGpuDirtyBytes(args_at(i), sizeof(backing)) &&
              LibKernel::Memory::TryReadBacking(args_at(i), backing.data(), sizeof(backing)) &&
              backing == stale_args,
          "fixture lost GPU-only arguments or changed the deliberately stale CPU backing");
      const auto& code = i == matching_cases ? filtered_consumer : consumer;
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
          .num_thread_x = 4, .num_thread_y = 1, .num_thread_z = 1,
          .wave_size = 64, .user_sgpr = 8, .tgid_x_en = true});
      set_buffer(0, output_at(i), output_words * sizeof(u32));
      set_buffer(4, args_at(i), 3u * sizeof(u32));
      ShaderComputeInputInfo input{};
      Require(name, "real consumer module",
          context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input) != nullptr &&
              (input.stage.module->info.shader_hash == expected_hash) == (i < matching_cases),
          "fixture filter does not match the actual selected consumer module");
      Causal::ContextScope origin({.cause = origins[i], .guest = origins[i],
                                   .point = 1000u + i, .request = 2000u + i}, Causal::Enabled());
      processor.DispatchIndirect(args_at(i), 0x41u);
      if (diagnostic && i < matching_cases) {
        Causal::Snapshot causal{};
        Require(name, "actual recorded dispatch identity",
            Causal::RuntimeRecorder().ReadSnapshot(causal) && causal.focus_available &&
                causal.focus_valid && causal.focus.event == Causal::Event::TargetDispatchIndirectRecorded &&
                causal.focus.data[0] == expected_hash &&
                causal.focus.data[3] == origins[i].sequence &&
                causal.focus.data[4] == origins[i].domain && causal.focus.data[5] == origins[i].id,
            "actual indirect dispatch did not retain its exact guest-origin event");
        dispatch_sequences[i] = causal.focus.sequence;
        dispatch_ticks[i] = scheduler.CurrentTick();
        Require(name, "recording tick", causal.focus.tick == dispatch_ticks[i],
            "dispatch marker used a different native recording tick");
        vk::DescriptorBufferInfo native{};
        Require(name, "actual argument owner", cache.CanonicalBinding(args_at(i), 12, native),
            "GPU-written arguments lost their canonical native owner");
        const VkBuffer handle = static_cast<VkBuffer>(native.buffer);
        static_assert(sizeof(handle) <= sizeof(native_handles[i]));
        std::memcpy(&native_handles[i], &handle, sizeof(handle));
        native_offsets[i] = native.offset;
      }
    }
    // Independent native readback verifies final argument bytes and all output
    // guards. It never asks the CPU backing to stand in for executed GPU work.
    const auto command = scheduler.Current().Handle();
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eTransfer, {},
        vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                          .dstAccessMask = vk::AccessFlagBits::eTransferRead}, {}, {});
    for (size_t i = 0; i < groups_x.size(); ++i) {
      const auto copy = [&](uint64_t address, uint64_t bytes, uint64_t destination) {
        vk::DescriptorBufferInfo native{};
        Require(name, "canonical GPU readback", cache.CanonicalBinding(address, bytes, native),
            "final argument/output canaries lost their native buffer");
        command.copyBuffer(native.buffer, readback->Handle(),
            vk::BufferCopy{native.offset, destination, bytes});
      };
      copy(args_at(i) - sizeof(u32), args_copy_bytes, i * case_copy_bytes);
      copy(output_at(i) - sizeof(u32), output_copy_bytes,
           i * case_copy_bytes + args_copy_bytes);
    }
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eHost, {},
        vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                          .dstAccessMask = vk::AccessFlagBits::eHostRead}, {}, {});
    scheduler.NoteRecordedWork();
    scheduler.DeferOperation([&] { normal_callbacks.fetch_add(1, std::memory_order_relaxed); });
    SubmitInfo blocked_submit;
    blocked_submit.AddWait(gate, 1);
    producer_tick = scheduler.Submit(blocked_submit);
  });
  idle();
  if (diagnostic) {
    gpu.SendCommandSync([&] {
      (void)cache.PeekCompletedIndirectArgs();
      blocked = cache.ReadIndirectArgsDiagnostics();
      if (control_run) control_blocked = Control::Read(*cache.IndirectArgsObserver());
    });
    Require(name, "no blocked payload read", blocked.observed_total == 0 &&
        normal_callbacks.load(std::memory_order_relaxed) == 0,
        "blocked dispatch bytes were observed or a pending normal callback ran");
    release(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (master.KnownGpuTick() < producer_tick && std::chrono::steady_clock::now() < deadline) {
      master.Refresh(); // Test-only query, never part of the passive observer.
      std::this_thread::yield();
    }
    Require(name, "query-only completion", master.KnownGpuTick() >= producer_tick &&
        master.ReadFaultPeekHostWait().completed_tick < producer_tick,
        "fixture did not separate queried native completion from actual host waiting");
    gpu.SendCommandSync([&] {
      (void)cache.PeekCompletedIndirectArgs();
      query_only = cache.ReadIndirectArgsDiagnostics();
      if (control_run) control_query = Control::Read(*cache.IndirectArgsObserver());
      scheduler.BeginCommand();
      SubmitInfo blocked_submit;
      blocked_submit.AddWait(gate, 2);
      host_wait_tick = scheduler.Submit(blocked_submit);
    });
    idle();
    Require(name, "query cannot authorize reading", query_only.observed_total == 0 &&
        normal_callbacks.load(std::memory_order_relaxed) == 0,
        "query completion published argument bytes or consumed a normal callback");
    std::atomic_bool waiter_finished{false};
    std::jthread waiter([&] {
      master.Wait(host_wait_tick);
      waiter_finished.store(true, std::memory_order_release);
    });
    const auto deadline_enter = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto wait = master.ReadFaultPeekHostWait();
    while (wait.entered_tick < host_wait_tick &&
           !waiter_finished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline_enter) {
      std::this_thread::yield();
      wait = master.ReadFaultPeekHostWait();
    }
    const bool entered_exact = wait.entered_tick == host_wait_tick &&
        wait.completed_tick < producer_tick;
    if (entered_exact) {
      gpu.SendCommandSync([&] {
        (void)cache.PeekCompletedIndirectArgs();
        entered_only = cache.ReadIndirectArgsDiagnostics();
        if (control_run) control_entered = Control::Read(*cache.IndirectArgsObserver());
      });
    }
    // Hold only this test's GPU-owner callback before completing the native
    // gate. Otherwise the automatic idle hook could publish before the foreign
    // caller, making a missing owner guard pass an AlreadyObserved check.
    std::atomic_bool owner_entered{false}, owner_release{false}, owner_timed_out{false};
    std::jthread owner_hold([&] {
      gpu.SendCommandSync([&] {
        owner_entered.store(true, std::memory_order_release);
        const auto hold_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!owner_release.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < hold_deadline)
          std::this_thread::yield();
        if (!owner_release.load(std::memory_order_acquire))
          owner_timed_out.store(true, std::memory_order_release);
      });
    });
    const auto owner_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!owner_entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < owner_deadline)
      std::this_thread::yield();
    const bool owner_held = owner_entered.load(std::memory_order_acquire) &&
                           !owner_timed_out.load(std::memory_order_acquire);
    release(2);
    waiter.join(); // Release/join our native waiter before reporting any prerequisite failure.
    Observer::Snapshot foreign_before{}, foreign_after{};
    bool foreign_published = false;
    if (owner_held && !owner_timed_out.load(std::memory_order_acquire)) {
      foreign_before = cache.ReadIndirectArgsDiagnostics();
      foreign_published = cache.PeekCompletedIndirectArgs(); // Main thread, not this renderer's owner.
      if (control_run) control_foreign = Control::Read(*cache.IndirectArgsObserver());
      foreign_after = cache.ReadIndirectArgsDiagnostics();
    }
    owner_release.store(true, std::memory_order_release);
    owner_hold.join(); // No assertion leaves our GPU-owner callback held.
    Require(name, "existing actual native wait", entered_exact &&
        waiter_finished.load(std::memory_order_acquire) &&
        master.ReadFaultPeekHostWait().completed_tick >= host_wait_tick,
        "fixture did not enter and finish the already-existing native host wait");
    const auto same_parent = [](const auto& a, const auto& b) {
      return a.sequence == b.sequence && a.domain == b.domain && a.id == b.id && a.family == b.family;
    };
    const auto same_dispatch = [&](const Observer::Dispatch& a, const Observer::Dispatch& b) {
      return a.valid == b.valid && a.ordinal == b.ordinal && a.sequence == b.sequence &&
          a.domain == b.domain && a.lifetime == b.lifetime && a.tick == b.tick &&
          a.shader == b.shader && a.guest_address == b.guest_address &&
          a.native_buffer == b.native_buffer && a.native_offset == b.native_offset &&
          a.recording_thread == b.recording_thread && same_parent(a.cause, b.cause) &&
          same_parent(a.guest, b.guest) && a.point == b.point && a.request == b.request;
    };
    bool foreign_unchanged = foreign_before.enabled == foreign_after.enabled &&
        foreign_before.available == foreign_after.available &&
        foreign_before.status == foreign_after.status &&
        foreign_before.selected_shader == foreign_after.selected_shader &&
        foreign_before.matched_total == foreign_after.matched_total &&
        foreign_before.copied_total == foreign_after.copied_total &&
        foreign_before.observed_total == 0 && foreign_after.observed_total == 0 &&
        foreign_before.capacity_skipped == foreign_after.capacity_skipped &&
        foreign_before.identity_errors == foreign_after.identity_errors &&
        foreign_before.cached_known_tick == foreign_after.cached_known_tick &&
        foreign_before.host_wait_entered_tick == foreign_after.host_wait_entered_tick &&
        foreign_before.host_waited_tick == foreign_after.host_waited_tick &&
        same_dispatch(foreign_before.latest_match, foreign_after.latest_match);
    for (size_t slot = 0; slot < Observer::Capacity; ++slot) {
      const auto& before_slot = foreign_before.slots[slot];
      const auto& after_slot = foreign_after.slots[slot];
      foreign_unchanged &= !before_slot.valid && !after_slot.valid &&
          before_slot.groups == after_slot.groups &&
          before_slot.observed_known_tick == after_slot.observed_known_tick &&
          before_slot.observed_host_waited_tick == after_slot.observed_host_waited_tick &&
          same_dispatch(before_slot.dispatch, after_slot.dispatch);
    }
    Require(name, "foreign owner cannot publish completed slots",
        owner_held && !owner_timed_out.load(std::memory_order_acquire) &&
            !foreign_published && foreign_unchanged,
        "foreign caller published or changed host-wait-authorized slots before owner publication");
    Require(name, "entered is not completed", entered_only.observed_total == 0 &&
        normal_callbacks.load(std::memory_order_relaxed) == 0,
        "wait entry alone published arguments or drained a pending normal callback");
    idle();
    const auto before = scheduler.ReadCausalState();
    gpu.SendCommandSync([&] {
      (void)cache.PeekCompletedIndirectArgs();
      observed = cache.ReadIndirectArgsDiagnostics();
      if (control_run) control_observed = Control::Read(*cache.IndirectArgsObserver());
      (void)cache.PeekCompletedIndirectArgs();
      repeated = cache.ReadIndirectArgsDiagnostics();
      if (control_run) control_repeated = Control::Read(*cache.IndirectArgsObserver());
    });
    const bool foreign_owner = cache.PeekCompletedIndirectArgs();
    const auto after = scheduler.ReadCausalState();
    Require(name, "owner-only passive observation", !foreign_owner &&
        before.available && after.available && before.normal_queued == after.normal_queued &&
        normal_callbacks.load(std::memory_order_relaxed) == 0,
        "passive observation changed callback policy or allowed a foreign owner");
  } else {
    release(1);
    master.Wait(producer_tick);
    idle();
    observed = cache.ReadIndirectArgsDiagnostics();
  }

  readback->Invalidate(0, readback_bytes);
  for (size_t i = 0; i < groups_x.size(); ++i) {
    std::array<u32, 5> actual_args{};
    std::array<u32, output_words + 2> actual_output{};
    std::memcpy(actual_args.data(), readback->Mapped().data() + i * case_copy_bytes,
                sizeof(actual_args));
    std::memcpy(actual_output.data(),
                readback->Mapped().data() + i * case_copy_bytes + args_copy_bytes,
                sizeof(actual_output));
    Require(name, "real GPU argument and guard bytes",
        actual_args == std::array<u32, 5>{sentinel, groups_x[i] == 0 ? 0u : 7u, 1, 1, sentinel},
        "GPU producer/target argument mutation or argument canary check failed");
    Require(name, "real GPU output canaries",
        actual_output.front() == sentinel && actual_output.back() == sentinel,
        "indirect consumer changed an output guard word");
    for (u32 lane = 0; lane < output_words; ++lane)
      Require(name, "zero/nonzero indirect GPU work",
          actual_output[lane + 1] == (lane < groups_x[i] * 4u ? lane + 1u : sentinel),
          "actual indirect work or its untouched tail disagreed with GPU-written arguments");
    std::array<u32, 3> backing{};
    Require(name, "no guest readback side effect",
        LibKernel::Memory::TryReadBacking(args_at(i), backing.data(), sizeof(backing)) &&
            backing == stale_args,
        "native observation unexpectedly published arguments into guest CPU backing");
  }
  Require(name, "native completion is not a callback drain",
      master.KnownGpuTick() >= producer_tick && normal_callbacks.load(std::memory_order_relaxed) == 0,
      "native completion/observation consumed the unrelated normal callback");
  gpu.SendCommandSync([&] { scheduler.BeginCommand(); scheduler.PopPendingOperations(); });
  idle();
  Require(name, "controlled normal callback", normal_callbacks.load(std::memory_order_relaxed) == 1,
      "the explicit test-only Pop did not execute exactly the pending control callback");
  std::printf("[phase]   %s: ten GPU-written zero/nonzero dispatches, rewritten arguments, guards, and controlled callback passed\n", name);

  if (diagnostic && !control_run) {
    // Intentional RED occurs only after all actual GPU/output/wait controls above.
    Require(name, "persistent exact indirect arguments",
        observed.enabled && observed.available && observed.observed_total == Observer::Capacity,
        "real GPU controls passed, but exact pre-dispatch argument observations are unavailable");
    Require(name, "bounded immutable capacity",
        observed.selected_shader == expected_hash && observed.matched_total == matching_cases &&
            observed.copied_total == Observer::Capacity && observed.capacity_skipped == 1 &&
            observed.identity_errors == 0 && repeated.observed_total == Observer::Capacity &&
            repeated.matched_total == matching_cases && repeated.capacity_skipped == 1,
        "observer overwrote a slot, repeated a capture, or hid a filtered/capacity-lost match");
    const auto same_parent = [](const auto& a, const auto& b) {
      return a.sequence == b.sequence && a.domain == b.domain && a.id == b.id && a.family == b.family;
    };
    const auto check_identity = [&](const Observer::Dispatch& item, size_t i) {
      return item.valid && item.ordinal == i + 1 && item.sequence == dispatch_sequences[i] &&
          item.domain == Causal::Domain(&master) && item.lifetime == scheduler.CausalLifetimeSequence() &&
          item.tick == dispatch_ticks[i] && item.shader == expected_hash &&
          item.guest_address == args_at(i) && item.native_buffer == native_handles[i] &&
          item.native_offset == native_offsets[i] && item.recording_thread == recording_thread &&
          same_parent(item.cause, origins[i]) && same_parent(item.guest, origins[i]) &&
          item.point == 1000u + i && item.request == 2000u + i;
    };
    for (size_t i = 0; i < Observer::Capacity; ++i) {
      const auto& slot = observed.slots[i];
      Require(name, "exact joined native argument payload",
          slot.valid && check_identity(slot.dispatch, i) &&
              slot.groups == std::array<u32, 3>{groups_x[i], 1, 1} &&
              slot.observed_known_tick >= producer_tick &&
              slot.observed_host_waited_tick >= host_wait_tick &&
              repeated.slots[i].valid && check_identity(repeated.slots[i].dispatch, i) &&
              repeated.slots[i].groups == slot.groups,
          "an observation used later/stale arguments or the wrong dispatch/guest/native identity");
    }
    Require(name, "uncaptured latest match is explicit",
        check_identity(observed.latest_match, matching_cases - 1),
        "the ninth matching dispatch was silently represented by an older captured payload");
    Require(name, "insufficient host-read proof remains explicit",
        blocked.available && blocked.status == Observer::Status::NativeIncomplete &&
            query_only.available && query_only.status == Observer::Status::HostWaitIncomplete &&
            query_only.cached_known_tick >= producer_tick &&
            query_only.host_waited_tick < producer_tick &&
            entered_only.available && entered_only.status == Observer::Status::HostWaitIncomplete &&
            entered_only.host_wait_entered_tick == host_wait_tick &&
            entered_only.host_waited_tick < producer_tick,
        "blocked/query-only/wait-entry-only states concealed incomplete host synchronization");
  } else if (control_run) {
    Require(name, "control never publishes typed argument bytes",
        observed.enabled && observed.available && observed.matched_total == matching_cases &&
            observed.copied_total == 0 && observed.observed_total == 0 &&
            observed.capacity_skipped == 1 && observed.identity_errors == 0 &&
            repeated.copied_total == 0 && repeated.observed_total == 0 &&
            !control_foreign.available && !control_foreign.raw_scratch_read,
        "control metadata/scratch bytes were confused with captured arguments or a foreign owner read them");
    Require(name, "bounded control capacity",
        control_observed.available && control_repeated.available &&
            control_observed.recorded_total == Observer::Capacity &&
            control_repeated.recorded_total == Observer::Capacity,
        "control did not retain exactly the first eight of nine matching dispatches");
    const auto same_parent = [](const auto& a, const auto& b) {
      return a.sequence == b.sequence && a.domain == b.domain && a.id == b.id && a.family == b.family;
    };
    const auto check_identity = [&](const Observer::Dispatch& item, size_t i) {
      return item.valid && item.ordinal == i + 1 && item.sequence == dispatch_sequences[i] &&
          item.domain == Causal::Domain(&master) && item.lifetime == scheduler.CausalLifetimeSequence() &&
          item.tick == dispatch_ticks[i] && item.shader == expected_hash &&
          item.guest_address == args_at(i) && item.native_buffer == native_handles[i] &&
          item.native_offset == native_offsets[i] && item.recording_thread == recording_thread &&
          same_parent(item.cause, origins[i]) && same_parent(item.guest, origins[i]) &&
          item.point == 1000u + i && item.request == 2000u + i;
    };
    for (size_t i = 0; i < Observer::Capacity; ++i) {
      Require(name, "untyped exact control dispatch identity",
          check_identity(control_observed.dispatches[i], i) &&
              check_identity(control_repeated.dispatches[i], i) &&
              !observed.slots[i].valid && !repeated.slots[i].valid &&
              observed.slots[i].groups == std::array<u32, 3>{} &&
              repeated.slots[i].groups == std::array<u32, 3>{},
          "control changed argument snapshot semantics or lost its actual dispatch identity");
    }
    Require(name, "uncaptured ninth control match remains explicit",
        check_identity(observed.latest_match, matching_cases - 1),
        "latest match was silently represented by an earlier controlled dispatch");
    if (control_mode == Control::Mode::ImmutableScratchCopy) {
      Require(name, "raw scratch requires actual completed host wait",
          !control_blocked.raw_scratch_read && !control_query.raw_scratch_read &&
              !control_entered.raw_scratch_read && control_observed.raw_scratch_read &&
              control_repeated.raw_scratch_read &&
              control_observed.cached_known_tick >= producer_tick &&
              control_observed.actual_host_waited_tick >= host_wait_tick,
          "scratch bytes were read before native completion and the actual host-wait witness");
      const std::array<u32, 5> expected_scratch {
          scratch_canary, scratch_payload[0], scratch_payload[1], scratch_payload[2], scratch_canary};
      Require(name, "immutable initialized scratch source",
          control_observed.source_words == expected_scratch &&
              control_repeated.source_words == expected_scratch &&
              control_observed.scratch_source_handle != 0 &&
              control_observed.scratch_destination_handle != 0 &&
              control_observed.scratch_source_handle != control_observed.scratch_destination_handle,
          "scratch source/guards or distinct native allocation identity changed");
      for (size_t i = 0; i < Observer::Capacity; ++i) {
        Require(name, "actual scratch copy and per-slot canaries",
            control_observed.slots[i] == expected_scratch &&
                control_repeated.slots[i] == expected_scratch &&
                control_observed.scratch_source_handle != native_handles[i] &&
                control_observed.scratch_destination_handle != native_handles[i],
            "scratch transfer, its twelve-byte range, immutable slot, or native ownership check failed");
      }
      std::printf("[control] immutable-scratch-copy: copies=8 bytes_per_copy=12 typed_arguments=not-captured guards=16 checked\n");
    } else {
      Require(name, "barrier-only has no raw payload",
          !control_observed.raw_scratch_read && !control_repeated.raw_scratch_read,
          "barrier-only control unexpectedly read a payload");
      std::printf("[control] composed-broad-barrier: barriers=8 copies=0 typed_arguments=not-captured; not an exact-equivalence claim\n");
    }
  } else {
    Require(name, "diagnostic allocation/default-off",
        cache.IndirectArgsObserver() == nullptr && !observed.enabled && !observed.available &&
            observed.matched_total == 0 && observed.observed_total == 0,
        "missing filter or required child gate still created an observer");
  }
  gpu.SendCommandSync([&] {
    readback.reset();
    context.UnmapMemory(base, size);
    scheduler.Finish();
  });
  LibKernel::Memory::InstallGpuResources(nullptr);
  context.ShutdownGpu();
  m_runtime_context.device.destroySemaphore(gate, nullptr);
  Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
      "indirect diagnostic test memory release failed");
  std::printf("[gpu]     %-32s ok\n", name);
}

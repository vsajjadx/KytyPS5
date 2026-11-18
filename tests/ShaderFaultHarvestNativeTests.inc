// Included inside VulkanHarness. Real shader miss, native completion, and host
// harvest are separately observed. This test never changes runtime queue policy.
// --fault-harvest-only uses KYTY_CAUSAL_DIAG=1 and require_witness=true.
void CheckFaultHarvestIdle(bool require_witness) {
  namespace Causal = Libs::Graphics::CausalDiagnostics;
  const bool diagnostic = Causal::Enabled();
  const bool passive = FaultPeekEnabled();
  Require("FaultHarvestIdle", "diagnostic test opt-in",
          !require_witness || diagnostic, "typed witness case requires KYTY_CAUSAL_DIAG=1");
  Require("FaultHarvestIdle", "external gate precondition", !GpuSyncDebugEnabled(),
          "external timeline gate requires KYTY_GPU_SYNC to be disabled");
  constexpr uintptr_t base = 0x000000020b600000ull;
  constexpr uint64_t size = 0x40000;
  constexpr uint64_t table = base + 0x100, leaf_a = base + 0x1000,
                     leaf_b = base + 0x1100, leaf_c = base + 0x1200,
                     scratch = base + 0x3000, output_guard = base + 0x4000,
                     output = output_guard + sizeof(u32);
  constexpr u32 lanes = 64, sentinel = 0xa5a5a5a5u, value_a = 0x11u,
                value_b = 0x22u, value_c = 0x33u;
  constexpr uint64_t output_bytes = (lanes + 2u) * sizeof(u32);
  constexpr uint64_t table_bytes = 8u * sizeof(u32), leaf_bytes = 3u * sizeof(u32);
  constexpr uint64_t table_copy = output_bytes, leaf_a_copy = table_copy + table_bytes,
                     leaf_b_copy = leaf_a_copy + leaf_bytes,
                     readback_bytes = leaf_b_copy + leaf_bytes;
    std::vector<u32> code{
        EncodeSop1(0x04, 20, 126),                    // s_mov_b64 s[20:21], exec
        EncodeVop2(0x1b, 1, InlineU32(1), 0),         // v_and_b32 v1, 1, v0   (per-lane index)
        EncodeVop2(0x1a, 2, InlineU32(2), 0),         // v_lshlrev_b32 v2, 2, v0
        EncodeVop1(0x01, 6, InlineU32(0)),            // v_mov_b32 v6, 0
        EncodeVop1(0x01, 3, 6),                       // v_mov_b32 v3, s6 (new entry-1 lo)
        EncodeVop1(0x01, 4, 7),                       // v_mov_b32 v4, s7 (new entry-1 hi)
        EncodeMubuf0(0x1c, 16, false, false), EncodeMubuf1(3, 3, 0),  // store v3 -> s[12:15]+16
        EncodeMubuf0(0x1c, 20, false, false), EncodeMubuf1(4, 3, 0),  // store v4 -> s[12:15]+20
        EncodeVop1(0x01, 8, 16),                      // v_mov_b32 v8, s16 (new entry-0 lo)
        EncodeVop1(0x01, 9, 17),                      // v_mov_b32 v9, s17 (new entry-0 hi)
        EncodeMubuf0(0x1c, 0, false, false), EncodeMubuf1(8, 3, 0),   // store v8 -> s[12:15]+0
        EncodeMubuf0(0x1c, 4, false, false), EncodeMubuf1(9, 3, 0),   // store v9 -> s[12:15]+4
        EncodeSopp(0x0c, 0),                          // s_waitcnt
        0xf4800000u, 0xfa000000u};                    // s_dcache_inv (captured f5edf70dbb1f6057 encoding)
    const auto loop = code.size();
    code.insert(code.end(), {
        EncodeVop1(0x02, 4, Vgpr(1)),                 // v_readfirstlane_b32 s4, v1
        EncodeSop1(0x04, 18, 126),                    // s_mov_b64 s[18:19], exec
        EncodeVopc(0xd2, 4, 1),                       // v_cmpx_eq_u32 s4, v1
        EncodeSop2(0x1e, 5, 4, InlineU32(4)),         // s_lshl_b32 s5, s4, 4
        EncodeSmem0(0x09, 14, 4), 5u << 25u,          // s_buffer_load_dwordx2 s[14:15], s[8:11], soffset s5
        EncodeSopp(0x0c, 0),                          // s_waitcnt
        EncodeSop1(0x21, 14, 14),                     // s_swappc_b64 s[14:15], s[14:15]
        EncodeSop2(0x15, 18, 18, 126),                // s_andn2_b64 s[18:19], s[18:19], exec
        EncodeSop1(0x04, 126, 18)});                  // s_mov_b64 exec, s[18:19]
    code.push_back(EncodeSopp(0x09, static_cast<u32>(static_cast<int32_t>(loop) -
                                                     static_cast<int32_t>(code.size() + 1u)) & 0xffffu));
    code.insert(code.end(), {
        EncodeSop1(0x04, 126, 20),                    // s_mov_b64 exec, s[20:21]
        EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 2)});  // buffer_store_dword v6, v2, s[0:3]
  AppendEnd(&code);
  ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
      {.type = Prospero::ShaderBinaryType::kCs,
       .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
  EnsureRuntimeContext();
  for (const bool unknown : {false, true}) {
    const char* name = unknown ? "FaultHarvestIdleUnknown" : "FaultHarvestIdleNull";
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
        "fault-harvest test allocation failed");
    void* mapped = reinterpret_cast<void*>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, size, 0x3, 0x10, physical, size) == 0 &&
        mapped == reinterpret_cast<void*>(base), "fault-harvest test mapping failed");
    std::memset(mapped, 0, size);
    const std::array<u32, 3> leaf_words_a{
        EncodeVop1(0x01, 6, 255), value_a, EncodeSop1(0x21, 125, 14)};
    const std::array<u32, 3> leaf_words_b{
        EncodeVop1(0x01, 6, 255), value_b, EncodeSop1(0x21, 125, 14)};
    const std::array<u32, 3> leaf_words_c{
        EncodeVop1(0x01, 6, 255), value_c, EncodeSop1(0x21, 125, 14)};
    LibKernel::Memory::WriteBacking(leaf_a, leaf_words_a.data(), leaf_bytes);
    LibKernel::Memory::WriteBacking(leaf_b, leaf_words_b.data(), leaf_bytes);
    LibKernel::Memory::WriteBacking(leaf_c, leaf_words_c.data(), leaf_bytes);
    const uint64_t initial_second = unknown ? leaf_b : 0;
    std::array<u32, 8> table_words{
        static_cast<u32>(leaf_a), static_cast<u32>(leaf_a >> 32u), 0, 0,
        static_cast<u32>(initial_second), static_cast<u32>(initial_second >> 32u), 0, 0};
    LibKernel::Memory::WriteBacking(table, table_words.data(), sizeof(table_words));
    std::array<u32, lanes + 2u> output_words;
    output_words.fill(sentinel);
    LibKernel::Memory::WriteBacking(output_guard, output_words.data(), sizeof(output_words));

    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    auto& gpu = context.GetGpu();
    auto& scheduler = context.GetCommandScheduler();
    auto& cache = context.GetBufferCache();
    const auto& faults = BufferCacheTestAccess::Faults(cache);
    CommandProcessor processor(context, 0);
    std::unique_ptr<Libs::Graphics::Buffer> readback;
    vk::Semaphore gate = nullptr;
    uint64_t producer_tick = 0, harvest_tick = 0, unrelated_tick = 0, host_wait_tick = 0;
    uint32_t recording_thread = 0;
    std::atomic_uint unrelated_callbacks{0};
    FaultManager::HarvestSnapshot before_gc{}, queued{}, native_done{}, completed{}, after_unrelated{};
    FaultManager::PeekSnapshot blocked_peek{}, query_peek{}, entered_peek{},
        observed_peek{}, repeat_peek{}, harvested_peek{};
    const uint64_t misses_before = faults.CallTableMisses();
    uint64_t expected_shader_hash = 0;
    const auto idle = [&] {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      bool observed = false;
      do {
        const auto state = GuestGpuTestAccess::Snapshot(gpu);
        if (state.consistent && !state.processing && state.commands == 0 &&
            state.pending_commands == 0 && state.submissions == 0) {
          observed = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (std::chrono::steady_clock::now() < deadline);
      Require(name, "renderer idle", observed, "renderer did not return to an empty idle queue");
    };
    const auto release = [&](uint64_t value) {
      vk::SemaphoreSignalInfo signal{};
      signal.sType = vk::StructureType::eSemaphoreSignalInfo;
      signal.semaphore = gate;
      signal.value = value;
      Require(name, "native gate release",
          m_runtime_context.device.signalSemaphore(&signal) == vk::Result::eSuccess,
          "host could not release the already-submitted native gate");
    };
    gpu.SendCommandSync([&] {
      processor.BufferInit();
      context.MapMemory(base, size);
      auto& shaders = processor.GetShCtx();
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
          .num_thread_x = lanes, .num_thread_y = 1, .num_thread_z = 1,
          .wave_size = 64, .user_sgpr = 18});
      const auto descriptor = [](uint64_t address, u32 bytes) {
        ShaderBufferResource result{};
        result.UpdateAddress48(address);
        result.fields[2] = bytes;
        result.fields[3] = DstSel(4, 5, 6, 7) |
            (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
        return result;
      };
      const auto set_vsharp = [&](u32 reg, uint64_t address, u32 bytes) {
        const auto value = descriptor(address, bytes);
        for (u32 word = 0; word < 4; ++word) shaders.SetCsUserSgpr(reg + word, value.fields[word]);
      };
      set_vsharp(0, output, lanes * sizeof(u32));
      set_vsharp(8, table, table_bytes);
      set_vsharp(12, unknown ? table : scratch, table_bytes);
      const uint64_t replacement = unknown ? leaf_c : 0;
      shaders.SetCsUserSgpr(6, static_cast<u32>(replacement));
      shaders.SetCsUserSgpr(7, static_cast<u32>(replacement >> 32u));
      shaders.SetCsUserSgpr(16, static_cast<u32>(leaf_a));
      shaders.SetCsUserSgpr(17, static_cast<u32>(leaf_a >> 32u));
      // Canonical owners permit a direct native copy without ReadMemory/Finish/Pop.
      cache.PrepareBdaRange(base, size);
      context.PrepareBda();
      ShaderComputeInputInfo input{};
      const auto* program = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
      Require(name, "real table-call module", program != nullptr && input.stage.call_closure,
          "the deliberate GPU miss fixture did not bind a real table-call module");
      expected_shader_hash = input.stage.call_closure->shader_hash;
      readback = std::make_unique<Libs::Graphics::Buffer>(
          context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
          vk::BufferUsageFlagBits::eTransferDst, readback_bytes);
      vk::SemaphoreTypeCreateInfo type{};
      type.sType = vk::StructureType::eSemaphoreTypeCreateInfo;
      type.semaphoreType = vk::SemaphoreType::eTimeline;
      vk::SemaphoreCreateInfo create{};
      create.sType = vk::StructureType::eSemaphoreCreateInfo;
      create.pNext = &type;
      Require(name, "native gate creation",
          m_runtime_context.device.createSemaphore(&create, nullptr, &gate) == vk::Result::eSuccess,
          "could not create the external native gate");
      processor.DispatchDirect(1, 1, 1, 0x41u);
      const auto command = scheduler.Current().Handle();
      command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
          vk::PipelineStageFlagBits::eTransfer, {},
          vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                            .dstAccessMask = vk::AccessFlagBits::eTransferRead}, {}, {});
      const auto copy = [&](uint64_t address, uint64_t bytes, uint64_t destination) {
        vk::DescriptorBufferInfo native{};
        Require(name, "canonical native copy", cache.CanonicalBinding(address, bytes, native),
            "GPU output/guard source lost its canonical owner");
        command.copyBuffer(native.buffer, readback->Handle(),
            vk::BufferCopy{native.offset, destination, bytes});
      };
      copy(output_guard, output_bytes, 0);
      copy(table, table_bytes, table_copy);
      copy(leaf_a, leaf_bytes, leaf_a_copy);
      copy(leaf_b, leaf_bytes, leaf_b_copy);
      command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
          vk::PipelineStageFlagBits::eHost, {},
          vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                            .dstAccessMask = vk::AccessFlagBits::eHostRead}, {}, {});
      scheduler.NoteRecordedWork();
      SubmitInfo blocked;
      blocked.AddWait(gate, 1);
      producer_tick = scheduler.Submit(blocked);
    });
    idle();
    release(1);
    // MasterSemaphore::Wait never consumes normal callbacks; the worker is free to idle.
    scheduler.GetMasterSemaphore().Wait(producer_tick);
    idle();
    Require(name, "native producer completion",
        scheduler.GetMasterSemaphore().KnownGpuTick() >= producer_tick &&
            faults.CallTableMisses() == misses_before,
        "native completion unexpectedly required or executed host fault harvest");
    readback->Invalidate(0, readback_bytes);
    std::memcpy(output_words.data(), readback->Mapped().data(), sizeof(output_words));
    Require(name, "native output canaries",
        output_words.front() == sentinel && output_words.back() == sentinel,
        "the real shader overwrote the output guard words");
    for (u32 lane = 0; lane < lanes; ++lane)
      Require(name, "native callable/miss outputs",
          output_words[lane + 1u] == ((lane & 1u) == 0 ? value_a : 0u),
          "the deliberate missing call did not retain zero while known lanes called A");
    const uint64_t expected_missing = unknown ? leaf_c : 0;
    table_words[4] = static_cast<u32>(expected_missing);
    table_words[5] = static_cast<u32>(expected_missing >> 32u);
    Require(name, "GPU-selected table and native code canaries",
        std::memcmp(readback->Mapped().data() + table_copy, table_words.data(), table_bytes) == 0 &&
        std::memcmp(readback->Mapped().data() + leaf_a_copy, leaf_words_a.data(), leaf_bytes) == 0 &&
        std::memcmp(readback->Mapped().data() + leaf_b_copy, leaf_words_b.data(), leaf_bytes) == 0,
        "GPU pointer mutation or immutable callee-code oracle was not satisfied");
    before_gc = faults.ReadHarvestDiagnostics();

    gpu.SendCommandSync([&] {
      scheduler.BeginCommand();
      recording_thread = static_cast<uint32_t>(Common::Thread::GetThreadIdUnique());
      context.RunGarbageCollector(); // Real parser/copy/clear plus ordinary deferred callback.
      queued = faults.ReadHarvestDiagnostics();
      SubmitInfo blocked;
      blocked.AddWait(gate, 2);
      harvest_tick = scheduler.Submit(blocked);
    });
    idle();
    Require(name, "harvest gate held", faults.CallTableMisses() == misses_before,
        "host harvest ran while its native copy was still blocked");
    if (passive) {
      gpu.SendCommandSync([&] {
        (void)cache.PeekCompletedFaultDownloads();
        blocked_peek = faults.ReadFaultPeekDiagnostics();
      });
      Require(name, "no read of blocked staged bytes", !blocked_peek.areas[0].valid &&
          faults.CallTableMisses() == misses_before,
          "passive observation read bytes before the native copy completed");
      release(2);
      // A test-only native query proves that a numeric completion observation
      // alone does not grant the stronger existing-host-wait witness.
      auto& master = scheduler.GetMasterSemaphore();
      const auto query_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (master.KnownGpuTick() < harvest_tick &&
             std::chrono::steady_clock::now() < query_deadline) {
        master.Refresh();
        std::this_thread::yield();
      }
      Require(name, "query-only native completion", master.KnownGpuTick() >= harvest_tick &&
          master.ReadFaultPeekHostWait().completed_tick < harvest_tick,
          "query-only phase did not isolate native completion from a real host wait");
      gpu.SendCommandSync([&] {
        (void)cache.PeekCompletedFaultDownloads();
        query_peek = faults.ReadFaultPeekDiagnostics();
        scheduler.BeginCommand();
        SubmitInfo blocked;
        blocked.AddWait(gate, 3);
        host_wait_tick = scheduler.Submit(blocked);
      });
      idle();
      Require(name, "query-only peek refusal", !query_peek.areas[0].valid &&
          faults.CallTableMisses() == misses_before,
          "cached timeline completion was used as a host-read authorization");

      // The external gate remains held until the already-selected native wait
      // branch publishes its requested tick. No sleep determines this ordering.
      std::atomic_bool waiter_finished{false};
      std::jthread waiter([&] {
        master.Wait(host_wait_tick);
        waiter_finished.store(true, std::memory_order_release);
      });
      const auto enter_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      auto wait_state = master.ReadFaultPeekHostWait();
      while (wait_state.entered_tick < host_wait_tick &&
             !waiter_finished.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < enter_deadline) {
        std::this_thread::yield();
        wait_state = master.ReadFaultPeekHostWait();
      }
      const bool entered_exact = wait_state.entered_tick == host_wait_tick &&
          wait_state.completed_tick < harvest_tick;
      if (entered_exact) {
        gpu.SendCommandSync([&] {
          (void)cache.PeekCompletedFaultDownloads();
          entered_peek = faults.ReadFaultPeekDiagnostics();
        });
      }
      // Even a failed witness prerequisite releases this test's own gate and
      // joins its own waiter before reporting failure. No process is terminated.
      release(3);
      waiter.join();
      Require(name, "actual native wait branch", entered_exact &&
          waiter_finished.load(std::memory_order_acquire) &&
          master.ReadFaultPeekHostWait().completed_tick >= host_wait_tick,
          "the fixture did not enter and complete the existing real native wait branch");
      Require(name, "entered is not completed", !entered_peek.areas[0].valid &&
          faults.CallTableMisses() == misses_before,
          "entering a native wait was mistaken for completed host synchronization");
      idle();

      const auto pending_before = scheduler.ReadCausalState();
      gpu.SendCommandSync([&] {
        (void)cache.PeekCompletedFaultDownloads();
        observed_peek = faults.ReadFaultPeekDiagnostics();
        (void)cache.PeekCompletedFaultDownloads();
        repeat_peek = faults.ReadFaultPeekDiagnostics();
      });
      const bool wrong_owner_read = cache.PeekCompletedFaultDownloads();
      const auto pending_after = scheduler.ReadCausalState();
      Require(name, "passive observation keeps normal callback pending",
          pending_before.available && pending_after.available &&
          pending_before.normal_queued == pending_after.normal_queued &&
          faults.CallTableMisses() == misses_before && !wrong_owner_read,
          "passive observation consumed a callback, changed the host counter, or admitted a foreign owner");
    } else {
      release(2);
      scheduler.GetMasterSemaphore().Wait(harvest_tick);
      idle();
    }
    native_done = faults.ReadHarvestDiagnostics();
    Require(name, "native copy complete without host harvest",
        scheduler.GetMasterSemaphore().KnownGpuTick() >= harvest_tick &&
            faults.CallTableMisses() == misses_before,
        "native completion was conflated with running the pending normal harvest callback");

    // This explicit Pop is a test control, never an idle-path/runtime policy change.
    gpu.SendCommandSync([&] {
      scheduler.BeginCommand();
      scheduler.PopPendingOperations();
      completed = faults.ReadHarvestDiagnostics();
      harvested_peek = faults.ReadFaultPeekDiagnostics();
    });
    idle();
    const uint64_t actual_misses = faults.CallTableMisses() - misses_before;
    const auto actual_miss = faults.LastCallTableMiss();
    Require(name, "controlled actual harvest",
        actual_misses != 0 && actual_miss.target == expected_missing &&
            actual_miss.shader_hash == expected_shader_hash && actual_miss.pc != 0,
        "controlled Pop did not publish the genuine shader-produced miss tuple");
    std::printf("[phase]   %s: native tick %llu complete, host miss count stayed unchanged until controlled harvest tick %llu\n",
        name, static_cast<unsigned long long>(producer_tick),
        static_cast<unsigned long long>(harvest_tick));

    // A later unrelated normal callback must not impersonate a newer fault harvest.
    gpu.SendCommandSync([&] {
      scheduler.DeferOperation([&] { unrelated_callbacks.fetch_add(1, std::memory_order_relaxed); });
      unrelated_tick = scheduler.Submit();
    });
    idle();
    scheduler.GetMasterSemaphore().Wait(unrelated_tick);
    gpu.SendCommandSync([&] {
      scheduler.BeginCommand();
      scheduler.PopPendingOperations();
      after_unrelated = faults.ReadHarvestDiagnostics();
    });
    idle();
    Require(name, "unrelated normal callback control",
        unrelated_callbacks.load(std::memory_order_relaxed) == 1 &&
            faults.CallTableMisses() == misses_before + actual_misses,
        "unrelated normal callback changed or replayed the harvested miss");

    if (diagnostic) {
      // Intentional RED is here, after real GPU separation and the controlled harvest.
      Require(name, "typed persistent harvest witness",
          before_gc.enabled && before_gc.available && queued.enabled && queued.available &&
          native_done.enabled && native_done.available && completed.enabled && completed.available &&
          after_unrelated.enabled && after_unrelated.available,
          "real native completion/idle separation passed, but typed harvest witness is unavailable");
      const auto& pending = queued.areas[0];
      const auto& still_pending = native_done.areas[0];
      const auto& done = completed.last_completed;
      const auto same_operation = [](const auto& a, const auto& b) {
        return a.valid && b.valid && a.area == b.area && a.domain == b.domain &&
            a.lifetime == b.lifetime && a.operation_id == b.operation_id && a.tick == b.tick &&
            a.enqueue_sequence == b.enqueue_sequence && a.enqueue_thread == b.enqueue_thread;
      };
      Require(name, "no fabricated pre-GC harvest",
          before_gc.queued_total == 0 && before_gc.completed_total == 0 &&
              !before_gc.last_completed.valid && before_gc.cumulative_call_table_misses == misses_before,
          "an unscheduled native miss was labeled as harvested");
      Require(name, "exact accepted pending identity",
          pending.valid && pending.area == 0 && pending.phase == FaultManager::HarvestPhase::Queued &&
              pending.domain == Causal::Domain(&scheduler.GetMasterSemaphore()) &&
              pending.lifetime == scheduler.CausalLifetimeSequence() && pending.operation_id != 0 &&
              pending.tick == harvest_tick && pending.enqueue_sequence != 0 &&
              pending.enqueue_thread == recording_thread && queued.queued_total == 1 &&
              queued.completed_total == 0 && queued.identity_errors == 0,
          "pending witness did not identify the actual accepted native operation");
      Require(name, "native completion is not host completion",
          same_operation(pending, still_pending) &&
              still_pending.phase == FaultManager::HarvestPhase::Queued &&
              native_done.completed_total == 0 && !native_done.last_completed.valid &&
              native_done.cumulative_call_table_misses == misses_before,
          "completed native work was mislabeled as a completed host harvest");
      Require(name, "completed exact miss and page counts",
          same_operation(pending, done) && done.phase == FaultManager::HarvestPhase::Completed &&
              completed.queued_total == 1 && completed.completed_total == 1 &&
              completed.identity_errors == 0 && done.miss_invocations == actual_misses &&
              done.call_miss.shader_hash == actual_miss.shader_hash &&
              done.call_miss.target == actual_miss.target && done.call_miss.pc == actual_miss.pc &&
              done.raw_page_faults == 0 && done.page_faults == 0 &&
              completed.cumulative_call_table_misses == misses_before + actual_misses &&
              same_operation(done, completed.areas[0]),
          "completed witness preceded actual processing or changed the already-read payload");
      Require(name, "persistent last fault harvest",
          same_operation(done, after_unrelated.last_completed) &&
              after_unrelated.last_completed.phase == FaultManager::HarvestPhase::Completed &&
              after_unrelated.completed_total == 1 &&
              after_unrelated.cumulative_call_table_misses == misses_before + actual_misses &&
              unrelated_tick > done.tick,
          "a generic later callback displaced the last proven fault harvest");
      if (passive) {
        // Intended RED is after the genuine miss, query-only/native-wait
        // separation, unchanged pending callback, and controlled actual harvest.
        const auto& peek = observed_peek.areas[0];
        Require(name, "persistent passive staged observation",
            observed_peek.enabled && observed_peek.available && peek.valid &&
                peek.staged.phase == FaultManager::HarvestPhase::Queued &&
                same_operation(peek.staged, pending) &&
                peek.staged.miss_invocations == actual_misses &&
                peek.staged.call_miss.shader_hash == actual_miss.shader_hash &&
                peek.staged.call_miss.target == actual_miss.target &&
                peek.staged.call_miss.pc == actual_miss.pc &&
                peek.staged.raw_page_faults == done.raw_page_faults &&
                peek.staged.page_faults == done.page_faults &&
                peek.observed_known_tick >= harvest_tick &&
                peek.observed_host_waited_tick >= host_wait_tick,
            "real GPU miss and controlled harvest passed, but completed staged bytes were not passively observed");
        Require(name, "insufficient witnesses remain explicit",
            blocked_peek.enabled && blocked_peek.available &&
                blocked_peek.status == FaultManager::PeekStatus::NativeIncomplete &&
                query_peek.enabled && query_peek.available &&
                query_peek.status == FaultManager::PeekStatus::HostWaitIncomplete &&
                query_peek.cached_known_tick >= harvest_tick &&
                query_peek.host_waited_tick < harvest_tick &&
                entered_peek.enabled && entered_peek.available &&
                entered_peek.status == FaultManager::PeekStatus::HostWaitIncomplete &&
                entered_peek.host_wait_entered_tick == host_wait_tick &&
                entered_peek.host_waited_tick < harvest_tick,
            "blocked, query-only or entered-only states concealed insufficient host-read proof");
        Require(name, "one observation per accepted operation",
            observed_peek.observed_total == 1 && repeat_peek.observed_total == 1 &&
                harvested_peek.observed_total == 1 && repeat_peek.areas[0].valid &&
                harvested_peek.areas[0].valid &&
                same_operation(peek.staged, repeat_peek.areas[0].staged) &&
                same_operation(peek.staged, harvested_peek.areas[0].staged) &&
                repeat_peek.areas[0].staged.miss_invocations == actual_misses &&
                harvested_peek.areas[0].staged.miss_invocations == actual_misses,
            "repeated peeks or real callback processing replayed or replaced the passive observation");
        std::printf("[phase]   %s: staged miss observed after actual wait tick %llu while normal harvest stayed pending\n",
            name, static_cast<unsigned long long>(host_wait_tick));
      }
    } else {
      Require(name, "disabled harvest snapshot",
          !before_gc.enabled && !queued.enabled && !native_done.enabled && !completed.enabled &&
              !after_unrelated.enabled && !before_gc.available && !completed.available,
          "disabled diagnostics exposed or initialized a harvest witness");
    }
    if (!passive) {
      const auto disabled_peek = faults.ReadFaultPeekDiagnostics();
      const auto disabled_wait = scheduler.GetMasterSemaphore().ReadFaultPeekHostWait();
      Require(name, "passive diagnostics default off",
          !disabled_peek.enabled && !disabled_peek.available && disabled_peek.observed_total == 0 &&
              disabled_wait.entered_tick == 0 && disabled_wait.completed_tick == 0,
          "passive state was accessed without both exact child diagnostic flags");
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
        "fault-harvest test direct memory release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }
}

// Real GuestGpu scheduling regressions. This include is inside VulkanHarness.
void CheckGuestGpuQueueSelection(bool deferred_case = true,
                                bool control_cases = true) {
  EnsureRuntimeContext();
  constexpr auto deadline = std::chrono::seconds(2);
  const auto write_packet = [](uint32_t *destination, uint32_t value) {
    const auto address = reinterpret_cast<uint64_t>(destination);
    return std::array<uint32_t, 5>{
        KYTY_PM4(5, Pm4::IT_WRITE_DATA, 0), 0,
        static_cast<uint32_t>(address),
        static_cast<uint32_t>(address >> 32u), value};
  };
  const auto wait_packet = [&](uint32_t *label, uint32_t *destination,
                               uint32_t value) {
    const auto address = reinterpret_cast<uint64_t>(label);
    std::array<uint32_t, 12> packet{};
    packet[0] = KYTY_PM4(7, Pm4::IT_WAIT_REG_MEM, 0);
    packet[1] = 0x10u | 3u;
    packet[2] = static_cast<uint32_t>(address);
    packet[3] = static_cast<uint32_t>(address >> 32u);
    packet[4] = 1;
    packet[5] = UINT32_MAX;
    const auto suffix = write_packet(destination, value);
    std::copy(suffix.begin(), suffix.end(), packet.begin() + 7);
    return packet;
  };
  const auto sync = [&](GuestGpu &gpu, const char *name, auto &&operation,
                        const std::function<void()> &cleanup = {}) {
    std::binary_semaphore done{0};
    std::jthread caller([&] {
      gpu.SendCommandSync([&] { operation(); });
      done.release();
    });
    const bool completed = done.try_acquire_for(deadline);
    if (!completed && cleanup) cleanup();
    Require(name, "bounded host command", completed,
            "GPU host command did not return before the fixture deadline");
    caller.join();
  };
  const auto idle = [&](GuestGpu &gpu, const char *name,
                        const std::function<void()> &cleanup = {}) {
    std::binary_semaphore done{0};
    std::jthread caller([&] {
      gpu.WaitForIdle();
      done.release();
    });
    const bool completed = done.try_acquire_for(deadline);
    if (!completed && cleanup) cleanup();
    Require(name, "bounded queue drain", completed,
            "guest queue did not drain before the fixture deadline");
    caller.join();
  };
  const auto state = [&](GuestGpu &gpu, const char *name, auto &&predicate,
                         const std::function<void()> &cleanup = {}) {
    const auto until = std::chrono::steady_clock::now() + deadline;
    auto snapshot = GuestGpuTestAccess::Snapshot(gpu);
    while (!predicate(snapshot) && std::chrono::steady_clock::now() < until) {
      std::this_thread::yield();
      snapshot = GuestGpuTestAccess::Snapshot(gpu);
    }
    if ((!snapshot.consistent || !predicate(snapshot)) && cleanup) cleanup();
    Require(name, "queue accounting", snapshot.consistent,
            "submission/command counters disagree with their locked queues");
    Require(name, "bounded queue state", predicate(snapshot),
            "expected real queue state was not reached before the deadline");
    return snapshot;
  };
  const auto finish = [&](RenderContext &context, const char *name) {
    auto &gpu = context.GetGpu();
    idle(gpu, name);
    sync(gpu, name, [&] {
      auto &scheduler = context.GetCommandScheduler();
      scheduler.Finish();
      scheduler.WaitPriorityOperations(scheduler.CurrentTick() - 1);
    });
  };
  const auto stop = [&](RenderContext &context, const char *name) {
    std::binary_semaphore done{0};
    std::jthread caller([&] {
      context.GetGpu().Shutdown();
      done.release();
    });
    Require(name, "bounded shutdown", done.try_acquire_for(deadline),
            "guest GPU shutdown did not return before the deadline");
    caller.join();
    context.ShutdownGpu();
  };
  const auto report = [](const char *name) {
    std::printf("[host]    %-32s ok\n", name);
  };

  if (deferred_case) {
    constexpr const char *name = "GuestGpuAllBlockedDeferred";
    const char *threshold = std::getenv("KYTY_SUBMIT_DRAW_THRESHOLD");
    Require(name, "deferred fixture configuration",
            threshold == nullptr || std::strtoul(threshold, nullptr, 10) > 1,
            "this regression requires the normal deferred-submit threshold");
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    auto &gpu = context.GetGpu();
    auto &scheduler = context.GetCommandScheduler();
    CommandProcessor setup(context, 0);
    std::unique_ptr<Libs::Graphics::Buffer> output;
    std::binary_semaphore staged{0}, release_lane{0}, native_completed{0};
    std::atomic_uint completed_callbacks{0};
    uint64_t recorded_tick = 0;
    constexpr uint32_t guard = 0xa5a5a5a5u;
    constexpr uint32_t marker = 0x13579bdfu;
    gpu.SendCommand([&] {
      setup.BufferInit();
      output = std::make_unique<Libs::Graphics::Buffer>(
          context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
          vk::BufferUsageFlagBits::eTransferDst, 64);
      std::array<uint32_t, 16> initial;
      initial.fill(guard);
      std::memcpy(output->Mapped().data(), initial.data(), sizeof(initial));
      output->Flush(0, sizeof(initial));
      auto command = scheduler.Current().Handle();
      command.fillBuffer(output->Handle(), 16, 4, marker);
      command.pipelineBarrier(
          vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
          {}, vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                .dstAccessMask = vk::AccessFlagBits::eHostRead},
          {}, {});
      scheduler.NoteRecordedWork();
      recorded_tick = scheduler.CurrentTick();
      scheduler.DeferPriorityOperation([&] {
        completed_callbacks.fetch_add(1, std::memory_order_relaxed);
        native_completed.release();
      });
      staged.release();
      release_lane.acquire();
    });
    const bool prepared = staged.try_acquire_for(deadline);
    if (!prepared) release_lane.release();
    Require(name, "staging", prepared, "native work was not staged");
    uint32_t prefix = 0, graphics_label = 0, compute_label = 0;
    uint32_t graphics_suffix = 0, compute_suffix = 0;
    const auto prefix_list = write_packet(&prefix, 11);
    const auto graphics_wait = wait_packet(&graphics_label, &graphics_suffix, 22);
    const auto compute_wait = wait_packet(&compute_label, &compute_suffix, 33);
    gpu.Submit(prefix_list);
    gpu.Submit(graphics_wait);
    gpu.SubmitCompute(0x20, compute_wait);
    std::printf("[phase]   %s: native tick %llu, prefix and two first-packet waits queued\n",
                name, static_cast<unsigned long long>(recorded_tick));
    std::fflush(stdout);
    release_lane.release();
    const bool flushed = native_completed.try_acquire_for(deadline);
    if (!flushed) {
      // Only cleanup after the failed oracle may enqueue new work.
      gpu.SendCommand([&] { graphics_label = compute_label = 1; });
    }
    Require(name, "deferred native completion", flushed,
            "all-blocked deferred work was not submitted before external release");
    const auto release_labels = [&] {
      gpu.SendCommand([&] { graphics_label = compute_label = 1; });
    };
    state(gpu, name, [](const auto &s) {
      return s.submissions == 2 && s.blocked_heads == 2 && !s.processing;
    }, release_labels);
    bool prefix_and_suffix = false;
    sync(gpu, name, [&] {
      prefix_and_suffix = prefix == 11 && graphics_suffix == 0 && compute_suffix == 0;
    });
    output->Invalidate(0, 64);
    std::array<uint32_t, 16> actual{};
    std::memcpy(actual.data(), output->Mapped().data(), sizeof(actual));
    bool canaries = true;
    for (size_t i = 0; i < actual.size(); ++i)
      canaries &= actual[i] == (i == 4 ? marker : guard);
    sync(gpu, name, [&] { graphics_label = compute_label = 1; });
    finish(context, name);
    const auto drained = GuestGpuTestAccess::Snapshot(gpu);
    Require(name, "native output and canaries", canaries,
            "deferred native fill or neighboring canaries are incorrect");
    Require(name, "blocked suffix preservation", prefix_and_suffix,
            "a blocked wait executed its suffix before label release");
    Require(name, "resume and accounting",
            graphics_suffix == 22 && compute_suffix == 33 &&
                completed_callbacks.load() == 1 && drained.consistent &&
                drained.submissions == 0 && drained.commands == 0,
            "queue recovery lost, duplicated or miscounted work");
    sync(gpu, name, [&] { output.reset(); });
    stop(context, name);
    report(name);
  }

  if (!control_cases) return;

  {
    constexpr const char *name = "GuestGpuAllBlockedNoDeferred";
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    auto &gpu = context.GetGpu();
    std::binary_semaphore staged{0}, release_lane{0};
    gpu.SendCommand([&] { staged.release(); release_lane.acquire(); });
    const bool entered = staged.try_acquire_for(deadline);
    if (!entered) release_lane.release();
    Require(name, "staging", entered, "host lane did not enter");
    uint32_t graphics_label = 0, compute_label = 0;
    uint32_t graphics_value = 0, compute_value = 0;
    const auto graphics_wait = wait_packet(&graphics_label, &graphics_value, 41);
    const auto compute_wait = wait_packet(&compute_label, &compute_value, 42);
    gpu.Submit(graphics_wait);
    gpu.SubmitCompute(0x20, compute_wait);
    release_lane.release();
    const auto release_labels = [&] {
      gpu.SendCommand([&] { graphics_label = compute_label = 1; });
    };
    state(gpu, name, [](const auto &s) {
      return s.submissions == 2 && s.blocked_heads == 2 && !s.processing;
    }, release_labels);
    const bool no_submit = context.GetCommandScheduler().CurrentTick() == 1;
    if (!no_submit) release_labels();
    Require(name, "no invented native submit", no_submit,
            "blocked first packets submitted native work without a deferred buffer");
    bool suffixes_blocked = false;
    sync(gpu, name, [&] {
      suffixes_blocked = graphics_value == 0 && compute_value == 0;
      graphics_label = compute_label = 1;
    });
    finish(context, name);
    Require(name, "host wake and resume",
            suffixes_blocked && graphics_value == 41 && compute_value == 42,
            "host-lane wake lost blocked work or ran its suffix early");
    state(gpu, name, [](const auto &s) {
      return s.submissions == 0 && s.commands == 0 && !s.processing;
    });
    stop(context, name);
    report(name);
  }

  {
    constexpr const char *name = "GuestGpuRunnableQueueFifo";
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    auto &gpu = context.GetGpu();
    std::binary_semaphore staged{0}, release_lane{0};
    gpu.SendCommand([&] { staged.release(); release_lane.acquire(); });
    const bool entered = staged.try_acquire_for(deadline);
    if (!entered) release_lane.release();
    Require(name, "staging", entered, "host lane did not enter");
    std::array<uint32_t, 3> values{};
    std::array<std::array<uint32_t, 5>, 6> packets;
    for (uint32_t i = 0; i < 3; ++i) {
      packets[i * 2] = write_packet(&values[i], i * 10 + 11);
      packets[i * 2 + 1] = write_packet(&values[i], i * 10 + 12);
    }
    gpu.Submit(packets[0]);
    gpu.Submit(packets[1]);
    gpu.SubmitCompute(0x20, packets[2]);
    gpu.SubmitCompute(0x20, packets[3]);
    gpu.SubmitCompute(0x57, packets[4]);
    gpu.SubmitCompute(0x57, packets[5]);
    const auto queued = GuestGpuTestAccess::Snapshot(gpu);
    const bool queued_correct = queued.consistent && queued.submissions == 6 &&
                                queued.runnable_heads == 3 && queued.blocked_heads == 0;
    release_lane.release();
    Require(name, "queued runnable heads", queued_correct,
            "fixture did not queue graphics plus first/last compute queues");
    finish(context, name);
    Require(name, "per-queue FIFO",
            values == std::array<uint32_t, 3>{12, 22, 32},
            "runnable queue selection reordered or lost a submission");
    state(gpu, name, [](const auto &s) {
      return s.submissions == 0 && s.commands == 0 && !s.processing;
    });
    stop(context, name);
    report(name);
  }

  {
    constexpr const char *name = "GuestGpuHostCommandPrecedence";
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    auto &gpu = context.GetGpu();
    std::binary_semaphore staged{0}, release_lane{0};
    gpu.SendCommand([&] { staged.release(); release_lane.acquire(); });
    const bool entered = staged.try_acquire_for(deadline);
    if (!entered) release_lane.release();
    Require(name, "staging", entered, "host lane did not enter");
    uint32_t value = 0;
    uint32_t seen_by_command = UINT32_MAX;
    const auto packet = write_packet(&value, 51);
    gpu.Submit(packet);
    gpu.SendCommand([&] { seen_by_command = value; });
    const auto queued = GuestGpuTestAccess::Snapshot(gpu);
    const bool queued_correct = queued.consistent && queued.submissions == 1 &&
                                queued.commands == 1;
    release_lane.release();
    Require(name, "both lanes queued", queued_correct,
            "fixture did not queue both a host command and a submission");
    finish(context, name);
    Require(name, "command selection", seen_by_command == 0 && value == 51,
            "runnable guest submission overtook the pending host command");
    stop(context, name);
    report(name);
  }

  {
    constexpr const char *name = "GuestGpuEmptyQueue";
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    auto &gpu = context.GetGpu();
    idle(gpu, name);
    const auto empty = GuestGpuTestAccess::Snapshot(gpu);
    Require(name, "empty idle",
            empty.consistent && empty.submissions == 0 && empty.commands == 0 &&
                !empty.processing && context.GetCommandScheduler().CurrentTick() == 1,
            "an empty guest queue invented work or failed to become idle");
    report(name);
    stop(context, "GuestGpuEmptyShutdown");
    report("GuestGpuEmptyShutdown");
  }

  {
    constexpr const char *name = "GuestGpuPriorityNativeIdle";
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    auto &gpu = context.GetGpu();
    auto &scheduler = context.GetCommandScheduler();
    CommandProcessor setup(context, 0);
    vk::SemaphoreTypeCreateInfo type{};
    type.sType = vk::StructureType::eSemaphoreTypeCreateInfo;
    type.semaphoreType = vk::SemaphoreType::eTimeline;
    vk::SemaphoreCreateInfo create{};
    create.sType = vk::StructureType::eSemaphoreCreateInfo;
    create.pNext = &type;
    vk::Semaphore gate = nullptr;
    Require(name, "timeline create",
            m_runtime_context.device.createSemaphore(&create, nullptr, &gate) ==
                vk::Result::eSuccess, "failed to create native timeline gate");
    bool gate_released = false;
    const auto release_native = [&] {
      if (gate_released) return;
      vk::SemaphoreSignalInfo info{};
      info.sType = vk::StructureType::eSemaphoreSignalInfo;
      info.semaphore = gate;
      info.value = 1;
      Require(name, "timeline release",
              m_runtime_context.device.signalSemaphore(&info) == vk::Result::eSuccess,
              "failed to release native timeline gate");
      gate_released = true;
    };
    std::binary_semaphore native_completed{0};
    std::atomic_uint native_callbacks{0};
    uint64_t callback_tick = 0;
    sync(gpu, name, [&] {
      setup.BufferInit();
      callback_tick = scheduler.CurrentTick();
      scheduler.DeferPriorityOperation([&] {
        native_callbacks.fetch_add(1, std::memory_order_relaxed);
        native_completed.release();
      });
      SubmitInfo blocked;
      blocked.AddWait(gate, 1);
      scheduler.Flush(blocked);
    }, release_native);

    std::binary_semaphore first_returned{0};
    std::jthread first([&] { gpu.SuspendPoint(); first_returned.release(); });
    const bool first_ok = first_returned.try_acquire_for(deadline);
    if (!first_ok) release_native();
    Require(name, "first suspend point", first_ok,
            "initial suspend-point credit waited for native completion");
    first.join();
    idle(gpu, name, release_native);
    state(gpu, name, [](const auto &s) {
      return s.submissions == 0 && s.commands == 0 && !s.processing;
    }, release_native);
    std::binary_semaphore second_entered{0}, second_returned{0};
    std::jthread second([&] {
      second_entered.release();
      gpu.SuspendPoint();
      second_returned.release();
    });
    const bool second_started = second_entered.try_acquire_for(deadline);
    if (!second_started) release_native();
    Require(name, "second caller entered", second_started,
            "second suspend-point caller did not start");
    const bool overtook = second_returned.try_acquire_for(std::chrono::milliseconds(100));
    const bool callback_overtook = native_callbacks.load() != 0;
    if (overtook || callback_overtook) release_native();
    Require(name, "native gate held", !overtook && !callback_overtook,
            "a completion or second suspend point bypassed the held native timeline");
    const auto before_release = GuestGpuTestAccess::Snapshot(gpu);
    const bool queue_idle = before_release.consistent && before_release.submissions == 0 &&
                            before_release.commands == 0 && !before_release.processing;
    if (!queue_idle) release_native();
    Require(name, "host queue idle before native release", queue_idle,
            "fixture still had external queued work before native release");

    // Critical oracle: only signal the already-submitted external native gate.
    // No new guest/host work, Pop, Finish or Drain is supplied before both
    // observations. The second caller's own marker follows its acquired credit.
    release_native();
    const bool callback_completed = native_completed.try_acquire_for(deadline);
    const bool second_completed = second_returned.try_acquire_for(deadline);
    Require(name, "native callback without new work", callback_completed,
            "completed native priority callback required new guest or host work");
    Require(name, "suspend credit without new work", second_completed,
            "waiting suspend point required external work after native release");
    second.join();
    finish(context, name);
    Require(name, "exact callbacks and markers",
            native_callbacks.load() == 1 && gpu.GetFrameNum() == 2,
            "native idle recovery duplicated a callback or suspend marker");
    std::printf("[phase]   %s: callback tick %llu serviced with no external wake work\n",
                name, static_cast<unsigned long long>(callback_tick));
    stop(context, name);
    m_runtime_context.device.destroySemaphore(gate, nullptr);
    report(name);
  }

  {
    constexpr const char *name = "GuestGpuConcurrentQueueShutdown";
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    auto &gpu = context.GetGpu();
    auto &scheduler = context.GetCommandScheduler();
    CommandProcessor setup(context, 0);
    vk::SemaphoreTypeCreateInfo type{};
    type.sType = vk::StructureType::eSemaphoreTypeCreateInfo;
    type.semaphoreType = vk::SemaphoreType::eTimeline;
    vk::SemaphoreCreateInfo create{};
    create.sType = vk::StructureType::eSemaphoreCreateInfo;
    create.pNext = &type;
    vk::Semaphore gate = nullptr;
    Require(name, "timeline create",
            m_runtime_context.device.createSemaphore(&create, nullptr, &gate) ==
                vk::Result::eSuccess, "failed to create native timeline gate");
    bool native_released = false;
    const auto release_native = [&] {
      if (native_released) return;
      vk::SemaphoreSignalInfo info{};
      info.sType = vk::StructureType::eSemaphoreSignalInfo;
      info.semaphore = gate;
      info.value = 1;
      Require(name, "timeline release",
              m_runtime_context.device.signalSemaphore(&info) == vk::Result::eSuccess,
              "failed to release native timeline gate");
      native_released = true;
    };
    std::binary_semaphore staged{0}, release_lane{0}, native_completed{0};
    std::atomic_bool lane_released{false};
    const auto release_host = [&] {
      if (!lane_released.exchange(true)) release_lane.release();
    };
    const auto cleanup = [&] { release_host(); release_native(); };
    std::atomic_uint native_callbacks{0}, host_commands{0};
    gpu.SendCommand([&] {
      setup.BufferInit();
      SubmitInfo blocked;
      blocked.AddWait(gate, 1);
      scheduler.Flush(blocked);
      scheduler.DeferPriorityOperation([&] {
        native_callbacks.fetch_add(1, std::memory_order_relaxed);
        native_completed.release();
      });
      staged.release();
      release_lane.acquire();
    });
    const bool entered = staged.try_acquire_for(deadline);
    if (!entered) cleanup();
    Require(name, "staging", entered, "host lane did not enter");

    constexpr uint32_t producers = 4;
    constexpr uint32_t lists_per_producer = 16;
    std::array<uint32_t, producers> values{};
    std::array<std::array<std::array<uint32_t, 5>, lists_per_producer>, producers> packets;
    for (uint32_t i = 0; i < producers; ++i)
      for (uint32_t j = 0; j < lists_per_producer; ++j)
        packets[i][j] = write_packet(&values[i], i * 100 + j + 1);
    std::counting_semaphore<producers> producer_done{0};
    std::array<std::jthread, producers> producer_threads;
    for (uint32_t i = 0; i < producers; ++i) {
      producer_threads[i] = std::jthread([&, i] {
        for (uint32_t j = 0; j < lists_per_producer; ++j) {
          gpu.SubmitCompute(0x20 + i, packets[i][j]);
          gpu.SendCommand([&] { host_commands.fetch_add(1, std::memory_order_relaxed); });
        }
        producer_done.release();
      });
    }
    bool producers_finished = true;
    for (uint32_t i = 0; i < producers; ++i)
      producers_finished &= producer_done.try_acquire_for(deadline);
    if (!producers_finished) cleanup();
    Require(name, "concurrent enqueue", producers_finished,
            "concurrent producers failed to queue their owned work");
    for (auto &producer : producer_threads) producer.join();
    const auto queued = GuestGpuTestAccess::Snapshot(gpu);
    const bool queued_correct = queued.consistent &&
        queued.submissions == producers * lists_per_producer &&
        queued.commands == producers * lists_per_producer &&
        queued.runnable_heads == producers;
    if (!queued_correct) cleanup();
    Require(name, "concurrent accounting", queued_correct,
            "concurrent enqueue lost or miscounted a command/submission");

    // Producers have finished before closing admission. Both shutdown callers
    // must serialize and drain the accepted work; no post-shutdown enqueue occurs.
    std::counting_semaphore<2> shutdown_entered{0}, shutdown_done{0};
    std::array<std::jthread, 2> shutdown_threads;
    for (auto &caller : shutdown_threads) {
      caller = std::jthread([&] {
        shutdown_entered.release();
        gpu.Shutdown();
        shutdown_done.release();
      });
    }
    bool shutdown_started = shutdown_entered.try_acquire_for(deadline);
    shutdown_started &= shutdown_entered.try_acquire_for(deadline);
    if (!shutdown_started) cleanup();
    Require(name, "shutdown callers", shutdown_started,
            "concurrent shutdown callers did not start");
    state(gpu, name, [](const auto &s) { return s.stopping; }, cleanup);
    release_host();
    idle(gpu, name, release_native);
    const auto drained = GuestGpuTestAccess::Snapshot(gpu);
    bool values_correct = drained.consistent && drained.submissions == 0 &&
                          drained.commands == 0 &&
                          host_commands.load() == producers * lists_per_producer;
    for (uint32_t i = 0; i < producers; ++i)
      values_correct &= values[i] == i * 100 + lists_per_producer;
    const bool overtook_native =
        shutdown_done.try_acquire_for(std::chrono::milliseconds(100));
    const bool callback_overtook = native_callbacks.load() != 0;
    release_native();
    bool callbacks_done = native_completed.try_acquire_for(deadline);
    uint32_t completed_shutdowns = overtook_native ? 1 : 0;
    while (completed_shutdowns < 2 && shutdown_done.try_acquire_for(deadline))
      ++completed_shutdowns;
    Require(name, "bounded concurrent shutdown", completed_shutdowns == 2,
            "shutdown did not drain after native gate release");
    for (auto &caller : shutdown_threads) caller.join();
    Require(name, "accepted queue drain", values_correct,
            "shutdown lost accepted commands or per-queue FIFO results");
    Require(name, "native completion before shutdown",
            !overtook_native && !callback_overtook && callbacks_done &&
                native_callbacks.load() == 1,
            "shutdown returned before native completion or lost its callback");
    context.ShutdownGpu();
    m_runtime_context.device.destroySemaphore(gate, nullptr);
    report(name);
  }
}

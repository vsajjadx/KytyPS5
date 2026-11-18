// Included inside VulkanHarness. One real nonzero indirect dispatch. The ledger
// itself only records metadata; the explicit final copy/wait below belongs to
// this private native test. No observer, gate, callback-policy, or game change.
void CheckIndirectArgsLedgerNative() {
  namespace L = Libs::Graphics::IndirectArgsLedger;
  namespace Causal = Libs::Graphics::CausalDiagnostics;
  constexpr const char* name = "IndirectArgsLedgerNative";
  constexpr uint64_t base = 0x000000020bc00000ull;
  constexpr uint64_t mapping_bytes = 0x100000, mapping_alignment = 0x10000;
  constexpr uint64_t page = BufferCache::CACHING_PAGESIZE;
  constexpr uint64_t args = base + page - 4;
  constexpr uint64_t output_page = base + 4 * page;
  constexpr uint64_t output = output_page + 4;
  constexpr uint32_t canary = 0xa5a5a5a5u, poison = 0xcdcdcdcdu;
  constexpr std::array<uint64_t, 2> watches{args, output};
  constexpr std::array<uint32_t, 3> input_words{2, 1, 1};
  constexpr std::array<uint64_t, 3> old_offsets{16380, 0, 4};
  constexpr std::array<uint64_t, 3> new_offsets{16380, 16384, 16388};
  constexpr uint64_t host_wait_timeout_ns = 30'000'000'000ull;
  // Literal independent oracle: two groups of four, output = global lane + 1.
  constexpr std::array<uint32_t, 21> expected_readback{
      canary, canary, 2, 1, 1, canary,
      canary, 1, 2, 3, 4, 5, 6, 7, 8, canary, canary, canary, canary, canary,
      canary};
  constexpr uint64_t args_copy_bytes = 20, output_copy_bytes = 56;
  constexpr uint64_t readback_bytes = 84;
  static_assert(page == 16384 && args % page == page - 4 && output % page == 4);
  static_assert(sizeof(expected_readback) == readback_bytes);

  Require(name, "fresh child gates", Causal::Enabled() && !FaultPeekEnabled() &&
      !IndirectArgsDiagnosticEnabled() && !GpuSyncDebugEnabled(),
      "requires causal=1, fault-peek=0, no argument observer, and GPU-sync=0");
  Require(name, "no prior recorder", L::Active() == nullptr,
      "a recorder was already installed in this child");

  std::vector<u32> code;
  AppendVop3(&code, 0x346u, 1, 4, InlineU32(2), Vgpr(0));
  code.insert(code.end(), {
      EncodeVop2(0x25u, 2, InlineU32(1), 1),
      EncodeVop2(0x1au, 1, InlineU32(2), 1),
      EncodeMubuf0(0x1cu), EncodeMubuf1(2, 0, 1)});
  AppendEnd(&code);
  ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
      {.type = Prospero::ShaderBinaryType::kCs,
       .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
  EnsureRuntimeContext(); // This does not construct the test's RenderContext.

  int64_t physical = -1;
  Require(name, "allocate exact test range", LibKernel::Memory::KernelAllocateDirectMemory(
      0, LibKernel::Memory::KernelGetDirectMemorySize(), mapping_bytes,
      mapping_alignment, 0, &physical) == 0, "native ledger allocation failed");
  void* mapped = reinterpret_cast<void*>(base);
  Require(name, "map prearmed addresses", LibKernel::Memory::KernelMapDirectMemory(
      &mapped, mapping_bytes, 0x3, 0x10, physical, mapping_alignment) == 0 &&
      mapped == reinterpret_cast<void*>(base), "exact prearmed mapping was unavailable");

  // The watches never change. This one-shot installation precedes every native
  // owner whose generation is claimed, including BufferCache's upload stream.
  auto recorder = std::make_unique<L::Recorder>(watches);
  Require(name, "pre-resource recorder install", recorder->Valid() && recorder->Install(),
      "fixed watched ranges or one-shot recorder installation failed");

  const auto history = [&]() {
    const auto coverage = recorder->ReadCoverage();
    Require(name, "complete bounded metadata history",
        coverage.event_tickets <= L::EventCapacity && coverage.dropped_events == 0 &&
        !coverage.event_history_full && coverage.reservation_tickets <= L::ReservationCapacity &&
        coverage.dropped_reservations == 0 && !coverage.reservation_history_full,
        "an event or reservation history gap invalidates lineage");
    Require(name, "coverage remains scoped",
        !coverage.arbitrary_guest_stores_observed && !coverage.kernel_backing_writes_observed &&
        !coverage.all_native_commands_observed && !coverage.native_execution_observed,
        "metadata coverage must not claim complete writes or native execution");
    std::vector<L::Event> events;
    events.reserve(static_cast<size_t>(coverage.event_tickets));
    for (size_t i = 0; i < coverage.event_tickets; ++i) {
      L::Event event{};
      Require(name, "published immutable event", recorder->ReadEvent(i, event) &&
          event.journal_sequence == i + 1, "event slot was absent or unpublished");
      events.push_back(event);
    }
    return events;
  };
  const auto one = [&](const std::vector<L::Event>& events, L::Kind kind,
                       uint8_t watch, uint8_t word) {
    L::Event found{};
    size_t matches = 0;
    for (const auto& event : events) {
      if (event.kind == kind && event.watch == watch && event.word == word) {
        found = event;
        ++matches;
      }
    }
    Require(name, "one exact per-dword edge", matches == 1,
        "required edge was missing, duplicated, or a different path was taken");
    Require(name, "full exact watched dword",
        found.byte_mask == 15 && found.intersection_bytes == 4 &&
        found.word_address == watches[watch] + uint64_t{word} * 4 &&
        found.intersection_guest_begin == found.word_address &&
        found.guest_mapping_generation == 0,
        "per-dword range/mask changed or a guest-mapping generation was invented");
    return found;
  };
  const auto same_span = [](const L::NativeSpan& a, const L::NativeSpan& b) {
    return L::SameResource(a.resource, b.resource) &&
        a.resource.bytes == b.resource.bytes && a.offset == b.offset && a.bytes == b.bytes &&
        a.identity_known && b.identity_known && a.reservation == b.reservation &&
        a.reservation_state == b.reservation_state;
  };
  const auto native_identity = [&](const L::NativeSpan& span) {
    return span.identity_known && span.resource.capture == recorder->Capture() &&
        span.resource.generation != 0 && span.resource.handle != 0 &&
        L::ValidRange(span.offset, span.bytes) && span.offset <= span.resource.bytes &&
        span.bytes <= span.resource.bytes - span.offset;
  };
  const auto no_native_position = [](const L::Position& p) {
    return p.native_domain == 0 && p.causal_lifetime == 0 && p.command_buffer == 0 &&
        p.tick == 0 && p.tracked_command_ordinal == 0 && !p.owner_witness;
  };
  const auto same_recording = [](const L::Position& a, const L::Position& b) {
    return a.owner_witness && b.owner_witness && a.native_domain != 0 &&
        a.native_domain == b.native_domain && a.causal_lifetime != 0 &&
        a.causal_lifetime == b.causal_lifetime && a.command_buffer != 0 &&
        a.command_buffer == b.command_buffer && a.tick != 0 && a.tick == b.tick;
  };
  const auto no_payload = [](const L::Event& event) {
    return event.payload_meaning == L::PayloadMeaning::None &&
        event.payload_mask == 0 && event.payload == 0;
  };
  struct InputVersion {
    uint64_t version;
    std::array<uint32_t, 3> words;
    std::array<L::NativeSpan, 3> native;
  };
  struct Capture {
    InputVersion original, migrated;
    L::NativeSpan upload_owner, output_owner, readback_owner;
    std::array<L::Event, 3> stores, snapshots, uploads, migrations, consumers, candidates;
    std::vector<L::Event> events;
    vk::Semaphore semaphore;
    uint64_t native_domain, causal_lifetime, command, tick, shader, module;
  };
  std::unique_ptr<const Capture> captured; // published once; never retarget an old tuple
  std::array<uint32_t, expected_readback.size()> actual{};
  uint64_t submitted_tick = 0;
  {
    // Recorder outlives the GPU owner, scheduler, all buffers and deferred deletion.
    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    auto& gpu = context.GetGpu();
    auto& scheduler = context.GetCommandScheduler();
    auto& master = scheduler.GetMasterSemaphore();
    auto& cache = context.GetBufferCache();
    CommandProcessor processor(context, 0);
    std::unique_ptr<Libs::Graphics::Buffer> readback;
    gpu.SendCommandSync([&] {
      processor.BufferInit();
      context.MapMemory(base, mapping_bytes); // range registration only; no full-range preparation
      Require(name, "passive ledger only", cache.IndirectArgsObserver() == nullptr,
          "argument observer would add work to the tested path");
      Require(name, "real serialized owner", scheduler.Active() &&
          GuestGpu::IsGpuThreadFor(scheduler.Context()), "not the active GPU owner");

      auto& shaders = processor.GetShCtx();
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
          .num_thread_x = 4, .num_thread_y = 1, .num_thread_z = 1,
          .wave_size = 64, .user_sgpr = 4, .tgid_x_en = true});
      ShaderBufferResource descriptor{};
      descriptor.UpdateAddress48(output);
      descriptor.fields[2] = 12u * sizeof(u32);
      descriptor.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      for (u32 word = 0; word < 4; ++word) shaders.SetCsUserSgpr(word, descriptor.fields[word]);
      ShaderComputeInputInfo input{};
      input.indirect_args_address = args;
      Require(name, "actual module and ready pipeline",
          context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input) != nullptr &&
          context.GetPipelineCache().GetComputePipeline(input) != nullptr,
          "consumer compilation was absent/pending; no skipped dispatch is a pass");
      const uint64_t shader = input.stage.module->info.shader_hash;
      const uint64_t module = input.stage.module->id;
      Require(name, "output-only fixture program", !input.stage.module->info.info.uses_dma,
          "unexpected BDA preparation would invalidate the planned cache shape");

      const uint64_t tick = scheduler.CurrentTick();
      const auto command = scheduler.Current().Handle();
      const uint64_t command_bits = L::HandleBits(static_cast<VkCommandBuffer>(command));
      const uint64_t domain = Causal::Domain(&master);
      const uint64_t lifetime = scheduler.CausalLifetimeSequence();
      Require(name, "native lifetime identity", tick != 0 && command_bits != 0 &&
          domain != 0 && lifetime != 0, "recording lifetime metadata unavailable");
      const auto stable_tick = [&] {
        Require(name, "one retained recording tick", scheduler.Active() &&
            scheduler.CurrentTick() == tick && scheduler.Current().Handle() == command,
            "recording restarted/submitted; captured owners must not be retargeted");
      };
      const auto& upload_buffer = cache.GetUtilityBuffer(MemoryUsage::Upload);
      const auto upload_owner = upload_buffer.LedgerSpan(0, 1);
      Require(name, "upload allocation born in capture", native_identity(upload_owner) &&
          upload_owner.resource.handle ==
              L::HandleBits(static_cast<VkBuffer>(upload_buffer.Handle())),
          "preexisting/unknown upload resource generation cannot be relabeled");

      // Real CPU writes, before any native read. No arbitrary guest reread is
      // used by the ledger. The constant source version is retained to completion.
      Require(name, "initial CPU ownership", !cache.IsRegionGpuModified(base, 2 * page) &&
          !cache.IsRegionGpuModified(output_page, page), "fresh pages were GPU-dirty");
      cache.FillBuffer(base, 2 * page, canary, false);
      cache.FillBuffer(output_page, page, canary, false);
      processor.WriteData(reinterpret_cast<uint32_t*>(args), input_words.data(), 3, 0);

      // Upload two complete adjacent pages separately. Their normal readonly
      // dirty tracking becomes clean, avoiding a later private argument upload.
      const auto left = cache.ObtainBuffer(base, page, false, false, {}, false);
      stable_tick();
      const auto left_span = left.first->LedgerSpan(left.second, page);
      const auto right = cache.ObtainBuffer(base + page, page, false, false, {}, false);
      stable_tick();
      const auto right_span = right.first->LedgerSpan(right.second, page);
      Require(name, "two real disjoint owners", native_identity(left_span) &&
          native_identity(right_span) && !L::SameResource(left_span.resource, right_span.resource) &&
          left_span.resource.handle == L::HandleBits(static_cast<VkBuffer>(left.first->Handle())) &&
          right_span.resource.handle == L::HandleBits(static_cast<VkBuffer>(right.first->Handle())) &&
          left.first->CpuAddress() == base && left.first->Size() == page &&
          right.first->CpuAddress() == base + page && right.first->Size() == page &&
          left_span.offset == 0 && right_span.offset == 0,
          "adjacent pages were not distinct native owners");
      const InputVersion original{1, input_words,
          {L::Slice(left_span, page - 4, 4), L::Slice(right_span, 0, 4),
           L::Slice(right_span, 4, 4)}};
      for (uint8_t word = 0; word < 3; ++word) {
        Require(name, "literal old native offsets", original.native[word].offset == old_offsets[word] &&
            original.native[word].bytes == 4 && original.native[word].reservation_state ==
                L::ReservationState::NotSuballocated,
            "old per-dword identity differs from the actual two-page layout");
      }
      const auto output_binding = cache.ObtainBuffer(output_page, page, false, false, {}, false);
      stable_tick();
      const auto output_owner = output_binding.first->LedgerSpan(output_binding.second, page);
      const auto output_handle = output_binding.first->Handle();
      Require(name, "fixed output owner", native_identity(output_owner) &&
          output_binding.first->CpuAddress() == output_page &&
          output_binding.first->Size() == page && output_owner.offset == 0 &&
          output_owner.resource.handle == L::HandleBits(static_cast<VkBuffer>(output_handle)),
          "output owner/offset was not fixed before dispatch");
      Require(name, "full-page upload cleared CPU dirtiness",
          !cache.IsRegionCpuModified(args, 12), "arguments would use a different upload owner");

      const auto before_merge = history();
      std::array<L::Event, 3> stores{}, snapshots{}, uploads{};
      for (uint8_t word = 0; word < 3; ++word) {
        stores[word] = one(before_merge, L::Kind::CpuStoreCommitted, 0, word);
        snapshots[word] = one(before_merge, L::Kind::HostPayloadSnapshot, 0, word);
        uploads[word] = one(before_merge, L::Kind::TransferCopyRecorded, 0, word);
        const auto& store = stores[word];
        const auto& snapshot = snapshots[word];
        const auto& copy = uploads[word];
        Require(name, "committed literal CPU input", store.payload_meaning ==
            L::PayloadMeaning::CpuCommittedBytes && store.payload_mask == 15 &&
            store.payload == input_words[word] && no_native_position(store.position),
            "actual WriteData store payload/domain was not the input version");
        Require(name, "owned staging snapshot", snapshot.payload_meaning ==
            L::PayloadMeaning::HostSnapshotBytes && snapshot.payload_mask == 15 &&
            snapshot.payload == input_words[word] && no_native_position(snapshot.position) &&
            native_identity(snapshot.destination) &&
            L::SameResource(snapshot.destination.resource, upload_owner.resource) &&
            snapshot.destination.reservation_state == L::ReservationState::Known &&
            snapshot.destination.reservation != 0,
            "staging snapshot lacks exact generation/reservation/value");
        Require(name, "actual upload source lineage", no_payload(copy) &&
            same_span(copy.source, snapshot.destination) &&
            same_span(copy.destination, original.native[word]) &&
            store.host_thread_token == snapshot.host_thread_token &&
            snapshot.host_thread_token == copy.host_thread_token &&
            store.host_action_ordinal < snapshot.host_action_ordinal &&
            snapshot.host_action_ordinal < copy.host_action_ordinal,
            "recorded upload does not match its committed store and staging snapshot");
      }

      // This ordinary cross-page obtain causes actual JoinOverlap migrations.
      // No prefill of the whole mapping and no fixture-forced bad cache behavior.
      const auto merged = cache.ObtainBuffer(args, 12, false, false, {}, false);
      stable_tick();
      const auto merged_span = merged.first->LedgerSpan(merged.second, 12);
      const auto merged_handle = merged.first->Handle();
      Require(name, "real merged owner", native_identity(merged_span) &&
          merged.first->CpuAddress() == base && merged.first->Size() == 2 * page &&
          merged_span.offset == page - 4 &&
          merged_span.resource.handle == L::HandleBits(static_cast<VkBuffer>(merged_handle)) &&
          !L::SameResource(merged_span.resource, left_span.resource) &&
          !L::SameResource(merged_span.resource, right_span.resource),
          "no real allocation-generation transition occurred");
      const auto after_merge = history();
      std::array<L::Event, 3> migrations{};
      std::array<L::NativeSpan, 3> migrated_words{};
      for (uint8_t word = 0; word < 3; ++word) {
        const auto expected = L::Slice(merged_span, uint64_t{word} * 4, 4);
        migrations[word] = one(after_merge, L::Kind::MigrationCopyRecorded, 0, word);
        const auto& migration = migrations[word];
        Require(name, "exact recorded migration edge", no_payload(migration) &&
            same_span(migration.source, original.native[word]) &&
            same_span(migration.destination, expected) &&
            migration.source_guest_begin == args + uint64_t{word} * 4 &&
            migration.operation_guest_begin == (word == 0 ? base : base + page) &&
            migration.operation_bytes == page &&
            migration.source.offset == old_offsets[word] &&
            migration.destination.offset == new_offsets[word] &&
            same_recording(uploads[word].position, migration.position) &&
            uploads[word].host_thread_token == migration.host_thread_token &&
            uploads[word].position.tracked_command_ordinal <
                migration.position.tracked_command_ordinal,
            "new owner has no complete old-generation-to-new-generation edge");
        migrated_words[word] = migration.destination;
      }
      // Create a NEW immutable expectation only after every real migration edge
      // is checked. original is const and is never updated to the new owner.
      const InputVersion migrated{2, input_words, migrated_words};
      Require(name, "original version retained", original.version == 1 &&
          migrated.version == 2 && original.words == input_words &&
          !cache.IsRegionCpuModified(args, 12),
          "stale source version changed or arguments became CPU-dirty");

      readback = std::make_unique<Libs::Graphics::Buffer>(context.GetGraphics(), scheduler,
          MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, readback_bytes);
      const auto readback_owner = readback->LedgerSpan(0, readback_bytes);
      Require(name, "owned guarded readback", native_identity(readback_owner) &&
          readback_owner.resource.handle == L::HandleBits(static_cast<VkBuffer>(readback->Handle())) &&
          readback->Mapped().size() >= readback_bytes,
          "fresh readback allocation is unmapped or lacks a generation");
      std::array<uint32_t, expected_readback.size()> initial;
      initial.fill(poison); // an omitted copy cannot imitate the expected canaries
      initial.front() = canary;
      initial.back() = canary;
      std::memcpy(readback->Mapped().data(), initial.data(), sizeof(initial));
      readback->Flush(0, readback_bytes);

      processor.DispatchIndirect(args, 0x41u); // exactly one nonzero workload
      stable_tick();
      const auto after_dispatch = history();
      std::array<L::Event, 3> consumers{}, candidates{};
      for (uint8_t word = 0; word < 3; ++word) {
        consumers[word] = one(after_dispatch, L::Kind::IndirectConsumerRecorded, 0, word);
        candidates[word] = one(after_dispatch, L::Kind::DescriptorWriteCandidate, 1, word);
        const auto& consumer = consumers[word];
        const auto& candidate = candidates[word];
        Require(name, "new generation consumed; old span not consumed", no_payload(consumer) &&
            same_span(consumer.source, migrated.native[word]) &&
            !same_span(consumer.source, original.native[word]) &&
            same_recording(migrations[word].position, consumer.position) &&
            migrations[word].position.tracked_command_ordinal <
                consumer.position.tracked_command_ordinal &&
            consumer.position.native_domain == domain &&
            consumer.position.causal_lifetime == lifetime && consumer.position.tick == tick &&
            consumer.position.command_buffer == command_bits &&
            consumer.shader.shader == shader && consumer.shader.module == module,
            "consumer skipped, used a stale/unknown owner, or lost recorded order");
        Require(name, "descriptor overlap remains a candidate", no_payload(candidate) &&
            candidate.destination.resource.handle == output_owner.resource.handle &&
            candidate.destination.offset == output_owner.offset + 4 + uint64_t{word} * 4 &&
            candidate.destination.bytes == 4 && !candidate.destination.identity_known &&
            candidate.destination.resource.capture == 0 &&
            candidate.destination.resource.generation == 0 &&
            candidate.destination.reservation == 0 &&
            candidate.destination.reservation_state == L::ReservationState::Unknown &&
            same_recording(consumer.position, candidate.position) &&
            candidate.position.tracked_command_ordinal == consumer.position.tracked_command_ordinal &&
            candidate.shader.shader == shader && candidate.shader.module == module,
            "descriptor candidate was mistyped as a store or its logical offset changed");
      }
      for (const auto& event : after_dispatch) {
        Require(name, "input is not a target writable descriptor",
            event.kind != L::Kind::DescriptorWriteCandidate || event.watch != 0,
            "output-only shader unexpectedly advertised the argument input as writable");
      }
      // Recheck the captured staging reservations in complete metadata history;
      // no later reservation may silently reuse bytes from this source version.
      for (const auto& snapshot : snapshots) {
        const auto& span = snapshot.destination;
        Require(name, "source reservation still immutable at record time",
            same_span(span, recorder->Span(span.resource, span.offset, span.bytes, true)),
            "staging reservation was reused before submission");
      }

      // Final fixture-only readback from CAPTURED handles/ranges. There is no
      // canonical-owner lookup after dispatch and no dereference of old slots.
      command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
          vk::PipelineStageFlagBits::eTransfer, {},
          vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                            .dstAccessMask = vk::AccessFlagBits::eTransferRead}, {}, {});
      command.copyBuffer(merged_handle, readback->Handle(),
          vk::BufferCopy{merged_span.offset - 4, 4, args_copy_bytes});
      command.copyBuffer(output_handle, readback->Handle(),
          vk::BufferCopy{output_owner.offset, 24, output_copy_bytes});
      command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
          vk::PipelineStageFlagBits::eHost, {},
          vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                            .dstAccessMask = vk::AccessFlagBits::eHostRead}, {}, {});
      scheduler.NoteRecordedWork();
      stable_tick();
      captured = std::make_unique<const Capture>(Capture{
          original, migrated, upload_owner, output_owner, readback_owner,
          stores, snapshots, uploads, migrations, consumers, candidates,
          history(), master.Handle(), domain, lifetime, command_bits, tick, shader, module});
      submitted_tick = scheduler.Submit();
      Require(name, "exact submitted tick", submitted_tick == captured->tick,
          "submitted completion does not cover the captured recording lifetime");
    });

    Require(name, "immutable completion capsule", captured != nullptr && submitted_tick != 0 &&
        captured->semaphore == master.Handle() &&
        captured->native_domain == Causal::Domain(&master) &&
        captured->causal_lifetime == scheduler.CausalLifetimeSequence(),
        "captured submission owner/lifetime changed");
    const auto semaphore = captured->semaphore;
    const auto waited_tick = captured->tick;
    vk::SemaphoreWaitInfo wait_info{};
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &semaphore;
    wait_info.pValues = &waited_tick;
    // Deliberately call the native API: MasterSemaphore::Wait can return from
    // cached/query completion. This does NOT claim its FaultPeek witness changed.
    const auto wait_result = m_runtime_context.device.waitSemaphores(
        &wait_info, host_wait_timeout_ns);
    std::printf("[ledger-native-wait] tick=%llu result=%d timeout_ns=%llu\n",
        static_cast<unsigned long long>(waited_tick), static_cast<int>(wait_result),
        static_cast<unsigned long long>(host_wait_timeout_ns));
    std::fflush(stdout);
    Require(name, "actual exact native host wait", wait_result == vk::Result::eSuccess,
        "native timeline host wait did not succeed");
    (void)history(); // complete bounded history still required after the real wait
    for (const auto& snapshot : captured->snapshots) {
      const auto& span = snapshot.destination;
      Require(name, "captured source reservation survived completion",
          same_span(span, recorder->Span(span.resource, span.offset, span.bytes, true)),
          "source history changed; recording-time payload is not consumption proof");
    }
    Require(name, "same owned readback after completion",
        same_span(captured->readback_owner, readback->LedgerSpan(0, readback_bytes)),
        "readback owner changed; stale identity must invalidate rather than retarget");
    readback->Invalidate(0, readback_bytes);
    std::memcpy(actual.data(), readback->Mapped().data(), sizeof(actual));
    for (size_t i = 0; i < actual.size(); ++i) {
      if (actual[i] != expected_readback[i]) {
        std::printf("[ledger-readback-mismatch] word=%zu expected=%08x actual=%08x\n",
            i, expected_readback[i], actual[i]);
      }
      Require(name, "native results and all canaries", actual[i] == expected_readback[i],
          "indirect work, retained input, output tail, or source/destination guard failed");
    }
    std::printf("[ledger-native] dispatches=1 groups=2,1,1 local=4,1,1 "
        "writes=8 actual_host_waits=1_fixture_owned tick=%llu semaphore=%016llx "
        "domain=%016llx lifetime=%llu shader=%016llx module=%llu\n",
        static_cast<unsigned long long>(captured->tick),
        static_cast<unsigned long long>(L::HandleBits(static_cast<VkSemaphore>(captured->semaphore))),
        static_cast<unsigned long long>(captured->native_domain),
        static_cast<unsigned long long>(captured->causal_lifetime),
        static_cast<unsigned long long>(captured->shader),
        static_cast<unsigned long long>(captured->module));
    for (size_t i = 0; i < actual.size(); ++i)
      std::printf("[ledger-readback] word=%zu value=%08x\n", i, actual[i]);

    gpu.SendCommandSync([&] {
      readback.reset(); // exact native completion was established above
      context.UnmapMemory(base, mapping_bytes);
      scheduler.Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
  } // Context/scheduler resources and producer thread are gone before Uninstall.

  const auto final_coverage = recorder->ReadCoverage();
  Require(name, "quiescent uninstall", recorder->Uninstall() && L::Active() == nullptr,
      "recorder could not be detached after resource and producer teardown");
  Require(name, "release test mapping", LibKernel::Memory::KernelCheckedReleaseDirectMemory(
      physical, mapping_bytes) == 0, "native ledger mapping release failed");
  for (const auto& event : captured->events) {
    std::printf("[ledger-event] seq=%llu kind=%u meaning=%u watch=%u word=%u mask=%x "
        "payload_mask=%x payload=%08x guest=%016llx operation=%016llx/%llu "
        "source_guest=%016llx thread=%llu action=%llu "
        "src=%llu:%llu:%016llx+%llu/%llu:r%llu dst=%llu:%llu:%016llx+%llu/%llu:r%llu "
        "domain=%016llx lifetime=%llu command=%016llx tick=%llu tracked_ordinal=%llu\n",
        static_cast<unsigned long long>(event.journal_sequence),
        static_cast<unsigned>(event.kind), static_cast<unsigned>(event.payload_meaning),
        static_cast<unsigned>(event.watch), static_cast<unsigned>(event.word),
        static_cast<unsigned>(event.byte_mask), static_cast<unsigned>(event.payload_mask), event.payload,
        static_cast<unsigned long long>(event.word_address),
        static_cast<unsigned long long>(event.operation_guest_begin),
        static_cast<unsigned long long>(event.operation_bytes),
        static_cast<unsigned long long>(event.source_guest_begin),
        static_cast<unsigned long long>(event.host_thread_token),
        static_cast<unsigned long long>(event.host_action_ordinal),
        static_cast<unsigned long long>(event.source.resource.capture),
        static_cast<unsigned long long>(event.source.resource.generation),
        static_cast<unsigned long long>(event.source.resource.handle),
        static_cast<unsigned long long>(event.source.offset),
        static_cast<unsigned long long>(event.source.bytes),
        static_cast<unsigned long long>(event.source.reservation),
        static_cast<unsigned long long>(event.destination.resource.capture),
        static_cast<unsigned long long>(event.destination.resource.generation),
        static_cast<unsigned long long>(event.destination.resource.handle),
        static_cast<unsigned long long>(event.destination.offset),
        static_cast<unsigned long long>(event.destination.bytes),
        static_cast<unsigned long long>(event.destination.reservation),
        static_cast<unsigned long long>(event.position.native_domain),
        static_cast<unsigned long long>(event.position.causal_lifetime),
        static_cast<unsigned long long>(event.position.command_buffer),
        static_cast<unsigned long long>(event.position.tick),
        static_cast<unsigned long long>(event.position.tracked_command_ordinal));
  }
  std::printf("[ledger-native] cpu_store_words=3 staging_snapshot_words=3 upload_edges=3 "
      "migration_edges=3 consumer_words=3 descriptor_candidate_words=3 "
      "old_version=1 new_version=2 old_spans_not_consumed=3 "
      "capture_cut=%zu final_events=%llu dropped_events=%llu dropped_reservations=%llu "
      "native_execution_in_ledger=false; actual completion/readback is separate evidence\n",
      captured->events.size(), static_cast<unsigned long long>(final_coverage.event_tickets),
      static_cast<unsigned long long>(final_coverage.dropped_events),
      static_cast<unsigned long long>(final_coverage.dropped_reservations));
  std::printf("[gpu]     %-32s ok\n", name);
}

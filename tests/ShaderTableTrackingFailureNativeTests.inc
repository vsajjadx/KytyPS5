// Native pipeline fixture: a TABLE-linked resource-tracking refusal fails only the dispatch.
// Paste/include inside VulkanHarness in tests/ShaderRecompilerComputeTests.cpp (next to
// CheckTableSelectedCall). Uses table_tracking_failure::GoodLeaf/BadLeaf/Caller from
// ShaderTableTrackingFailureCfgTests.inc semantics, re-encoded here with this harness's helpers.
//
// Rounds (every failure: nullptr, empty stage, all 64 output lanes + 16 tail canaries untouched,
// no callable miss, no module/cache poison):
//   P1 mapped positive FIRST (proves the output path before any refusal; saves id_p1).
//   P2 (fresh program) cold failure, cold repeat (re-translated from stashed source), cold recovery.
//   P1 reuses id_p1 -> cached good->bad (translation failure) -> unchanged repeats refused BEFORE any
//   module lookup (cached_rejections grows, translation_failures does not: the stale-plan regression)
//   -> good again binds the ORIGINAL module -> bad again -> bad callee CODE changed to valid code
//   relinks a new module (code-changed recovery) -> original table binds the original module.
// Each round looks the program up TWICE: run()'s explicit GetComputeProgram, then DispatchDirect's
// own lookup (renderCompute.cpp:376). Exact counters below include both lookups:
//   cold failure: both lookups re-translate the stashed source (+2 translations per round);
//   cached changed->bad: the explicit lookup translates (+1), the dispatch lookup sees the now
//   unchanged failed source and is refused before FindModule (+1 cached); unchanged: +2 cached.
void CheckTableTrackingSoftFailure() {
  constexpr const char *name = "TableTrackingSoftFailure";
  constexpr uintptr_t base = 0x0000000209400000ull;
  constexpr uint64_t size = 0x10000;
  constexpr uint64_t table = base + 0x100;
  constexpr uint64_t leaf_a = base + 0x1000, leaf_b = base + 0x1100, leaf_bad = base + 0x1200;
  constexpr uint64_t output = base + 0x4000;
  constexpr u32 lanes = 64, tail_words = 16, sentinel = 0xa5a5a5a5u;
  constexpr u32 value_a = 0x11u, value_b = 0x22u, value_fixed = 0x44u;
  const auto good_leaf = [](u32 value) {
    return std::vector<u32>{EncodeVop1(0x01, 6, 255), value, EncodeSop1(0x21, 125, 14)};
  };
  // V# base word from the caller's lane-varying v2 (GPU data), then a store through it.
  const std::vector<u32> bad_leaf{
      EncodeVop1(0x02, 24, Vgpr(2)), EncodeSMovB32(25, InlineU32(0)),
      EncodeSMovB32(26, 255), 0x100u, EncodeSMovB32(27, 255), 0x00014facu,
      EncodeMubuf0(0x1c, 0, false, false), EncodeMubuf1(6, 6, 0), EncodeSop1(0x21, 125, 14)};
  const auto make_caller = [](bool extra_nop) {
    std::vector<u32> code;
    if (extra_nop) code.push_back(EncodeSopp(0x00)); // distinct program hash for the cold rounds
    code.insert(code.end(), {EncodeSop1(0x04, 40, 126), EncodeVop2(0x1b, 1, InlineU32(1), 0),
                             EncodeVop2(0x1a, 2, InlineU32(2), 0), EncodeVop1(0x01, 6, InlineU32(0))});
    const auto loop = code.size();
    code.insert(code.end(), {EncodeVop1(0x02, 4, Vgpr(1)), EncodeSop1(0x04, 42, 126), EncodeVopc(0xd2, 4, 1),
                             EncodeSop2(0x1e, 5, 4, InlineU32(4)), EncodeSmem0(0x09, 14, 4), 5u << 25u,
                             EncodeSopp(0x0c, 0), EncodeSop1(0x21, 14, 14), EncodeSop2(0x15, 42, 42, 126),
                             EncodeSop1(0x04, 126, 42)});
    code.push_back(EncodeSopp(0x09, static_cast<u32>(static_cast<int32_t>(loop) -
                                                     static_cast<int32_t>(code.size() + 1u)) & 0xffffu));
    code.insert(code.end(), {EncodeSop1(0x04, 126, 40), EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 2)});
    AppendEnd(&code);
    return code;
  };
  const auto p1 = make_caller(false), p2 = make_caller(true);
  for (const auto *program_code : {&p1, &p2})
    ShaderMapUserData(reinterpret_cast<uint64_t>(program_code->data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<u32>(program_code->size() * sizeof(u32))});
  EnsureRuntimeContext();
  int64_t physical = -1;
  Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
      0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
      "tracking-failure allocation failed");
  void *mapped = reinterpret_cast<void *>(base);
  Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
      &mapped, size, 0x3, 0x10, physical, size) == 0 && mapped == reinterpret_cast<void *>(base),
      "tracking-failure mapping failed");
  std::memset(mapped, 0, size);
  const auto place = [](uint64_t address, const std::vector<u32> &code) {
    LibKernel::Memory::WriteBacking(address, code.data(), code.size() * sizeof(u32));
  };
  place(leaf_a, good_leaf(value_a));
  place(leaf_b, good_leaf(value_b));
  place(leaf_bad, bad_leaf);

  RenderContext context(m_runtime_context);
  context.InitializeGpu(nullptr);
  LibKernel::Memory::InstallGpuResources(&context);
  context.GetGpu().SendCommandSync([&] {
    CommandProcessor processor(context, 0);
    processor.BufferInit();
    context.MapMemory(base, size);
    auto &cache = context.GetBufferCache();
    auto &pipelines = context.GetPipelineCache();
    auto &shaders = processor.GetShCtx();
    const auto &faults = BufferCacheTestAccess::Faults(cache);
    const auto raw_descriptor = [](uint64_t address, u32 bytes) {
      ShaderBufferResource v{};
      v.UpdateAddress48(address);
      v.fields[2] = bytes;
      v.fields[3] = DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      return v;
    };
    const auto set_vsharp = [&](u32 sgpr, const ShaderBufferResource &v) {
      for (u32 i = 0; i < 4; ++i) shaders.SetCsUserSgpr(sgpr + i, v.fields[i]);
    };
    const auto bind = [&](const std::vector<u32> &program_code) {
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(program_code.data()),
                           .num_thread_x = lanes, .num_thread_y = 1, .num_thread_z = 1,
                           .wave_size = 64, .user_sgpr = 12});
      set_vsharp(0, raw_descriptor(output, lanes * sizeof(u32)));
      set_vsharp(8, raw_descriptor(table, 2u * 16u));
    };
    const auto write_entry = [&](u32 index, uint64_t target) {
      Require(name, "refresh table entry", context.InvalidateMemory(table + index * 16u, 8),
              "tracking-failure table escaped mapped memory");
      LibKernel::Memory::WriteBacking(table + index * 16u, &target, sizeof(target));
    };
    // expect_module=false: nullptr, empty stage, every output/tail word unchanged.
    const auto run = [&](const std::string &round, bool expect_module, u32 even, u32 odd) {
      constexpr u32 words = lanes + tail_words;
      Require(name, "reset output", context.InvalidateMemory(output, words * 4u), "output escaped mapped memory");
      std::array<u32, words> actual{};
      actual.fill(sentinel);
      LibKernel::Memory::WriteBacking(output, actual.data(), sizeof(actual));
      const auto misses_before = faults.CallTableMisses();
      ShaderComputeInputInfo input{};
      const auto *program = pipelines.GetComputeProgram(shaders.GetCs(), input);
      Require(name, round.c_str(), (program != nullptr) == expect_module, round + ": unexpected module binding");
      if (!expect_module)
        Require(name, round.c_str(), input.stage.module == nullptr && input.stage.resources == nullptr &&
                    !input.stage.call_closure,
                round + ": a refused dispatch kept a bound stage");
      processor.DispatchDirect(1, 1, 1, 0x41u);
      context.RunGarbageCollector();
      context.GetCommandScheduler().Finish();
      cache.ReadMemory(output, sizeof(actual));
      Require(name, "read output", LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)),
              "output was not readable");
      for (u32 lane = 0; lane < lanes; ++lane) {
        const u32 expected = expect_module ? ((lane & 1u) == 0u ? even : odd) : sentinel;
        Require(name, round.c_str(), actual[lane] == expected,
                round + " lane " + std::to_string(lane) + " expected " + Hex(expected) + ", got " + Hex(actual[lane]));
      }
      for (u32 i = lanes; i < words; ++i)
        Require(name, round.c_str(), actual[i] == sentinel, round + ": tail canary " + std::to_string(i) + " changed");
      Require(name, round.c_str(), faults.CallTableMisses() == misses_before, round + ": unexpected callable miss");
      return program != nullptr ? program->id : 0;
    };
    const auto counters = [&] { return pipelines.CallTracking(); };
    const auto expect_counters = [&](const char *round, uint64_t translations, uint64_t cached) {
      const auto now = counters();
      Require(name, round, now.translation_failures == translations && now.cached_rejections == cached,
              std::string(round) + ": tracking counters " + std::to_string(now.translation_failures) + "/" +
                  std::to_string(now.cached_rejections) + " expected " + std::to_string(translations) + "/" +
                  std::to_string(cached));
    };
    const auto base_counters = counters();
    const auto t0 = base_counters.translation_failures, c0 = base_counters.cached_rejections;

    // P1 mapped positive first.
    bind(p1);
    write_entry(0, leaf_a);
    write_entry(1, leaf_b);
    const auto id_p1 = run("P1 positive", true, value_a, value_b);
    expect_counters("P1 positive", t0, c0);

    // P2: cold failure, cold repeat, cold recovery.
    bind(p2);
    write_entry(1, leaf_bad);
    (void)run("P2 cold failure", false, 0, 0);
    expect_counters("P2 cold failure", t0 + 2, c0);
    (void)run("P2 cold repeat", false, 0, 0);
    expect_counters("P2 cold repeat", t0 + 4, c0);
    write_entry(1, leaf_b);
    const auto id_p2 = run("P2 cold recovery", true, value_a, value_b);
    expect_counters("P2 cold recovery", t0 + 4, c0);

    // P1: reuses its original module, cached good->bad, unchanged repeats refused before lookup.
    bind(p1);
    Require(name, "P1 reuse", run("P1 positive reuse", true, value_a, value_b) == id_p1,
            "P1 did not reuse its original module after the P2 rounds");
    Require(name, "distinct programs", id_p1 != 0 && id_p2 != 0 && id_p1 != id_p2, "P1/P2 shared a module");
    expect_counters("P1 positive reuse", t0 + 4, c0);
    write_entry(1, leaf_bad);
    (void)run("P1 cached good->bad", false, 0, 0);
    expect_counters("P1 cached good->bad", t0 + 5, c0 + 1);
    (void)run("P1 unchanged failed source", false, 0, 0);
    expect_counters("P1 unchanged failed source", t0 + 5, c0 + 3); // refused before FindModule
    (void)run("P1 unchanged failed source again", false, 0, 0);
    expect_counters("P1 unchanged failed source again", t0 + 5, c0 + 5);
    write_entry(1, leaf_b);
    Require(name, "P1 recovery", run("P1 bad->good", true, value_a, value_b) == id_p1,
            "the original module was not preserved for recovery");
    Require(name, "P1 cleared", run("P1 good repeat", true, value_a, value_b) == id_p1,
            "a recovered source kept its failure state");
    expect_counters("P1 recovered", t0 + 5, c0 + 5);
    // Code-changed recovery: same table row, the bad callee's CODE becomes valid.
    write_entry(1, leaf_bad);
    (void)run("P1 bad again", false, 0, 0);
    expect_counters("P1 bad again", t0 + 6, c0 + 6);
    Require(name, "rewrite callee", context.InvalidateMemory(leaf_bad, bad_leaf.size() * sizeof(u32)),
            "callee rewrite escaped mapped memory");
    std::vector<u32> fixed = good_leaf(value_fixed);
    fixed.resize(bad_leaf.size(), EncodeSopp(0x00));
    place(leaf_bad, fixed);
    const auto id_fixed = run("P1 code-changed recovery", true, value_a, value_fixed);
    Require(name, "code-changed module", id_fixed != 0 && id_fixed != id_p1, "changed callee code reused a module");
    expect_counters("P1 code-changed recovery", t0 + 6, c0 + 6);
    write_entry(1, leaf_b);
    Require(name, "P1 original", run("P1 original table", true, value_a, value_b) == id_p1,
            "the original table did not rebind its module");
    // P2 still binds its own module (no cross-program poison).
    bind(p2);
    Require(name, "P2 intact", run("P2 after P1 rounds", true, value_a, value_b) == id_p2,
            "P1 failures poisoned P2");
    context.UnmapMemory(base, size);
    context.GetCommandScheduler().Finish();
  });
  LibKernel::Memory::InstallGpuResources(nullptr);
  context.ShutdownGpu();
  Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
          "tracking-failure release failed");
  std::printf("[gpu]     %-32s ok\n", name);
}

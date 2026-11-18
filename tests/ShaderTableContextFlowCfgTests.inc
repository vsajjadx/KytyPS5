// Exercise the real callable/image path with definitions before each load.
void TestTableContextControlFlow() {
  using namespace ShaderRecompiler;
  namespace t = table_call_context;
  t::Memory memory;
  FakeGuestMemory fake(t::Read, &memory);
  t::Entry(memory, 0, t::leaf_a, t::context_a);
  t::Entry(memory, 1, t::leaf_a, t::context_a2);
  t::ImageTable(memory, t::context_a, t::images_a, 0x40u);
  t::ImageTable(memory, t::context_a2, t::images_a, 0x40u);
  std::array<uint32_t, 64> user_data{};
  const uint64_t output = t::table + 0x4000u;
  user_data[0] = static_cast<uint32_t>(output); user_data[1] = output >> 32u;
  user_data[2] = 256u; user_data[3] = 0x00014facu;
  user_data[8] = static_cast<uint32_t>(t::table); user_data[9] = t::table >> 32u;
  user_data[10] = 32u; user_data[11] = 3u << 28u;
  auto options = MakeCompileOptions(ShaderType::Compute);
  ShaderComputeInputInfo compute = *options.input_info.compute;
  compute.thread_ids_num = 1u;
  compute.threads_num[0] = 64u; compute.threads_num[1] = compute.threads_num[2] = 1u;
  options.input_info.compute = &compute;
  options.wave_size = 64u; options.user_data = user_data;
  IR::SrtRuntime runtime{.user_data = user_data, .userdata = &memory,
                        .read_specialization_memory = t::Read};
  auto source = PrepareShaderSource(t::Caller(), options);
  std::string reason;
  const Decoder::Program* linked = nullptr;
  const auto load = [&](const std::vector<uint32_t>& leaf) {
    t::Place(memory, t::leaf_a, leaf);
    linked = TryRefreshShaderSource(source, runtime, reason);
    Check(linked != nullptr && reason.empty(), "context flow fixture failed native linking");
  };
  const auto evaluate = [&](bool accepted, bool compile = false) {
    bool rejected = false;
    try {
      auto translated = TranslateProgram(*linked, options, true);
      translated.program.source_reads = source.reads;
      Check(accepted, "clobbered context reached a specialized image descriptor");
      const auto plan = IR::ExtractResourcePlan(translated.program);
      IR::ResourceSnapshot snapshot;
      IR::ResourceSpecialization specialization;
      Check(IR::MaterializeResources(plan, runtime, snapshot, specialization),
            "flow-proved context image resources did not materialize");
      std::unordered_set<uint32_t> images;
      for (const auto& image: snapshot.images) if (image.dwords[0]) images.insert(image.dwords[0]);
      Check(images == std::unordered_set<uint32_t>{0x40u, 0x41u},
            "flow-proved context selected incorrect image descriptors");
      if (compile) {
        auto compiled = CompileProgram(std::move(translated), options, specialization);
        CheckSpirvBinaryValidates(compiled.spirv);
      }
    } catch (const IR::ResourceTrackingFailure& failure) {
      rejected = true;
      Check(!accepted, "a context clobber after the load invalidated a valid reaching entry pair");
      Check(failure.reason.find("GetImageResource") != std::string::npos,
            "flow refusal did not retain the unproved image descriptor diagnostic");
    }
    Check(rejected != accepted, "context flow acceptance disagreed with reaching definitions");
  };

  auto later_clobber = t::Leaf();
  later_clobber.insert(later_clobber.begin(), {EncodeSopp(0x04, 1), EncodeSopp(0x00, 0)});
  later_clobber.insert(later_clobber.end() - 1, EncodeSMovB32(16, 128));
  load(later_clobber);
  evaluate(true, true); // both incoming paths preserve the pair; only a later write clobbers it

  auto incoming_clobber = later_clobber;
  incoming_clobber[1] = EncodeSMovB32(17, 128);
  load(incoming_clobber);
  evaluate(false); // one incoming path writes just the high half

  for (const int target_word: {2, 0}) {
    auto backedge = later_clobber;
    const auto branch_word = static_cast<int>(backedge.size()) - 1;
    backedge.insert(backedge.end() - 1,
                    EncodeSopp(0x04, static_cast<uint16_t>(target_word - branch_word - 1)));
    load(backedge);
    evaluate(false); // later write can reach the load again, including through the entry block
  }

  auto own_base = t::Leaf();
  own_base[0] = EncodeSmem0(0x02, 16, 8); // s[16:19] = load(s[16:17] + 16)
  own_base[5] = EncodeSmem0(0x0b, 32, 8); // use the V# now in s[16:19]
  own_base.insert(own_base.begin(), {EncodeSopp(0x04, 1), EncodeSopp(0x00, 0)});
  load(own_base);
  evaluate(true, true); // the load reads its old base before writing its destinations

  auto unreachable = later_clobber;
  unreachable[0] = EncodeSopp(0x02, static_cast<uint16_t>(unreachable.size() - 3u));
  load(unreachable);
  Check(source.linked->context_loads.empty() && source.call_data.empty(),
        "unreachable context load created an immutable data obligation");
  auto dead = TranslateProgram(*linked, options, true);
  Check(dead.program.info.images.empty(), "unreachable image load survived CFG pruning");

  auto interior_target = later_clobber;
  interior_target[0] = EncodeSopp(0x04, 2); // target the second DWORD of S_LOAD, not an instruction
  load(interior_target);
  Check(source.linked->context_loads.empty() && source.call_data.empty(),
        "optional context analysis accepted an interior-of-instruction branch target");

  // An optional context probe must not add preservation requirements to an
  // earlier independent call and turn native linking into a new fatal refusal.
  constexpr uint64_t earlier = t::table + 0x1200u;
  t::Place(memory, earlier, std::vector<uint32_t>{EncodeSMovB32(16, 128), EncodeSop1(0x21, 125, 12)});
  t::Place(memory, t::leaf_a, t::Leaf());
  auto two_calls = t::Caller();
  two_calls.insert(two_calls.begin() + 10,
      {EncodeSMovB32(12, 255), static_cast<uint32_t>(earlier),
       EncodeSMovB32(13, 255), static_cast<uint32_t>(earlier >> 32u), EncodeSop1(0x21, 12, 12)});
  auto independent = PrepareShaderSource(two_calls, options);
  Check(independent.calls.size() == 2u && !independent.calls[0].preserved_registers[16] &&
        !independent.calls[0].preserved_registers[17],
        "optional context probes imposed new preservation requirements on an earlier call");
  Check(TryRefreshShaderSource(independent, runtime, reason) != nullptr && reason.empty(),
        "an earlier context-clobbering call changed valid native linking into a refusal");
  Check(independent.linked->context_loads.empty(),
        "context provenance incorrectly crossed a potentially clobbering earlier call");
  std::printf("[host] TableContextControlFlow ok (8 reaching-definition/compatibility scenarios)\n");
}

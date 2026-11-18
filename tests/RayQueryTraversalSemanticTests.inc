  // Follow-on semantic negatives, kept separate from the first counter/inactive checkpoint.
  // Each compiles real guest ISA before asking the production traversal matcher to refuse it.
  template <typename Trace>
  void CheckRayQueryTraversalSemanticInputs(const Trace &traced, const char *selected = nullptr) {
    namespace IR = ShaderRecompiler::IR;
    constexpr const char *name = "RayQueryTraversal semantic inputs";
    const auto base = RayQueryTraversalWithoutCounter(RayQueryTraversalProgram());
    const auto run = [&](const char *label) { return !selected || std::strcmp(selected, label) == 0; };
    Require(name, "selected case", !selected || run("predicate") || run("payload") ||
                run("lds") || run("global"), "unknown semantic test selector");
    const auto rebranch = [&](std::vector<u32> &code) {
      const auto loop = std::ranges::find(code, 0xd7000024u);
      const auto branch = std::ranges::find_if(code, [](u32 word) {
        return (word & 0xffff0000u) == 0xbf890000u;
      });
      Require(name, "loop branch", loop != code.end() && branch != code.end(),
              "fixture lost its loop branch");
      *branch = EncodeSopp(0x09, static_cast<u32>(loop - branch - 1));
    };
    if (run("predicate")) {
      auto code = base;
      const auto commit = std::ranges::find(code, 0x7e0a0325u); // v_mov v5,v37
      Require(name, "commit", commit != code.end(), "fixture lost its extent commit");
      // Accept only during the first two software iterations; the counter is not exported.
      code.insert(commit, {EncodeVopc(0xc6, InlineU32(2), 31), 0x877e6a7eu});
      rebranch(code);
      Require(name, "counter-dependent acceptance", !traced(code, ShaderType::Compute),
              "an unexported iteration counter changed native hit acceptance");
    }
    if (run("payload")) {
      auto code = base;
      const auto commit = std::ranges::find(code, 0x7e340330u); // v_mov v26,v48
      Require(name, "payload commit", commit != code.end(), "fixture lost its payload commit");
      // Same accepted-hit predicate as extent, but the exported payload accumulates history.
      *commit = EncodeVop2(0x25, 26, InlineU32(1), 26);
      Require(name, "history-dependent payload", !traced(code, ShaderType::Compute),
              "sharing the extent predicate admitted an order-dependent accumulator");
    }
    if (run("lds")) {
      auto code = base;
      const auto output = std::ranges::find(code, EncodeVop1(0x01, 45, InlineU32(0)));
      Require(name, "LDS export", output != code.end(), "fixture lost its output word");
      const auto at = static_cast<size_t>(output - code.begin());
      code.erase(output);
      code.insert(code.begin() + at, {EncodeVop1(0x01, 37, InlineU32(0)),
                                     EncodeDs0(0x36), EncodeDs1(45, 0, 37),
                                     0xbf8cc07fu});
      Require(name, "observable LDS stack", !traced(code, ShaderType::Compute),
              "skipped stack writes changed a shared-memory value read after the loop");
    }
    if (run("global")) {
      auto ir = PackedTraversalIR(base, 32);
      IR::Inst *push = nullptr, *address = nullptr;
      for (auto *block : ir.blocks) {
        for (auto &inst : *block) {
          if (!push && inst.GetOpcode() == IR::ValueOpcode::WriteSharedU32) push = &inst;
          if (address || inst.GetOpcode() != IR::ValueOpcode::LoadAddressU32) continue;
          const auto &memory = ir.memory_info[inst.Flags<IR::MemoryFlags>().index];
          if (memory.kind == IR::ResourceKind::Global && memory.address_is_full) address = &inst;
        }
      }
      Require(name, "global write fixture", push && address,
              "fixture lost its stack push or external global address");
      auto *block = push->Parent();
      const auto at = std::ranges::find_if(*block, [&](const IR::Inst &inst) { return &inst == push; });
      (void)block->PrependNewInst(at, IR::ValueOpcode::StoreAddressU32,
                           {address->Arg(0), address->Arg(1), address->Arg(2),
                            push->Arg(1), push->Arg(2)}, address->Flags<uint64_t>());
      Require(name, "global box write", IR::TraceTraversalsWithRayQueries(ir) == 0,
              "a global write of an intersection word was mistaken for an unobservable stack");
    }
  }


  // The original BVH emitter returns a zero vector when its execution predicate is false.
  // Compile a real mixed-active wave, then make component 1 observable only for its inactive
  // lanes. That outside use is intentionally ignored by the participating-lane output gate.
  // Evaluate the small scalar output slice before/after the actual production transform;
  // this is a host IR semantic regression, not a claim of another GPU measurement.
  void CheckRayQueryTraversalInactiveExtract() {
    namespace IR = ShaderRecompiler::IR;
    using Op = IR::ValueOpcode;
    constexpr const char *name = "RayQueryTraversal inactive extract";
    auto ir = PackedTraversalIR(RayQueryTraversalInactiveProgram(), 64);
    IR::Inst *bvh = nullptr, *component = nullptr, *output = nullptr;
    for (auto *block : ir.blocks) {
      for (auto &inst : *block) {
        if (inst.GetOpcode() == Op::BvhIntersect) bvh = &inst;
        if (inst.GetOpcode() == Op::StoreBufferU32x2)
          output = inst.Arg(4).Resolve().TryInstruction();
      }
    }
    Require(name, "fixture", bvh && output &&
                output->GetOpcode() == Op::CompositeConstructU32x2,
            "compiled fixture lost its intersection or output words");
    for (const auto &use : bvh->Uses()) {
      auto *extract = use.user;
      if (extract->GetOpcode() == Op::CompositeExtractU32x4 &&
          extract->Arg(1).IsImmediate() && extract->Arg(1).U32() == 1) component = extract;
    }
    auto *exec = bvh->Arg(2).Resolve().TryInstruction();
    Require(name, "active lanes", component && exec && exec->GetOpcode() == Op::Phi &&
                exec->NumArgs() == 2, "compiled fixture lost its mixed-active loop");
    IR::Value entry{};
    for (size_t i = 0; i < exec->NumArgs(); ++i)
      if (exec->PhiBlock(i) != bvh->Parent()) entry = exec->Arg(i).Resolve();
    Require(name, "entry lanes", entry.TryInstruction() &&
                entry.TryInstruction()->Parent() != exec->Parent(),
            "fixture did not identify its nonconstant entry predicate");
    auto *block = output->Parent();
    const auto at = std::ranges::find_if(*block, [&](const IR::Inst &inst) { return &inst == output; });
    auto inactive = block->PrependNewInst(at, Op::LogicalNot, {entry});
    auto exported = block->PrependNewInst(at, Op::SelectU32,
                        {IR::Value(&*inactive), IR::Value(component), IR::Value(0u)});
    output->SetArg(1, IR::Value(&*exported));
    const auto evaluate = [&](auto &&self, IR::Value value, u32 depth) -> u32 {
      Require(name, "bounded output slice", depth < 32, "output slice was unexpectedly cyclic");
      value = value.Resolve();
      if (value == entry || value.TryInstruction() == exec) return 0; // An inactive lane.
      if (value.IsImmediate())
        return value.GetType() == IR::Type::U1 ? u32(value.U1()) : value.U32();
      const auto *inst = value.TryInstruction();
      Require(name, "typed output slice", inst != nullptr, "output slice has no producer");
      const auto arg = [&](u32 index) { return self(self, inst->Arg(index), depth + 1); };
      switch (inst->GetOpcode()) {
        case Op::RayQueryValid: return 1; // A GPU-valid mirror is shared by the whole wave.
        case Op::LogicalNot: return !arg(0);
        case Op::LogicalAnd: return arg(0) && arg(1);
        case Op::LogicalOr: return arg(0) || arg(1);
        case Op::SelectU1:
        case Op::SelectU32: return arg(0) ? arg(1) : arg(2);
        case Op::CompositeExtractU32x4: {
          const auto *source = inst->Arg(0).Resolve().TryInstruction();
          Require(name, "inactive BVH", source == bvh &&
                      self(self, bvh->Arg(2), depth + 1) == 0,
                  "output slice did not read a false-exec BVH component");
          return 0; // spirvEmitterBvh: false exec emits ConstantU32CompositeZero(4).
        }
        default: Require(name, "known scalar operation", false,
                         "output slice contains an unexpected operation");
      }
      return 0;
    };
    Require(name, "original inactive result", evaluate(evaluate, output->Arg(1), 0) == 0,
            "the unlowered false-exec BVH did not export zero");
    const auto lowered = IR::TraceTraversalsWithRayQueries(ir);
    const auto result = evaluate(evaluate, output->Arg(1), 0);
    if (result != 0)
      std::printf("[host]    inactive component 1: lowered %u, result 0x%x (expected zero)\n",
                  lowered, result);
    Require(name, "preserved inactive component", result == 0,
            "a valid mirror replaced an inactive lane's zero BVH component with float one");
    std::printf("[host]    inactive BVH component preserved (lowered %u)\n", lowered);
  }

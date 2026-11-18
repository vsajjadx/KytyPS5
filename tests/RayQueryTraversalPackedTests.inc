  // A synthetic format-10 traversal. The original fixture supplies the control flow and
  // triangle/any-hit code; these edits change only the documented instance layout and
  // make the loop carry the already-shifted low/high BVH address. No title shader is embedded.
  static std::vector<u32> RayQueryTraversalPackedProgram() {
    constexpr const char *name = "RayQueryTraversal packed program";
    auto code = RayQueryTraversalWithoutCounter(RayQueryTraversalProgram64());
    const auto replace = [&](std::initializer_list<u32> before,
                             std::initializer_list<u32> after, size_t wanted = 1) {
      size_t count = 0, begin = 0;
      while (begin < code.size()) {
        const auto at = std::search(code.begin() + begin, code.end(), before.begin(), before.end());
        if (at == code.end()) break;
        const auto index = static_cast<size_t>(at - code.begin());
        code.erase(at, at + before.size());
        code.insert(code.begin() + index, after);
        begin = index + after.size();
        ++count;
      }
      Require(name, "fixture edit", count == wanted, "synthetic instruction pattern changed");
    };
    replace({0xf4040201u, 0xfa000028u}, {0xf4040201u, 0xfa000030u});
    // Compute the TLAS >> 3 once, keep it in v[72:73], and restore both words together.
    size_t pairs = 0;
    for (size_t i = 0; i + 1 < code.size(); ++i) {
      if (code[i] != 0x7e1e0202u || code[i + 1] != 0x7e200203u) continue;
      std::vector<u32> extra;
      if (pairs++ == 0) {
        extra = {EncodeVop3Word0(0x300, 72),
                 EncodeVop3Word1(InlineU32(3), Vgpr(15), InlineU32(0))};
      }
      extra.push_back(EncodeVop1(0x01, 70, Vgpr(72)));
      extra.push_back(EncodeVop1(0x01, 71, Vgpr(73)));
      code.insert(code.begin() + i + 2, extra.begin(), extra.end());
      i += extra.size() + 1;
    }
    Require(name, "TLAS restore", pairs == 2, "fixture lost its initial or restored TLAS pair");
    replace({0xd7000024u, 0x02021e83u},
            {EncodeVop1(0x01, 36, Vgpr(70)), EncodeVop1(0x01, 37, Vgpr(71))});
    replace({0xdc348000u, 0x447d0034u}, {0xdc348030u, 0x447d0034u});
    // The mask is nonzero iff the upper byte of +60 is nonzero. The comparison plus
    // and-not uses the same guarded inverted comparison shape as the observed guest IR.
    replace({0x34486e95u, 0x7d284880u},
            {EncodeVopc(0xc6, 255, 55), 0x00ffffffu, EncodeSop2(0x15, 126, 126, 106)});
    replace({0x7e1e0344u, 0x7e200345u},
            {EncodeVop1(0x01, 70, Vgpr(68)),
             EncodeVop2(0x1b, 71, 255, 69), 0x003fffffu,
             EncodeVop3Word0(0x2ff, 15),
             EncodeVop3Word1(InlineU32(3), Vgpr(70), InlineU32(0))});
    // Format 10 stores the inverse rows at +0..+47; +48/+52 is the packed BLAS.
    replace({0xdc388008u, 0x387d0034u}, {0xdc388000u, 0x387d0034u});
    replace({0xdc388018u, 0x3c7d0034u}, {0xdc388010u, 0x3c7d0034u});
    replace({0xdc388028u, 0x407d0034u}, {0xdc388020u, 0x407d0034u});
    replace({0x05065138u}, {EncodeVop3Word1(Vgpr(56), Vgpr(40), Vgpr(59))});
    replace({0x560c533bu}, {EncodeVop2(0x2b, 6, Vgpr(57), 41)});
    replace({0x560c553eu}, {EncodeVop2(0x2b, 6, Vgpr(58), 42)});
    replace({0x050a5139u}, {EncodeVop3Word1(Vgpr(60), Vgpr(40), Vgpr(63))});
    replace({0x560e533cu}, {EncodeVop2(0x2b, 7, Vgpr(61), 41)});
    replace({0x560e553fu}, {EncodeVop2(0x2b, 7, Vgpr(62), 42)});
    replace({0x050e513au}, {EncodeVop3Word1(Vgpr(64), Vgpr(40), Vgpr(67))});
    replace({0x5610533du}, {EncodeVop2(0x2b, 8, Vgpr(65), 41)});
    replace({0x56105540u}, {EncodeVop2(0x2b, 8, Vgpr(66), 42)});
    replace({0x56125b3bu}, {EncodeVop2(0x2b, 9, Vgpr(57), 45)});
    replace({0x56125d3eu}, {EncodeVop2(0x2b, 9, Vgpr(58), 46)});
    replace({0x10145939u}, {EncodeVop2(0x08, 10, Vgpr(60), 44)});
    replace({0x56145b3cu}, {EncodeVop2(0x2b, 10, Vgpr(61), 45)});
    replace({0x56145d3fu}, {EncodeVop2(0x2b, 10, Vgpr(62), 46)});
    replace({0x1016593au}, {EncodeVop2(0x08, 11, Vgpr(64), 44)});
    replace({0x56165b3du}, {EncodeVop2(0x2b, 11, Vgpr(65), 45)});
    replace({0x56165d40u}, {EncodeVop2(0x2b, 11, Vgpr(66), 46)});
    const auto loop = std::ranges::find(code, EncodeVop1(0x01, 36, Vgpr(70)));
    const auto branch = std::ranges::find_if(code, [](u32 word) {
      return (word & 0xffff0000u) == 0xbf890000u;
    });
    Require(name, "loop branch", loop != code.end() && branch != code.end(),
            "fixture lost its loop branch");
    *branch = EncodeSopp(0x09, static_cast<u32>(loop - branch - 1));
    return code;
  }

  static ShaderRecompiler::IR::Program PackedTraversalIR(std::span<const u32> code,
                                                        u32 wave = 64) {
    static constexpr ShaderProfile profile{.maintenance8 = true};
    ShaderComputeInputInfo compute{};
    compute.wave_size = wave;
    compute.host_subgroup_size = 32;
    compute.threads_num[0] = wave == 64 ? 64 : 8;
    compute.threads_num[1] = compute.threads_num[2] = 1;
    compute.thread_ids_num = 1;
    compute.lds_size_dwords = 1024;
    const std::array<u32, 64> user_data{};
    ShaderRecompiler::CompileOptions options;
    options.stage = ShaderType::Compute;
    options.wave_size = wave;
    options.profile = &profile;
    options.user_data = user_data;
    options.input_info.compute = &compute;
    auto translated = ShaderRecompiler::TranslateProgram(code, options);
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    specialization.buffers.resize(translated.program.info.buffers.size());
    return ShaderRecompiler::CompileProgram(std::move(translated), options, specialization).program;
  }

  struct PackedTraversalAddress {
    ShaderRecompiler::IR::Inst *sum, *high, *low_phi, *high_phi;
    ShaderRecompiler::IR::Value active;
  };

  static PackedTraversalAddress PackedTraversalAddressOf(ShaderRecompiler::IR::Program &ir) {
    namespace IR = ShaderRecompiler::IR;
    constexpr const char *name = "RayQueryTraversal packed IR";
    IR::Inst *bvh = nullptr;
    for (auto *block : ir.blocks)
      for (auto &inst : *block)
        if (inst.GetOpcode() == IR::ValueOpcode::BvhIntersect) bvh = &inst;
    Require(name, "intersection", bvh != nullptr, "compiled fixture lost its intersection");
    auto *address = bvh->Arg(1).Resolve().TryInstruction();
    const auto active = bvh->Arg(2).Resolve();
    auto *sum = IR::ResolveActiveU32(address->Arg(0), active).TryInstruction();
    auto *high = IR::ResolveActiveU32(address->Arg(1), active).TryInstruction();
    Require(name, "address sums", sum && high &&
                sum->GetOpcode() == IR::ValueOpcode::IAdd32 &&
                high->GetOpcode() == IR::ValueOpcode::IAdd32,
            "compiled fixture lost its split address sums");
    auto *low_phi = IR::ResolveActiveU32(sum->Arg(1), active).TryInstruction();
    auto *high_phi = IR::ResolveActiveU32(high->Arg(0), active).TryInstruction();
    Require(name, "shifted phis", low_phi && high_phi &&
                low_phi->GetOpcode() == IR::ValueOpcode::Phi &&
                high_phi->GetOpcode() == IR::ValueOpcode::Phi &&
                low_phi->Parent() == high_phi->Parent(),
            "compiled fixture did not carry both shifted address words");
    return {sum, high, low_phi, high_phi, active};
  }

  static ShaderRecompiler::IR::Inst *PackedTraversalLoad(ShaderRecompiler::IR::Program &ir,
                                                        u32 offset) {
    namespace IR = ShaderRecompiler::IR;
    IR::Inst *found = nullptr;
    for (auto *block : ir.blocks) {
      for (auto &inst : *block) {
        if (inst.GetOpcode() != IR::ValueOpcode::LoadAddressU32) continue;
        const auto &memory = ir.memory_info[inst.Flags<IR::MemoryFlags>().index];
        if (memory.kind != IR::ResourceKind::Global || !memory.address_is_full ||
            memory.offset != offset) continue;
        Require("RayQueryTraversal packed IR", "unique instance load", !found,
                "fixture has an ambiguous instance load");
        found = &inst;
      }
    }
    Require("RayQueryTraversal packed IR", "instance load", found != nullptr,
            "fixture lost a packed instance load");
    return found;
  }

  void CheckRayQueryTraversalPacked() {
    namespace IR = ShaderRecompiler::IR;
    using Op = IR::ValueOpcode;
    constexpr const char *name = "RayQueryTraversal format 10";
    const auto code = RayQueryTraversalPackedProgram();
    const auto immediate = [](IR::Value value, u32 wanted) {
      value = value.Resolve();
      return value.IsImmediate() && value.GetType() == IR::Type::U32 && value.U32() == wanted;
    };
    const auto find = [&](IR::Program &ir, Op opcode, u32 constant) -> IR::Inst * {
      IR::Inst *found = nullptr;
      for (auto *block : ir.blocks) {
        for (auto &inst : *block) {
          if (inst.GetOpcode() != opcode) continue;
          for (size_t i = 0; i < inst.NumArgs(); ++i) {
            if (!immediate(inst.Arg(i), constant)) continue;
            Require(name, "unique fixture operation", !found, "fixture operation is ambiguous");
            found = &inst;
            break;
          }
        }
      }
      Require(name, "fixture operation", found != nullptr, "fixture operation is missing");
      return found;
    };
    const auto reject = [&](const char *label, auto mutate) {
      auto ir = PackedTraversalIR(code);
      mutate(ir);
      Require(name, label, IR::TraceTraversalsWithRayQueries(ir) == 0,
              "a non-equivalent packed traversal was accepted");
    };
    reject("missing low carry", [&](IR::Program &ir) {
      auto address = PackedTraversalAddressOf(ir);
      address.high->SetArg(1, IR::Value(0u));
    });
    reject("carry from another sum", [&](IR::Program &ir) {
      auto address = PackedTraversalAddressOf(ir);
      auto *carry = address.high->Arg(1).Resolve().TryInstruction();
      Require(name, "carry select", carry && carry->GetOpcode() == Op::SelectU32,
              "fixture lost its carry select");
      auto *condition = carry->Arg(0).Resolve().TryInstruction();
      if (condition && condition->GetOpcode() == Op::LogicalAnd)
        condition = (condition->Arg(0).Resolve() == address.active ?
                         condition->Arg(1) : condition->Arg(0)).Resolve().TryInstruction();
      Require(name, "carry compare", condition && condition->GetOpcode() == Op::ULessThan32,
              "fixture lost its carry comparison");
      condition->SetArg(0, IR::Value(address.low_phi));
    });
    reject("wrong entry high component", [&](IR::Program &ir) {
      auto address = PackedTraversalAddressOf(ir);
      auto *header = address.high_phi->Parent();
      for (size_t i = 0; i < address.high_phi->NumArgs(); ++i) {
        if (address.high_phi->PhiBlock(i) == header->terminator.continue_block) continue;
        size_t low = 0;
        while (low < address.low_phi->NumArgs() &&
               address.low_phi->PhiBlock(low) != address.high_phi->PhiBlock(i)) ++low;
        Require(name, "entry predecessor", low < address.low_phi->NumArgs(),
                "the low/high entry predecessors differ");
        // Substitute the low word on this exact predecessor. This remains a wrong
        // 64-bit entry address even if constant folding simplifies the high extract.
        address.high_phi->SetArg(i, address.low_phi->Arg(low));
      }
    });
    reject("different high predecessor value", [&](IR::Program &ir) {
      auto address = PackedTraversalAddressOf(ir);
      auto *latch = address.high_phi->Parent()->terminator.continue_block;
      for (size_t i = 0; i < address.high_phi->NumArgs(); ++i) {
        if (address.high_phi->PhiBlock(i) != latch) continue;
        address.high_phi->SetArg(i, IR::Value(0u));
      }
    });
    reject("unpaired TLAS reset", [&](IR::Program &ir) {
      auto address = PackedTraversalAddressOf(ir);
      auto *header = address.high_phi->Parent();
      auto *exec = address.active.TryInstruction();
      IR::Value entry;
      for (size_t i = 0; i < address.high_phi->NumArgs(); ++i) {
        if (address.high_phi->PhiBlock(i) == header->terminator.continue_block) continue;
        for (size_t j = 0; j < exec->NumArgs(); ++j)
          if (exec->PhiBlock(j) == address.high_phi->PhiBlock(i))
            entry = IR::ResolveActiveU32(address.high_phi->Arg(i), exec->Arg(j));
      }
      IR::Inst *reset = nullptr;
      bool in_loop = false;
      for (auto *block : ir.blocks) {
        if (block == header) in_loop = true;
        if (!in_loop) continue;
        for (auto &inst : *block)
          if (inst.GetOpcode() == Op::SelectU32 && inst.Arg(1).Resolve() == entry)
            reset = &inst;
        if (block == header->terminator.continue_block) break;
      }
      Require(name, "TLAS reset select", reset != nullptr, "fixture lost its TLAS reset");
      reset->SetArg(1, IR::Value(address.low_phi));
    });
    reject("different high Select condition", [&](IR::Program &ir) {
      auto *masked = find(ir, Op::BitwiseAnd32, 0x003fffffu);
      IR::Inst *select = nullptr;
      for (const auto &use : masked->Uses())
        if (use.user->GetOpcode() == Op::SelectU32 && use.operand == 1) select = use.user;
      Require(name, "packed high select", select != nullptr, "fixture lost its packed high select");
      auto *block = select->Parent();
      const auto at = std::ranges::find_if(*block, [&](const IR::Inst &inst) { return &inst == select; });
      auto &inverted = *block->PrependNewInst(at, Op::LogicalNot, {select->Arg(0)});
      select->SetArg(0, IR::Value(&inverted));
    });
    for (const u32 mask : {0xffffffffu, 0x007fffffu}) {
      reject(mask == 0xffffffffu ? "unmasked packed high flags" : "wrong packed high width",
             [&](IR::Program &ir) {
        auto *masked = find(ir, Op::BitwiseAnd32, 0x003fffffu);
        for (size_t i = 0; i < 2; ++i)
          if (immediate(masked->Arg(i), 0x003fffffu)) masked->SetArg(i, IR::Value(mask));
      });
    }
    reject("wrong packed low offset", [&](IR::Program &ir) {
      auto *load = PackedTraversalLoad(ir, 48);
      ir.memory_info[load->Flags<IR::MemoryFlags>().index].offset = 44;
    });
    reject("wrong packed high offset", [&](IR::Program &ir) {
      auto *load = PackedTraversalLoad(ir, 52);
      ir.memory_info[load->Flags<IR::MemoryFlags>().index].offset = 48;
    });
    reject("different high instance address", [&](IR::Program &ir) {
      PackedTraversalLoad(ir, 52)->SetArg(1, IR::Value(0u));
    });
    reject("wrong mask threshold", [&](IR::Program &ir) {
      auto *test = find(ir, Op::UGreaterThanEqual32, 0x00ffffffu);
      for (size_t i = 0; i < 2; ++i)
        if (immediate(test->Arg(i), 0x00ffffffu)) test->SetArg(i, IR::Value(0x0000ffffu));
    });
    reject("mask from another instance word", [&](IR::Program &ir) {
      auto *load = PackedTraversalLoad(ir, 60);
      ir.memory_info[load->Flags<IR::MemoryFlags>().index].offset = 56;
    });
    // Quarantine is explicit even for structurally equivalent packed programs. Execute
    // only valid fixtures in software and verify their hit outputs; invalid variants above
    // remain host refusal cases. These tests make no native guest-conversion claim.
    CheckRayQueryTraversal("RayQueryTraversal format 10 wave64", code, true, 10, false, false);
    CheckRayQueryTraversal("RayQueryTraversal format 10 low carry", code, true, 10, true, false);
    CheckRayQueryTraversal("RayQueryTraversal format 10 wave32", code, false, 10, false, false);
    for (const u32 wave : {32u, 64u}) {
      auto ir = PackedTraversalIR(code, wave);
      Require(name, "compiled packed traversal", IR::TraceTraversalsWithRayQueries(ir) == 0,
              "an unproved packed traversal bypassed the quarantine");
    }
    // Reordering phi operands while preserving their predecessor identities is equivalent.
    // This catches implementations that correlate low/high phis by vector index alone.
    {
      auto ir = PackedTraversalIR(code);
      auto address = PackedTraversalAddressOf(ir);
      auto *header = address.high_phi->Parent();
      auto &reordered = *header->PrependNewInst(header->begin(), Op::Phi);
      for (size_t i = address.high_phi->NumArgs(); i-- > 0;)
        reordered.AddPhiOperand(address.high_phi->PhiBlock(i), address.high_phi->Arg(i));
      address.high_phi->ReplaceUsesWith(IR::Value(&reordered));
      Require(name, "reordered phi predecessors", IR::TraceTraversalsWithRayQueries(ir) == 0,
              "equivalent predecessor order bypassed the quarantine");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

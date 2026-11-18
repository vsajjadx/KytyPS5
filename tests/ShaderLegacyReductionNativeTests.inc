// Historical reduction opcodes loaded from fixed wire IDs, emitted through the production
// backend and checked on the GPU. The generic frontend remains the ordinary translation path.
void CheckLegacyReductionOpcodes(VulkanHarness& vulkan) {
  using namespace ShaderRecompiler::IR;
  struct LegacyReduction {
    uint16_t wire;
    const char* name;
    const char* spirv;
    ValueOpcode operation;
    ShaderOpcode vector_opcode, scalar_opcode;
    u32 vector_encoding, scalar_encoding, identity;
  };
  const std::array reductions{
      LegacyReduction{212u, "SubgroupUMinU32", "OpGroupNonUniformUMin",
                      ValueOpcode::UMin32, ShaderOpcode::V_MIN_U32,
                      ShaderOpcode::S_MIN_U32, 0x13u, 0x07u, UINT32_MAX},
      LegacyReduction{213u, "SubgroupUMaxU32", "OpGroupNonUniformUMax",
                      ValueOpcode::UMax32, ShaderOpcode::V_MAX_U32,
                      ShaderOpcode::S_MAX_U32, 0x14u, 0x09u, 0u},
      LegacyReduction{214u, "SubgroupOrU32", "OpGroupNonUniformBitwiseOr",
                      ValueOpcode::BitwiseOr32, ShaderOpcode::V_OR_B32,
                      ShaderOpcode::S_OR_B32, 0x1cu, 0x10u, 0u}};
  for (const auto& reduction : reductions) {
    // Start with a known one-U32 instruction envelope, then explicitly restore its
    // historical wire ID before the typed production reader.
    ResourcePlan wire_source;
    auto& prototype =
        wire_source.value_storage.emplace_back(ValueOpcode::SubgroupReduceU32, 0u);
    prototype.SetArg(0u, Value(0u));
    auto wire_bytes = SaveResourcePlan(wire_source);
    Require(reduction.name, "historical wire identity",
            ValueOpcodeName(static_cast<ValueOpcode>(reduction.wire)) == reduction.name &&
                wire_bytes.size() >= 14u,
            "historical reduction opcode no longer has its persisted meaning");
    wire_bytes[4] = static_cast<uint8_t>(reduction.wire);
    wire_bytes[5] = static_cast<uint8_t>(reduction.wire >> 8u);
    for (const u32 wave : {32u, 64u}) {
      for (const bool masked : {false, true}) {
        const std::string name = "Legacy" + std::string(reduction.name) +
                                 std::to_string(wave) + (masked ? "Masked" : "Full");
        TestCase test;
        test.name = name.c_str();
        // Historical flags have no operation-selection meaning. Sparse cases deliberately
        // preserve nonzero bits through the wire reader and must execute the same operation.
        const uint64_t legacy_flags = masked ? UINT64_C(0x0123456789abcdef) : uint64_t{0};
        auto legacy_bytes = wire_bytes;
        for (size_t byte = 0u; byte < sizeof(legacy_flags); ++byte)
          legacy_bytes[6u + byte] = static_cast<uint8_t>(legacy_flags >> (8u * byte));
        ResourcePlan loaded;
        Require(test.name, "historical typed load",
                LoadResourcePlan(legacy_bytes, loaded) && loaded.value_storage.size() == 1u &&
                    SaveResourcePlan(loaded) == legacy_bytes,
                "historical reduction did not survive the real plan reader");
        const auto& restored = loaded.value_storage.front();
        Require(test.name, "historical typed signature",
                restored.GetType() == Type::U32 && restored.NumArgs() == 1u &&
                    restored.Flags<uint64_t>() == legacy_flags,
                "historical reduction result, operands or unused flags changed");
        const uint64_t all_lanes = UINT64_MAX >> (64u - wave);
        // Exclude lane0 and the final four lanes. OR therefore loses its highest
        // contributing bit; min/max also move away from the final lane.
        const uint64_t exec = masked ? all_lanes & ~(uint64_t{1} | (uint64_t{0xf} << (wave - 4u)))
                                     : all_lanes;
        auto& code = test.code;
        code.push_back(EncodeVop2(0x1au, 20u, InlineU32(2u), 0u));
        const u32 value_shift = wave == 64u ? 26u : 27u;
        if (reduction.operation == ValueOpcode::UMin32) {
          // Cross the sign bit between halves so a signed minimum is observably wrong.
          AppendVMovU32(&code, 13u, wave - 1u);
          code.push_back(EncodeVop2(0x26u, 13u, Vgpr(13u), 0u));
          code.push_back(EncodeVop2(0x1au, 13u, InlineU32(value_shift), 13u));
        } else if (reduction.operation == ValueOpcode::UMax32) {
          // The highest unsigned values are in the upper guest half.
          code.push_back(EncodeVop2(0x1au, 13u, InlineU32(value_shift), 0u));
        } else {
          code.push_back(EncodeVop2(0x16u, 13u, InlineU32(2u), 0u));
          AppendVop3(&code, 0x11au, 13u, Vgpr(13u), InlineU32(1u));
        }
        if (masked) {
          AppendSMovLiteral(&code, 126u, static_cast<u32>(exec));
          AppendSMovLiteral(&code, 127u, static_cast<u32>(exec >> 32u));
        }
        // The ordinary RDNA2 prefix-scan idiom, with endpoint lanes appropriate
        // to each guest wave width. Its generic replacement is the control.
        code.insert(code.end(), {EncodeSop1(0x04u, 24u, 126u),
                                 EncodeSop1(0x28u, 106u, 126u)});
        AppendSMovLiteral(&code, 26u, reduction.identity);
        AppendVop3(&code, 0x101u, 13u, 26u, Vgpr(13u), 24u);
        for (const u32 shift : {1u, 2u, 4u, 8u}) {
          code.push_back(EncodeVop2(reduction.vector_encoding, 13u, 250u, 13u));
          code.push_back(EncodeVop2Dpp(13u, 0x110u + shift));
        }
        if (wave == 64u) {
          code.insert(code.end(), {0xd778100cu, 0x0305830du,
                                  EncodeVop2(reduction.vector_encoding, 13u, Vgpr(13u), 12u)});
        }
        code.push_back(EncodeSop1(0x04u, 126u, 106u));
        AppendVop3(&code, 0x360u, 6u, Vgpr(13u), InlineU32(wave / 2u - 1u));
        AppendVop3(&code, 0x360u, 7u, Vgpr(13u), InlineU32(wave - 1u));
        code.push_back(EncodeSop2(reduction.scalar_encoding, 8u, 6u, 7u));
        code.push_back(EncodeVop1(0x01u, 0u, 8u));
        AppendBufferStoreDword(&code, 0u, 20u);
        AppendEnd(&code);
        test.has_compute_info = true;
        test.compute_info.wave_size = static_cast<uint8_t>(wave);
        test.compute_info.threads_num[0] = wave;
        test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
        test.compute_info.thread_ids_num = 1u;
        test.compute_info.lds_size_dwords = 0u;
        test.initial.assign(wave, 0xa5a5a5a5u);
        u32 expected = reduction.identity;
        for (u32 lane = 0u; lane < wave; ++lane) {
          if ((exec & (uint64_t{1} << lane)) == 0u) continue;
          if (reduction.operation == ValueOpcode::UMin32)
            expected = std::min(expected, (wave - 1u - lane) << value_shift);
          else if (reduction.operation == ValueOpcode::UMax32)
            expected = std::max(expected, lane << value_shift);
          else expected |= 1u << (lane / 4u);
        }
        test.expected = test.initial;
        for (u32 lane = 0u; lane < wave; ++lane)
          if ((exec & (uint64_t{1} << lane)) != 0u) test.expected[lane] = expected;
        test.opcodes = {reduction.vector_opcode, reduction.scalar_opcode,
                        ShaderOpcode::V_READLANE_B32, ShaderOpcode::BUFFER_STORE_DWORD};
        test.required_spirv = {reduction.spirv, "GroupNonUniformArithmetic"};
        auto compiled = CompileCase(test, vulkan.SubgroupSize());
        size_t replaced = 0u;
        for (auto* block : compiled.program.blocks) {
          for (auto it = block->begin(); it != block->end(); ++it) {
            if (it->GetOpcode() != ValueOpcode::SubgroupReduceU32) continue;
            Require(test.name, "generic reduction control",
                    it->Flags<ValueOpcode>() == reduction.operation,
                    "translated scan selected the wrong reduction operation");
            auto& legacy = *block->PrependNewInst(
                it, restored.GetOpcode(), {it->Arg(0u)}, restored.Flags<uint64_t>());
            it->ReplaceUsesWith(Value(&legacy));
            ++replaced;
          }
        }
        Require(test.name, "legacy operation replacement", replaced == 1u,
                "real guest reduction did not produce exactly one replacement site");
        EliminateDeadCode(compiled.program.blocks);
        auto compute = test.compute_info;
        compute.host_subgroup_size = vulkan.SubgroupSize();
        compiled.program.shader_info_complete = false;
        CollectShaderInfo(compiled.program, {.compute = &compute});
        Require(test.name, "legacy subgroup capability", compiled.program.info.subgroup_arithmetic,
                "legacy opcode did not restore subgroup arithmetic capability");
        compiled.spirv = ShaderRecompiler::Spirv::EmitProgram(
            compiled.program, {.compute = &compute}, 0u, *test.profile, true);
        ValidateSpirv(test.name, compiled.spirv);
        CheckSpirvEmission(test, compiled.spirv);
        spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
        std::string disassembly;
        Require(test.name, "one whole-wave native reduction",
                tools.Disassemble(compiled.spirv, &disassembly) &&
                    CountText(disassembly, reduction.spirv) == 1u,
                "legacy reduction was lost or emitted independently for both emulated halves");
        Require(test.name, "stable binding layout",
                compiled.packed_user_data.size() == compiled.program.bindings.ShaderDataDwords(),
                "legacy replacement changed the prepared shader-data layout");
        auto output = vulkan.CreateStorageBuffer(test.name, test.initial, test.initial.size());
        vulkan.Dispatch(test, compiled, output);
        const auto actual = vulkan.ReadBuffer(test.name, output, test.expected.size());
        vulkan.DestroyBuffer(&output);
        CompareWords(test, "historical reduction across the full guest wave and inactive-lane canaries",
                     test.expected, actual);
        std::printf("[compute] %-32s ok\n", test.name);
      }
    }
  }
}

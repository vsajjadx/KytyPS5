// The complete historical wire schema is independent of the current opcode list.
// The fixed IDs are consumed by serialized ResourcePlan values.
void TestLegacyValueOpcodeSchema() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct WireOpcode {
    uint16_t id;
    std::string_view name;
    Type result;
    size_t arity;
    std::array<Type, 16> arguments;
  };
  static constexpr std::array<WireOpcode, 344> schema{{
#include "LegacyValueOpcodeWireSchema.inc"
  }};
  Check(static_cast<size_t>(ValueOpcode::Count) >= schema.size(),
        "historical opcode schema was truncated");
  for (size_t index = 0; index < schema.size(); ++index) {
    const auto& row = schema[index];
    Check(row.id == index, "historical opcode fixture has a gap or duplicate ID");
    const auto opcode = static_cast<ValueOpcode>(row.id);
    const auto actual_name = ValueOpcodeName(opcode);
    if (actual_name != row.name) {
      std::fprintf(stderr,
                   "LegacyValueOpcodeSchema: wire=%u expected=%.*s actual=%.*s\n",
                   unsigned(row.id), int(row.name.size()), row.name.data(),
                   int(actual_name.size()), actual_name.data());
    }
    Check(actual_name == row.name, "historical opcode name changed at its persisted ID");
    Check(TypeOf(opcode) == row.result && NumArgsOf(opcode) == row.arity,
          "historical opcode result or arity changed");
    for (size_t arg = 0; arg < row.arity; ++arg)
      Check(ArgTypeOf(opcode, arg) == row.arguments[arg],
            "historical opcode argument type changed");
    Writer writer;
    Io(writer, opcode);
    const std::vector<uint8_t> expected{
        static_cast<uint8_t>(row.id), static_cast<uint8_t>(row.id >> 8u)};
    Check(writer.bytes == expected, "historical opcode header wire bytes changed");
  }
  struct AppendedOpcode { ValueOpcode opcode; uint16_t wire; };
  constexpr std::array appended{
      AppendedOpcode{ValueOpcode::SubgroupReduceU32, 344u},
      AppendedOpcode{ValueOpcode::RayQueryValid, 345u},
      AppendedOpcode{ValueOpcode::RayQueryInitialize, 346u},
      AppendedOpcode{ValueOpcode::RayQueryProceed, 347u},
      AppendedOpcode{ValueOpcode::RayQueryCandidate, 348u},
      AppendedOpcode{ValueOpcode::RayQueryCandidateInstance, 349u},
      AppendedOpcode{ValueOpcode::RayQueryConfirm, 350u}};
  for (const auto& row : appended) {
    Check(static_cast<uint16_t>(row.opcode) == row.wire,
          "appended opcode changed its persisted ID");
    Writer writer;
    Io(writer, row.opcode);
    const std::vector<uint8_t> expected{
        static_cast<uint8_t>(row.wire), static_cast<uint8_t>(row.wire >> 8u)};
    Check(writer.bytes == expected, "appended opcode header wire bytes changed");
  }
  std::puts("LegacyValueOpcodeSchema: 344 historical signatures and seven appended IDs passed");
}

void TestLegacyReductionWirePlans() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct Reduction {
    uint16_t wire;
    std::string_view name;
  };
  constexpr std::array reductions{
      Reduction{212u, "SubgroupUMinU32"}, Reduction{213u, "SubgroupUMaxU32"},
      Reduction{214u, "SubgroupOrU32"}};
  for (const auto& reduction : reductions) {
    const auto opcode = static_cast<ValueOpcode>(reduction.wire);
    Check(ValueOpcodeName(opcode) == reduction.name && TypeOf(opcode) == Type::U32 &&
              NumArgsOf(opcode) == 1u && ArgTypeOf(opcode, 0u) == Type::U32,
          "historical reduction ID lost its typed signature");
    for (const uint64_t flags : {uint64_t{0}, uint64_t{0x0123456789abcdef}}) {
      ResourcePlan source;
      auto& inst = source.value_storage.emplace_back(opcode, flags);
      inst.SetArg(0u, Value(uint32_t{0x81234567u}));
      auto bytes = SaveResourcePlan(source);
      Check(bytes.size() >= 14u, "legacy reduction instruction header missing");
      bytes[4] = static_cast<uint8_t>(reduction.wire);
      bytes[5] = static_cast<uint8_t>(reduction.wire >> 8u);
      ResourcePlan loaded;
      Check(LoadResourcePlan(bytes, loaded) && loaded.value_storage.size() == 1u,
            "historical reduction wire plan failed typed loading");
      const auto& restored = loaded.value_storage.front();
      Check(restored.GetOpcode() == opcode && restored.Flags<uint64_t>() == flags &&
                restored.Arg(0u).U32() == 0x81234567u &&
                SaveResourcePlan(loaded) == bytes,
            "historical reduction opcode/flags/operand bytes changed on roundtrip");
    }
  }
  std::puts("LegacyReductionWirePlans: three historical opcodes and flag roundtrips passed");
}

void TestLegacyValueOpcodePlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  TestLegacyValueOpcodeSchema();
  TestLegacyReductionWirePlans();
  // Raw enum values are persisted wire IDs. Refuse a known schema mismatch before
  // feeding those bytes into the typed IR reader.
  Check(ValueOpcodeName(static_cast<ValueOpcode>(341u)) == "SelectU32",
        "persisted opcode341 no longer names SelectU32");
  Check(static_cast<uint16_t>(ValueOpcode::DataCacheInvalidate) == 336u &&
            static_cast<uint16_t>(ValueOpcode::Barrier) == 337u &&
            static_cast<uint16_t>(ValueOpcode::Sendmsg) == 338u &&
            static_cast<uint16_t>(ValueOpcode::TtraceData) == 339u &&
            static_cast<uint16_t>(ValueOpcode::InstPrefetch) == 340u,
        "legacy trailing opcode wire IDs changed");
  ResourcePlan source;
  auto& select = source.value_storage.emplace_back(ValueOpcode::SelectU32, 0u);
  select.SetArg(0, Value(true));
  select.SetArg(1, Value(uint32_t{0x11223344u}));
  select.SetArg(2, Value(uint32_t{0x55667788u}));
  auto legacy = SaveResourcePlan(source);
  Check(legacy.size() >= 14u, "legacy plan instruction header missing");
  // Preserve the old instruction ID explicitly in the current plan envelope.
  // Whole-cache BuildStamp invalidation remains a separate compatibility gate.
  legacy[4] = 0x55u; legacy[5] = 0x01u; // 341, little endian as the Windows store.
  ResourcePlan loaded;
  Check(LoadResourcePlan(legacy, loaded) && loaded.value_storage.size() == 1u,
        "old SelectU32 wire plan failed typed loading");
  const auto& restored = loaded.value_storage.front();
  Check(restored.GetOpcode() == ValueOpcode::SelectU32 && restored.Arg(0).U1() &&
            restored.Arg(1).U32() == 0x11223344u && restored.Arg(2).U32() == 0x55667788u &&
            SaveResourcePlan(loaded) == legacy,
        "old SelectU32 wire plan changed opcode/arguments/bytes on roundtrip");
  std::puts("LegacyValueOpcodePlan passed");
}

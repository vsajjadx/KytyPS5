// Paste in ResourceMaterializationTests.cpp anonymous namespace; main can invoke
// TestLegacyValueOpcodePlan() through --legacy-opcodes-only or its ordinary suite.
void TestLegacyValueOpcodePlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // Raw enum values are the persisted wire IDs, independent of host-only call modules.
  // Refuse a known schema mismatch BEFORE feeding those bytes into the IR reader.
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
  // IoValues: uint32 count, then {uint16 opcode,uint64 flags}; encode the OLD
  // SelectU32 ID explicitly rather than deriving our test bytes from the new enum.
  legacy[4] = 0x55u; legacy[5] = 0x01u; // 341, little endian as existing Windows store
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

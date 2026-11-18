void TestScratchAddressSpaceValidation() {
  const auto emit_store = [](Fixture& f, Value handle, ResourceKind kind,
                             uint32_t bits = 32u, uint32_t offset = 0u) {
    const auto op = bits == 8u ? ValueOpcode::StoreAddressU8 :
                    bits == 16u ? ValueOpcode::StoreAddressU16 : ValueOpcode::StoreAddressU32;
    const auto data = bits == 8u ? Value(uint8_t{0x5a}) :
                      bits == 16u ? Value(uint16_t{0x5aa5}) : Value(0x5aa55aa5u);
    f.Emit(op, {handle, Value(offset), Value(0u), data, Value(true)},
           f.AddMemory({.kind = kind, .data_bits = bits}, 4u));
  };
  for (const uint32_t bits : {8u, 16u, 32u}) {
    for (const uint32_t offset : {0u, 252u, 256u, 0xfffffffcu}) {
      Fixture f;
      f.program.scratch_dwords = 64u;
      emit_store(f, f.Emit(ValueOpcode::GetScratchResource), ResourceKind::Scratch, bits, offset);
      f.PlanAndTrack();
      Check(f.program.resource_tracking_complete && !f.program.has_address_writes &&
                f.program.info.buffers.empty() && f.program.descriptor_sources.empty(),
            "typed scratch was mistaken for a guest-address write");
      Check(f.program.memory_info[0].kind == ResourceKind::Scratch,
            "scratch tracking changed the isolated address space");
    }
  }
  {
    Fixture f;
    f.program.scratch_dwords = 64u;
    emit_store(f, f.Address(Value(0u), Value(0u)), ResourceKind::Scratch);
    CheckFatal([&] { f.PlanAndTrack(); }, "scratch operation requires GetScratchResource",
               "scratch accepted a physical-address handle");
    Check(!f.program.resource_tracking_complete && f.program.info.buffers.empty() &&
              f.program.descriptor_sources.empty(),
          "malformed scratch handle published partial resource state");
  }
  {
    Fixture f;
    emit_store(f, f.Emit(ValueOpcode::GetScratchResource), ResourceKind::Scratch);
    CheckFatal([&] { f.PlanAndTrack(); },
               "scratch operation requires a nonzero per-thread scratch size",
               "scratch without an allocation was accepted");
    Check(!f.program.resource_tracking_complete && f.program.info.buffers.empty() &&
              f.program.descriptor_sources.empty(),
          "missing scratch allocation published partial resource state");
  }
  {
    Fixture f;
    // The typed IR constructor fixes GetScratchResource's arity at zero. A
    // malformed operand is rejected at that boundary, before tracking can trust it.
    CheckFatal([&] {
      f.block->AppendNewInst(ValueOpcode::GetScratchResource, {Value(0u)});
    }, "typed IR assertion", "scratch accepted a nonzero-arity handle");
  }
  for (const auto kind : {ResourceKind::Flat, ResourceKind::Global}) {
    Fixture f;
    f.program.scratch_dwords = 64u;
    emit_store(f, f.Address(Value(0u), Value(0u)), kind);
    f.PlanAndTrack();
    Check(f.program.resource_tracking_complete && f.program.has_address_writes &&
              f.program.memory_info[0].kind == kind,
          "nonlocal address writes inherited the scratch exemption");
  }
}

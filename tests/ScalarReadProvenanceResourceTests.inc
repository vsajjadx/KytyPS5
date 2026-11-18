// A native V# bank identifies where a descriptor came from, not the address read
// after all four explicit descriptor words have been resolved.
Value ScalarBankRead(Fixture& f, const std::array<Value, 4>& table,
                     MemoryInfo memory, uint32_t pc, uint32_t offset = 0u) {
  return f.Emit(ValueOpcode::ReadConstBuffer,
                {f.Buffer(table), Value(offset)}, f.AddMemory(memory, pc));
}

void UseScalarDescriptorWord(Fixture& f, Value word, uint32_t pc) {
  const auto descriptor = f.Buffer({word, Value(0u), Value(64u),
      Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
            (static_cast<uint32_t>(Libs::Graphics::Prospero::BufferFormat::k32UInt) << 12u) |
            (2u << 28u))});
  f.Emit(ValueOpcode::LoadBufferU32,
         {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
         f.AddMemory({.kind = ResourceKind::Buffer}, pc));
}

void TestScalarReadBankInterning() {
  Fixture f;
  const std::array<Value, 4> table{f.UserData(0), f.UserData(1),
                                   f.UserData(2), f.UserData(3)};
  for (uint32_t bank : {0u, 9u}) {
    const auto read = ScalarBankRead(f, table,
        {.kind = ResourceKind::ScalarBuffer, .resource = bank, .offset = 4u},
        4u + bank * 4u);
    UseScalarDescriptorWord(f, read, 64u + bank * 4u);
  }
  f.PlanAndTrack();
  Check(f.program.srt_reads.size() == 1u && f.program.info.buffers.size() == 1u,
        "native V# bank provenance duplicated an identical scalar read/buffer");
  Check(f.program.memory_info[0].resource == 0u &&
            f.program.memory_info[2].resource == 9u,
        "scalar interning overwrote native provenance metadata");
  auto plan = ExtractResourcePlan(f.program);
  TestMemory memory;
  memory.words[1] = 0x4000u;
  const std::array<uint32_t, 4> user{0x1000u, 0u, 32u, 0u};
  SrtRuntime runtime{.user_data = user};
  FakeGuestMemory fake(ReadTestMemory, &memory);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (uint32_t base : {0x4000u, 0x5000u, 0x4000u}) {
    memory.words[1] = base;
    memory.reads = 0u;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers.size() == 1u &&
              snapshot.buffers[0].dwords[0] == base &&
              snapshot.flattened_srt == std::vector<uint32_t>{base} &&
              memory.reads == 1u,
          "shared scalar read lost current backing bytes or read memoization");
  }
}

void TestScalarReadBankManyAliases() {
  Fixture f;
  const std::array<Value, 4> table{f.UserData(0), f.UserData(1),
                                   f.UserData(2), f.UserData(3)};
  // 24 architectural banks and three byte offsets used to create 72 roots.
  // The existing 64-root limit stays intact; these are only three resources.
  for (uint32_t offset = 0u; offset < 3u; ++offset) {
    for (uint32_t bank = 0u; bank < 24u; ++bank) {
      const uint32_t pc = 4u + (offset * 24u + bank) * 8u;
      const auto read = ScalarBankRead(f, table,
          {.kind = ResourceKind::ScalarBuffer, .resource = bank,
           .offset = offset * 4u}, pc);
      UseScalarDescriptorWord(f, read, pc + 4u);
    }
  }
  f.PlanAndTrack();
  Check(f.program.srt_reads.size() == 3u && f.program.info.buffers.size() == 3u,
        "repeated native banks exhausted the resource limit instead of sharing reads");
  auto plan = ExtractResourcePlan(f.program);
  TestMemory memory;
  memory.words = {0x4000u, 0x5000u, 0x6000u};
  const std::array<uint32_t, 4> user{0x1000u, 0u, 32u, 0u};
  FakeGuestMemory fake(ReadTestMemory, &memory);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {.user_data = user}, snapshot, specialization) &&
            snapshot.buffers.size() == 3u && memory.reads == 3u,
        "many scalar aliases did not materialize all distinct resource values");
  for (uint32_t i = 0u; i < 3u; ++i)
    Check(snapshot.buffers[i].dwords[0] == memory.words[i],
          "scalar alias interning conflated distinct byte offsets");
}

void TestScalarReadBankNestedDescriptors() {
  Fixture f;
  const std::array<Value, 4> root{f.UserData(0), f.UserData(1),
                                  f.UserData(2), f.UserData(3)};
  for (uint32_t bank : {2u, 14u}) {
    std::array<Value, 4> table;
    for (uint32_t word = 0u; word < 4u; ++word)
      table[word] = ScalarBankRead(f, root,
          {.kind = ResourceKind::ScalarBuffer, .resource = bank,
           .offset = word * 4u, .component_index = word, .component_count = 4u},
          4u + bank * 16u);
    const auto payload = ScalarBankRead(f, table,
        {.kind = ResourceKind::ScalarBuffer, .resource = bank + 1u},
        8u + bank * 16u);
    UseScalarDescriptorWord(f, payload, 12u + bank * 16u);
  }
  f.PlanAndTrack();
  Check(f.program.srt_reads.size() == 5u && f.program.info.buffers.size() == 1u,
        "nested copied descriptors did not share bottom-up scalar reads");
  LinearTestMemory memory;
  memory.words[0] = 0x1100u;
  memory.words[2] = 16u;
  memory.words[0x100u / 4u] = 0x5000u;
  const std::array<uint32_t, 4> user{0x1000u, 0u, 16u, 0u};
  auto plan = ExtractResourcePlan(f.program);
  FakeGuestMemory fake(ReadLinearTestMemory, &memory);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {.user_data = user}, snapshot, specialization) &&
            snapshot.buffers.size() == 1u && snapshot.buffers[0].dwords[0] == 0x5000u &&
            memory.reads == 5u,
        "shared nested descriptor did not retain exact native read semantics");
}

void TestScalarReadBankSemanticSeparation() {
  // These fields/operands can affect native bounds or the meaning of a read.
  // Only the architectural source bank is allowed to differ in the new equality.
  for (uint32_t change = 0u; change < 10u; ++change) {
    Fixture f;
    const std::array<Value, 4> original{f.UserData(0), f.UserData(1),
                                       f.UserData(2), f.UserData(3)};
    for (uint32_t side = 0u; side < 2u; ++side) {
      auto table = original;
      MemoryInfo scalar{.kind = ResourceKind::ScalarBuffer, .resource = side * 7u,
                        .component_count = 4u};
      uint32_t offset = 0u;
      if (side) {
        if (change < 4u)
          table[change] = f.Emit(ValueOpcode::IAdd32,
              {table[change], Value(change == 1u ? 4u << 16u :
                                     change == 3u ? 1u << 28u : 4u)});
        if (change == 4u) scalar.offset = 4u;
        if (change == 5u) offset = 4u;
        if (change == 6u) scalar.component_index = 1u;
        if (change == 7u) scalar.component_count = 8u;
        if (change == 8u) scalar.coherent = true;
        if (change == 9u) table[0] = f.UserData(4u);
      }
      const auto read = ScalarBankRead(f, table, scalar, 4u + side * 8u, offset);
      UseScalarDescriptorWord(f, read, 8u + side * 8u);
    }
    f.PlanAndTrack();
    Check(f.program.srt_reads.size() == 2u && f.program.info.buffers.size() == 2u,
          "scalar bank interning erased descriptor/read semantics");
  }
}

void TestScalarReadBankCoincidentPayloads() {
  Fixture f;
  for (uint32_t side = 0u; side < 2u; ++side) {
    std::array<Value, 4> table;
    for (uint32_t word = 0u; word < 4u; ++word)
      table[word] = f.UserData(side * 4u + word);
    const auto read = ScalarBankRead(f, table,
        {.kind = ResourceKind::ScalarBuffer, .resource = side * 6u},
        4u + side * 8u);
    UseScalarDescriptorWord(f, read, 8u + side * 8u);
  }
  f.PlanAndTrack();
  Check(f.program.srt_reads.size() == 2u && f.program.info.buffers.size() == 2u,
        "independent descriptor expressions were merged before materialization");
  auto plan = ExtractResourcePlan(f.program);
  TestMemory memory;
  memory.words[0] = 0x4000u;
  memory.words[4] = 0x5000u;
  std::array<uint32_t, 8> user{0x1000u, 0u, 32u, 0u, 0x1000u, 0u, 32u, 0u};
  FakeGuestMemory fake(ReadTestMemory, &memory);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (uint32_t second : {0x1000u, 0x1010u, 0x1000u}) {
    user[4] = second;
    Check(MaterializeResources(plan, {.user_data = user}, snapshot, specialization) &&
              snapshot.buffers.size() == 2u &&
              snapshot.buffers[0].dwords[0] == 0x4000u &&
              snapshot.buffers[1].dwords[0] == (second == 0x1000u ? 0x4000u : 0x5000u),
          "equal current payloads conflated descriptors that later diverged");
  }
}

void TestScalarReadBankRuntimeOnly() {
  Fixture f;
  const std::array<Value, 4> table{f.UserData(0), f.UserData(1),
                                   f.UserData(2), f.UserData(3)};
  const auto planned = ScalarBankRead(f, table,
      {.kind = ResourceKind::ScalarBuffer, .resource = 2u}, 4u);
  UseScalarDescriptorWord(f, planned, 8u);
  const auto native = ScalarBankRead(f, table,
      {.kind = ResourceKind::ScalarBuffer, .resource = 3u, .gpu_runtime_only = true}, 12u);
  const auto native_index = native.Instruction()->Flags<MemoryFlags>().index;
  f.Emit(ValueOpcode::ReferenceU32, {native});
  f.PlanAndTrack();
  Check(f.program.srt_reads.size() == 1u &&
            native.ResolveInstruction()->GetOpcode() == ValueOpcode::ReadConstBuffer &&
            f.program.memory_info[native_index].gpu_runtime_only &&
            !f.program.memory_info[native_index].planning_only,
        "runtime-only scalar read was merged into a host-planned read");
}

void TestScalarReadBankProvenance() {
  TestScalarReadBankInterning();
  TestScalarReadBankManyAliases();
  TestScalarReadBankNestedDescriptors();
  TestScalarReadBankSemanticSeparation();
  TestScalarReadBankCoincidentPayloads();
  TestScalarReadBankRuntimeOnly();
}

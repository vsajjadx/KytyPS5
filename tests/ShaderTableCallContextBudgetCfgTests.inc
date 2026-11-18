namespace table_context_budget {
struct Memory {
  table_call_context::Memory ordinary;
  uint64_t base = table_call_context::table + 0x1000000u, bytes = 0;
  uint64_t attempts = 0, pointer_dwords = 0, companions = 0;
  bool fail_batches = false, fail_companions = false, late_code_read = false, partial_tail = false;
};
inline bool Read(void* cookie, uint64_t address, std::span<uint32_t> words) {
  auto& m = *static_cast<Memory*>(cookie);
  namespace t = table_call_context;
  if (address >= m.base && address - m.base < m.bytes) {
    m.attempts += words.size_bytes();
    if (words.size_bytes() > m.bytes - (address - m.base)) return false;
    const bool companion = (address - m.base) % 16u == 8u && words.size() == 2u;
    if (companion) { ++m.companions; if (m.fail_companions) return false; }
    if (m.fail_batches && words.size() > 2u) return false;
    if (words.size() == 1u && (address - m.base) % 16u < 8u) ++m.pointer_dwords;
    for (size_t i = 0; i < words.size(); ++i) {
      const auto offset = address - m.base + i * 4u;
      const auto row = offset / 16u, word = offset % 16u / 4u;
      const bool live = row == 0u || row + 1u == m.bytes / 16u;
      const uint64_t value = !live ? 0u : word < 2u ? t::leaf_a : row == 0u ? t::context_a : t::context_a2;
      words[i] = static_cast<uint32_t>(value >> ((word & 1u) * 32u));
      if (m.partial_tail && offset + 4u == m.bytes) words[i] = 0xcafebabeu;
    }
    return true;
  }
  if (address >= t::leaf_a && address < t::leaf_b && m.companions != 0u)
    m.late_code_read = true;
  return t::Read(&m.ordinary, address, words);
}
} // namespace table_context_budget

void TestTableContextBudgets() {
  using namespace ShaderRecompiler;
  namespace t = table_call_context;
  table_context_budget::Memory memory;
  FakeGuestMemory fake(table_context_budget::Read, &memory);
  t::Place(memory.ordinary, t::leaf_a, t::Leaf());
  t::ImageTable(memory.ordinary, t::context_a, t::images_a, 0x40u);
  t::ImageTable(memory.ordinary, t::context_a2, t::images_a, 0x40u);
  std::array<uint32_t, 64> user_data {};
  const uint64_t output = t::table + 0x4000u;
  user_data[0] = static_cast<uint32_t>(output); user_data[1] = output >> 32u;
  user_data[2] = 256u; user_data[3] = 0x00014facu;
  user_data[8] = static_cast<uint32_t>(memory.base); user_data[9] = memory.base >> 32u;
  user_data[11] = 3u << 28u;
  auto options = MakeCompileOptions(ShaderType::Compute);
  ShaderComputeInputInfo compute = *options.input_info.compute;
  compute.thread_ids_num = 1u;
  compute.threads_num[0] = 64u; compute.threads_num[1] = compute.threads_num[2] = 1u;
  options.input_info.compute = &compute;
  options.wave_size = 64u; options.user_data = user_data;
  IR::SrtRuntime runtime {.user_data = user_data, .userdata = &memory,
                          .read_specialization_memory = table_context_budget::Read};
  memory.bytes = user_data[10] = 8192u * 16u;
  memory.fail_batches = true;
  auto source = PrepareShaderSource(t::Caller(), options);
  std::string reason;
  const auto* linked = TryRefreshShaderSource(source, runtime, reason);
  Check(linked != nullptr && reason.empty(), "optional companion probes consumed mandatory fallback credits");
  Check(memory.pointer_dwords == 16384u && memory.companions != 0u && !memory.late_code_read,
        "mandatory fallback/code reads did not finish before optional probes");
  Check(source.linked->context_loads.size() == 1u && !source.call_data.empty(),
        "exactly exhausted fallback credits incorrectly disabled an otherwise bounded payload proof");
  auto translated = TranslateProgram(*linked, options, true);
  translated.program.source_reads = source.reads;
  const auto plan = IR::ExtractResourcePlan(translated.program);
  IR::ResourceSnapshot snapshot;
  IR::ResourceSpecialization specialization;
  Check(IR::MaterializeResources(plan, runtime, snapshot, specialization),
        "large fallback table context images did not materialize");

  memory.bytes = user_data[10] = 8u << 20u;
  memory.fail_batches = false;
  memory.attempts = memory.pointer_dwords = memory.companions = 0u;
  memory.late_code_read = false;
  linked = TryRefreshShaderSource(source, runtime, reason);
  Check(linked != nullptr && reason.empty(),
        "optional payload probes changed a valid attempt-bound table link into a refusal");
  Check(memory.attempts == (8u << 20u) && memory.companions != 0u && !memory.late_code_read,
        "attempt-bound context probes exceeded the limit or preceded a required code read");
  Check(source.linked->context_loads.empty() && source.call_data.empty(),
        "attempt-bound table retained an unproved context projection");

  memory.bytes = user_data[10] = 8192u * 16u;
  memory.fail_batches = memory.fail_companions = true;
  memory.attempts = memory.pointer_dwords = memory.companions = 0u;
  memory.late_code_read = false;
  linked = TryRefreshShaderSource(source, runtime, reason);
  Check(linked != nullptr && reason.empty(),
        "unreadable optional companion data rejected a valid native call table");
  Check(memory.pointer_dwords == 16384u && memory.companions != 0u && !memory.late_code_read,
        "unreadable companion handling changed mandatory fallback/code read order");
  Check(source.linked->context_loads.empty() && source.call_data.empty(),
        "unreadable optional companions retained a context projection");
  memory.bytes = user_data[10] = 60u; // 3 complete rows and a 12-byte null row
  memory.fail_batches = memory.fail_companions = false;
  memory.partial_tail = true;
  memory.attempts = memory.pointer_dwords = memory.companions = 0u;
  memory.late_code_read = false;
  linked = TryRefreshShaderSource(source, runtime, reason);
  Check(linked != nullptr && reason.empty(), "partial null record changed native linking");
  Check(!source.linked->context_loads.empty() && !source.call_data.empty(),
        "readable partial null record unnecessarily lost the bounded context proof");
  const uint64_t tail = memory.base + memory.bytes - 4u;
  Check(std::ranges::any_of(source.call_data, [&](const auto& range) {
    return tail >= range.address && (tail - range.address) / 4u < range.words.size() &&
           range.words[(tail - range.address) / 4u] == 0xcafebabeu;
  }), "partial null record captured an unread zero instead of its actual companion DWORD");
  Check(!memory.late_code_read, "partial companion probe preceded a required code read");
  std::printf("[host] TableContextBudgets ok (4 compatibility/bound scenarios)\n");
} // TEST_CONTEXT_BUDGET_COMPLETE

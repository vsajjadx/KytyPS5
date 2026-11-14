// TABLE_CALL_LARGE_ROUNDS_BEGIN
// Included after snippets/R14-prepared-storage.inc in CheckTableSelectedCall(). Synthetic
// 8193-row {pointer, payload} table at base + 0x18000 (the fixture maps 0x40000 bytes).
{
  constexpr uint64_t large_table = base + 0x18000;
  constexpr u32 large_rows = 8193;
  static_assert(large_table + uint64_t{large_rows} * 16u <= base + size);
  constexpr uint64_t final_row = large_table + uint64_t{large_rows - 1u} * 16u;
  set_mutation(scratch, 0);
  bind_code(large_code, 23);
  const auto set_large = [&](u32 shift, u32 bytes = large_rows * 16u) {
    set_vsharp(8, raw_descriptor(large_table, bytes));
    shaders.SetCsUserSgpr(22, shift);
  };
  std::vector<uint64_t> rows_data(uint64_t{large_rows} * 2u); // {pointer, payload} per row
  const auto publish = [&] {
    Require(name, "large table refresh", context.InvalidateMemory(large_table, rows_data.size() * 8u),
            "large call table escaped mapped memory");
    LibKernel::Memory::WriteBacking(large_table, rows_data.data(), rows_data.size() * 8u);
  };
  const auto standard = [&] {
    for (u32 row = 0; row < large_rows; ++row) {
      rows_data[row * 2u] = row % 64u == 0u && row + 1u < large_rows ? leaf_a : 0u; // repeated A
      rows_data[row * 2u + 1u] = 0xd00d000000000000ull | row;                       // nonzero payload
    }
    rows_data[(large_rows - 1u) * 2u] = leaf_b; // the only B: final row
    publish();
  };
  standard();
  // LR1 odd lanes select the FINAL row (8192): its callee must be linked, not sampled away.
  set_large(13);
  const auto large_id = run("large table final row", true, value_a, value_b, {});
  // LR2 first row of the second 64 KiB batch (4096) and first row past the former cap (256).
  set_large(12);
  (void)run("large table batch-boundary row", true, value_a, value_a, {});
  rows_data[4096u * 2u] = leaf_c;
  publish();
  Require(name, "large rekey", run("large table late callee", true, value_a, value_c, {}) != large_id,
          "a changed large table reused the old link");
  rows_data[4096u * 2u] = leaf_a;
  publish();
  set_large(8);
  (void)run("large table row 256", true, value_a, value_a, {});
  // LR3 GPU writes the final row twice before enumeration: ordered reads see the LAST write.
  set_large(13);
  {
    auto [buffer, offset] = cache.ObtainBuffer(final_row, 8, true);
    buffer->Fill(offset, 4, static_cast<u32>(leaf_b));
    buffer->Fill(offset + 4, 4, static_cast<u32>(leaf_b >> 32u));
    buffer->Fill(offset, 4, static_cast<u32>(leaf_c));
    buffer->Fill(offset + 4, 4, static_cast<u32>(leaf_c >> 32u));
  }
  (void)run("large gpu-written final row", true, value_a, value_c, {});
  Require(name, "large ordered read", !cache.HasGpuDirtyBytes(final_row, 8),
          "the large table was not read in order");
  standard();
  // LR4 the dispatch writes an UNLINKED pointer (C) into the final row before its loads: the
  // observable miss records exactly C (row 8191 receives its existing null).
  set_mutation(final_row - 16u, leaf_c, 0);
  (void)run("large same-dispatch unknown pointer", true, value_a, 0, {leaf_c});
  set_mutation(scratch, 0);
  standard();
  // LR5 a sentinel-sized WRITTEN binding starting in row 0's payload gap is bounded by the next
  // pointer dependency (row 1), never rejected: dependencies cover pointer DWORDs only. Its stores
  // land in row 0/1 payloads only.
  set_vsharp(12, raw_descriptor(large_table + 8u, UINT32_MAX));
  (void)run("large payload-gap sentinel writer", true, value_a, value_b, {});
  set_mutation(scratch, 0);
  standard();
  // LR6 33 distinct callees (A, 31 unknown, B) fail closed; restoration recovers the same module.
  for (u32 k = 0; k < 31u; ++k) rows_data[(4080u + k) * 2u] = 0x0000dead00000000ull + uint64_t{k} * 0x100u;
  publish();
  (void)run("large table 33 callees fails", false, sentinel, sentinel, {});
  standard();
  Require(name, "large recovery", run("large table recovers", true, value_a, value_b, {}) == large_id,
          "the restored large table did not bind its module");
  // LR7 a physical alias of a table page fails explicitly; unmapping recovers.
  {
    void *large_alias = reinterpret_cast<void *>(alias_va);
    Require(name, "large alias map", LibKernel::Memory::KernelMapDirectMemory(
        &large_alias, 0x4000, 0x3, 0x10, physical + 0x1c000, 0x4000) == 0, "large table alias mapping failed");
    (void)run("aliased large table fails", false, sentinel, sentinel, {});
    Require(name, "large alias unmap", LibKernel::Memory::KernelMunmap(alias_va, 0x4000) == 0,
            "large table alias unmap failed");
  }
  Require(name, "alias recovery", run("large table after alias", true, value_a, value_b, {}) == large_id,
          "the large table did not recover after its alias was removed");
  // LR8 9,600,000-byte descriptor exceeds the 8 MiB scan budget: refused before any table read
  // (the range is not even mapped), then recovery.
  set_large(13, 600000u * 16u);
  (void)run("over-budget large table fails", false, sentinel, sentinel, {});
  set_large(13);
  Require(name, "budget recovery", run("large table after budget", true, value_a, value_b, {}) == large_id,
          "the large table did not recover after a budget refusal");
  // Restore the small fixture.
  bind_code(code);
  set_table_records(2);
  write_entry(0, leaf_a);
  write_entry(1, leaf_b);
  Require(name, "small after large", run("small table after large", true, value_a, value_b, {}) == id_ab,
          "the small table did not rebind its module after the large rounds");
}
// TABLE_CALL_LARGE_ROUNDS_END

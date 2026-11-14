// TABLE_CALL_LARGE_CFG_BEGIN
// Included at the end of the existing failure cases in TestTableSelectedLeafCall()
// (tests/ShaderTableCallCfgTests.inc). Synthetic tables only.
// Tables above 256 potential entries: every row is enumerated through bounded batched transport;
// only pointer DWORDs become dependencies; budgets refuse before any table read.
{
  constexpr uint64_t large = base + 0x1000;
  constexpr uint64_t large_end = base + memory_bytes;
  using Reads = std::vector<std::pair<uint64_t, uint64_t>>;
  const auto set_table = [&](uint64_t table, uint64_t bytes, uint32_t raw_stride = 0) {
    user_data[0] = static_cast<uint32_t>(table);
    user_data[1] = static_cast<uint32_t>(table >> 32u) | (raw_stride << 16u);
    user_data[2] = static_cast<uint32_t>(bytes);
  };
  const auto restore_small = [&] { set_table(base, 48); };
  // Zeroes every pointer of `rows` 16-byte rows and fills their 8-byte payloads with nonzero data.
  const auto reset_rows = [&](uint32_t rows) {
    for (uint32_t row = 0; row < rows + 1u; ++row) {
      const uint64_t address = large + uint64_t{row} * 16u;
      if (address + 16u > large_end) break;
      at(address) = 0; at(address + 4) = 0;
      at(address + 8) = 0xd00d0000u | row; at(address + 12) = ~row;
    }
  };
  const auto row_ptr = [&](uint32_t row, uint64_t target) {
    at(large + uint64_t{row} * 16u) = static_cast<uint32_t>(target);
    at(large + uint64_t{row} * 16u + 4) = static_cast<uint32_t>(target >> 32u);
  };
  const auto chain_of = [&](const Decoder::Program &program) {
    std::vector<uint64_t> chain;
    for (const auto *select : selects(program))
      chain.push_back(uint64_t{select->raw[1]} | (uint64_t{select->raw[2]} << 32u));
    return chain;
  };
  const auto link = [&](ShaderSource &target_source, const char *text) {
    std::string reason;
    const auto *linked = TryRefreshShaderSource(target_source, runtime, reason);
    Check(linked != nullptr && reason.empty(), text);
    if (linked == nullptr) std::printf("  failure: %s\n", reason.c_str());
    return linked != nullptr ? chain_of(*linked) : std::vector<uint64_t>{};
  };
  const auto refuse = [&](ShaderSource &target_source, const char *expected, const char *pc,
                          const char *text) {
    std::string reason;
    Check(TryRefreshShaderSource(target_source, runtime, reason) == nullptr &&
              reason.find(expected) != std::string::npos && reason.find(pc) != std::string::npos,
          text);
    if (reason.find(expected) == std::string::npos) std::printf("  failure: %s\n", reason.c_str());
  };
  const auto table_reads = [&](const ShaderSource &target_source) {
    Reads result;
    for (const auto &read : target_source.reads)
      if (read.first >= large && read.first < large_end) result.push_back(read);
    return result;
  };
  // No dependency may cover a payload byte of a 16-byte row (FindBuffers would then clamp or
  // reject a sentinel writer that starts in that payload gap).
  const auto pointer_only = [&](const Reads &reads) {
    return std::ranges::all_of(reads, [](const auto &read) {
      const uint64_t offset = (read.first - large) % 16u;
      return read.second != 0 && offset + read.second <= 8u;
    });
  };
  const auto has = [](const std::string &text, const char *needle) {
    return text.find(needle) != std::string::npos;
  };

  // Give L1 a different valid cached membership. Moving to a larger table with the SAME
  // {A,B} set correctly reuses its link; {A,C} -> {A,B} must genuinely relink below.
  entry(1, leaf_c);
  Check(link(source, "large-table precondition did not link") == std::vector<uint64_t>{leaf_a, leaf_c},
        "large-table precondition did not establish distinct cached callees");
  entry(1, leaf_b);

  // L1: 8193 rows (0..8192), batches of 4096 rows. Row 0 = A, every 64th row repeats A, the
  // only B is the FINAL row. Nothing past row 256 may be sampled away.
  constexpr uint32_t rows = 8193;
  reset_rows(rows);
  row_ptr(0, leaf_a | 3u);
  for (uint32_t row = 64; row < rows - 1u; row += 64) row_ptr(row, leaf_a);
  row_ptr(rows - 1u, leaf_b);
  set_table(large, uint64_t{rows} * 16u);
  memory.largest = 0;
  const auto revision = source.revision;
  Check(link(source, "large table did not link") == std::vector<uint64_t>{leaf_a, leaf_b},
        "large table lost its final-row callee");
  Check(source.revision == revision + 1, "large table did not relink");
  {
    const auto reads = table_reads(source);
    Check(reads.size() == rows && reads.front() == std::pair<uint64_t, uint64_t>{large, 8} &&
              reads.back() == std::pair<uint64_t, uint64_t>{large + uint64_t{rows - 1u} * 16u, 8} &&
              pointer_only(reads),
          "large table dependencies are not exactly its pointer DWORDs");
    Check(memory.largest == 4095u * 16u + 8u, "large table transport was not batched in 64 KiB windows");
    Check(has(source.call_diagnostic, "entries=8193") && has(source.call_diagnostic, "chunks=3") &&
              has(source.call_diagnostic, "fallback_dwords=0") &&
              has(source.call_diagnostic, "dependency_ranges=8193"),
          "large table diagnostic is incomplete");
  }
  // Unchanged: reused, same dependencies.
  Check(link(source, "unchanged large table") == std::vector<uint64_t>{leaf_a, leaf_b} &&
            source.revision == revision + 1,
        "unchanged large table relinked");
  // L2: late-row change (A->B->C->B) relinks; a callee only at the batch-boundary row 4096 counts.
  row_ptr(rows - 1u, leaf_c);
  Check(link(source, "late change") == std::vector<uint64_t>{leaf_a, leaf_c}, "late row change missed");
  row_ptr(rows - 1u, 0);
  row_ptr(4096, leaf_b);
  Check(link(source, "boundary row") == std::vector<uint64_t>{leaf_a, leaf_b}, "batch-boundary row missed");
  row_ptr(4096, 0);
  row_ptr(256, leaf_b); // first row beyond the former cap
  Check(link(source, "row 256") == std::vector<uint64_t>{leaf_a, leaf_b}, "row 256 missed");
  row_ptr(256, 0);
  row_ptr(rows - 1u, leaf_b);
  Check(link(source, "restored") == std::vector<uint64_t>{leaf_a, leaf_b}, "restored table did not link");
  // L2b: same membership, changed callee BYTES (code A->B->A) relinks and carries fresh code.
  {
    const auto before = source.revision;
    const uint32_t original = at(leaf_b + 4);
    std::string reason;
    leaf(leaf_b, 0x66666666u);
    Check(TryRefreshShaderSource(source, runtime, reason) != nullptr && source.revision == before + 1 &&
              std::ranges::find(source.linked->code, 0x66666666u) != source.linked->code.end(),
          "large table kept stale callee code");
    leaf(leaf_b, original);
    Check(TryRefreshShaderSource(source, runtime, reason) != nullptr && source.revision == before + 2 &&
              std::ranges::find(source.linked->code, original) != source.linked->code.end() &&
              std::ranges::find(source.linked->code, 0x66666666u) == source.linked->code.end(),
          "restored callee code reused the changed link");
  }

  // L3: exactly 32 distinct callees across the batch boundary link; a 33rd (final row) fails.
  std::vector<uint64_t> extras;
  for (uint32_t k = 0; k < 31u; ++k) {
    const uint64_t address = base + 0x400 + uint64_t{k} * 16u;
    leaf(address, 0x55000000u + k);
    extras.push_back(address);
    row_ptr(4081u + k, address);      // rows 4081..4111 straddle row 4096
    row_ptr(6000u + k, address);      // repeated
  }
  row_ptr(rows - 1u, 0);
  {
    std::vector<uint64_t> expected{leaf_a};
    expected.insert(expected.end(), extras.begin(), extras.end());
    Check(link(source, "32 callees") == expected, "exactly 32 distinct callees did not link in row order");
  }
  row_ptr(rows - 1u, leaf_b);
  refuse(source, "more than 32 callees", "0x0000001c", "33 distinct callees were accepted");
  for (uint32_t k = 0; k < 31u; ++k) { row_ptr(4081u + k, 0); row_ptr(6000u + k, 0); }
  Check(link(source, "after 33") == std::vector<uint64_t>{leaf_a, leaf_b}, "33-callee failure poisoned recovery");

  // L4: a hole over a POINTER DWORD (row 5000 hi) fails explicitly after exact fallback.
  memory.holes = {{large + 5000u * 16u + 4u, large + 5000u * 16u + 8u}};
  refuse(source, "is unreadable", "0x0000001c", "unreadable pointer row was skipped");
  // L5: a hole over PAYLOAD bytes only (rows 100 and 5000): failed batches fall back to exact
  // pointer reads; the result and dependencies are unchanged. 2 * 8192 DWORDs = the whole budget.
  memory.holes = {{large + 100u * 16u + 8u, large + 100u * 16u + 16u},
                  {large + 5000u * 16u + 8u, large + 5000u * 16u + 16u}};
  Check(link(source, "payload holes") == std::vector<uint64_t>{leaf_a, leaf_b}, "payload-only hole rejected readable pointers");
  {
    const auto reads = table_reads(source);
    Check(reads.size() == rows && pointer_only(reads) && has(source.call_diagnostic, "fallback_dwords=16384"),
          "fallback changed the dependency journal or its count");
  }
  // One more failed batch (the final row) exceeds the fallback budget: explicit refusal.
  memory.holes.push_back({large + uint64_t{rows - 1u} * 16u, large + uint64_t{rows - 1u} * 16u + 4u});
  refuse(source, "fallback budget", "0x0000001c", "fallback budget was not enforced");
  memory.holes.clear();
  Check(link(source, "holes cleared") == std::vector<uint64_t>{leaf_a, leaf_b}, "hole recovery failed");

  // L6: scan-byte budget boundary. 524288 rows span 8388600 <= 8 MiB: accepted. One more row
  // spans 8388616: refused BEFORE any read attempt.
  reset_rows(524288u);
  row_ptr(0, leaf_a);
  row_ptr(524287u, leaf_b);
  set_table(large, 524288ull * 16u);
  Check(link(source, "8 MiB boundary") == std::vector<uint64_t>{leaf_a, leaf_b} &&
            has(source.call_diagnostic, "scan_bytes=8388600"),
        "largest admitted table did not link");
  set_table(large, 524289ull * 16u);
  {
    const auto attempts = memory.attempts;
    refuse(source, "scan bytes", "0x0000001c", "over-budget table was accepted");
    Check(memory.attempts == attempts, "over-budget table read memory before refusing");
  }
  // L6b: the observed geometry: raw V# stride 16 x 360000 records = 5,760,000 bytes, the only B
  // in row 359999. 88 transport windows; 360000 pointer-only dependency ranges (journal cost).
  reset_rows(360000u);
  row_ptr(0, leaf_a);
  row_ptr(359999u, leaf_b);
  set_table(large, 360000u, 16u);
  memory.largest = 0;
  Check(link(source, "observed geometry") == std::vector<uint64_t>{leaf_a, leaf_b} &&
            has(source.call_diagnostic, "bytes=5760000 ") && has(source.call_diagnostic, "entries=360000 ") &&
            has(source.call_diagnostic, "scan_bytes=5759992 ") && has(source.call_diagnostic, "chunks=88 ") &&
            has(source.call_diagnostic, "dependency_ranges=360000") && memory.largest == 4095u * 16u + 8u,
        "the observed 5,760,000-byte geometry did not enumerate every row in bounded windows");
  {
    const auto reads = table_reads(source);
    Check(reads.size() == 360000u && pointer_only(reads) &&
              reads.back() == std::pair<uint64_t, uint64_t>{large + 359999ull * 16u, 8},
          "observed geometry dependencies are not exactly its pointer DWORDs");
  }
  // L7: maximal descriptor (raw stride 16383, records 0xffffffff): reach clamps to 2^32;
  // refused by checked arithmetic and the budget, no allocation or read.
  set_table(large, 0xffffffffull, 16383u);
  {
    const auto attempts = memory.attempts;
    refuse(source, "scan bytes", "0x0000001c", "maximal descriptor was accepted");
    Check(memory.attempts == attempts, "maximal descriptor read memory before refusing");
  }
  reset_rows(524288u);

  for (uint32_t word = 0; word < 302u * 4u; ++word) at(large + uint64_t{word} * 4u) = 0; // no payload data
  // L8: nonzero addend/word offset and a split final pointer. (X << 4) + 4 with SMEM offset 8:
  // pointers at 12 + 16n, so each hi word is the next row's first word.
  uint32_t offset_shader[std::size(shader) + 1];
  std::copy(std::begin(shader), std::begin(shader) + 2, offset_shader);
  offset_shader[2] = EncodeSop2(0x00, 4, 4, 132); // s_add_u32 s4, s4, 4
  offset_shader[3] = shader[2];
  offset_shader[4] = (4u << 25u) | 8u;            // soffset s4, offset +8
  std::copy(std::begin(shader) + 4, std::end(shader), offset_shader + 5);
  auto offset_source = PrepareShaderSource(offset_shader, options);
  Check(offset_source.unsupported_call_pc == UINT32_MAX && offset_source.calls.size() == 1 &&
            offset_source.calls[0].table_stride == 16 && offset_source.calls[0].table_addend == 4 &&
            offset_source.calls[0].table_word_offset == 8,
        "offset table call was not recognized");
  const auto offset_ptr = [&](uint32_t row, uint64_t target) {
    at(large + 12u + uint64_t{row} * 16u) = static_cast<uint32_t>(target);
    at(large + 16u + uint64_t{row} * 16u) = static_cast<uint32_t>(target >> 32u);
  };
  offset_ptr(0, leaf_a);
  offset_ptr(299, leaf_b);       // last of 300 entries: lo at 4796, hi at 4800
  set_table(large, 4804);        // hi in range
  Check(link(offset_source, "offset table") == std::vector<uint64_t>{leaf_a, leaf_b}, "offset table lost a callee");
  {
    const auto reads = table_reads(offset_source);
    Check(reads.size() == 300 && reads.front() == std::pair<uint64_t, uint64_t>{large + 12, 8} &&
              reads.back() == std::pair<uint64_t, uint64_t>{large + 4796, 8},
          "offset table dependencies are not its pointer DWORDs");
  }
  set_table(large, 4803);        // misaligned size: the hi DWORD needs 4804 bytes and reads zero
  {
    std::string reason;
    char lo_only[32];
    std::snprintf(lo_only, sizeof(lo_only), "callee 0x%x", static_cast<uint32_t>(leaf_b));
    Check(TryRefreshShaderSource(offset_source, runtime, reason) == nullptr && has(reason, lo_only) &&
              has(reason, "0x00000020"),
          "a split final pointer did not read its out-of-range hi DWORD as zero");
    const auto reads = table_reads(offset_source);
    Check(!reads.empty() && reads.back() == std::pair<uint64_t, uint64_t>{large + 4796, 4} &&
              std::ranges::all_of(reads, [&](const auto &read) { return read.first + read.second <= large + 4800; }),
          "an out-of-range DWORD was read or journaled");
  }
  offset_ptr(299, 0);
  offset_ptr(0, 0);
  at(large + 16u + 299u * 16u) = 0;

  // L9: stride 8 is dense: adjacent pointers merge into ONE dependency range, never wider.
  uint32_t dense_shader[std::size(shader)];
  std::copy(std::begin(shader), std::end(shader), dense_shader);
  dense_shader[1] = EncodeSop2(0x1e, 4, 4, 131);  // << 3
  auto dense_source = PrepareShaderSource(dense_shader, options);
  Check(dense_source.unsupported_call_pc == UINT32_MAX && dense_source.calls.size() == 1 &&
            dense_source.calls[0].table_stride == 8,
        "dense table call was not recognized");
  for (uint32_t row = 0; row < 600; ++row) {
    at(large + row * 8u) = 0;
    at(large + row * 8u + 4u) = 0;
  }
  at(large) = static_cast<uint32_t>(leaf_a);
  at(large + 4) = static_cast<uint32_t>(leaf_a >> 32u);
  at(large + 599u * 8u) = static_cast<uint32_t>(leaf_b);
  at(large + 599u * 8u + 4u) = static_cast<uint32_t>(leaf_b >> 32u);
  at(large + 600u * 8u) = 0xffffffffu;           // first byte past the table: never read
  set_table(large, 600u * 8u);
  Check(link(dense_source, "dense table") == std::vector<uint64_t>{leaf_a, leaf_b} &&
            table_reads(dense_source) == Reads{{large, 4800}},
        "dense table dependencies were split or widened");
  at(large + 600u * 8u) = 0;
  reset_rows(600u);

  // L10: aggregate budget across two sites in one refresh (index in s21; the leaves write s20).
  const uint32_t two_sites[] = {
      EncodeVop1(0x02, 21, 256),            // 0x00 v_readfirstlane_b32 s21, v0
      EncodeSop2(0x1e, 21, 21, 132),        // 0x04 s_lshl_b32 s21, s21, 4
      EncodeSmem0(0x0a, 8, 0), 21u << 25u,  // 0x08 s_buffer_load_dwordx4 s[8:11], s[0:3], soffset s21
      EncodeSopp(0x0c, 0),                  // 0x10
      EncodeSMovB32(14, 8),                 // 0x14
      EncodeSMovB32(15, 9),                 // 0x18
      EncodeSop1(0x21, 14, 14),             // 0x1c s_swappc_b64 (site 1)
      EncodeSmem0(0x0a, 8, 2), 21u << 25u,  // 0x20 s_buffer_load_dwordx4 s[8:11], s[4:7], soffset s21
      EncodeSopp(0x0c, 0),                  // 0x28
      EncodeSMovB32(14, 8),                 // 0x2c
      EncodeSMovB32(15, 9),                 // 0x30
      EncodeSop1(0x21, 14, 14),             // 0x34 s_swappc_b64 (site 2)
      EncodeSopp(0x01)};                    // 0x38 s_endpgm
  auto two_source = PrepareShaderSource(two_sites, options);
  Check(two_source.unsupported_call_pc == UINT32_MAX && two_source.calls.size() == 2 &&
            two_source.calls[0].kind == ShaderSource::Call::Kind::Table &&
            two_source.calls[1].kind == ShaderSource::Call::Kind::Table,
        "two-site table caller was not recognized");
  reset_rows(rows);
  row_ptr(0, leaf_a);
  row_ptr(rows - 1u, leaf_b);
  set_table(large, uint64_t{rows} * 16u);
  user_data[4] = user_data[0]; user_data[5] = user_data[1]; user_data[6] = user_data[2]; user_data[7] = user_data[3];
  Check(link(two_source, "two large sites") == std::vector<uint64_t>{leaf_a, leaf_b, leaf_a, leaf_b},
        "two large sites did not both enumerate");
  // Site 1 alone fits (8388600), site 2 needs 4792 more than the remaining 8 bytes.
  reset_rows(524288u);
  row_ptr(0, leaf_a);
  row_ptr(524287u, leaf_b);
  set_table(large, 524288ull * 16u);
  user_data[6] = 300u * 16u;
  {
    std::string reason;
    Check(TryRefreshShaderSource(two_source, runtime, reason) == nullptr &&
              has(reason, "remaining budget 8 of 8388608") && has(reason, "0x00000034"),
          "aggregate scan budget was not enforced across sites");
  }
  // ---- rev3: strict table ATTEMPT budget and large-refresh code budgets ----
  const auto set_site = [&](uint32_t first_sgpr, uint64_t table, uint64_t bytes) {
    user_data[first_sgpr] = static_cast<uint32_t>(table);
    user_data[first_sgpr + 1u] = static_cast<uint32_t>(table >> 32u);
    user_data[first_sgpr + 2u] = static_cast<uint32_t>(bytes);
    user_data[first_sgpr + 3u] = 3u << 28u;
  };
  // R3a: the 8 MiB-span table (524288 rows, 128 windows of 65528 bytes) with a payload-only hole.
  reset_rows(524288u);
  row_ptr(0, leaf_a);
  row_ptr(524287u, leaf_b);
  set_table(large, 524288ull * 16u);
  Check(link(source, "dense 8 MiB positive") == std::vector<uint64_t>{leaf_a, leaf_b} &&
            has(source.call_diagnostic, "table_attempt_bytes=8387584"),
        "the hole-free 8 MiB-span table no longer links with its exact attempt total");
  {
    // Early failed window: 127 windows (incl. the failed one) + 8192 fallback DWORDs are requested;
    // the 128th window would exceed 8 MiB and is never attempted.
    memory.holes = {{large + 100u * 16u + 8u, large + 100u * 16u + 16u}};
    const auto requested = memory.table_requested;
    refuse(source, "(8354824 bytes requested before this 65528-byte read)", "0x0000001c",
           "a failed window and its fallback escaped the strict attempt budget");
    Check(memory.table_requested - requested == 8354824u, "table requests exceeded or misreported the budget");
    // Late failed window: its fallback DWORDs stop exactly at the remaining 1024 bytes.
    memory.holes = {{large + 524000u * 16u + 8u, large + 524000u * 16u + 16u}};
    const auto late = memory.table_requested;
    refuse(source, "(8388608 bytes requested before this 4-byte read)", "0x0000001c",
           "fallback requests did not stop at the remaining attempt budget");
    Check(memory.table_requested - late == 8388608u, "fallback requests exceeded 8 MiB");
    memory.holes.clear();
    Check(link(source, "attempt recovery") == std::vector<uint64_t>{leaf_a, leaf_b}, "attempt refusal poisoned recovery");
  }
  // R3b: mixed sites in ONE refresh (two_sites: site 1 = s[0:3] at 0x1c, site 2 = s[4:7] at 0x34).
  // small -> large: a 3-entry table (6 DWORDs = 24 bytes, 6 ranges) first. The large site may then
  // use at most 8388584 span bytes and 524282 dependency ranges: 524282 rows fit exactly
  // (span 8388504, 128 windows = 8387488 requested bytes).
  reset_rows(524288u);
  row_ptr(0, leaf_a);
  row_ptr(524281u, leaf_b);
  set_site(0, base, 48);
  set_site(4, large, 524282ull * 16u);
  Check(link(two_source, "small then large") == std::vector<uint64_t>{leaf_a, leaf_b, leaf_a, leaf_b} &&
            has(two_source.call_diagnostic, "table_attempt_bytes=8387512"),
        "small->large refresh within all budgets did not link");
  set_site(4, large, 524283ull * 16u); // one range too many after the small site's 6
  {
    std::string reason;
    Check(TryRefreshShaderSource(two_source, runtime, reason) == nullptr &&
              has(reason, "needs 524283 dependency ranges (remaining budget 524282 of 524288)") &&
              has(reason, "0x00000034"),
          "earlier small dependency records were not charged against the large site");
  }
  set_site(4, large, 524288ull * 16u); // span 8388600 > 8388584 left after the small site's 24
  {
    const auto requested = memory.table_requested;
    std::string reason;
    Check(TryRefreshShaderSource(two_source, runtime, reason) == nullptr &&
              has(reason, "remaining budget 8388584 of 8388608") && has(reason, "0x00000034"),
          "earlier small work was not charged against the large site");
    Check(memory.table_requested == requested, "a refused large site requested table bytes");
  }
  // large -> small: 524287-row table (8387568 requested) then a 256-entry table (512 DWORDs):
  // the 261st DWORD would pass 8 MiB.
  row_ptr(524281u, 0);
  row_ptr(524286u, leaf_b);
  set_site(0, large, 524287ull * 16u);
  set_site(4, large, 256u * 16u);
  {
    std::string reason;
    Check(TryRefreshShaderSource(two_source, runtime, reason) == nullptr &&
              has(reason, "(8388608 bytes requested before this 4-byte read)") && has(reason, "0x00000034"),
          "later small work escaped the strict attempt budget");
  }
  set_site(4, base, 48);
  Check(link(two_source, "large then small") == std::vector<uint64_t>{leaf_a, leaf_b, leaf_a, leaf_b} &&
            has(two_source.call_diagnostic, "table_attempt_bytes=8387592"),
        "large->small refresh within the budget did not link");
  // The same 256-entry second site without any large site keeps the legacy (unbounded) policy.
  set_site(0, large, 256u * 16u);
  set_site(4, large, 256u * 16u);
  Check(link(two_source, "pure small") == std::vector<uint64_t>{leaf_a, leaf_a} &&
            !has(two_source.call_diagnostic, "table_attempt_bytes"),
        "a pure-small refresh changed behavior");
  reset_rows(524288u);

  // Long leaves: 59999 s_nop + s_setpc = 240000 bytes each, 256 KiB apart from base + 0x802000.
  constexpr uint64_t leaf_region = base + 0x802000;
  const auto long_leaf = [&](uint32_t k) {
    const uint64_t address = leaf_region + uint64_t{k} * 0x40000u;
    for (uint32_t w = 0; w + 1u < 60000u; ++w) at(address + uint64_t{w} * 4u) = EncodeSopp(0x00);
    at(address + 59999ull * 4u) = EncodeSop1(0x21, 125, 14); // s_setpc_b64 s[14:15]
    return address;
  };
  // R3c: 40 sites share ONE long leaf. Unique decode is 240000 bytes; each site copies it.
  std::vector<uint32_t> many;
  for (uint32_t k = 0; k < 40u; ++k)
    many.insert(many.end(), {EncodeVop1(0x02, 4, 256), EncodeSop2(0x1e, 4, 4, 132), EncodeSmem0(0x0a, 8, 0),
                             4u << 25u, EncodeSopp(0x0c, 0), EncodeSMovB32(14, 8), EncodeSMovB32(15, 9),
                             EncodeSop1(0x21, 14, 14)}); // site k: s_swappc at 32k + 0x1c
  many.push_back(EncodeSopp(0x01));
  const uint64_t shared = long_leaf(0);
  reset_rows(257u);
  row_ptr(0, shared);
  {
    auto many_source = PrepareShaderSource(many, options);
    Check(many_source.unsupported_call_pc == UINT32_MAX && many_source.calls.size() == 40,
          "40-site caller was not recognized");
    set_table(large, 257u * 16u); // every site large: linked = 1284 + 40 * 240040 > 8 MiB
    std::string reason;
    Check(TryRefreshShaderSource(many_source, runtime, reason) == nullptr &&
              has(reason, "pc 0x0000045c: linked code needs 240040 more bytes") &&
              has(reason, "unique decoded callee bytes 240000") &&
              many_source.revision == 0 && !many_source.linked,
          "duplicated leaf copies were not refused before layout");
    set_table(large, 256u * 16u); // all <= 256: legacy behavior, 9602884 linked bytes
    Check(TryRefreshShaderSource(many_source, runtime, reason) != nullptr && reason.empty() &&
              many_source.revision == 1 && many_source.linked->code.size() * 4u == 9602884u,
          "the paired all-small 40-site program changed behavior");
    // R4: the cached >8 MiB pure-small link must not be reused by a LARGE refresh with the same
    // membership and callee bytes; it falls through to the bounded rebuild and refuses.
    set_table(large, 257u * 16u);
    Check(TryRefreshShaderSource(many_source, runtime, reason) == nullptr &&
              has(reason, "pc 0x0000045c: linked code needs 240040 more bytes") &&
              many_source.revision == 1 && many_source.linked &&
              many_source.linked->code.size() * 4u == 9602884u,
          "a large refresh reused an over-budget pure-small link");
    // Back to all <= 256: the original cached link is reused unchanged.
    set_table(large, 256u * 16u);
    Check(TryRefreshShaderSource(many_source, runtime, reason) != nullptr && reason.empty() &&
              many_source.revision == 1 && many_source.linked->code.size() * 4u == 9602884u,
          "the pure-small refresh lost its cached link after a large refusal");
    // Large again with the SAME target, its code shortened to 1000 words: the oversized cache
    // does not poison the valid input; it relinks within budget (1284 + 40 * (28 + 12 + 4000)).
    at(shared + 999u * 4u) = EncodeSop1(0x21, 125, 14); // s_setpc_b64 s[14:15]
    set_table(large, 257u * 16u);
    Check(TryRefreshShaderSource(many_source, runtime, reason) != nullptr && reason.empty() &&
              many_source.revision == 2 && many_source.linked->code.size() * 4u == 162884u,
          "an oversized cached link poisoned a later valid large refresh");
    at(shared + 999u * 4u) = EncodeSopp(0x00);
  }
  // R3d: two large sites with 18 DISTINCT long leaves each (36 unique, <= 32 per site):
  // unique decode passes 8 MiB at the 17th leaf of site 2.
  {
    reset_rows(257u);
    constexpr uint64_t second = large + 0x10000;
    for (uint32_t row = 0; row < 258u; ++row) {
      at(second + row * 16u) = 0; at(second + row * 16u + 4u) = 0;
    }
    std::vector<uint64_t> first_leaves, second_leaves;
    for (uint32_t k = 0; k < 36u; ++k) {
      const uint64_t address = long_leaf(k);
      const uint64_t row_base = (k < 18u ? large : second) + uint64_t{k % 18u} * 16u;
      at(row_base) = static_cast<uint32_t>(address);
      at(row_base + 4u) = static_cast<uint32_t>(address >> 32u);
      (k < 18u ? first_leaves : second_leaves).push_back(address);
    }
    set_site(0, large, 257u * 16u);
    set_site(4, second, 257u * 16u);
    const auto revision_before = two_source.revision;
    std::string reason;
    Check(TryRefreshShaderSource(two_source, runtime, reason) == nullptr &&
              has(reason, "decoded callee code budget") && has(reason, "0x00000034") &&
              two_source.revision == revision_before,
          "distinct callees across sites escaped the unique decode budget");
    // Two fewer callees at site 2 (34 unique, 8160000 decoded bytes) recover.
    for (uint32_t row = 16; row < 18u; ++row) {
      at(second + row * 16u) = 0; at(second + row * 16u + 4u) = 0;
    }
    second_leaves.resize(16);
    std::vector<uint64_t> expected(first_leaves);
    expected.insert(expected.end(), second_leaves.begin(), second_leaves.end());
    Check(link(two_source, "decode recovery") == expected && two_source.revision == revision_before + 1,
          "valid distinct callees did not recover after the decode refusal");
    for (uint32_t row = 0; row < 18u; ++row) {
      at(second + row * 16u) = 0; at(second + row * 16u + 4u) = 0;
    }
  }
  reset_rows(524288u);
  user_data[4] = user_data[5] = user_data[6] = user_data[7] = 0;
  reset_rows(524288u);
  restore_small();
  std::string restored;
  Check(TryRefreshShaderSource(source, runtime, restored) != nullptr && restored.empty(),
        "small table did not recover after the large-table cases");
}
// TABLE_CALL_LARGE_CFG_END

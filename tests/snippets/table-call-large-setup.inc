// TABLE_CALL_LARGE_SETUP_BEGIN
// Included after snippets/table-call-flatlocal-setup.inc in CheckTableSelectedCall().
// Same waterfall caller, but the per-lane index is (lane & 1) << s22: odd lanes select row
// 1 << s22 of a large table (row 256, the batch-boundary row 4096, or the final row 8192).
std::vector<u32> large_code(code);
large_code.insert(large_code.begin() + 2, EncodeVop2(0x1a, 1, 22, 1)); // v_lshlrev_b32 v1, s22, v1
ShaderMapUserData(reinterpret_cast<uint64_t>(large_code.data()),
    {.type = Prospero::ShaderBinaryType::kCs,
     .code_size_bytes = static_cast<u32>(large_code.size() * sizeof(u32))});
// TABLE_CALL_LARGE_SETUP_END

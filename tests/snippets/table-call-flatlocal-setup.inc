// CALLABLE_FLATLOCAL_SETUP_BEGIN
// Insert AFTER AppendEnd(&flat_code), before the existing code/flat_code map loop.
// The final three words are the original output MUBUF plus S_ENDPGM.
Require(name, "local caller source shape", code.size() >= 3u,
        "table-call fixture has no final output store");
std::vector<u32> local_code(code.begin(), code.end() - 3);
local_code.push_back(EncodeSop1(0x0b, 26, InlineU32(1))); // upper aperture80000000
local_code.push_back(EncodeVop2(0x1a, 20, InlineU32(2), 0)); // local byteoffset=lane*4
// Lane ID is runtime but always0..63 here: MED3(80000000,70000000,lane)
// still selects private70000000 for EVERY lane. Keep the same two-constant
// proof shape as the real shader; three immediates are not folded by current IR.
local_code.push_back(EncodeVop1(0x01, 22, Vgpr(0)));
AppendVop3(&local_code, 0x159, 21, 26, 255, Vgpr(22));
local_code.push_back(0x70000000u); // MED3 clamps high to private70000000
AppendVMovLiteral(&local_code, 30, 0x1000u);
local_code.push_back(EncodeVop2(0x25, 30, Vgpr(6), 30)); // actual callee result+1000
local_code.insert(local_code.end(), {
    EncodeFlat0(0x1c, 0), EncodeFlat1(0, 125, 30, 20)}); // initialize private scratch
AppendVMovLiteral(&local_code, 30, 0x2000u);
local_code.push_back(EncodeVop2(0x25, 30, Vgpr(6), 30));
local_code.insert(local_code.end(), {EncodeDs0(0x0d), EncodeDs1(0, 30, 20)}); // initialize LDS
local_code.push_back(EncodeVop2(0x1b, 22, InlineU32(1), 0));
local_code.push_back(EncodeVop2(0x1a, 22, InlineU32(31), 22));
AppendVop3(&local_code, 0x159, 21, 26, 255, Vgpr(22));
local_code.push_back(0x70000000u); // even lanes private, odd shared
AppendVMovLiteral(&local_code, 30, 0x3000u);
local_code.push_back(EncodeVop2(0x25, 30, Vgpr(6), 30));
local_code.insert(local_code.end(), {EncodeFlat0(0x1c, 0), EncodeFlat1(0, 125, 30, 20),
    EncodeFlat0(0x0c, 0), EncodeFlat1(34, 125, 0, 20)});
// Only even lanes overwrite their selected local word. Odd LDS contents and v34 stay intact.
AppendSMovLiteral(&local_code, 126, 0x55555555u);
AppendSMovLiteral(&local_code, 127, 0x55555555u);
AppendVMovLiteral(&local_code, 30, 0x4000u);
local_code.push_back(EncodeVop2(0x25, 30, Vgpr(6), 30));
local_code.insert(local_code.end(), {EncodeFlat0(0x1c, 0), EncodeFlat1(0, 125, 30, 20),
    EncodeFlat0(0x0c, 0), EncodeFlat1(34, 125, 0, 20)});
local_code.push_back(EncodeSop1(0x04, 126, 20)); // original saved EXEC (s20:21)
local_code.insert(local_code.end(), {
    EncodeFlat0(0x0c, 1), EncodeFlat1(32, 125, 0, 20), // explicit private read
    EncodeDs0(0x36), EncodeDs1(33, 0, 20)});           // explicit LDS read
// Generic AppendStoreVgprAtLaneDwordOffset targets SGPR48:51. This native
// fixture binds its output V# at SGPR0:3, matching the original caller store.
for (const auto [value, base_word] : {std::pair{34u, 0u}, std::pair{32u, lanes},
                                    std::pair{33u, 2u * lanes}, std::pair{6u, 3u * lanes}}) {
  AppendVMovU32(&local_code, 31, base_word);
  local_code.push_back(EncodeVop2(0x25, 31, Vgpr(0), 31));
  local_code.push_back(EncodeVop2(0x1a, 31, InlineU32(2), 31));
  local_code.insert(local_code.end(), {EncodeMubuf0(0x1c), EncodeMubuf1(value, 0, 31)});
}
AppendEnd(&local_code);
// Local stores do not justify even one extra GLOBAL store.
std::vector<u32> mixed_code(local_code.begin(), local_code.end() - 1);
mixed_code.insert(mixed_code.end(), {
    EncodeFlat0(0x1c, 2), EncodeFlat1(0, 0, 6, 2)}); // GLOBAL through output base s0:1
AppendEnd(&mixed_code);
// Unknown high word is deliberately not clamped. Its actual address remains safe
// in this fixture; the policy must reject it before the output can be touched.
std::vector<u32> unknown_code(local_code.begin(), local_code.end() - 1);
AppendVMovLiteral(&unknown_code, 28, static_cast<u32>(output));
unknown_code.push_back(EncodeVop2(0x25, 28, Vgpr(2), 28));
unknown_code.push_back(EncodeVop2(0x16, 29, InlineU32(6), 0)); // lane>>6, runtime value
AppendVMovLiteral(&unknown_code, 30, static_cast<u32>(output >> 32u));
unknown_code.push_back(EncodeVop2(0x25, 29, Vgpr(30), 29));
unknown_code.insert(unknown_code.end(), {EncodeFlat0(0x1c, 0), EncodeFlat1(0, 125, 6, 28)});
AppendEnd(&unknown_code);
// A clamp executed only for even lanes cannot justify an all-lane store later.
std::vector<u32> divergent_code(local_code.begin(), local_code.end() - 1);
AppendVMovLiteral(&divergent_code, 28, static_cast<u32>(output));
divergent_code.push_back(EncodeVop2(0x25, 28, Vgpr(2), 28));
AppendVMovLiteral(&divergent_code, 29, static_cast<u32>(output >> 32u));
AppendSMovLiteral(&divergent_code, 126, 0x55555555u);
AppendSMovLiteral(&divergent_code, 127, 0x55555555u);
AppendVop3(&divergent_code, 0x159, 29, 26, 255, Vgpr(22));
divergent_code.push_back(0x70000000u);
divergent_code.push_back(EncodeSop1(0x04, 126, 20));
divergent_code.insert(divergent_code.end(), {EncodeFlat0(0x1c, 0), EncodeFlat1(0, 125, 6, 28)});
AppendEnd(&divergent_code);
std::vector<u32> local_callee_code(code); // same caller, only leaf will contain local stores
for (const auto *extra : {&local_code, &mixed_code, &unknown_code, &divergent_code, &local_callee_code})
  ShaderMapUserData(reinterpret_cast<uint64_t>(extra->data()),
      {.type = Prospero::ShaderBinaryType::kCs,
       .code_size_bytes = static_cast<u32>(extra->size() * sizeof(u32)),
       .scratch_size_dwords = lanes});
// CALLABLE_FLATLOCAL_SETUP_END

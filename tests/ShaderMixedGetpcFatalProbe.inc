// Dedicated child process only: a TABLE site populates the address-keyed decoded
// leaf cache before a Uniform site requests the SAME callee address.
void ProbeMixedTableUniformGetpcFatal() {
  using namespace ShaderRecompiler;
  constexpr uint64_t table = 0xfe00d0000ull, address = table + 0x1000;
  std::unordered_map<uint64_t, uint32_t> memory {
      {table, static_cast<uint32_t>(address)}, {table + 4, static_cast<uint32_t>(address >> 32u)},
      {table + 16, static_cast<uint32_t>(address)}, {table + 20, static_cast<uint32_t>(address >> 32u)},
      {address, EncodeSop1(0x1f, 20, 0)}, {address + 4, EncodeSop1(0x21, 125, 14)}};
  const auto read = +[](void* data, uint64_t at, std::span<uint32_t> words) {
    const auto& m = *static_cast<std::unordered_map<uint64_t, uint32_t>*>(data);
    for (size_t i = 0; i < words.size(); ++i) {
      const auto found = m.find(at + i * 4u);
      if (found == m.end()) return false;
      words[i] = found->second;
    }
    return true;
  };
  std::array<uint32_t, 64> user_data {};
  user_data[0] = static_cast<uint32_t>(address); user_data[1] = address >> 32u;
  user_data[8] = static_cast<uint32_t>(table); user_data[9] = table >> 32u;
  user_data[10] = 32; user_data[11] = 0x00014facu;
  const uint32_t caller[] = {EncodeVop1(0x02, 4, 256), EncodeSop2(0x1e, 4, 4, 132),
      EncodeSmem0(0x09, 14, 4), 4u << 25u, EncodeSopp(0x0c, 0), EncodeSop1(0x21, 14, 14),
      EncodeSMovB32(14, 0), EncodeSMovB32(15, 1), EncodeSop1(0x21, 14, 14), EncodeSopp(0x01)};
  auto options = MakeCompileOptions(ShaderType::Compute); options.user_data = user_data;
  auto source = PrepareShaderSource(caller, options);
  Check(source.calls.size() == 2 && source.calls[0].kind == ShaderSource::Call::Kind::Table &&
            source.calls[1].kind == ShaderSource::Call::Kind::Uniform,
        "mixed fatal probe lost its site policies");
  IR::SrtRuntime runtime {.user_data = user_data, .userdata = &memory, .read_specialization_memory = read};
  std::fprintf(stderr, "MixedTableUniformGetpcFatal: invoking SAME address after TABLE decode\n");
  std::fflush(stderr);
  std::string reason;
  (void)TryRefreshShaderSource(source, runtime, reason);
  Check(false, "TABLE address-cache bypass permitted a uniform GETPC");
}

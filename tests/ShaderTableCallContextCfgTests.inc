// Synthetic GCN regression: a GPU-selected callable record passes a companion
// context pointer to a leaf which uses it to locate a GPU-selected texture table.
// All guest addresses and descriptor contents below are created by this fixture.
namespace table_call_context {
constexpr uint64_t table = 0xfe0200000ull;
constexpr uint64_t leaf_a = table + 0x1000, leaf_b = table + 0x1100;
constexpr uint64_t context_a = table + 0x2000, context_b = table + 0x2100;
constexpr uint64_t context_a2 = table + 0x2200;
constexpr uint64_t images_a = table + 0x3000, images_b = table + 0x3100;

using Memory = table_tracking_failure::Memory;
using table_tracking_failure::Read;
using table_tracking_failure::Place;

inline std::vector<uint32_t> Caller() {
  return {
      EncodeVop2(0x1b, 1, 129, 0),                    // v_and_b32 v1, 1, v0
      EncodeVop1(0x02, 4, 257),                       // s4 = readfirstlane(v1)
      EncodeSop2(0x1e, 5, 4, 132),                   // row * 16
      EncodeSmem0(0x0a, 24, 4), 5u << 25u,           // s[24:27] = {fnptr, context}
      EncodeSopp(0x0c, 0),
      EncodeSMovB32(14, 24), EncodeSMovB32(15, 25),
      EncodeSMovB32(16, 26), EncodeSMovB32(17, 27),
      EncodeSop1(0x21, 14, 14),                      // call s[14:15]
      EncodeMubuf0(0x1c), EncodeMubuf1(6, 0, 0),      // keep image result live
      EncodeSopp(0x01)};
}

inline std::vector<uint32_t> Leaf() {
  return {
      EncodeSmem0(0x02, 24, 8), (125u << 25u) | 16u, // context -> image-table V#
      EncodeSopp(0x0c, 0),
      EncodeVop1(0x02, 28, 257),                      // GPU-selected texture row
      EncodeSop2(0x1e, 28, 28, 133),                  // row * 32
      EncodeSmem0(0x0b, 32, 12), 28u << 25u,         // eight image descriptor words
      EncodeSopp(0x0c, 0),
      EncodeVop1(0x01, 2, 128), EncodeVop1(0x01, 3, 128),
      EncodeMimg0(0x00, 0x1), EncodeMimg1(6, 8, 0, 2),
      EncodeSop1(0x21, 125, 14)};
}

inline void Entry(Memory& memory, uint32_t row, uint64_t target, uint64_t context) {
  const uint64_t at = table + row * 16u;
  memory.words[at] = static_cast<uint32_t>(target);
  memory.words[at + 4u] = static_cast<uint32_t>(target >> 32u);
  memory.words[at + 8u] = static_cast<uint32_t>(context);
  memory.words[at + 12u] = static_cast<uint32_t>(context >> 32u);
}

inline void ImageTable(Memory& memory, uint64_t context, uint64_t images, uint32_t first) {
  memory.words[context + 16u] = static_cast<uint32_t>(images);
  memory.words[context + 20u] = static_cast<uint32_t>(images >> 32u);
  memory.words[context + 24u] = 64u;
  memory.words[context + 28u] = 3u << 28u;
  for (uint32_t row = 0; row < 2u; ++row) {
    std::array<uint32_t, 8> words {};
    words[0] = first + row;
    words[1] = static_cast<uint32_t>(Prospero::BufferFormat::k32Float) << 20u;
    words[2] = 3u | (3u << 14u);
    words[3] = DstSel(4, 5, 6, 7) | (static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u);
    for (size_t word = 0; word < words.size(); ++word)
      memory.words[images + row * 32u + word * 4u] = words[word];
  }
}
} // namespace table_call_context

void TestTableSelectedContextImage() {
  using namespace ShaderRecompiler;
  namespace t = table_call_context;
  t::Memory memory;
  FakeGuestMemory fake(t::Read, &memory);
  t::Place(memory, t::leaf_a, t::Leaf());
  t::Place(memory, t::leaf_b, t::Leaf());
  // One callee receives distinct contexts with the same payload. Another callee
  // has a different payload; neither a global pointer nor a global V# is valid.
  t::Entry(memory, 0u, t::leaf_a, t::context_a);
  t::Entry(memory, 1u, t::leaf_a, t::context_a2);
  t::Entry(memory, 2u, t::leaf_b, t::context_b);
  t::Entry(memory, 3u, 0u, 0u);
  t::ImageTable(memory, t::context_a, t::images_a, 0x40u);
  t::ImageTable(memory, t::context_a2, t::images_a, 0x40u);
  t::ImageTable(memory, t::context_b, t::images_b, 0x80u);
  std::array<uint32_t, 64> user_data {};
  const uint64_t output = t::table + 0x4000u;
  user_data[0] = static_cast<uint32_t>(output); user_data[1] = output >> 32u;
  user_data[2] = 256u; user_data[3] = 0x00014facu;
  user_data[8] = static_cast<uint32_t>(t::table); user_data[9] = t::table >> 32u;
  user_data[10] = 64u; user_data[11] = 3u << 28u;
  auto options = MakeCompileOptions(ShaderType::Compute);
  ShaderComputeInputInfo compute = *options.input_info.compute;
  compute.thread_ids_num = 1u;
  compute.threads_num[0] = 64u; compute.threads_num[1] = compute.threads_num[2] = 1u;
  options.input_info.compute = &compute;
  options.wave_size = 64u;
  options.user_data = user_data;
  IR::SrtRuntime runtime {.user_data = user_data, .userdata = &memory,
                          .read_specialization_memory = t::Read};
  auto source = PrepareShaderSource(t::Caller(), options);
  Check(source.calls.size() == 1u && source.calls[0].kind == ShaderSource::Call::Kind::Table,
        "context-image fixture lost its GPU-selected call");
  std::string reason;
  const auto* linked = TryRefreshShaderSource(source, runtime, reason);
  Check(linked != nullptr && reason.empty(), "context-image call table failed to link");
  const auto refresh = [&] {
    linked = TryRefreshShaderSource(source, runtime, reason);
    Check(linked != nullptr && reason.empty(), "context-image refresh failed to link");
  };
  const auto evaluate = [&](const std::unordered_set<uint32_t>& expected, bool compile = false) {
    bool rejected = false;
    try {
      auto translated = TranslateProgram(*linked, options, true);
      translated.program.source_reads = source.reads;
      Check(!expected.empty(), "non-invariant context descriptors were accepted");
      const auto plan = IR::ExtractResourcePlan(translated.program);
      IR::ResourceSnapshot snapshot;
      IR::ResourceSpecialization specialization;
      Check(IR::MaterializeResources(plan, runtime, snapshot, specialization),
            "context-selected image tables did not materialize");
      std::unordered_set<uint32_t> addresses;
      for (const auto& image: snapshot.images) if (image.dwords[0] != 0u) addresses.insert(image.dwords[0]);
      Check(addresses == expected, "context-selected image candidates mixed or lost callees' texture tables");
      if (compile) {
        auto compiled = CompileProgram(std::move(translated), options, specialization);
        CheckSpirvBinaryValidates(compiled.spirv);
      }
    } catch (const IR::ResourceTrackingFailure& failure) {
      rejected = true;
      if (!expected.empty()) std::fprintf(stderr, "TableSelectedContextImage: %s\n", failure.message.c_str());
      Check(expected.empty(), "a table-selected callable's invariant context is not available to descriptor tracking");
      Check(failure.reason.find("GetImageResource") != std::string::npos,
            "context rejection did not identify its unprovable image descriptor");
    }
    Check(rejected == expected.empty(), "context descriptor acceptance did not match its proof");
  };
  const auto captured = [&](uint64_t address, uint32_t value) {
    return std::ranges::any_of(source.call_data, [&](const auto& range) {
      return address >= range.address && (address - range.address) % 4u == 0u &&
          (address - range.address) / 4u < range.words.size() &&
          range.words[(address - range.address) / 4u] == value;
    });
  };
  const std::unordered_set<uint32_t> initial {0x40u, 0x41u, 0x80u, 0x81u};
  evaluate(initial, true);
  Check(source.calls[0].context_registers[16], "live companion SGPR pair was not proved");
  Check(source.linked->context_loads.size() == 2u, "per-callee fixed descriptor loads were not projected");
  uint64_t captured_bytes = 0;
  for (const auto& range: source.call_data) captured_bytes += range.words.size() * 4u;
  Check(captured_bytes == 64u + 3u * 16u, "record/payload closure duplicated or omitted bytes");
  Check(captured(t::table + 48u, 0u) && captured(t::table + 60u, 0u),
        "null table record was absent from the immutable closure");
  const auto first_revision = source.revision;
  const auto first_code = source.linked->code;
  refresh();
  Check(source.revision == first_revision && source.linked->code == first_code,
        "unchanged context proof did not reuse its linked program");

  // A changed companion pointer with identical payload refreshes the dispatch
  // closure while reusing the same semantically identical compiled program.
  const uint64_t context_a3 = t::table + 0x2300u;
  t::ImageTable(memory, context_a3, t::images_a, 0x40u);
  t::Entry(memory, 1u, t::leaf_a, context_a3);
  refresh();
  Check(source.revision == first_revision && captured(t::table + 24u, static_cast<uint32_t>(context_a3)),
        "context-only table mutation did not refresh the owned record snapshot");
  Check(captured(context_a3 + 16u, static_cast<uint32_t>(t::images_a)),
        "new context payload was not captured after a table mutation");
  evaluate(initial);
  t::Entry(memory, 1u, t::leaf_a, t::context_a2);
  t::Entry(memory, 3u, t::leaf_a, t::context_a2);
  refresh();
  Check(captured(t::table + 48u, static_cast<uint32_t>(t::leaf_a)),
        "null-to-live record mutation escaped complete enumeration");
  evaluate(initial);
  t::Entry(memory, 3u, 0u, 0u);

  // Uniform payload changes must enter the module key even though function
  // addresses and original raw leaf instructions are unchanged.
  const uint64_t images_c = t::table + 0x3300u;
  t::ImageTable(memory, t::context_a, images_c, 0x90u);
  t::ImageTable(memory, t::context_a2, images_c, 0x90u);
  Check(captured(t::context_a + 16u, static_cast<uint32_t>(t::images_a)),
        "captured payload borrowed mutable guest memory instead of owning its words");
  refresh();
  Check(source.revision > first_revision && source.linked->code != first_code,
        "payload-only change reused the preceding module key");
  evaluate({0x90u, 0x91u, 0x80u, 0x81u}, true);
  t::ImageTable(memory, t::context_a2, t::images_a, 0x40u);
  refresh();
  evaluate({}); // same callee, distinct non-invariant payloads
  t::ImageTable(memory, t::context_a, t::images_a, 0x40u);
  refresh();
  evaluate(initial);

  // A reader failure or invalid address must preserve the original load and
  // yield the established typed descriptor refusal, never invent a V#.
  memory.words.erase(t::context_a2 + 20u);
  refresh();
  evaluate({});
  t::ImageTable(memory, t::context_a2, t::images_a, 0x40u);
  t::Entry(memory, 1u, t::leaf_a, t::context_a2 | 1u);
  refresh();
  evaluate({});
  t::Entry(memory, 1u, t::leaf_a, 0x0000fffffffffff0ull);
  refresh();
  evaluate({});
  t::Entry(memory, 1u, t::leaf_a, t::context_a2);

  // Control flow alone does not lose a context pair preserved by the whole
  // leaf. A redefinition of either half on one edge does invalidate it.
  auto branch_leaf = t::Leaf();
  branch_leaf.insert(branch_leaf.begin(), {EncodeSopp(0x04, 1), EncodeSopp(0x00, 0)});
  t::Place(memory, t::leaf_a, branch_leaf);
  const auto before_leaf_change = source.revision;
  refresh();
  Check(source.revision > before_leaf_change, "changed original leaf code failed to relink");
  evaluate(initial, true);
  branch_leaf[1] = EncodeSMovB32(17, 128); // only one half, and only one branch arm
  t::Place(memory, t::leaf_a, branch_leaf);
  refresh();
  evaluate({});
  branch_leaf[1] = EncodeSMovB32(16, 128); // the low half must be rejected as well
  t::Place(memory, t::leaf_a, branch_leaf);
  refresh();
  evaluate({});
  t::Place(memory, t::leaf_a, t::Leaf());
  refresh();
  evaluate(initial);
  std::printf("[host]    TableSelectedContextImage       ok (13 proof/invalidation scenarios)\n");
}

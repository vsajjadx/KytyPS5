// CALLABLE_FLATLOCAL_ROUNDS_BEGIN
// Insert immediately BEFORE the existing R13 GLOBAL-store negative; safe local-only RED runs first.
set_mutation(scratch, 0);
write_entry(0, leaf_a); write_entry(1, leaf_b);
const auto bind_local = [&](const std::vector<u32> &program_code) {
  shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(program_code.data()),
                       .num_thread_x = lanes, .num_thread_y = 1, .num_thread_z = 1,
                       .wave_size = 64, .user_sgpr = 18, .lds_size = 1});
};
constexpr u32 local_words = 4u * lanes + 1u;
const auto run_local = [&](const char *label, const std::vector<u32> &program_code, bool expect_module) {
  bind_local(program_code);
  set_vsharp(0, raw_descriptor(output, local_words * sizeof(u32)));
  Require(name, label, context.InvalidateMemory(output, local_words * sizeof(u32)),
          "local fixture output escaped mapping");
  std::array<u32, local_words> actual{}; actual.fill(sentinel);
  LibKernel::Memory::WriteBacking(output, actual.data(), sizeof(actual));
  const auto missed_before = faults.CallTableMisses();
  ShaderComputeInputInfo input{};
  const auto *module = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), input);
  Require(name, label, (module != nullptr) == expect_module,
          std::string(label) + ": unexpected module binding (local-only must execute, unknown/global must reject)");
  if (expect_module) {
    Require(name, "local-only has no BDA/global memory", !module->info.info.uses_dma &&
                !module->info.has_address_writes,
            "local-only caller retained a global/BDA address write");
    const auto *plan = PipelineCacheTestAccess::ActivityPlan(context.GetPipelineCache(), input.stage.resources);
    u32 classified = 0;
    if (plan != nullptr) for (const auto &memory : plan->memory_info)
      if (memory.kind == ShaderRecompiler::IR::ResourceKind::FlatLocal) {
        Require(name, "local-only DWORD metadata", memory.data_bits == 32u && memory.data_dwords == 1u,
                "the local-only proof unexpectedly accepted a different memory width");
        ++classified;
      }
    Require(name, "local-only real post-track classification", plan != nullptr && classified >= 4u,
            "the actual local memory instructions did not become FlatLocal");
  }
  processor.DispatchDirect(1, 1, 1, 0x41u);
  context.RunGarbageCollector();
  context.GetCommandScheduler().Finish();
  cache.ReadMemory(output, sizeof(actual));
  Require(name, label, LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)),
          "local fixture output not readable");
  for (u32 lane = 0; lane < lanes; ++lane) {
    const u32 marker = (lane & 1u) == 0u ? value_a : value_b;
    const std::array expected{marker + ((lane & 1u) == 0u ? 0x4000u : 0x3000u),
                             marker + ((lane & 1u) == 0u ? 0x4000u : 0x1000u),
                             marker + ((lane & 1u) == 0u ? 0x2000u : 0x3000u), marker};
    for (u32 column = 0; column < 4; ++column) {
      const auto value = expect_module ? expected[column] : sentinel;
      Require(name, label, actual[column * lanes + lane] == value,
          std::string(label) + " column " + std::to_string(column) + " lane " + std::to_string(lane) +
          " expected " + Hex(value) + " got " + Hex(actual[column * lanes + lane]));
    }
  }
  Require(name, "local-only tail canary", actual.back() == sentinel, "local stores escaped output bound");
  Require(name, label, faults.CallTableMisses() == missed_before,
          "known local-only/fail-closed dispatch emitted an unexpected callable miss");
};
run_local("caller private/LDS FlatLocal executes", local_code, true);
run_local("mixed local/global remains rejected", mixed_code, false);
run_local("unknown-high Flat remains rejected", unknown_code, false);
run_local("narrow EXEC clamp cannot certify wider store", divergent_code, false);
run_local("local-only caller recovers", local_code, true);
// The linked leaf, not just the caller, must obey the same policy.
const uint64_t local_leaf = base + 0x1500;
std::vector<u32> leaf_local{EncodeSop1(0x0b, 26, InlineU32(1)),
                          EncodeVop2(0x1a, 20, InlineU32(2), 0)};
leaf_local.push_back(EncodeVop2(0x1b, 22, InlineU32(1), 0));
leaf_local.push_back(EncodeVop2(0x1a, 22, InlineU32(31), 22));
AppendVop3(&leaf_local, 0x159, 21, 26, 255, Vgpr(22));
leaf_local.push_back(0x70000000u);
AppendVMovU32(&leaf_local, 30, value_c);
leaf_local.insert(leaf_local.end(), {EncodeFlat0(0x1c, 0), EncodeFlat1(0, 125, 30, 20),
    EncodeFlat0(0x0c, 0), EncodeFlat1(6, 125, 0, 20), EncodeSop1(0x21, 125, 14)});
Require(name, "local leaf write", context.InvalidateMemory(local_leaf, 0x100), "local leaf escaped mapping");
LibKernel::Memory::WriteBacking(local_leaf, leaf_local.data(), leaf_local.size() * sizeof(u32));
write_entry(0, local_leaf); write_entry(1, local_leaf);
bind_local(local_callee_code);
set_vsharp(0, raw_descriptor(output, lanes * sizeof(u32)));
ShaderComputeInputInfo local_leaf_input{};
const auto *local_leaf_module = context.GetPipelineCache().GetComputeProgram(shaders.GetCs(), local_leaf_input);
Require(name, "linked local callee has no BDA/global write", local_leaf_module != nullptr &&
            !local_leaf_module->info.info.uses_dma && !local_leaf_module->info.has_address_writes,
        "local callee did not retain a certified local-only memory path");
const auto local_leaf_id = run("callee private/LDS FlatLocal executes", true, value_c, value_c, {});
// Same target address, but an actual GLOBAL write: must not inherit local proof.
std::vector<u32> leaf_global{EncodeFlat0(0x1c, 2), EncodeFlat1(0, 0, 6, 2),
                           EncodeVop1(0x01, 6, InlineU32(value_c)), EncodeSop1(0x21, 125, 14)};
Require(name, "global leaf rewrite", context.InvalidateMemory(local_leaf, 0x100), "global leaf escaped mapping");
LibKernel::Memory::WriteBacking(local_leaf, leaf_global.data(), leaf_global.size() * sizeof(u32));
(void)run("callee global store stays rejected", false, sentinel, sentinel, {});
(void)run("unchanged global callee remains rejected", false, sentinel, sentinel, {});
Require(name, "restore local leaf", context.InvalidateMemory(local_leaf, 0x100), "local leaf escaped mapping");
LibKernel::Memory::WriteBacking(local_leaf, leaf_local.data(), leaf_local.size() * sizeof(u32));
Require(name, "same cached caller recovers local callee",
        run("callee local-only recovers", true, value_c, value_c, {}) == local_leaf_id,
        "global rejection poisoned the restored local-callee module cache");
// Preserve the original R13/restoration cases exactly after the extension.
write_entry(0, leaf_a); write_entry(1, leaf_b);
set_vsharp(0, raw_descriptor(output, lanes * sizeof(u32)));
// CALLABLE_FLATLOCAL_ROUNDS_END

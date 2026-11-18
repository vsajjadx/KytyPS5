# Upstream provenance for the 2026-11-18 integration

This ledger records source revisions, authors, adaptation boundaries, and regression coverage. Gameplay and performance results require separate, scene-identified runtime evidence. Source compatibility and passing synthetic GPU tests establish narrower properties.

## KytyPS5 changes

The following changes were adapted into this fork's existing renderer and shader architecture. Existing guest-address guards, atomic BDA fault publication, call-table handling, launcher preferences, and frame-generation lifecycle changes are preserved.

| Area | Upstream revision and author | Adaptation and regression coverage |
| --- | --- | --- |
| Clean read-only memory tracking | [db3f2758](https://github.com/KytyPS5/KytyPS5/commit/db3f2758c0266095704a91c490eb90e2f4953df9), nmzik | Avoid the region lock for CPU-clean read-only pages; dirty and read-only behavior tests. |
| Byte permutation | [11d834dd](https://github.com/KytyPS5/KytyPS5/commit/11d834dd820c093118d37177c6162f8092134cf2), nmzik | Translate V_PERM_B32 with real GPU result checks. |
| Masked byte SAD | [174ec912](https://github.com/KytyPS5/KytyPS5/commit/174ec91265d58bdb8d86ef2f47a439c38e1f2e9b), nmzik | Translate V_MSAD_U8; unsigned saturation is checked separately. |
| Whole-wave reductions | [1e871718](https://github.com/KytyPS5/KytyPS5/commit/1e871718fae020bb085fdb1c9cc7d5cb4c7b296d), Senaxx | Add, signed min/max, AND and XOR reductions with Wave32/Wave64 and provenance cases. |
| Dirty-bit publication | [9d718b0a](https://github.com/KytyPS5/KytyPS5/commit/9d718b0a248ff08797686690d94d0e1bcae8bada), Jean-Philippe | Acquire/release ordering for lock-free tracker queries; publication-race and bit-array regressions. |
| Package metadata | [e4cc27d1](https://github.com/KytyPS5/KytyPS5/commit/e4cc27d11b16c63b1e0126bad808baad824b7dc3), nmzik | PlayGo fallback for a package without chunk metadata. |
| Multichannel Opus | [0db0e022](https://github.com/KytyPS5/KytyPS5/commit/0db0e022c102496597369aef8a604f35220f03b6), Senaxx | Mapping-family-1 decoding through eight channels in element order. |
| Image transfer state | [006e33bd](https://github.com/KytyPS5/KytyPS5/commit/006e33bd57ed9edd76bcc24f6c8e5e8cacdb9e9d), nmzik | Preserve transfer layout until the next reader; actual GPU transfer-state regression. |
| Native BVH traversal | [93478a25](https://github.com/KytyPS5/KytyPS5/commit/93478a252428f8ce6af854d3605890226f763478), nmzik | Reviewed and regression-tested, then quarantined after observable semantic mismatches and unproved traversal contracts. The automatic guest-loop pass preserves software IR and returns zero conversions, including when native RT is requested. |
| Traversal diagnostics | [91f4cd04](https://github.com/KytyPS5/KytyPS5/commit/91f4cd047e9b1342d0e6f7b37d4a546f5909a922), nmzik | Diagnostic source was used to investigate retained software loops. Production automatic conversion is disabled; a native-RT option or fallback message does not establish accelerated guest traversal. |
| BVH header formats | [98aa7e01](https://github.com/KytyPS5/KytyPS5/commit/98aa7e01fdf3784193c49ecd95e3fa656ce6439f), nmzik | Native mirror support for header formats 3 and 10 alongside format 1. |
| Wave64 traversal | [e34dc034](https://github.com/KytyPS5/KytyPS5/commit/e34dc034eefc4b3abc9851fc5c62721f52b02d91), nmzik | Masked BLAS-header moves and Wave64 infrastructure reviewed. Guest-loop conversion is disabled; software traversal regressions and standalone native query/mirror regressions are reported separately. |
| BDA uploads | [704357ab](https://github.com/KytyPS5/KytyPS5/commit/704357ab530dbee9e1b539a08eaf9dbc6df7cbed), nmzik | Reupload every CPU-written BDA page while retaining this fork's address guards, atomic faults, padding and call-table behavior. |
| HDR overlay bounds | [18809cb0](https://github.com/KytyPS5/KytyPS5/commit/18809cb06de178a876ce657c5a2f950249a2c33f), XxOtakuXx | Clamp unpremultiplied PQ input before decoding. The GPU regression uses alpha 1/3 with color above alpha; this is an independent rendering correction. |

The formatted-buffer adaptation comes from [zinosaure/KytyPS5, 1e4b3ec2](https://github.com/zinosaure/KytyPS5/commit/1e4b3ec2c1ee65e9e5b767aa1766c70b017319a0), authored by Gino D. It supports indirect formatted X/XY/XYZ/XYZW loads for 32-bit and packed 16-bit formats. Its tests cover selectors, bounds, EXEC masking and canaries. The adapted fixture uses a scalar loop to keep shader generation bounded. The source revision's LICENSE is GPL version 2; its SHA-256 is `f9c375a1be4a41f7b70301dd83c91cb89e41567478859b77eef375a52d782505`.

## AnyPS5 shader semantics

[boykopovar/AnyPS5](https://github.com/boykopovar/AnyPS5) is a native relinker and replacement-library project. Its shader instruction semantics were adapted to KytyPS5's existing IR and SPIR-V backend. The reviewed instruction source snapshot was `bef2f1f77d46b84bdc579078bd856bbdc1cfe7c1`, in `core/shader/recompiler/Translation/src/IntegerInstructions.cpp`.

| Semantics | Source revision | Author |
| --- | --- | --- |
| V_LERP_U8 per-byte rounding | [4a4a1edd](https://github.com/boykopovar/AnyPS5/commit/4a4a1eddc1c920133b343fab9bf2d172163688e2) | DotDebian / Quentin Waguet |
| V_SAD_U8, V_SAD_HI_U8, V_SAD_U16 and masked byte SAD | [9f2319b4](https://github.com/boykopovar/AnyPS5/commit/9f2319b46eda1cecff980eb8472aedcc2d70dfa9) | DotDebian / Quentin Waguet |
| Unsigned saturating clamp for integer SAD results | [fa996bbc](https://github.com/boykopovar/AnyPS5/commit/fa996bbccb8f7302cd8270b6aa8961725c8589a7) | Adrià Franch |
| Refactored subword SAD helper inspected for equivalent behavior | [f9a07425](https://github.com/boykopovar/AnyPS5/commit/f9a074250f8029fed870bee43db611e4a6e31af7) | Adrià Franch |
| Existing scalar SAD basis reviewed for saturation consistency | [3efd3759](https://github.com/boykopovar/AnyPS5/commit/3efd37596da5f81c7224054d776928f583a142b8) | boykopovar |

Both project roots use GPL version 2. Regression inputs include edge values, randomized operands, overflow/clamp cases, masked byte behavior, and actual Wave32/Wave64 GPU execution.

A later source review through AnyPS5 `a6c68e1953bc70602e71c197cd504ab85fbe62b7` (2026-11-18 01:21:53 UTC) covered 138 additional commits and 181 changed paths. That delta contains no shader-recompiler, AGC backend or memory-runtime implementation change relevant to these ports. RDNA2 gfx1036 oracle documentation in that delta is explicitly distinct from PS5 measurements.

A bounded official-head check at 2026-11-18 05:44:31 UTC found KytyPS5 unchanged at `2638b1b07613dc32a4708d2cd61a75711b84b6cf`. AnyPS5 had advanced to `56c87adbdc3846f7259511f758ea5d31e4d26c20`; the additional 112 commits and 160 changed paths contained no matching AGC, shader, graphics or execution-memory implementation delta to transfer. The driver host-import/write-watch and packed-16 texture-oracle documentation was reviewed as diagnostic context, with its NVIDIA/AMD observations kept separate from PS5 and this fork's measurements. This is a dated review boundary, not an assertion about future commits.

## Launcher graphics profile and frame generation

The optional UFC6 profile applies only to PPSA23566 version 01.004.000, with the expected original patch bytes at all three guarded patch locations. It offers a fixed 960×540 guest scale, disabled dynamic resolution and depth of field, and 1080p output. It defaults to off and preserves user cheat-plan conflicts and saved settings.

Its patch source is [PORTHEON release 7.0](https://github.com/XAndromedaXR/PORTHEON/releases/tag/7.0), archive member `PORTHEON-Released/games/UFC6/src/ufc6.c`, copyright 2026 Marcelo Miyake (XAndromedaXR), GPL-2.0-or-later. The archive member, the locally retained source, and the declared SHA-256 agree: `f4e25c3bb80478dd156a594a0dffd0fc625d40da76e3e72b0321bd457cdc4cd8`. All selected original/replacement bytes and offsets were compared against that source. See the bundled profile NOTICE for attribution.

The FSR/Vulkan dependency is pinned to the owned fork at [8ee4f83c](https://github.com/vsajjadx/UniversalScalers-Vulkan/commit/8ee4f83cf512c42e26f02a3abb54e5408a0051a6), preserving the existing frame-lifecycle changes. Frame-generation tests establish lifecycle and accounting behavior. Base game frames and generated frames must be reported separately in runtime measurements.

## Call-context correctness work

The new compute table-call work proves companion-pointer provenance and invariant fixed scalar-load payloads per callee. It retains the mandatory pointer scan and code-validation budgets, treats optional proof failure conservatively, refreshes per-dispatch snapshots on module reuse, and includes semantic projected words in linked-code identity.

Owned data snapshots are enforced against actual canonical GPU buffer owners. Every overlapping SSBO uses the same owner as BDA; ordered copies occur after binding commit and remain alive through the consumer tick. Tests exercise CPU A→B→A with stale GPU B, all four table/payload SSBO/BDA aliases, physical aliasing, native owner replacement, preceding GPU writes, prepared writable extents and staging lifetime.

Native acceleration structures are separate representations. The combination of captured call data and an accepted native ray-query binding is refused until acceleration-structure input disjointness can be established. That limitation is explicit and does not silently overwrite native AS inputs.

## Sampled image-heap dependencies for captured calls

The original renderer work admits only read-only sampled heaps whose actual stage-bound source tables, derived slot views and native image inputs are disjoint from protected call code, data and table ranges. It captures source/view/element identities before Commit and re-resolves them after Commit clears the active view list. Stable intentional null elements are handled explicitly; stale, changed or malformed views remain refusals. Existing canonical GPU copies, physical-alias checks, writable-image guards and native-AS limitations remain enforced.

The real Wolverine run exposed a further depth/stencil dependency boundary. The extension captures the native depth owner and its direct registered stencil-plane records, including data, independent stencil, active metadata and tracked ranges. IDs, serials, native handles, exact membership, reciprocal parent links, formats, mappings and compression policy must remain stable. A stencil plane need not have an independent native image because the cache resolves it to its depth owner. The successful post-Commit check adds no allocation, recursion, GPU wait or retained cache pointer. Unexpected nested graphs, invalid mappings and unresolved owners remain refused.

An unchanged-production native test first reproduced the valid depth/stencil sampling refusal. The corrected implementation then passed 36 actual GPU paths across Wave32 and Wave64, including selected sample values, disjoint associated depth, real HTile metadata, standalone stencil ranges without child records, source ABA, retired views, nulls and exact GPU/guest canaries. The historical-child overlap fixture keeps all requested and current-parent inputs disjoint and directly verifies that graph capture rejects the child range; the full dispatch guard may reject even earlier at strict image-backed currentness. A separate same-draw reassociation test preserves source bytes, view/draw identity and memory mapping/protection generations and rejects the changed parent-child graph after Commit.

These tests do not establish target-game dispatch or gameplay. The NullToLive case changes draw identity and is not an isolated same-draw null mutation oracle. Nonzero stencil subresource mappings have checked bounds but do not yet have a distinct native fixture.

## Private Scratch address-space validation

The closure write guard previously counted validated private Scratch stores as unrestricted guest-address writes. The original correction admits only completed resource tracking with a nonzero scratch allocation and a resolved zero-argument `GetScratchResource` handle. The existing certified FlatLocal case remains unchanged; physical Flat/Global stores and unknown or malformed cases remain refused.

The native regression first reproduced the original binding refusal before the private shader dispatched. After the narrow correction, actual Wave32/Wave64 execution verifies private first/last DWORD stores and reads, alternating EXEC, out-of-bounds accesses and per-lane output. Output and output canaries are checked by GPU readback; separate guest-backing comparisons verify callee/table/context byte preservation. Real Flat/Global writers and missing scratch allocations still fail and a subsequent valid bind recovers. Separate CPU tests establish the tracking prerequisites for byte, halfword and DWORD operations. These regressions do not alone establish gameplay.

## Scalar-read provenance deduplication

The original fix compares normalized temporary scalar-read metadata only after validating the resolved descriptor operands. A prior read's physical bank index is provenance, not a semantic difference once the explicit descriptor words and arguments agree. Removing that index only from the comparison prevents equivalent callee loads from consuming duplicate root slots. Stored metadata and generic value equivalence are unchanged. Distinct offsets, bounds, descriptor words, flags, components and other semantic fields remain separate, and the existing resource limit remains enforced.

The CPU regression failed before the production change and passes afterward. Actual native GPU fixtures exercise copied and distinct banks, table mutation, unchanged-module reuse, output values and canaries on both Wave32 and Wave64. Passing these regressions does not prove that the blocked Wolverine dispatch or gameplay works.

## Ray-query conversion quarantine

An exported-loop-counter test and later counter, payload, stack and address-space tests demonstrated incorrect matcher acceptance. These are host-side acceptance/refusal assertions. A separate native GPU regression demonstrated an inactive lane's preserved sentinel being overwritten. The conversion also lacks complete proofs for ray updates, liveness, push/pop ordering, bounds and equal-distance outputs. Consequently the production conversion entry point is a small documented no-op with no runtime override. Tests check unchanged IR and metadata and correct repeated software outputs. Standalone Vulkan ray-query and BVH-mirror tests exercise native backend infrastructure independently; they do not establish native conversion of UFC6's guest shader.

## Search and review coverage

The saved discovery scope catalogued 7,133 repositories, of which 7,099 were accessible, and metadata for 51,640 branches. History inspection of all 1,047 saved detached tips was completed, fetching 1,035 initially missing exact tips from 360 repository groups without fetch failures. The union contains 104,452 reachable commits above the two project bases; 100,312 additional commit/name-status records were indexed after the earlier review. This establishes coverage of the saved repository and history scope, not manual semantic review of every patch or completeness across inaccessible, private or future repositories.

A many-buffer-root candidate, [Nawid3333/KytyPS5, dec78125](https://github.com/Nawid3333/KytyPS5/commit/dec781257395b4b9a028aaff4c90729846ef1d24), was examined but not integrated. Removing the resource cap also requires a common device/pool-budget validator, including warm-up paths. The observed Wolverine root-count failure was first addressed by removing duplicate provenance roots without relaxing that cap.

## Validation records

The repository contains reproducible CPU and GPU regressions. Final build, runtime, release and installation results belong in the associated validation report. A successful build alone does not establish gameplay compatibility, and a changed scene or compilation state cannot establish an FPS gain.

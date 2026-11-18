# Gameplay validation checkpoint: 2026-11-18

The complete 144-case regression gate and the new native metadata-ledger fixture pass. Those tests used the uncommitted source state based on `96de01dcbed26fad88f3de46856708a62a70d713`, with a truthful `-dirty` build label. A release manifest must separately bind the final committed revision and its actually relinked binaries. Final candidate gameplay and matched performance comparison remain pending; these regression results do not establish working Wolverine gameplay or an FPS gain.

## Current gate

The successful gate receipt is `9b056b548bd4c2aa70c34b89ad1419189bb5bebff520fadba75c5b9e1d9b7ef1`, completed on 2026-11-18 at 19:41:30 +03:30.

| Gate | Observed result |
|---|---|
| Complete application, launcher, updater and test build | Exit 0; 566.909 seconds for the continuation |
| Registered regression suite | 144/144 passed, no failure/error/skipped case; 155.096 seconds |
| New native ledger child | Exit 0; exact 21-word readback and a successful native timeline wait; 0.458 seconds |
| Source consistency | Source/dependency fingerprints unchanged during the successful gate |
| Protected data | All 26 original-path guards valid; separate shadPS4 data/runtime guards unchanged |

The clean origin was a single `--clean-first` invocation. It initially failed after 99.099 seconds because one image-upload callback still accepted two parameters after the metadata helper gained a third `Buffer` argument. Adding an unnamed `const Buffer&` to that callback preserved its body and left that image-only payload path unobserved. The build then resumed its existing clean-origin objects without another clean operation. Both the failed receipt and the exact one-line source delta remain preserved.

The CTest execution driver round-tripped all 144 original command arrays, working directories and properties exactly, except for setting timeouts to zero to avoid terminating a process. The original generated test definitions were untouched. The previously completed 82-check metadata contract and six legal observer controls were not rerun.

| Tested dirty artifact | SHA256 |
|---|---|
| Core | `3151a3bc61668e64d0a8bfdb7a07025c2886d338f08f46c063cb9f6b1f0de772` |
| Native compute test executable | `10b538dc9f9988b173ed38fdef76b0638080430b004dd3fc05abdbc95f9431f2` |
| Launcher | `84c54a67f5b5bad21c9110b327f42ef39d23ae414aeb8f35c2d745606b197fef` |
| Updater | `46dc3feec4c9b7392cf51044d3f3645eef407e105d59912287d4acfda4c852bf` |

These are the actual tested development binaries, not silently relabeled release binaries. The postcommit provenance build and package manifest record any final executable changes separately.

### Earlier failure and focused repair history

The first full clean build completed successfully in 649.782 seconds (2,236 build actions). Its fresh CTest run passed 131 of 132 cases in 149.939 seconds. The failing `legacy_value_opcode_wire` test correctly rejected a changed persisted instruction ID: historical SelectU32 at wire ID 341 had moved. That historical gate failed and is preserved; the successful 144-case gate above supersedes it.

The expanded pre-fix ABI regression then compiled successfully and failed at historical wire ID 212 (`SubgroupUMinU32` decoded as `SubgroupReduceU32`). The expanded fixture freezes all 344 opcode names, result types, arities and argument types from baseline commit `96de01dcbed26fad88f3de46856708a62a70d713`. The repair preserves all 344 historical rows and appends the seven new operations at IDs 344 through 350. The focused and full ResourcePlan CPU suites now pass, including the exact old 212/213/214/341 encodings and flag roundtrips. All 12 native legacy-reduction cases also pass on Wave32/Wave64 with full/masked execution, nonzero ignored legacy flags, sign-crossing operands and canaries. The later complete 144-case gate above now passes.

The independently reported guest queue defect has a passing production-path regression: when all queue heads are blocked and deferred native work exists, a negative selected queue can fall through into unsigned queue indexing. A first-packet blocked PM4 wait makes no progress, so it can preserve deferred work left by an earlier submission. The production-path regression reached its armed native-tick/prefix/two-wait phase and then exited with access violation `0xC0000005` before the repair. Independent controls passed. A minimal `else` now restricts queue extraction, pop/count and round-robin updates to a valid selected queue, preserving the outside-lock deferred flush. With that sole runtime change, all eight queue cases pass, including native fill output/canaries, blocked suffix preservation and resume, host commands, empty idle/shutdown, native completion without externally supplied work, concurrent producers and two simultaneous shutdown callers. Existing SchedulerTimeline and SuspendPoint tests also pass. Its contribution to Wolverine's black screen is still not established.

### Queue and ABI artifact evidence

- Expanded wire-ABI RED: successful compile followed by the expected historical-ID assertion; executable SHA256 `c757e8cf75f9c9129d092d8db8558a393fce2ed4ba0a170dda5e4ab087c14bd6`.
- Wire-ABI CPU GREEN: focused and full suite exit 0; executable SHA256 `fe110a2164044d763782f2a97a0b2934e717243de919d60d37daba70bfe8e54e`.
- Queue RED / legacy native GREEN: build 0, controls 0, all 12 legacy cases 0, isolated deferred case `0xC0000005`; native test executable SHA256 `94638a93571ee1446aec63829654b0fb2546edef4ec37e1c846a6f80fa57ae8e`.
- Queue GREEN: build 0, all eight queue cases 0, existing SchedulerTimeline 0 and SuspendPoint 0; native test executable SHA256 `4156bb79823d27ffe3a31407c236b698d8c20e28b19c2221a5b06bf0f3705208`.
- Source hashes remained unchanged during each recorded build/run. All 26 protected originals remained valid, including append-only watchdog prefixes. The access-violation instruction address was not captured, and these synthetic tests are not a Wolverine gameplay result.
- The bounded diagnostic-recorder CPU suite passes its exact-enable/default-off, rollover, explicit loss/ancestry, repeated-rejection coalescing, target retention, concurrent snapshot integrity and report-bound checks. Its context/current-worker tests also pass nested, recursive, deferred, exceptional and cross-thread restoration, bounded parent loss, and failed callback preparation without FIFO ordinal drift.
- Actual Flip instrumentation RED: the existing `Pm4Flip` fixture completed, then the new missing diagnostic-chain assertion failed; native executable SHA256 `004b1e9377888e71e25709536f673b6e77a1c4ad99437b90b9b5e43a856e04e5`.
- Integrated causal GREEN: build exit 0 in 15.782 seconds; helper CPU, all eight queue cases with diagnostics both off and on, existing scheduler and suspend tests both off and on, and actual one-GPU/four-CPU Flip chains passed. Native executable SHA256 `64c263b6db8e74b8711de5375ac369dbb586dcdd434e84d138840b3d96bdd760`; helper SHA256 `1f4ee69b5f218a8046e00e4214f1842a81381ce89edd00e74d2d8722eeae5e58`. Source inputs stayed unchanged and all 26 protected paths remained valid. This was a focused gate at 136 registered cases; subsequent fault and indirect-observer cases raised the complete clean gate to 144, which now passes as recorded above.

## Wolverine: descriptor progress and remaining black screen

Game version: PPSA03671 v01.001.005. Tests use a separate runtime directory and copies of saves/cache inputs; the installed runtime and usual user paths remain unchanged.

The image-heap association change captures a fixed snapshot of a native parent and its directly registered stencil child, then validates identity, reciprocal association, allocation serials and bounded image/stencil/metadata ranges before and after resource commit. It does not retain view pointers across commit, add a GPU wait, or relax unsupported descriptor access into arbitrary host reads.

| Evidence | Result | Limit |
|---|---|---|
| Native pre-fix association regression (R14) | Expected failure at `TableCallHeap32DepthStencilAssociation`, after preceding cases passed | A focused renderer oracle, not a game run |
| Native repaired association regression (R15) | 18 paths at guest Wave32 and Wave64, 36 heap cases, passed; existing snapshot/context/scratch group also passed | The physical GPU uses native subgroup size 32; this is not physical native-64 coverage. Nonzero mip/layer mapping has bounds checks but no distinct native oracle in this set |
| R15 game launch 01, profiled | Guest virtual address reservation failed before startup; exit 321 | No shader, dispatch or gameplay result |
| R15 game launch 02, unprofiled | Target shader compiled in 104.2 ms; no final call-table refusal; normal close, exit 0 | Black image; frame counter 557 remained unchanged for approximately 285 seconds |
| R15 game launch 03, diagnostic | Target shader compiled in 102.4 ms; target dispatch-indirect recording marker observed; normal close, exit 0 | Black image; native completion and gameplay were not demonstrated |

R15 runtime executable SHA256: `3cf1b8bea81fb1606e09d0f6c94488b9dc452ee03a1ab3eb0b713601c6fb1639`.

The third run's late 15-second Tracy capture contains 1,483 completed `GuestGpu::EmptyQueueWait` scopes and 909 `FlipQueue::Flip` calls. Those calls are not proof of 909 presentations: Flip can return for missing group owner, a CPU member, non-Ready state, stale generation or a not-yet-due flip. An empty guest queue is not proof that native GPU work completed. The sole open captured zone is also the empty-queue wait, not an identified outstanding compute dispatch.

### R16: real-game causal comparison

Both 240-second child runs used the same runtime executable, SHA256 `ffce6d067ba6c0dda45c96a98349141ca90107f9ad1ee1224796542d76a433f7`, identical game arguments, copied original save inputs and an empty shader-cache seed. Frame generation, native RT, profiling and the in-process sampler were off. `KYTY_CAUSAL_DIAG=1`, call-table diagnostics and frame-class counters were on, so these are not FPS benchmarks. Only the second child's `KYTY_SUBMIT_DRAW_THRESHOLD=0` differed in the launch environment.

| Observation before the close request | Default threshold 64 | Child threshold 0 |
|---|---:|---:|
| Target recorded-dispatch native tick | 6,563 | 28,451 |
| Final successful-submit / cached-known native tick | 6,574 / 6,574 | 28,480 / 28,480 |
| Active / queued priority operations | 0 / 0 | 0 / 0 |
| Normal deferred operations still queued | 26 | 25 |
| Complete causal reports inside the pre-close boundary | 238 | 238 |
| Visible end state | Black, title frame 550 stalled | Black, title frame 551 stalled |
| Process close result | Exit 0 | Exit 0 |

The target's full shader hash, native semaphore domain and scheduler lifetime are retained. The default run retains its exact target-tick successful submit and timeline observation. The threshold-0 run has an explicit retention gap for that same-tick submit record, but a later successful timeline observation passes the target tick on the exact same domain and lifetime. This supports native timeline progress beyond the recorded target in both runs; recording a `vkCmdDispatchIndirect` alone did not establish that. These observations do not establish nonzero indirect workgroup counts, correct shader output, guest CPU consumption of a signal, physical presentation or gameplay. Timeline tick totals are not frame counts.

The last retained completed Flip request in the default run had exact Ready -> Dispatch -> GuestComplete ancestry. During the long stalled interval, the observed no-owner state had empty GPU, CPU and cancelled-request queues, and the priority worker was waiting for new work. This narrows the investigation away from an outstanding Flip group or priority callback in that interval. Earlier history rolled out of the bounded recorder; lost or unexported records remain explicit, and there is no claim of a complete historical causal chain.

Threshold 0 bypasses the existing deferred-work-count gate each time `BufferFlushDeferred` is called. It does not add a flush after every packet or every dispatch. It increased submission counts here but did not resolve the black screen. It remains a diagnostic child override, not a persistent setting or an enabled optimization.

Normal deferred operations include fault-buffer harvesting as well as resource retirement. Their pending count does not identify a blocker, and absence of a call-table miss log does not rule out an unharvested miss. These exact CLI arguments leave shared printf routing Silent, which suppresses the existing FaultManager harvest log; the compiler's call-table diagnostic flag does not override that routing. The next investigation follows the busy CPU threads and actual signal-consumption path without treating native completion as proof of a correct guest result.

### R17-R19: harvest identity, thread lifetimes and exact indirect arguments

Later runs added typed identities for ordinary fault-harvest callbacks. These distinguish an enqueued harvest from an executed harvest; native timeline completion does not itself run the CPU callback. No blanket drain, callback-priority change or new fault lookup was introduced. The R17 CPU-context attempt found that its owned game was already closing and produced no CPU context samples; later samples must not be attributed to it.

In R18 / execution E, the passive observer read the actual staged fault area only after an existing, successful native host wait covered that area's submission. The latest observed area was 3, operation 4,768, native tick 7,386, enqueue sequence 479,830, scheduler domain `0xff1b438a28`, lifetime 1. The staged call-table-miss and page-fault counts were both zero, while its ordinary harvest callback was still Queued. The last completed harvest was older (operation 1,408 at tick 2,121). Normal callbacks still queued numbered 26. This establishes the sampled buffer contents and callback state for this run, not correctness or execution of the target shader body.

The E CPU probe captured eight bounded samples on each of two explicitly identified threads. The producer samples stayed in a guest pause/helper-return-false loop; the main thread traversed a guest job table. The original SELF instruction at one producer-loop address had been replaced by a live trampoline jump, and the trampoline/state word was not captured. A specific signal or job dependency therefore remains unproven. E's instruction pointers and register values are not evidence about later execution F.

R19 / execution F used core SHA256 `043ca6eb348b59f37f85a1ba5b6417b6cbd51a4df6c65d58ff52cee79036e10b` in a separate runtime with copied inputs. Frame generation, native RT, Tracy and the sampler were off; causal, fault and exact-hash indirect diagnostics were on. The process closed normally with exit 0 after 122.332 seconds. The game window was verified in the foreground and black at 70.817 seconds; late images without a foreground witness are not treated as game-image evidence. The title counter stopped at frame 551. This is not an FPS benchmark or working gameplay.

| F observation | Exact identity/value |
|---|---|
| Target shader | `1a04c58a9fbc2052` |
| Recorded target dispatch | sequence 480,366; native tick 7,184 |
| Scheduler | domain `0xff5bbfc488`, lifetime 1 |
| Native indirect source | buffer `0xff26f76550`, byte offset 787,712 |
| Guest argument range | `[0x11d06d0500, 0x11d06d050c)` |
| Copied arguments immediately before the dispatch | `(x,y,z) = (0,1,1)` |
| Observer coverage at first publication | cached-known tick 7,196; actual successful host-wait tick 7,195 |
| Observer capacity/result | one matching copy, one observation, zero capacity skips and zero identity errors |
| Latest sampled fault callback | area 3, operation 4,700, tick 7,189, enqueue sequence 480,871; still Queued |
| Latest sampled worker | QueueWait; zero queued/active priority callbacks; 24 normal callbacks queued |

Zero in an indirect-dispatch dimension is legal Vulkan input and requests zero workgroups. It is not itself a defect. The observation is valid for this instrumented execution: the observer inserts a transfer read and synchronization, including AllCommands scopes. Those dependencies can affect execution and may hide a synchronization defect that would exist without the observer. No claim is made that the uninstrumented game consumed identical values, that the target shader body executed, that the producer is wrong, or that these values explain the black screen. The subsequent legal GPU controls below separate the copy and dependency modes in controlled fixtures; they do not establish neutrality in this game execution.

The next producer ledger must retain each dword's guest range, native allocation generation, byte offset and exact scheduler domain/lifetime/tick/recording ordinal. Confirmed CPU writes and submitted transfer writes must be separate from shader-descriptor overlaps, which are only candidate writers until execution/output is corroborated. The `renderCompute.cpp` write notification after dispatch is not the producer of that dispatch's input. Buffer merging during `ObtainBuffer` can be normal; overlap or a merge alone is not a corruption witness. Missing or rolled-out provenance remains unknown.

The causal tid and the `Pthread run begin` id use internal TLS `GetThreadIdUnique`; they are not the separately created `guest.thread_id`. F has explicit same-process, same-lifetime witnesses for the following mapping (process 29,280, creation FILETIME 134394798748710925):

| F role | OS TID | Thread creation FILETIME | Internal unique id | Explicit guest thread id |
|---|---:|---:|---:|---:|
| Last retained Commands producer | 30,508 | 134394798832604482 | 23 | 20 |
| Guest main | 16,664 | 134394798815997070 | 4 | 1 |

The producer's last retained Commands record is sequence 480,149, submission 264,678, queue 0, 16 dwords. The target dispatch instead joins to guest-slice sequence 480,330 and the older submission 263,788; its original parent sequence 476,263 is outside retained global history. These are distinct submissions. No F CPU-context probe was performed. OS IDs must be paired with process/thread creation evidence because they can be reused.

The exact-argument fixture first built and reached its real GPU phase, then failed on the missing persistent argument observation. The enabled GREEN run and all three disabled/prerequisite controls passed. It checks real GPU-produced arguments with stale CPU backing, zero and nonzero dimensions, later source rewrites, immutable eight-slot capacity, host-wait coverage, owner-thread publication and output canaries. Those existing cases establish the instrumented copy's identity and position, not neutrality of its added barriers. The source was unchanged during the focused run and all 26 original-path guards remained valid. The later complete 144-case clean gate passes as recorded above.

F's offline audit parsed 119 complete reports with zero invalid or partial reports. Its exact join includes sequence, scheduler domain/lifetime/tick, shader hash, native buffer/offset, internal thread, cause and guest ancestry. The narrow new offline parser also passed nine corruption controls. Report families are sampled under different synchronization and do not constitute one global atomic instant. Retention loss is still explicit.

Artifact receipts: focused indirect RED `8ff9638ebfe5068e6f10a95144200b3c4548d31aa89f804904f1de1ccc24c7b9`; focused GREEN `68e2db98d5d6e3e892394c00b0f15fdbd4ee24e43d78953f22c3e2a108a4e7bb`; F exact-argument audit `c9c0fed371e9b9fe8a497a0e100b65f1f12f4394688460960e84b2eeb5d3d0f4`.

### Subsequent legal controls and native metadata ledger

Six separate children exercised observer-off, composed broad-barrier, immutable scratch-copy and real-argument-copy modes, plus the disabled causal and disabled host-wait prerequisites. All retained the same ten controlled GPU-written zero/nonzero workloads and their output/canary expectations. Receipt: `37e362a233c7cc8fc883a0f8e2b15cc662057226740cbbea1b7bc3e00991fb3a`. These passing controls are fixture evidence; they do not prove observer neutrality or the cause of the game's black screen.

The later ledger fixture installed its immutable watched addresses before the native buffer owners were created. Actual CPU stores, owned staging snapshots, upload commands, two-page allocation migration and the indirect consumer were joined per dword using captured handles, allocation generations, offsets, reservations, scheduler domain/lifetime/tick and recorded order. The original tuple remained unchanged; a new expected version was accepted only after the actual migration edges matched. Descriptor overlap remained a candidate writer with no invented payload or allocation identity.

One nonzero indirect dispatch used groups `(2,1,1)` and local size `(4,1,1)`. Its captured semaphore/tick completed through an actual successful native wait. The subsequent readback checked the immutable `(2,1,1)` input, outputs 1 through 8, four untouched tail words and all source/destination guards: 21 DWORDs in total. The recorded old spans were not consumed in this fixture. This does not claim a general stale-generation rejection mechanism, arbitrary-write coverage or correctness of a game dispatch.

The production metadata hooks add no Vulkan command, barrier, GPU query, host wait or extra guest-memory read. Their recorder must be installed before the resources it claims and detached only after producers and owners are quiescent. Metadata is recording evidence, not execution evidence; the fixture's separate completion and readback establish its native result. The unobserved image-only upload payload and unknown guest-mapping generation remain explicit limitations.

### Diagnostic implementation limits

Exact child opt-in `KYTY_CAUSAL_DIAG=1` enables five fixed record families with 64 retained records each, a separately retained target, and a nonblocking live-scheduler observation. Reports go to stderr, at most once per second and 128 KiB per report. Producer contention, overwritten history, unavailable snapshots and parent loss are reported. Causal thread IDs are internal TLS unique IDs, distinct from both OS TIDs and guest thread IDs; the explicit three-way mapping above is required.

The causal-only path observes existing waits, submissions, publication routes and the single PM4 wait-label read without an added GPU query, guest-memory reread or scheduling barrier. The separately enabled exact-argument observer does add a copy and barriers, as described above; that limitation must not be hidden by the causal-only claim. Callback attribution uses the actual selected queue tick, and callback context is restored across asynchronous, nested and exceptional execution. A callback preparation record is an attempt until insertion/dequeue or execution corroborates acceptance.

The disabled path still has enable branches and added object storage: four full-width metadata fields per guest submission and an 80-byte context in an existing Flip request (the compiled x64 request size is 168 bytes versus 88). No zero-overhead claim is made. Diagnostic I/O and enabled callback capsules can affect timing, which is why these runs are excluded from base-FPS measurements. No collector thread or persistent environment change is introduced.

## UFC 6: gameplay and performance evidence

Game version: PPSA23566 v01.004.000. Host: RTX 3080 10 GB, Core i7-12700K, 32 GB RAM, Windows 11 build 26300. These observations do not establish results on other drivers, GPUs or operating systems.

### Eligible baseline

The original installed executable (SHA256 `78dfcd375074c3d9eff7fef65f80dded521e251b650c58cade3ce685b9b6bdaa`) reached visible Practice with Islam versus Islam, AI INERT, health/stamina display off and damage display on. During a foreground, no-input, compile-free epoch with frame generation off:

| Metric | Observed |
|---|---:|
| Complete diagnostic windows | 52 |
| Combined window duration | 104.664 s |
| Successful real-class presentations | 3,639 |
| Successful real-class presentation rate | 34.7684/s |
| Generated, delayed-real, repeat and blank presentations | 0 |
| Failed presentations, recreations, drops and superseded frames | 0 |

This measures real-class presentation calls over an idle Practice scene. It is not an independently measured simulation rate, distinct image count, physical display rate, 1% low, active-fight result or final candidate performance. Earlier mixed-camera samples of 18.17 versus 21.115 are not an eligible speedup comparison. The original process needed a stop fallback (exit 15), so its shutdown is not reported as clean exit 0.

Final candidate measurement must use the same seed, scene, camera, settings and resolution with no profiler, sampler, rendering capture, compilation or host-activity experiment during the measured epoch. Frame generation stays off for the baseline comparison. Actual movement and attack response must be observed separately; sending keys alone does not prove input was handled. Generated frame rates belong in a separate run and table.

### Readback and ray tracing investigation

A validated short R11 diagnostic trace contained 6,227 `SyncReadback` scopes, totaling 3.4870 seconds on the guest GPU thread; 6,223 contained waits totaled 3.1901 seconds. These are diagnostic elapsed intervals, not an additive critical-path frame budget. Top repeated consumers saw new writer generations, so the trace did not justify skipping their synchronization as redundant.

The automatic guest ray-query traversal conversion remains quarantined after a native inactive-lane sentinel failure. Existing native BVH mirror support and backend code remain; no UFC 6 hardware ray-tracing acceleration or improvement is claimed. The 540p profile is optional and disabled, and it is excluded from the matched baseline measurement.

The host-activity-proof experiment remains off by default. The last complete cumulative report recorded 60,416 eligible calls, 27,805 confirmed activity decisions and **zero reuse hits**; those counters include boot/navigation. Readbacks per captured Present scope did not decrease. These scope counts are not classified frame counts, and diagnostic logging changes overhead, so this is not an FPS comparison. It is not an enabled performance fix.

## Upstream review and provenance

See [upstream provenance](upstream-provenance-2026-11-18.md) for integrated changes, licenses, source commits, test coverage and deferred candidates. The saved review corpus includes 7,133 catalogued repositories (7,099 accessible), 51,640 branch metadata entries, all 1,047 saved detached tips and 104,452 reachable commits above the selected bases. This establishes coverage of that recorded corpus, not manual semantic review of every patch, private repositories, deleted content or every future fork.

The latest targeted official-head comparison reviewed 112 additional AnyPS5 commits through `56c87adbdc3846f7259511f758ea5d31e4d26c20`. It did not identify an additional relevant graphics/shader/execution-memory implementation to port from that delta. Useful earlier AnyPS5 and Kyty-family changes are already attributed in the provenance report. Unvalidated RT conversion, removal of resource bounds, asynchronous flip changes and speculative synchronization removal are not enabled merely because a fork contains them.

## Protected state and packaging gate

The protection manifest contains 26 distinct original paths, including 12 save files, one trophy file, settings/metadata and installed binaries. Earlier raw row counts duplicated relative and absolute references. Diagnostic appends are retained. The original prefixes of exactly `freeze_watchdog.log`, `asset_search.log`, `native_faults.log`, `pad_debug.log` and `title_confirm_debug.log` are checked; all other original files remain exact. The four later log-growth observations and their intact original prefixes are recorded separately. No original manifest was reset and no log was truncated.

No save migration prompt or permanent save/configuration path change is introduced. Release and local-update receipts must bind the successful test gate, final committed artifact identity, runtime dependency/license checks and staged launcher smoke result. Test/cache/trace/game artifacts are private test inputs and are excluded from redistributable packages.

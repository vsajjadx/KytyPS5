// Deterministic CPU fixture for frameGenerationPacing.h. A vblank-driven model of the videoOut loop
// (Flip first, else PresentPendingFrame) and of the presenter's FG decisions (the EMA, `wanted`
// window, generated-then-pending-real, release of the replaced main frame). Each scenario runs
// with the LEGACY rules (delay interval/2, always supersede, no tick stamp) and with the product
// rules. RED = the legacy run violates the scenario's invariant; GREEN = the product run keeps it.
// No GPU/WSI; the model mirrors swapchain.cpp branch by branch (see comments).
// rev2: overlay refresh (videoOut -> PresentLastFrame) while a real frame waits, and reset-seed
// generation accounting (FrameGenerationDiag::Stats, the product header itself).
#include "graphics/presentation/frameGenerationDiag.h"
#include "graphics/presentation/frameGenerationPacing.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace Libs::Graphics::FrameGenerationPacing;
using Ns    = std::chrono::nanoseconds;
using Clock = std::chrono::steady_clock;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static constexpr Ns kVblank {1'000'000'000 / 60}; // as swapchain.cpp computes it

struct Rules {
    bool legacy;
    bool overlay_fix = true; // rev2 PresentLastFrame: a waiting real is shown as DelayedReal, once
    Ns Delay(Ns interval) const { return legacy ? interval / 2 : PendingRealDelay(interval, kVblank); }
    bool Flush(bool pending, Clock::time_point now, Clock::time_point due) const {
        return legacy ? false : FlushPendingReal(pending, now, due);
    }
};

// kind: G generated, R real (immediate or delayed), P PresentLastFrame shown as Repeat.
struct Event { int vblank; char kind; int frame; };

struct Result {
    std::vector<Event> events;
    int drops = 0;
    int frames = 0;
    std::vector<int> flip_vblank; // vblank each guest frame was flipped (never deferred by the model)
};

// gaps[i]: vblanks between guest frame i and i+1; jitter_ns[k]: CPU time offset of vblank k's loop.
// overlay[k % size]: NeedsSystemOverlayRefresh() at vblank k (empty = never).
static Result Simulate(const Rules& rules, const std::vector<int>& gaps, const std::vector<int64_t>& jitter_ns,
                       int frame_count, const std::vector<bool>& overlay = {}) {
    Result r;
    std::vector<bool> flip_at;
    int at = 0;
    for (int f = 0; f < frame_count; ++f) {
        if (static_cast<int>(flip_at.size()) <= at) flip_at.resize(at + 1, false);
        flip_at[at] = true;
        at += gaps[f % gaps.size()];
    }
    const int vblanks = at + 8;
    flip_at.resize(vblanks, false);
    const auto start = Clock::time_point {} + std::chrono::seconds {1};
    Clock::time_point last_real {};
    Ns interval {0};
    bool pending = false;
    int pending_frame = -1;
    int main_frame    = -1; // guest frame held in layers[0]
    Clock::time_point due {};
    int next_frame = 0;
    for (int k = 0; k < vblanks; ++k) {
        const auto now = start + kVblank * k + Ns {jitter_ns.empty() ? 0 : jitter_ns[k % jitter_ns.size()]};
        if (flip_at[k]) {
            // Presenter::Present: release of the replaced main frame (flush or supersede).
            if (pending) {
                if (rules.Flush(true, now, due)) {
                    r.events.push_back({k, 'R', pending_frame});
                } else {
                    ++r.drops;
                }
                pending = false;
            }
            const int f = next_frame++;
            main_frame  = f; // layers[0] = real, also while it waits behind its generated frame
            r.flip_vblank.push_back(k);
            // PresentWithGeneratedFrame: EMA and `wanted` exactly as swapchain.cpp.
            if (last_real.time_since_epoch().count() != 0) {
                const Ns sample = std::chrono::duration_cast<Ns>(now - last_real);
                interval = interval.count() == 0 || sample > interval * 3 ? sample : (interval * 7 + sample) / 8;
            }
            last_real = now;
            const bool wanted = interval >= kVblank * 9 / 5 && interval <= std::chrono::milliseconds {150};
            if (wanted) {
                r.events.push_back({k, 'G', f});
                pending = true;
                pending_frame = f;
                due = now + rules.Delay(interval);
            } else {
                r.events.push_back({k, 'R', f});
            }
            continue;
        }
        // videoOut: no flip this vblank -> PresentPendingFrame.
        if (pending && now >= due) {
            r.events.push_back({k, 'R', pending_frame});
            pending = false;
            continue;
        }
        // videoOut.cpp: nothing presented and the overlay needs a refresh -> PresentLastFrame,
        // which presents layers[0] (the waiting real, if any).
        if (!overlay.empty() && overlay[k % overlay.size()] && main_frame >= 0) {
            if (pending && rules.overlay_fix) {
                r.events.push_back({k, 'R', pending_frame});
                pending = false;
            } else {
                r.events.push_back({k, 'P', main_frame});
            }
        }
    }
    r.frames = next_frame;
    return r;
}

// Every frame's real image is presented exactly once, in frame order, after its generated frame.
static bool EveryRealOnceInOrder(const Result& r) {
    std::vector<int> real_count(r.frames, 0);
    int last_real = -1;
    for (const auto& e : r.events) {
        if (e.kind != 'R') continue;
        if (e.frame <= last_real) return false;
        last_real = e.frame;
        ++real_count[e.frame];
    }
    return r.drops == 0 && std::ranges::all_of(real_count, [](int c) { return c == 1; });
}

// For generated frames: their real follows exactly `vblanks` later, with nothing in between.
static bool RealOneVblankAfterGenerated(const Result& r, int vblanks) {
    for (size_t i = 0; i < r.events.size(); ++i) {
        if (r.events[i].kind != 'G') continue;
        if (i + 1 >= r.events.size()) return false;
        const auto& next = r.events[i + 1];
        if (next.kind != 'R' || next.frame != r.events[i].frame || next.vblank - r.events[i].vblank != vblanks)
            return false;
    }
    return true;
}

// Diag kinds match the image: a Real/DelayedReal present shows a not-yet-shown game image, a
// Repeat shows one already shown (no early real labelled Repeat, no duplicate delayed real).
static bool HonestKinds(const Result& r) {
    std::vector<bool> shown(r.frames, false);
    for (const auto& e : r.events) {
        if (e.kind == 'G') continue;
        if (e.kind == 'R') {
            if (shown[e.frame]) return false;
            shown[e.frame] = true;
        } else if (!shown[e.frame]) {
            return false;
        }
    }
    return true;
}

static int GeneratedCount(const Result& r) {
    return static_cast<int>(std::ranges::count_if(r.events, [](const Event& e) { return e.kind == 'G'; }));
}

static std::vector<int64_t> Jitter(int count, int64_t amplitude_ns) {
    std::vector<int64_t> out;
    uint32_t state = 0x12345678u;
    for (int i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        out.push_back(static_cast<int64_t>(state % static_cast<uint32_t>(2 * amplitude_ns + 1)) - amplitude_ns);
    }
    return out;
}

static void Run(const char* name, const std::vector<int>& gaps, const std::vector<int64_t>& jitter, int frames,
                int expected_delay_vblanks) {
    const auto legacy = Simulate(Rules {true}, gaps, jitter, frames);
    const auto fixed  = Simulate(Rules {false}, gaps, jitter, frames);
    const bool legacy_ok = EveryRealOnceInOrder(legacy) &&
                           (expected_delay_vblanks == 0 || RealOneVblankAfterGenerated(legacy, expected_delay_vblanks));
    const bool fixed_ok  = EveryRealOnceInOrder(fixed) &&
                           (expected_delay_vblanks == 0 || RealOneVblankAfterGenerated(fixed, expected_delay_vblanks));
    // Guest flips are never deferred: frame i flips at the vblank it is ready in both runs.
    CHECK(legacy.flip_vblank == fixed.flip_vblank && fixed.frames == frames);
    CHECK(GeneratedCount(fixed) > frames / 2); // FG actually engaged in the scenario
    CHECK(HonestKinds(fixed));
    std::printf("%-28s legacy drops=%-3d %s | fixed drops=%-3d %s\n", name, legacy.drops, legacy_ok ? "ok " : "RED",
                fixed.drops, fixed_ok ? "GREEN" : "FAIL");
    CHECK(fixed_ok);
}

int main() {
    // Delay rule: nearest vblank to the midpoint, never negative.
    CHECK(PendingRealDelay(std::chrono::milliseconds {40}, kVblank) < kVblank);
    CHECK(PendingRealDelay(kVblank * 2, kVblank) == kVblank / 2);
    CHECK(PendingRealDelay(kVblank, kVblank).count() == 0 && PendingRealDelay(kVblank / 2, kVblank).count() == 0);
    const auto t = Clock::time_point {} + std::chrono::seconds {5};
    CHECK(FlushPendingReal(true, t, t) && !FlushPendingReal(true, t, t + Ns {1}) && !FlushPendingReal(false, t, t));

    // 25 FPS on 60 Hz: 2/3-vblank flips. Legacy holds every real 2 vblanks and drops it whenever
    // the next flip comes 2 vblanks later.
    {
        const std::vector<int> gaps {2, 3};
        const auto legacy = Simulate(Rules {true}, gaps, {}, 60);
        CHECK(legacy.drops > 0); // RED: real frames released unpresented
        Run("25fps 2/3 vblanks", gaps, {}, 60, 1);
    }
    // 30 FPS with +-0.5 ms loop jitter: legacy misses vblank +1 about half the time and collides.
    {
        const std::vector<int> gaps {2};
        const auto jitter = Jitter(97, 500'000);
        const auto legacy = Simulate(Rules {true}, gaps, jitter, 60);
        CHECK(legacy.drops > 0); // RED
        Run("30fps jitter", gaps, jitter, 60, 1);
    }
    // Pending collision: an occasional 1-vblank flip while the previous real is due. Legacy drops
    // it; the product presents it before releasing (R_n then G_n+1, same vblank).
    {
        const std::vector<int> gaps {2, 3, 1, 3, 2, 3, 1};
        const auto legacy = Simulate(Rules {true}, gaps, {}, 70);
        const auto fixed  = Simulate(Rules {false}, gaps, {}, 70);
        CHECK(legacy.drops > 0); // RED
        CHECK(EveryRealOnceInOrder(fixed)); // GREEN
        bool flushed_same_vblank = false;
        for (size_t i = 0; i + 1 < fixed.events.size(); ++i)
            flushed_same_vblank |= fixed.events[i].kind == 'R' && fixed.events[i + 1].kind == 'G' &&
                                   fixed.events[i].vblank == fixed.events[i + 1].vblank &&
                                   fixed.events[i].frame + 1 == fixed.events[i + 1].frame;
        CHECK(flushed_same_vblank);
        std::printf("%-28s legacy drops=%-3d RED | fixed drops=%-3d GREEN\n", "pending collision", legacy.drops,
                    fixed.drops);
    }
    // Slow producer (12 FPS, 5 vblanks, still within the 150 ms window): no drops either way, the
    // product shows the real at the vblank nearest the midpoint (2 vblanks after its generated).
    {
        const std::vector<int> gaps {5};
        const auto legacy = Simulate(Rules {true}, gaps, {}, 30);
        const auto fixed  = Simulate(Rules {false}, gaps, {}, 30);
        CHECK(EveryRealOnceInOrder(legacy) && EveryRealOnceInOrder(fixed));
        CHECK(RealOneVblankAfterGenerated(fixed, 2) && RealOneVblankAfterGenerated(legacy, 3));
        CHECK(legacy.flip_vblank == fixed.flip_vblank);
    }
    // rev2 overlay refresh before real_due (12 FPS, real due 2 vblanks after its generated frame).
    // v1 (and legacy) present the waiting real early as Repeat at +1 and again as DelayedReal at
    // +2; rev2 presents it once, as DelayedReal, at +1. Guest flips and generation are unchanged.
    {
        const std::vector<int> gaps {5};
        const std::vector<bool> every {true};
        const std::vector<bool> once {false, true, false, false, false, false, false, false, false, false};
        for (const auto* overlay : {&every, &once}) {
            const auto legacy = Simulate(Rules {true, false}, gaps, {}, 30, *overlay);
            const auto v1     = Simulate(Rules {false, false}, gaps, {}, 30, *overlay);
            const auto rev2   = Simulate(Rules {false, true}, gaps, {}, 30, *overlay);
            CHECK(!HonestKinds(legacy) && !HonestKinds(v1));        // RED: early Repeat + duplicate
            CHECK(HonestKinds(rev2) && EveryRealOnceInOrder(rev2)); // GREEN
            CHECK(rev2.flip_vblank == v1.flip_vblank && GeneratedCount(rev2) == GeneratedCount(v1));
            int early = 0, repeats = 0;
            for (size_t i = 0; i + 1 < rev2.events.size(); ++i) {
                const auto& g = rev2.events[i];
                const auto& n = rev2.events[i + 1];
                early += g.kind == 'G' && n.kind == 'R' && n.frame == g.frame && n.vblank == g.vblank + 1;
            }
            for (const auto& e : rev2.events) repeats += e.kind == 'P';
            CHECK(early > 0);                                       // the pre-due refresh happened
            CHECK(overlay == &once || repeats > 0);                 // later refreshes stay Repeat
            std::printf("%-28s v1 honest=%d RED | rev2 honest=%d early_delayed=%d repeats=%d GREEN\n",
                        overlay == &every ? "overlay every vblank" : "overlay once pre-due", HonestKinds(v1) ? 1 : 0,
                        HonestKinds(rev2) ? 1 : 0, early, repeats);
        }
    }
    // rev2 reset seed: Generate writes its output and returns false on the first frame after a
    // reset. Layout follows the write (below); diag counts it as gen_seeded, not gen_failed.
    {
        using Libs::Graphics::FrameGenerationDiag::Stats;
        Stats legacy, fixed;
        struct Gen { bool ok, written; };
        for (const Gen g : {Gen {false, true}, Gen {true, true}, Gen {true, true}, Gen {false, false}}) {
            legacy.NoteGeneration(g.ok); // the v1 call site
            fixed.NoteGeneration(g.ok, g.written);
        }
        // Observe counters through the product's public reporting path; no private test access.
        using Libs::Graphics::FrameGenerationDiag::PresentKind;
        char legacy_line[1024], line[1024];
        CHECK(!legacy.NotePresented(PresentKind::Blank, 1, legacy_line, sizeof(legacy_line)));
        CHECK(legacy.NotePresented(PresentKind::Blank, 2'000'000'001ull, legacy_line, sizeof(legacy_line)));
        const std::string legacy_text(legacy_line);
        CHECK(legacy_text.rfind("FG_DIAG v=1 window_ms=2000 frames=0 requested=0 eligible=0 gen_recorded=2 gen_failed=2 ", 0) == 0);
        const std::string no_seed = " source=callsite scanout=unknown gen_seeded=0";
        CHECK(legacy_text.size() > no_seed.size() &&
              legacy_text.compare(legacy_text.size() - no_seed.size(), no_seed.size(), no_seed) == 0); // RED v1: seed counted as failed
        CHECK(!fixed.NotePresented(PresentKind::Blank, 1, line, sizeof(line)));
        CHECK(fixed.NotePresented(PresentKind::Blank, 2'000'000'001ull, line, sizeof(line)));
        const std::string text(line);
        CHECK(text.rfind("FG_DIAG v=1 window_ms=2000 frames=0 requested=0 eligible=0 gen_recorded=2 gen_failed=1 ", 0) == 0);
        const std::string tail = " source=callsite scanout=unknown gen_seeded=1";
        CHECK(text.size() > tail.size() && text.compare(text.size() - tail.size(), tail.size(), tail) == 0);
        // The next public report proves the completed window reset ALL generation counters.
        CHECK(fixed.NotePresented(PresentKind::Blank, 4'000'000'001ull, line, sizeof(line)));
        const std::string reset_text(line);
        CHECK(reset_text.rfind("FG_DIAG v=1 window_ms=2000 frames=0 requested=0 eligible=0 gen_recorded=0 gen_failed=0 ", 0) == 0);
        CHECK(reset_text.size() > no_seed.size() &&
              reset_text.compare(reset_text.size() - no_seed.size(), no_seed.size(), no_seed) == 0);
        std::printf("%-28s v1 gen_failed=2 RED | rev2 recorded=2 failed=1 seeded=1 GREEN\n",
                    "reset seed accounting");
    }
    // Lifetime: a superseded/failed real or generated frame keeps the generation submission tick.
    {
        uint64_t real_tick = 5, generated_tick = 3; // stale ticks from earlier presents
        const uint64_t generation = 9;
        const uint64_t legacy_real_wait = real_tick; // legacy never stamped the generation tick
        CHECK(legacy_real_wait < generation);        // RED: pool reuse would not wait for generation
        real_tick      = RetainTick(real_tick, generation);
        generated_tick = RetainTick(generated_tick, generation);
        CHECK(real_tick == generation && generated_tick == generation); // GREEN
        CHECK(RetainTick(12, generation) == 12); // a later present tick is never lowered
    }
    // Reset/failure image state: Generate may write the output and still return false (first frame
    // after a reset); the recorded layout must follow the writes, not the return value.
    {
        enum Layout { Old, TransferSrc };
        struct Case { bool ok, written; Layout actual; };
        for (const Case c : {Case {true, true, TransferSrc}, Case {false, true, TransferSrc}, Case {false, false, Old}}) {
            const Layout legacy = c.ok ? TransferSrc : Old;
            const Layout fixed  = c.written ? TransferSrc : Old;
            CHECK(fixed == c.actual);
            if (!c.ok && c.written) CHECK(legacy != c.actual); // RED: reset frame left a stale layout
        }
    }
    std::printf("%s fg_pacing_fixture (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

// Standalone CPU fixture for frameGenerationDiag.h (std-only): fake present sequences, no GPU/WSI.
// Modes: "on" (KYTY_FG_DIAG=1) and "off" (unset). See RUNNER.txt.
#include "graphics/presentation/frameGenerationDiag.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace Libs::Graphics::FrameGenerationDiag;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static constexpr uint64_t Ms = 1'000'000ull;

static bool Has(const std::string& s, const char* t) { return s.find(t) != std::string::npos; }

static void Parsing() {
    CHECK(ParseEnabled("1"));
    CHECK(!ParseEnabled(nullptr) && !ParseEnabled("") && !ParseEnabled("0") && !ParseEnabled("10") &&
          !ParseEnabled("1 ") && !ParseEnabled("true"));
}

static void Sequence() {
    Stats stats;
    char line[1024];
    const uint64_t t0 = 1'000'000'000ull;
    // New frame 1: requested+eligible, generation recorded; generated then delayed real presented.
    stats.NoteRequest(true, true);
    stats.NoteGeneration(true);
    CHECK(!stats.NotePresented(PresentKind::Generated, t0, line, sizeof(line)));
    CHECK(!stats.NotePresented(PresentKind::DelayedReal, t0 + 20 * Ms, line, sizeof(line)));
    // New frame 2 replaces a still-pending delayed real; only its generated frame is presented.
    stats.NoteRequest(true, true);
    stats.NoteGeneration(true);
    stats.NoteSuperseded();
    CHECK(!stats.NotePresented(PresentKind::Generated, t0 + 40 * Ms, line, sizeof(line)));
    // New frame 3: requested but not eligible -> shown immediately as real.
    stats.NoteRequest(true, false);
    CHECK(!stats.NotePresented(PresentKind::Real, t0 + 60 * Ms, line, sizeof(line)));
    CHECK(!stats.NotePresented(PresentKind::Repeat, t0 + 70 * Ms, line, sizeof(line)));
    CHECK(!stats.NotePresented(PresentKind::Blank, t0 + 80 * Ms, line, sizeof(line)));
    stats.NoteFailed(PresentKind::Generated);
    stats.NoteRecreate(); // resize: the next interval spans a recreation and is not sampled
    CHECK(!stats.NotePresented(PresentKind::Real, t0 + 100 * Ms, line, sizeof(line)));
    CHECK(!stats.NotePresented(PresentKind::Real, t0 + 116 * Ms, line, sizeof(line)));
    // Closes the 2 s window. Samples: 20,20,20,10,10,16,1884 ms (n=7).
    CHECK(stats.NotePresented(PresentKind::Real, t0 + 2000 * Ms, line, sizeof(line)));
    const std::string text(line);
    CHECK(text.rfind("FG_DIAG v=1 window_ms=2000 frames=3 requested=3 eligible=2 gen_recorded=2 gen_failed=0 "
                     "superseded_real=1 ok_real=4 ok_generated=2 ok_delayed_real=1 ok_repeat=1 ok_blank=1 "
                     "fail_real=0 fail_generated=1 fail_delayed_real=0 fail_repeat=0 fail_blank=0 recreates=1 "
                     "interval_us n=7 p50=20000 p90=1884000 p99=1884000 max=1884000 dropped=0 "
                     "source=callsite scanout=unknown", 0) == 0);
    CHECK(text.find('\n') == std::string::npos);
    std::printf("sample: %s\n", line);

    // Next window: counters restarted; 600 presents at 1 ms overflow the 512 fixed samples.
    const uint64_t w1 = t0 + 2000 * Ms;
    for (uint64_t i = 1; i <= 600; ++i)
        CHECK(!stats.NotePresented(PresentKind::Real, w1 + i * Ms, line, sizeof(line)));
    CHECK(stats.NotePresented(PresentKind::Real, w1 + 2000 * Ms, line, sizeof(line)));
    const std::string overflow(line);
    CHECK(Has(overflow, " frames=0 requested=0 eligible=0 gen_recorded=0 gen_failed=0 superseded_real=0 ok_real=601 "
                        "ok_generated=0 ok_delayed_real=0 ok_repeat=0 ok_blank=0 "));
    CHECK(Has(overflow, " recreates=0 interval_us n=512 p50=1000 p90=1000 p99=1000 max=1000 dropped=89 "));

    // Empty interval set reports na, not zeros.
    Stats single;
    CHECK(single.NotePresented(PresentKind::Repeat, 5, line, sizeof(line)) == false);
    CHECK(single.NotePresented(PresentKind::Repeat, 5 + 2000 * Ms, line, sizeof(line)));
    CHECK(Has(line, "ok_repeat=2") && Has(line, "interval_us n=1 p50=2000000"));
    Stats empty;
    empty.NoteRecreate();
    CHECK(!empty.NotePresented(PresentKind::Blank, 7, line, sizeof(line)));
    empty.NoteRecreate();
    CHECK(empty.NotePresented(PresentKind::Blank, 7 + 2001 * Ms, line, sizeof(line)));
    CHECK(Has(line, "window_ms=2001") && Has(line, "recreates=2 interval_us n=0 p50=na p90=na p99=na max=na"));

    // A too-small buffer gets the bounded truncation marker prefix, never an overrun.
    Stats tiny;
    char small[16];
    std::memset(small, 'x', sizeof(small));
    CHECK(!tiny.NotePresented(PresentKind::Real, 9, small, sizeof(small)));
    CHECK(tiny.NotePresented(PresentKind::Real, 9 + 2000 * Ms, small, sizeof(small)));
    CHECK(std::strlen(small) == sizeof(small) - 1u && std::strncmp(small, "FG_DIAG v=1 ", 12) == 0);
}

int main(int argc, char** argv) {
    const bool on = argc >= 2 && std::strcmp(argv[1], "on") == 0;
    const bool off = argc >= 2 && std::strcmp(argv[1], "off") == 0;
    if (!on && !off) { std::printf("usage: fg_diag_fixture on|off\n"); return 2; }
    CHECK(Enabled() == on); // product hooks are all gated by Enabled(): OFF collects/formats nothing
    Parsing();
    Sequence();
    std::printf("%s fg_diag_fixture %s (%d failures)\n", g_fail ? "FAIL" : "PASS", on ? "on" : "off", g_fail);
    return g_fail ? 1 : 0;
}

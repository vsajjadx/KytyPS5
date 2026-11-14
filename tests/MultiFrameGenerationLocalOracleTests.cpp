#include "MultiFrameGenerationLocalOracle.h"
#include <cstdio>
int main() {
    namespace Local = MultiFgLocalOracle;
    int failures = 0;
    const auto check = [&](bool okay, const char* label) {
        if (!okay) { ++failures; std::printf("FAIL %s\n", label); }
    };
    for (const auto kind : {Local::Kind::OppositePanels, Local::Kind::Occlusion}) {
        const Local::Scene scene {kind};
        for (uint32_t factor = 2; factor <= 6; ++factor)
            for (uint32_t ordinal = 1; ordinal < factor; ++ordinal) {
                const double t = double(ordinal) / factor;
                const auto reference = Local::Frame(scene, 11 + t);
                check(Local::AllOkay(Local::Position(Local::Packed(reference, scene), scene, factor, ordinal)),
                      "exact independent reference rejected");
                const auto known = Local::KnownWarp(scene, t);
                const auto results = Local::Position(Local::Packed(known, scene), scene, factor, ordinal);
                check(Local::AllOkay(results), "per-region known vectors rejected");
                const auto blend = Local::Blend(scene, t);
                check(!Local::AllOkay(Local::Position(Local::Packed(blend, scene), scene, factor, ordinal)),
                      "endpoint crossfade accepted");
                const auto previous = Local::Frame(scene, 11);
                check(!Local::AllOkay(Local::Position(Local::Packed(previous, scene), scene, factor, ordinal)),
                      "previous endpoint/zero-global-median accepted");
                const auto global = Local::KnownWarp(scene, t, true, 12, -6);
                check(!Local::AllOkay(Local::Position(Local::Packed(global, scene), scene, factor, ordinal)),
                      "one global motion accepted for different local motions");
                for (const auto& r : results)
                    std::printf("LOCAL_CPU scene=%u factor=%u ordinal=%u region=%s valid=%llu mae=%.6f rmse=%.6f\n",
                        static_cast<unsigned>(kind), factor, ordinal, Local::Name(r.region),
                        static_cast<unsigned long long>(r.pixels), r.mae, r.rmse);
            }
    }
    std::printf("%s local oracle (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

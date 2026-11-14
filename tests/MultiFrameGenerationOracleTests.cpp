#include "MultiFrameGenerationOracle.h"
#include <cstdio>

int main() {
    using namespace MultiFgOracle;
    int failures = 0;
    const auto check = [&](bool okay, const char* what) {
        if (!okay) { ++failures; std::printf("FAIL %s\n", what); }
    };
    for (uint32_t variant = 0; variant < 2; ++variant) {
        Scene scene {257, 131, variant, variant ? -12.0 : 12.0, variant ? 6.0 : -6.0, 11};
        const auto previous = Frame(scene, scene.previous_frame);
        const auto current = Frame(scene, scene.previous_frame + 1);
        for (uint32_t factor = 2; factor <= 6; ++factor) {
            std::vector<std::vector<Pixel>> exact;
            for (uint32_t i = 1; i < factor; ++i) {
                exact.push_back(Frame(scene, scene.previous_frame + double(i) / factor));
                check(Position(Packed(exact.back(), scene), scene, factor, i).okay, "correct reference rejected");
                check(!Position(Packed(previous, scene), scene, factor, i).okay, "previous endpoint accepted");
                check(!Position(Packed(current, scene), scene, factor, i).okay, "current endpoint accepted");
                const auto blend = Blend(scene, double(i) / factor);
                check(!Position(Packed(blend, scene), scene, factor, i).okay, "endpoint cross-fade accepted");
            }
            for (uint32_t i = 1; i < factor; ++i)
                for (uint32_t j = 1; j < factor; ++j)
                    if (i != j) check(!Position(Packed(exact[j - 1], scene), scene, factor, i).okay,
                                      "duplicate/reversed/wrong temporal position accepted");
        }
        std::vector<Pixel> padded(size_t(scene.width + 19) * (scene.height + 7), Canary);
        for (uint32_t y = 0; y < scene.height; ++y)
            std::copy_n(current.data() + size_t(y) * scene.width, scene.width,
                        padded.data() + size_t(y) * (scene.width + 19));
        View physical {padded, scene.width + 19, scene.height + 7, scene.width + 19};
        check(OutsideLogicalUntouched(physical, scene.width, scene.height), "unchanged output border rejected");
        padded[scene.width] ^= 1;
        check(!OutsideLogicalUntouched(physical, scene.width, scene.height), "OOB border write accepted");
    }
    check(!Valid({{}, 1, 1, 1}), "empty view accepted");
    std::printf("%s MultiFrameGenerationOracle (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

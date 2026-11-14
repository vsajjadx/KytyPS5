#include "MultiFrameGenerationHudOracle.h"
#include <cstdio>
int main() {
    namespace Hud = MultiFgHudOracle;
    int failures = 0;
    const auto check = [&](bool okay,const char *label) {
        if(!okay){++failures;std::printf("FAIL HUD_CPU %s\n",label);}
    };
    uint64_t support=0,edge=0,ungraded=0;
    for(uint32_t y=0;y<Hud::Height;++y)for(uint32_t x=0;x<Hud::Width;++x) {
        const auto a=Hud::Hud()[size_t(y)*Hud::Width+x].alpha;
        if(a) {
            ++support; if(a!=255)++edge;
            const unsigned masks=unsigned(Hud::InMask(Hud::Region::GlyphOpaque,x,y))+
                unsigned(Hud::InMask(Hud::Region::Antialias,x,y))+
                unsigned(Hud::InMask(Hud::Region::PanelOpaque,x,y));
            if(masks!=1)++ungraded;
        }
    }
    check(support>10'000 && edge>1'000 && ungraded==0,"every static HUD/letter-edge pixel graded exactly once");
    for(uint32_t factor=2;factor<=6;++factor)for(uint32_t ordinal=1;ordinal<factor;++ordinal) {
        const double t=double(ordinal)/factor;
        const auto coverage=Hud::CoverageAt(t);
        check(coverage.ungraded_hud==0&&coverage.hud==support&&coverage.antialias==edge&&
              coverage.hud+coverage.background_observed+coverage.background_unobservable+
                  coverage.background_border_excluded==uint64_t(Hud::Width)*Hud::Height,
              "coverage partition excludes HUD or omits pixels");
        check(coverage.background_unobservable>0,"endpoint-HUD occlusion strips were not accounted");
        const auto &truth=Hud::Reference(factor,ordinal);
        const auto known=Hud::KnownLayerWarp(t);
        const auto positive=Hud::Position(Hud::Packed(known),factor,ordinal);
        check(Hud::AllOkay(Hud::Position(Hud::Packed(truth),factor,ordinal)),"exact fractional composite rejected");
        check(Hud::AllOkay(positive),"known separate-layer zero-HUD/moving-BG vectors rejected");
        const auto moving_hud=Hud::GlobalCompositeWarp(t,Hud::Dx,Hud::Dy);
        const auto moving_metrics=Hud::Position(Hud::Packed(moving_hud),factor,ordinal);
        check(!moving_metrics[0].okay && !moving_metrics[1].okay,
              "global background flow applied to HUD did not fail opaque and AA ROIs");
        const auto global_mean=Hud::GlobalCompositeWarp(t,Hud::Dx/2,Hud::Dy/2);
        const auto mean_metrics=Hud::Position(Hud::Packed(global_mean),factor,ordinal);
        check(!mean_metrics[4].okay,"one global mean vector accepted as true BG temporal position");
        for(uint32_t endpoint : {0u,factor}) {
            const auto result=Hud::Position(Hud::Packed(Hud::Reference(factor,endpoint)),factor,ordinal);
            check(!result[4].okay,"previous/current endpoint accepted in visible moving BG");
        }
        const auto blend=Hud::Blend(t);
        const auto blend_metrics=Hud::Position(Hud::Packed(blend),factor,ordinal);
        check(!blend_metrics[4].okay,"endpoint crossfade accepted in visible moving BG");
        auto broken_alpha=truth;
        for(uint32_t y=Hud::Margin;y<Hud::Height-Hud::Margin;++y)
            for(uint32_t x=Hud::Margin;x<Hud::Width-Hud::Margin;++x)
                if(Hud::InMask(Hud::Region::Antialias,x,y))broken_alpha[size_t(y)*Hud::Width+x]&=0x00ffffffu;
        check(!Hud::Position(Hud::Packed(broken_alpha),factor,ordinal)[1].okay,
              "nonopaque composited output alpha accepted on the antialiased ROI");
        for(const auto &metric:positive)
            std::printf("HUD_CPU factor=%u ordinal=%u region=%s pixels=%llu mae=%.6f rmse=%.6f okay=%u\n",
                factor,ordinal,Hud::Name(metric.region),static_cast<unsigned long long>(metric.pixels),
                metric.mae,metric.rmse,metric.okay?1u:0u);
    }
    std::printf("%s static-HUD oracle support=%llu edge=%llu ungraded=%llu failures=%d\n",
        failures?"FAIL":"PASS",static_cast<unsigned long long>(support),
        static_cast<unsigned long long>(edge),static_cast<unsigned long long>(ungraded),failures);
    return failures?1:0;
}

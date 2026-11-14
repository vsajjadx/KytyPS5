#pragma once
// Independent synthetic layer oracle. The emulator receives only composited
// RGBA8 endpoints; this private test mask/layer decomposition is never supplied.
#include "MultiFrameGenerationOracle.h"
#include <limits>

namespace MultiFgHudOracle {
using namespace MultiFgOracle;
constexpr uint32_t Width = 576, Height = 288;
constexpr double Dx = 12, Dy = -6;
constexpr uint32_t PreviousFrame = 11, Margin = 32;
enum class Region { GlyphOpaque, Antialias, PanelOpaque, NearBackground, VisibleBackground };
struct HudPixel { Pixel color = 0; uint8_t alpha = 0; bool panel = false; };
struct Metric {
    Region region;
    uint64_t pixels = 0;
    double mae = 0, rmse = 0, alternate_mse = INFINITY, blend_mse = INFINITY;
    bool okay = false;
};
inline const char *Name(Region region) {
    switch (region) {
    case Region::GlyphOpaque: return "opaque_glyph";
    case Region::Antialias: return "antialiased_edge";
    case Region::PanelOpaque: return "opaque_panel";
    case Region::NearBackground: return "near_hud_background";
    case Region::VisibleBackground: return "visible_background";
    }
    return "invalid";
}
// Authored five-by-seven glyph cells. Subpixel placement and noninteger scale
// produce known coverage without a font renderer, external font or game asset.
inline uint8_t Glyph(char c, uint32_t row) {
    constexpr std::array<uint8_t, 7> S {31,16,16,31,1,1,31}, T {31,4,4,4,4,4,4},
        A {14,17,17,31,17,17,17}, I {31,4,4,4,4,4,31}, C {15,16,16,16,16,16,15},
        H {17,17,17,31,17,17,17}, U {17,17,17,17,17,17,14}, D {30,17,17,17,17,17,30},
        N2 {30,1,1,14,16,16,31}, N4 {17,17,17,31,1,1,1}, N6 {15,16,16,31,17,17,31};
    switch (c) {
    case 'S': return S[row]; case 'T': return T[row]; case 'A': return A[row];
    case 'I': return I[row]; case 'C': return C[row]; case 'H': return H[row];
    case 'U': return U[row]; case 'D': return D[row]; case '2': return N2[row];
    case '4': return N4[row]; case '6': return N6[row]; default: return 0;
    }
}
inline bool Text(double x, double y, double top, const char *letters, uint32_t length) {
    constexpr double left = 52.25, scale = 3.5, advance = 6 * scale;
    if (x < left || y < top || y >= top + 7 * scale || x >= left + length * advance) return false;
    const uint32_t at = uint32_t((x - left) / advance);
    const double local = x - left - at * advance;
    if (local >= 5 * scale) return false;
    const uint32_t column = uint32_t(local / scale), row = uint32_t((y - top) / scale);
    return (Glyph(letters[at], row) & (1u << (4u - column))) != 0;
}
inline bool Line(double x, double y, double ay, double by) {
    constexpr double ax = 42.375, bx = 354.625, half_width = .8125;
    const double vx = bx - ax, vy = by - ay;
    const double t = std::clamp(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0);
    const double ex = x - ax - t * vx, ey = y - ay - t * vy;
    return ex * ex + ey * ey <= half_width * half_width;
}
inline uint32_t Category(double x, double y) {
    if (x >= 404.25 && x < 531.75 && y >= 44.375 && y < 128.625) return 4;
    if (Text(x, y, 52.375, "STATIC HUD", 10)) return 1;
    if (Text(x, y, 132.125, "HUD 246", 7)) return 2;
    if (Text(x, y, 204.625, "STATIC 246", 10)) return 3;
    if (Line(x, y, 98.375, 101.625)) return 1;
    if (Line(x, y, 181.625, 177.375)) return 2;
    if (Line(x, y, 246.25, 243.625)) return 3;
    return 0;
}
inline Pixel Color(uint32_t category, uint32_t x, uint32_t y) {
    if (category == 1) return Pack(246, 248, 251);
    if (category == 2) return Pack(250, 210, 36);
    if (category == 3) return Pack(30, 230, 250);
    // A static, nonuniform opaque HUD panel tests a genuine zero-motion region,
    // rather than inferring zero flow from an aperture-only uniform rectangle.
    return ((x / 3 + y / 5) & 1u) ? Pack(232, 58, 38) : Pack(28, 222, 238);
}
inline const std::vector<HudPixel> &Hud() {
    static const auto pixels = [] {
        std::vector<HudPixel> result(size_t(Width) * Height);
        for (uint32_t y = 0; y < Height; ++y) for (uint32_t x = 0; x < Width; ++x) {
            std::array<uint32_t, 5> coverage {};
            for (uint32_t sy = 0; sy < 8; ++sy) for (uint32_t sx = 0; sx < 8; ++sx)
                ++coverage[Category(x + (sx + .5) / 8, y + (sy + .5) / 8)];
            const uint32_t samples = 64 - coverage[0];
            if (!samples) continue;
            std::array<double, 3> rgb {};
            for (uint32_t category = 1; category <= 4; ++category)
                for (uint32_t channel = 0; channel < 3; ++channel)
                    rgb[channel] += coverage[category] * Byte(Color(category, x, y), channel);
            auto &pixel = result[size_t(y) * Width + x];
            pixel.color = Pack(rgb[0] / samples, rgb[1] / samples, rgb[2] / samples);
            pixel.alpha = uint8_t(std::lround(255.0 * samples / 64));
            pixel.panel = coverage[4] != 0;
        }
        return result;
    }();
    return pixels;
}
inline const std::vector<uint16_t> &Distance() {
    static const auto values = [] {
        std::vector<uint16_t> result(size_t(Width) * Height, uint16_t(4096));
        const auto &hud = Hud();
        for (size_t at = 0; at < hud.size(); ++at) if (hud[at].alpha) result[at] = 0;
        const auto update = [&](uint32_t x, uint32_t y, int nx, int ny) {
            if (nx >= 0 && ny >= 0 && nx < int(Width) && ny < int(Height))
                result[size_t(y) * Width + x] = std::min(result[size_t(y) * Width + x],
                    uint16_t(result[size_t(ny) * Width + nx] + 1));
        };
        for (uint32_t y = 0; y < Height; ++y) for (uint32_t x = 0; x < Width; ++x) {
            update(x,y,int(x)-1,y); update(x,y,x,int(y)-1);
            update(x,y,int(x)-1,int(y)-1); update(x,y,int(x)+1,int(y)-1);
        }
        for (int y = int(Height)-1; y >= 0; --y) for (int x = int(Width)-1; x >= 0; --x) {
            update(x,y,x+1,y); update(x,y,x,y+1); update(x,y,x+1,y+1); update(x,y,x-1,y+1);
        }
        return result;
    }();
    return values;
}
// BG truth is graded only when its actual bilinear source footprint is
// uncovered by static HUD in BOTH composited endpoints. This depends solely
// on input geometry, never measured GPU pixels or a failed metric.
inline bool BackgroundVisible(double x,double y) {
    if(x<0 || y<0 || x>Width-1 || y>Height-1)return false;
    const uint32_t x0=uint32_t(x),y0=uint32_t(y);
    const double fx=x-x0,fy=y-y0;
    for(uint32_t j=0;j<2;++j)for(uint32_t i=0;i<2;++i) {
        const double weight=(i?fx:1-fx)*(j?fy:1-fy);
        if(weight==0)continue;
        if(x0+i>=Width || y0+j>=Height || Hud()[size_t(y0+j)*Width+x0+i].alpha)return false;
    }
    return true;
}
inline bool BackgroundObserved(uint32_t x,uint32_t y,double t) {
    return BackgroundVisible(x-t*Dx,y-t*Dy) &&
        BackgroundVisible(x+(1-t)*Dx,y+(1-t)*Dy);
}
inline bool InMask(Region region, uint32_t x, uint32_t y, double t=.5) {
    if (x < Margin || y < Margin || x >= Width - Margin || y >= Height - Margin) return false;
    const auto at = size_t(y) * Width + x; const auto h = Hud()[at];
    switch (region) {
    case Region::GlyphOpaque: return h.alpha == 255 && !h.panel;
    case Region::Antialias: return h.alpha != 0 && h.alpha != 255;
    case Region::PanelOpaque: return h.alpha == 255 && h.panel;
    case Region::NearBackground: return h.alpha == 0 && Distance()[at] <= 12 && BackgroundObserved(x,y,t);
    case Region::VisibleBackground: return h.alpha == 0 && Distance()[at] > 12 && BackgroundObserved(x,y,t);
    }
    return false;
}
inline Pixel Composite(Pixel background, const HudPixel &hud) {
    const double a = double(hud.alpha) / 255;
    return Pack(a * Byte(hud.color,0) + (1-a) * Byte(background,0),
                a * Byte(hud.color,1) + (1-a) * Byte(background,1),
                a * Byte(hud.color,2) + (1-a) * Byte(background,2));
}
inline std::vector<Pixel> Background(double frame) {
    std::vector<Pixel> pixels(size_t(Width) * Height);
    for (uint32_t y = 0; y < Height; ++y) for (uint32_t x = 0; x < Width; ++x)
        pixels[size_t(y) * Width + x] = Sample(x - frame * Dx, y - frame * Dy, 0);
    return pixels;
}
inline std::vector<Pixel> Frame(double frame) {
    auto pixels = Background(frame); const auto &hud = Hud();
    for (size_t at = 0; at < pixels.size(); ++at) pixels[at] = Composite(pixels[at], hud[at]);
    return pixels;
}
inline View Packed(const std::vector<Pixel> &pixels) { return {pixels, Width, Height, Width}; }
inline const std::vector<Pixel> &Reference(uint32_t factor, uint32_t ordinal) {
    static std::array<std::array<std::vector<Pixel>, 7>, 7> cache;
    auto &pixels = cache[factor][ordinal];
    if (pixels.empty()) pixels = Frame(PreviousFrame + double(ordinal) / factor);
    return pixels;
}
inline std::vector<Pixel> Blend(double t) {
    const auto &a = Reference(2,0), &b = Reference(2,2);
    std::vector<Pixel> pixels(a.size());
    for (size_t at = 0; at < pixels.size(); ++at)
        pixels[at] = Pack((1-t)*Byte(a[at],0)+t*Byte(b[at],0),
                          (1-t)*Byte(a[at],1)+t*Byte(b[at],1),
                          (1-t)*Byte(a[at],2)+t*Byte(b[at],2));
    return pixels;
}
inline double MaskMse(View actual, const std::vector<Pixel> &reference, Region region, double t) {
    double error = 0; uint64_t count = 0;
    for (uint32_t y = Margin; y < Height-Margin; ++y)
        for (uint32_t x = Margin; x < Width-Margin; ++x) if (InMask(region,x,y,t))
            for (uint32_t c = 0; c < 3; ++c) {
                const double d = double(Byte(actual.words[size_t(y)*actual.stride+x],c))-
                    Byte(reference[size_t(y)*Width+x],c); error += d*d; ++count;
            }
    return count ? error/count : INFINITY;
}
inline std::array<Metric, 5> Position(View actual, uint32_t factor, uint32_t ordinal) {
    std::array<Metric, 5> result {{{Region::GlyphOpaque}, {Region::Antialias},
        {Region::PanelOpaque}, {Region::NearBackground}, {Region::VisibleBackground}}};
    if (factor<2 || factor>6 || ordinal==0 || ordinal>=factor || !Valid(actual) ||
        actual.width!=Width || actual.height!=Height) return result;
    const double t=double(ordinal)/factor;
    const auto &expected = Reference(factor,ordinal); const auto blend = Blend(t);
    for (auto &metric : result) {
        double absolute = 0; bool alpha = true;
        for (uint32_t y = Margin; y < Height-Margin; ++y)
            for (uint32_t x = Margin; x < Width-Margin; ++x) if (InMask(metric.region,x,y,t)) {
                const auto at = size_t(y)*Width+x;
                const auto p = actual.words[size_t(y)*actual.stride+x];
                alpha &= Byte(p,3)==255;
                for (uint32_t c=0;c<3;++c) absolute += std::abs(int(Byte(p,c))-int(Byte(expected[at],c)));
                ++metric.pixels;
            }
        const auto mse = MaskMse(actual,expected,metric.region,t);
        metric.mae = metric.pixels ? absolute/(3*metric.pixels) : INFINITY;
        metric.rmse = std::sqrt(mse); metric.blend_mse = MaskMse(actual,blend,metric.region,t);
        const uint64_t minimum = metric.region==Region::GlyphOpaque ? 1'000 :
            metric.region==Region::Antialias ? 1'000 : metric.region==Region::PanelOpaque ? 8'000 : 15'000;
        metric.okay = alpha && metric.pixels>=minimum && metric.mae<=8 && metric.rmse<=16;
        if (metric.region==Region::NearBackground || metric.region==Region::VisibleBackground) {
            for (uint32_t other=0;other<=factor;++other) if(other!=ordinal)
                metric.alternate_mse=std::min(metric.alternate_mse,MaskMse(actual,Reference(factor,other),metric.region,t));
            metric.okay &= mse < .60*metric.alternate_mse && mse+4 < .80*metric.blend_mse;
        }
    }
    return result;
}
struct Coverage {
    uint64_t hud=0,antialias=0,background_observed=0,background_unobservable=0,
        background_border_excluded=0,ungraded_hud=0;
};
inline Coverage CoverageAt(double t) {
    Coverage result;
    for(uint32_t y=0;y<Height;++y)for(uint32_t x=0;x<Width;++x) {
        const auto alpha=Hud()[size_t(y)*Width+x].alpha;
        const bool interior=x>=Margin&&y>=Margin&&x<Width-Margin&&y<Height-Margin;
        if(alpha){
            ++result.hud;if(alpha!=255)++result.antialias;
            if(!interior)++result.ungraded_hud;
        }else if(!interior)++result.background_border_excluded;
        else if(BackgroundObserved(x,y,t))++result.background_observed;
        else ++result.background_unobservable;
    }
    return result;
}
inline bool AllOkay(const std::array<Metric,5>& metrics) {
    return std::all_of(metrics.begin(),metrics.end(),[](const auto &m){return m.okay;});
}
// Bilinear CPU controls only; vectors/layers/masks never reach the renderer.
inline Pixel SamplePixels(View pixels,double x,double y) {
    x=std::clamp(x,0.0,double(pixels.width-1)); y=std::clamp(y,0.0,double(pixels.height-1));
    const uint32_t x0=uint32_t(x),y0=uint32_t(y),x1=std::min(x0+1,pixels.width-1),y1=std::min(y0+1,pixels.height-1);
    const double fx=x-x0,fy=y-y0; std::array<double,3> rgb{};
    for(uint32_t c=0;c<3;++c)
        rgb[c]=((1-fx)*Byte(pixels.words[size_t(y0)*pixels.stride+x0],c)+fx*Byte(pixels.words[size_t(y0)*pixels.stride+x1],c))*(1-fy)+
               ((1-fx)*Byte(pixels.words[size_t(y1)*pixels.stride+x0],c)+fx*Byte(pixels.words[size_t(y1)*pixels.stride+x1],c))*fy;
    return Pack(rgb[0],rgb[1],rgb[2]);
}
inline std::vector<Pixel> KnownLayerWarp(double t) {
    const auto background=Background(PreviousFrame); const auto view=Packed(background);
    std::vector<Pixel> output(background.size());
    for(uint32_t y=0;y<Height;++y)for(uint32_t x=0;x<Width;++x) {
        const auto at=size_t(y)*Width+x;
        output[at]=Composite(SamplePixels(view,x-t*Dx,y-t*Dy),Hud()[at]);
    }
    return output;
}
inline std::vector<Pixel> GlobalCompositeWarp(double t,double dx,double dy) {
    const auto &previous=Reference(2,0); const auto view=Packed(previous);
    std::vector<Pixel> output(previous.size());
    for(uint32_t y=0;y<Height;++y)for(uint32_t x=0;x<Width;++x)
        output[size_t(y)*Width+x]=SamplePixels(view,x-t*dx,y-t*dy);
    return output;
}
} // namespace MultiFgHudOracle

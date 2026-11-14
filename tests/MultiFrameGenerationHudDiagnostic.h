#pragma once
#include "MultiFrameGenerationHudOracle.h"
#include "MultiFrameGenerationDiagnostic.h"

namespace MultiFgHudDiagnostic {
inline std::filesystem::path Root() {
    const char *env=std::getenv("KYTY_MULTIFG_HUD_DUMP_DIR");
    if(!env||!*env)return {};
    const std::filesystem::path root(env),allowed("G:/ps4/Emu/kyty-perf");
    if(root.lexically_normal().parent_path()!=allowed.lexically_normal()||root.filename().empty()) {
        std::fprintf(stderr,"HUD dump requires a NEW immediate G:/ps4/Emu/kyty-perf child\n");std::abort();
    }
    static bool created=false;
    std::error_code error;
    if(!created){
        if(std::filesystem::exists(root,error)||error||!std::filesystem::create_directory(root,error)||error)std::abort();
        created=true;
    }
    return root;
}
inline void Save(uint32_t factor,uint32_t ordinal,const std::vector<uint32_t>& output) {
    const auto root=Root(); if(root.empty())return;
    const auto event=root/("factor"+std::to_string(factor)+"-ordinal"+std::to_string(ordinal));
    std::error_code error;
    if(!std::filesystem::create_directory(event,error)||error)std::abort();
    const auto &previous=MultiFgHudOracle::Reference(2,0),&current=MultiFgHudOracle::Reference(2,2);
    const auto &reference=MultiFgHudOracle::Reference(factor,ordinal);
    std::vector<uint32_t> alpha(previous.size());
    for(size_t at=0;at<alpha.size();++at){const auto a=MultiFgHudOracle::Hud()[at].alpha;alpha[at]=MultiFgOracle::Pack(a,a,a);}
    const bool saved=MultiFgDiagnostic::Ppm(event/"previous.ppm",MultiFgHudOracle::Packed(previous))&&
        MultiFgDiagnostic::Ppm(event/"current.ppm",MultiFgHudOracle::Packed(current))&&
        MultiFgDiagnostic::Ppm(event/"output.ppm",MultiFgHudOracle::Packed(output))&&
        MultiFgDiagnostic::Ppm(event/"reference.ppm",MultiFgHudOracle::Packed(reference))&&
        MultiFgDiagnostic::Ppm(event/"hud-alpha.ppm",MultiFgHudOracle::Packed(alpha));
    std::printf("MULTIFG_HUD_DUMP factor=%u ordinal=%u saved=%u path=%s\n",factor,ordinal,saved?1u:0u,event.string().c_str());
    std::fflush(stdout);if(!saved)std::abort();
}

struct FineCaptureInfo {
    uint64_t original_recorded=0,original_completed=0,copy_tick=0,after_recorded=0,after_completed=0;
    uint32_t original_state=0,original_flags=0,after_state=0,after_flags=0;
};
inline std::FILE* Exclusive(const std::filesystem::path& path) {
#ifdef _WIN32
    return _wfopen(path.c_str(),L"wbx");
#else
    return std::fopen(path.c_str(),"wbx");
#endif
}
inline bool Raw(const std::filesystem::path& path,const std::vector<uint32_t>& words) {
    auto *output=Exclusive(path);if(!output)return false;
    const bool written=std::fwrite(words.data(),sizeof(uint32_t),words.size(),output)==words.size();
    const bool flushed=std::fflush(output)==0;
    const bool closed=std::fclose(output)==0;
    return written&&flushed&&closed;
}
inline void SavePair(uint32_t factor,const std::array<std::vector<uint32_t>,2>& fields,
                     const FineCaptureInfo& info) {
    const auto root=Root();if(root.empty())return;
    constexpr uint64_t bytes=uint64_t(MultiFgHudOracle::Width)*MultiFgHudOracle::Height*16;
    if(fields[0].size()!=bytes/4||fields[1].size()!=bytes/4)std::abort();
    const auto event=root/("factor"+std::to_string(factor)+"-pair");
    std::error_code error;
    if(!std::filesystem::create_directory(event,error)||error)std::abort();
    const bool motion_saved=Raw(event/"fine-motion.rgba32i.tight.bin",fields[0]);
    const bool layer_saved=Raw(event/"static-layer.rgba32f.tight.bin",fields[1]);
    // P/C are known synthetic INPUT endpoints, not stored generator history readback.
    const auto &previous=MultiFgHudOracle::Reference(2,0),&current=MultiFgHudOracle::Reference(2,2);
    const bool endpoints_saved=MultiFgDiagnostic::Ppm(event/"previous-input-fixture.ppm",MultiFgHudOracle::Packed(previous))&&
        MultiFgDiagnostic::Ppm(event/"current-input-fixture.ppm",MultiFgHudOracle::Packed(current));
    auto *format=Exclusive(event/"format.json");if(!format)std::abort();
    const bool metadata_saved=std::fprintf(format,
        "{\n  \"schema\":\"kyty-multifg-hud-postcompleted-fine\",\"version\":1,\n"
        "  \"factor\":%u,\"width\":%u,\"height\":%u,\"layout\":\"tight row-major\",\"byte_order\":\"%s\",\n"
        "  \"original_recorded_tick\":%llu,\"original_completed_tick\":%llu,\"copy_tick\":%llu,\n"
        "  \"original_state\":%u,\"original_flags\":%u,\"after_state\":%u,\"after_flags\":%u,\n"
        "  \"after_recorded_tick\":%llu,\"after_completed_tick\":%llu,\n"
        "  \"endpoint_provenance\":\"synthetic input fixture endpoints11/12, not readback of stored generator history\",\n"
        "  \"motion\":{\"file\":\"fine-motion.rgba32i.tight.bin\",\"format\":\"R32G32B32A32_SINT\",\"bytes\":%llu,"
        "\"channels\":[\"primary_x\",\"primary_y\",\"alternate_x\",\"alternate_y\"],\"units\":\"current-to-previous pixel displacement\"},\n"
        "  \"layer\":{\"file\":\"static-layer.rgba32f.tight.bin\",\"format\":\"R32G32B32A32_SFLOAT\",\"bytes\":%llu,"
        "\"channels\":[\"K_r\",\"K_g\",\"K_b\",\"beta\"],\"meaning\":\"K.rgb stationary premultiplied contribution; beta moving-background weight; beta<0 invalid\"}\n}\n",
        factor,MultiFgHudOracle::Width,MultiFgHudOracle::Height,
        std::endian::native==std::endian::little?"little":"big",
        static_cast<unsigned long long>(info.original_recorded),static_cast<unsigned long long>(info.original_completed),
        static_cast<unsigned long long>(info.copy_tick),info.original_state,info.original_flags,info.after_state,info.after_flags,
        static_cast<unsigned long long>(info.after_recorded),static_cast<unsigned long long>(info.after_completed),
        static_cast<unsigned long long>(bytes),static_cast<unsigned long long>(bytes))>0;
    const bool metadata_flushed=std::fflush(format)==0,metadata_closed=std::fclose(format)==0;
    const bool saved=motion_saved&&layer_saved&&endpoints_saved&&metadata_saved&&metadata_flushed&&metadata_closed;
    std::printf("MULTIFG_HUD_FINE_DUMP factor=%u bytes_per_field=%llu copy_tick=%llu saved=%u path=%s\n",
        factor,static_cast<unsigned long long>(bytes),static_cast<unsigned long long>(info.copy_tick),saved?1u:0u,event.string().c_str());
    std::fflush(stdout);if(!saved)std::abort();
}
}

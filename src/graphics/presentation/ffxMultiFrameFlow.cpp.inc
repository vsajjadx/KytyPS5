// Copyright2026 KytyPS5 contributors. SPDX-License-Identifier: MIT
// Appended only in the build tree to the pinned MIT portable SDK context TU.
#include "graphics/presentation/ffxMultiFrameFlow.h"
extern "C" bool kytyFsrFinishOpticalFlow(FfxVkFsr3_3_1_6FrameGenerationContext* c,KytyFsrOpticalFlow* out) {
    if(!out)return false;*out={};
    if(!c||!c->initialized||!c->prepared||c->dispatched||!c->opticalFlowVector.image||!c->opticalFlowScd.image)return false;
    const auto describe=[](const OwnedSharedImage& i) {const auto& d=i.resource.description;return KytyFsrFlowImage{i.image,to_vk_format(static_cast<FfxApiSurfaceFormat>(d.format)),d.width,d.height,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT};};
    out->image=describe(c->opticalFlowVector);out->scene_change=describe(c->opticalFlowScd);out->frame_id=c->frameId;
    if(out->image.format!=VK_FORMAT_R16G16_SINT||out->scene_change.format!=VK_FORMAT_R32_UINT)return false;
    // Exactly the existing last-dispatch ownership handoff; OF already ran once in Prepare.
    c->retainedFrames.push_back({c->frameId,std::move(c->pendingImports)});
    c->color={};c->prepared=false;c->dispatched=false;c->renderWidth=0;c->renderHeight=0;
    return true;
}

// Scoped shader selection only during synchronous SDK context/pipeline creation.
// Nested/parallel legacy creation retains its prior per-thread mode on return.
extern "C" FfxVkFsr3_3_1_6FrameGenerationResult kytyFsrCreateSynchronizedMultiContext(
    const FfxVkFsr3_3_1_6FrameGenerationCreateInfo* info,
    FfxVkFsr3_3_1_6FrameGenerationContext** context) {
    struct Scope {
        bool previous=kyty_multi_scd_creation;
        Scope(){kyty_multi_scd_creation=true;}
        ~Scope(){kyty_multi_scd_creation=previous;}
    } scope;
    return ffxVkFsr3_3_1_6FrameGenerationContextCreate(info,context);
}

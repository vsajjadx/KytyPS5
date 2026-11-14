// Copyright2026 KytyPS5 contributors. SPDX-License-Identifier: GPL-2.0-only
// Independently authored motion reconstruction using the existing MIT AMD flow field.
#include "graphics/presentation/multiFrameGeneration.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include <cstring>
#include "gpu_tiler_shaders/multi_frame_warp_spv.h"
#include "gpu_tiler_shaders/multi_frame_refine_spv.h"
#include "gpu_tiler_shaders/multi_frame_motion_validate_spv.h"
#include <algorithm>
namespace Libs::Graphics {
namespace {
constexpr auto General=vk::ImageLayout::eGeneral;
void Barrier(vk::CommandBuffer cmd,vk::Image image,vk::ImageLayout before,vk::ImageLayout after,
             vk::AccessFlags2 src,vk::AccessFlags2 dst) {
    vk::ImageMemoryBarrier2 b{};
    b.srcStageMask=vk::PipelineStageFlagBits2::eAllCommands;b.dstStageMask=vk::PipelineStageFlagBits2::eAllCommands;
    b.srcAccessMask=src;b.dstAccessMask=dst;b.oldLayout=before;b.newLayout=after;
    b.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;b.image=image;
    b.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
    vk::DependencyInfo dep{};dep.imageMemoryBarrierCount=1;dep.pImageMemoryBarriers=&b;cmd.pipelineBarrier2(dep);
}
void Copy(vk::CommandBuffer cmd,vk::Image src,vk::ImageLayout sl,vk::Image dst,vk::ImageLayout dl,uint32_t w,uint32_t h) {
    vk::ImageBlit r{};r.srcSubresource={vk::ImageAspectFlagBits::eColor,0,0,1};r.dstSubresource=r.srcSubresource;
    r.srcOffsets[1]=vk::Offset3D{int32_t(w),int32_t(h),1};r.dstOffsets[1]=r.srcOffsets[1];
    cmd.blitImage(src,sl,dst,dl,1,&r,vk::Filter::eNearest);
}
// Keeps imported SDK images in GENERAL throughout; no SDK state/layout mutation.
void DiagnosticCopy(vk::CommandBuffer cmd,vk::Image image,uint32_t width,uint32_t height,
                    const MultiFrameDiagnosticBuffer& dst,uint32_t bytes_per_texel=4) {
    Barrier(cmd,image,General,General,vk::AccessFlagBits2::eMemoryWrite|vk::AccessFlagBits2::eMemoryRead,vk::AccessFlagBits2::eTransferRead);
    vk::BufferImageCopy region{};region.imageSubresource={vk::ImageAspectFlagBits::eColor,0,0,1};region.imageExtent={width,height,1};
    cmd.copyImageToBuffer(image,General,dst.buffer,1,&region);
    Barrier(cmd,image,General,General,vk::AccessFlagBits2::eTransferRead,vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite);
    vk::BufferMemoryBarrier2 b{};b.srcStageMask=vk::PipelineStageFlagBits2::eTransfer;b.srcAccessMask=vk::AccessFlagBits2::eTransferWrite;
    b.dstStageMask=vk::PipelineStageFlagBits2::eHost;b.dstAccessMask=vk::AccessFlagBits2::eHostRead;
    b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;b.buffer=dst.buffer;b.offset=0;b.size=uint64_t(width)*height*bytes_per_texel;
    vk::DependencyInfo dep{};dep.bufferMemoryBarrierCount=1;dep.pBufferMemoryBarriers=&b;cmd.pipelineBarrier2(dep);
}
struct Push { uint32_t width,height,flow_width,flow_height;float time;uint32_t copy_history; };
}
MultiFrameInterpolator::MultiFrameInterpolator(GraphicContext& g,CommandScheduler& s):m_graphics(g),m_scheduler(s){}
MultiFrameInterpolator::~MultiFrameInterpolator() {
    if(const auto retained=std::max(m_last_tick,m_scene_copy_tick))m_scheduler.Wait(retained);
    auto d=m_graphics.device;
    d.destroyImageView(m_current_view,nullptr);d.destroyImageView(m_flow_view,nullptr);d.destroyImageView(m_scene_view,nullptr);
    for(auto* t:{&m_previous,&m_outputs[0],&m_outputs[1],&m_outputs[2],&m_outputs[3],&m_outputs[4],&m_refined[0],&m_refined[1],&m_full_motion,&m_static_layer}) {
        d.destroyImageView(t->view,nullptr);if(t->image.image)m_graphics.DeleteImage(t->image);
    }
    d.destroyPipeline(m_pipeline,nullptr);d.destroyPipeline(m_refine_pipeline,nullptr);d.destroyPipeline(m_validate_pipeline,nullptr);d.destroyPipelineLayout(m_layout,nullptr);
    d.destroyDescriptorSetLayout(m_descriptors,nullptr);d.destroySampler(m_color_sampler,nullptr);d.destroySampler(m_flow_sampler,nullptr);
}
bool MultiFrameInterpolator::CreateView(vk::Image image,vk::Format format,vk::ImageView& out) {
    vk::ImageViewCreateInfo v{};v.image=image;v.viewType=vk::ImageViewType::e2D;v.format=format;
    v.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
    return m_graphics.device.createImageView(&v,nullptr,&out)==vk::Result::eSuccess;
}
bool MultiFrameInterpolator::Create(Target& t,vk::Format format,uint32_t width,uint32_t height) {
    vk::ImageCreateInfo i{};i.imageType=vk::ImageType::e2D;i.format=format;
    i.extent={width?width:m_width,height?height:m_height,1};i.mipLevels=1;i.arrayLayers=1;i.samples=vk::SampleCountFlagBits::e1;
    i.tiling=vk::ImageTiling::eOptimal;i.initialLayout=vk::ImageLayout::eUndefined;
    i.usage=vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst;
    return m_graphics.CreateImage(i,t.image)&&CreateView(t.image.image,i.format,t.view);
}
bool MultiFrameInterpolator::CreatePipeline() {
    auto d=m_graphics.device;
    vk::DescriptorSetLayoutBinding bindings[9]{};
    for(uint32_t i=0;i<9;++i)bindings[i]={i,i==5?vk::DescriptorType::eStorageBuffer:(i==3||i==7||i==8)?vk::DescriptorType::eStorageImage:vk::DescriptorType::eCombinedImageSampler,1,vk::ShaderStageFlagBits::eCompute,nullptr};
    vk::DescriptorSetLayoutCreateInfo ds{};ds.flags=vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;ds.bindingCount=9;ds.pBindings=bindings;
    if(d.createDescriptorSetLayout(&ds,nullptr,&m_descriptors)!=vk::Result::eSuccess)return false;
    vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute,0,sizeof(Push)};
    vk::PipelineLayoutCreateInfo pl{};pl.setLayoutCount=1;pl.pSetLayouts=&m_descriptors;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&push;
    if(d.createPipelineLayout(&pl,nullptr,&m_layout)!=vk::Result::eSuccess)return false;
    const auto build=[&](const uint32_t* code,size_t size,vk::Pipeline& pipeline) {
        vk::ShaderModuleCreateInfo sh{};sh.codeSize=size;sh.pCode=code;
        vk::ShaderModule module=nullptr;if(d.createShaderModule(&sh,nullptr,&module)!=vk::Result::eSuccess)return false;
        vk::ComputePipelineCreateInfo cp{};cp.stage.stage=vk::ShaderStageFlagBits::eCompute;cp.stage.module=module;cp.stage.pName="main";cp.layout=m_layout;
        const auto result=d.createComputePipelines(nullptr,1,&cp,nullptr,&pipeline);d.destroyShaderModule(module,nullptr);
        return result==vk::Result::eSuccess;
    };
    if(!build(MULTI_FRAME_WARP_SPV,sizeof(MULTI_FRAME_WARP_SPV),m_pipeline)||
       !build(MULTI_FRAME_REFINE_SPV,sizeof(MULTI_FRAME_REFINE_SPV),m_refine_pipeline)||
       !build(MULTI_FRAME_MOTION_VALIDATE_SPV,sizeof(MULTI_FRAME_MOTION_VALIDATE_SPV),m_validate_pipeline))return false;
    vk::SamplerCreateInfo si{};si.magFilter=si.minFilter=vk::Filter::eLinear;si.mipmapMode=vk::SamplerMipmapMode::eNearest;
    si.addressModeU=si.addressModeV=si.addressModeW=vk::SamplerAddressMode::eClampToEdge;
    if(d.createSampler(&si,nullptr,&m_color_sampler)!=vk::Result::eSuccess)return false;
    si.magFilter=si.minFilter=vk::Filter::eNearest;
    return d.createSampler(&si,nullptr,&m_flow_sampler)==vk::Result::eSuccess;
}
bool MultiFrameInterpolator::Initialize(uint32_t w,uint32_t h) {
    if(!w||!h||w>m_graphics.physical_device_properties.limits.maxImageDimension2D||h>m_graphics.physical_device_properties.limits.maxImageDimension2D)return false;
    m_width=w;m_height=h;
    if(!CreatePipeline()||!Create(m_previous))return false;
    m_feedback=std::make_unique<Buffer>(m_graphics,m_scheduler,MemoryUsage::Download,0,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst,4);
    if(m_feedback->Mapped().size()<4)return false;
    for(auto& o:m_outputs)if(!Create(o))return false;
    // Mandatory rgba32i storage format avoids requiring optional rg16i storage features.
    for(auto& f:m_refined)if(!Create(f,vk::Format::eR32G32B32A32Sint,(w+7)/8,(h+7)/8))return false;
    if(!Create(m_full_motion,vk::Format::eR32G32B32A32Sint,w,h)||!Create(m_static_layer,vk::Format::eR32G32B32A32Sfloat,w,h))return false;
    return true;
}
void MultiFrameInterpolator::Transit(vk::CommandBuffer cmd,Target& t,vk::ImageLayout layout,vk::AccessFlags2 access) {
    Barrier(cmd,t.image.image,t.layout,layout,vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite,access);t.layout=layout;
}
MultiFrameResult MultiFrameInterpolator::Record(vk::CommandBuffer cmd,vk::Image current,const KytyFsrOpticalFlow& flow,
                                               std::span<const vk::Image> outputs,bool pair_ready,const MultiFrameDiagnostics* diagnostics) {
    // All fallible view creation precedes writes to caller-owned images.
    if(!current||outputs.empty()||outputs.size()>5||flow.image.format!=VK_FORMAT_R16G16_SINT||!flow.image.image||flow.scene_change.format!=VK_FORMAT_R32_UINT||flow.scene_change.width<2||!flow.scene_change.image)return {};
    if(m_current_image!=current||m_flow_image!=vk::Image(flow.image.image)||m_scene_image!=vk::Image(flow.scene_change.image)) {
        if(m_last_tick&&!m_scheduler.IsFree(m_last_tick))return {};
        auto d=m_graphics.device;d.destroyImageView(m_current_view,nullptr);d.destroyImageView(m_flow_view,nullptr);d.destroyImageView(m_scene_view,nullptr);
        m_current_view=nullptr;m_flow_view=nullptr;m_scene_view=nullptr;m_current_image=current;m_flow_image=flow.image.image;m_scene_image=flow.scene_change.image;
        if(!CreateView(current,vk::Format::eR8G8B8A8Unorm,m_current_view)||!CreateView(m_flow_image,vk::Format::eR16G16Sint,m_flow_view)||!CreateView(m_scene_image,vk::Format::eR32Uint,m_scene_view)){m_current_image=nullptr;m_flow_image=nullptr;m_scene_image=nullptr;return {};}
    }
    if(diagnostics&&(!m_has_history||!pair_ready||diagnostics->optical_flow.capacity<uint64_t(flow.image.width)*flow.image.height*4))return {};
    m_has_fine=false;
    m_last_tick=m_scheduler.CurrentTick();
    cmd.fillBuffer(m_feedback->Handle(),0,4,0);
    vk::BufferMemoryBarrier2 fb{};fb.srcStageMask=vk::PipelineStageFlagBits2::eTransfer;fb.srcAccessMask=vk::AccessFlagBits2::eTransferWrite;
    fb.dstStageMask=vk::PipelineStageFlagBits2::eComputeShader;fb.dstAccessMask=vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite;
    fb.srcQueueFamilyIndex=fb.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;fb.buffer=m_feedback->Handle();fb.offset=0;fb.size=4;
    vk::DependencyInfo fd{};fd.bufferMemoryBarrierCount=1;fd.pBufferMemoryBarriers=&fb;cmd.pipelineBarrier2(fd);
    Barrier(cmd,current,General,General,vk::AccessFlagBits2::eMemoryWrite,vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eTransferRead);
    const bool generate=pair_ready&&m_has_history;
    // Valid descriptors/layouts are bound even during the history-only seed branch.
    Transit(cmd,m_full_motion,General,vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite);
    Transit(cmd,m_static_layer,General,vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite);
    if(generate) {
        Barrier(cmd,m_flow_image,General,General,vk::AccessFlagBits2::eMemoryWrite,vk::AccessFlagBits2::eShaderRead);
        Barrier(cmd,m_scene_image,General,General,vk::AccessFlagBits2::eMemoryWrite,vk::AccessFlagBits2::eShaderRead);
        Transit(cmd,m_previous,General,vk::AccessFlagBits2::eShaderRead);
        // Bounded local photometric validation is paid once per pair, never per subframe.
        // Four 3x3 candidate passes at strides 1,2,4,8; fixed 4x4 RGB-patch score.
        vk::ImageView refined_view=m_flow_view;
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute,m_refine_pipeline);
        for(uint32_t pass=0;pass<4;++pass) {
            auto& refined=m_refined[pass%2];Transit(cmd,refined,General,vk::AccessFlagBits2::eShaderWrite);
            const vk::DescriptorImageInfo images[]={
                {m_color_sampler,m_previous.view,General},{m_color_sampler,m_current_view,General},
                {m_flow_sampler,refined_view,General},{nullptr,refined.view,General}};
            vk::WriteDescriptorSet writes[4]{};
            for(uint32_t j=0;j<4;++j){writes[j].dstBinding=j;writes[j].descriptorCount=1;writes[j].descriptorType=j==3?vk::DescriptorType::eStorageImage:vk::DescriptorType::eCombinedImageSampler;writes[j].pImageInfo=&images[j];}
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,m_layout,0,4,writes);
            // Fixed spatial reach broadens without additional candidates, dispatches or history.
            const Push refine_push{m_width,m_height,flow.image.width,flow.image.height,0,1u<<pass};
            cmd.pushConstants(m_layout,vk::ShaderStageFlagBits::eCompute,0,sizeof(refine_push),&refine_push);
            cmd.dispatch((flow.image.width+7)/8,(flow.image.height+7)/8,1);
            Transit(cmd,refined,General,vk::AccessFlagBits2::eShaderRead);refined_view=refined.view;
        }
        // Full-resolution correspondence/layer validation runs ONCE per real pair.
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute,m_validate_pipeline);
        const vk::DescriptorImageInfo validation_images[]={
            {m_color_sampler,m_previous.view,General},{m_color_sampler,m_current_view,General},
            {m_flow_sampler,refined_view,General},{m_flow_sampler,m_flow_view,General},
            {nullptr,m_full_motion.view,General},{nullptr,m_static_layer.view,General}};
        const uint32_t validation_bindings[]={0,1,2,6,7,8};
        vk::WriteDescriptorSet validation_writes[6]{};
        for(uint32_t j=0;j<6;++j){validation_writes[j].dstBinding=validation_bindings[j];validation_writes[j].descriptorCount=1;validation_writes[j].descriptorType=j<4?vk::DescriptorType::eCombinedImageSampler:vk::DescriptorType::eStorageImage;validation_writes[j].pImageInfo=&validation_images[j];}
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,m_layout,0,6,validation_writes);
        const Push validation_push{m_width,m_height,flow.image.width,flow.image.height,0,0};
        cmd.pushConstants(m_layout,vk::ShaderStageFlagBits::eCompute,0,sizeof(validation_push),&validation_push);
        cmd.dispatch((m_width+7)/8,(m_height+7)/8,1);
        Transit(cmd,m_full_motion,General,vk::AccessFlagBits2::eShaderRead);
        Transit(cmd,m_static_layer,General,vk::AccessFlagBits2::eShaderRead);
        m_has_fine=true;
        if(diagnostics&&diagnostics->refined_flow.buffer)
            DiagnosticCopy(cmd,m_refined[1].image.image,flow.image.width,flow.image.height,diagnostics->refined_flow,16);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute,m_pipeline);
        for(uint32_t i=0;i<outputs.size();++i) {
            auto& dst=m_outputs[i];Transit(cmd,dst,General,vk::AccessFlagBits2::eShaderWrite);
            const vk::DescriptorImageInfo images[]={
                {m_color_sampler,m_previous.view,General},{m_color_sampler,m_current_view,General},
                {m_flow_sampler,refined_view,General},{nullptr,dst.view,General},{m_flow_sampler,m_scene_view,General}};
            vk::WriteDescriptorSet writes[9]{};
            for(uint32_t j=0;j<5;++j){writes[j].dstBinding=j;writes[j].descriptorCount=1;writes[j].descriptorType=j==3?vk::DescriptorType::eStorageImage:vk::DescriptorType::eCombinedImageSampler;writes[j].pImageInfo=&images[j];}
            vk::DescriptorBufferInfo feedback{m_feedback->Handle(),0,4};writes[5].dstBinding=5;writes[5].descriptorCount=1;writes[5].descriptorType=vk::DescriptorType::eStorageBuffer;writes[5].pBufferInfo=&feedback;
            const vk::DescriptorImageInfo extras[]={ {m_flow_sampler,m_flow_view,General},{nullptr,m_full_motion.view,General},{nullptr,m_static_layer.view,General} };
            for(uint32_t j=0;j<3;++j){writes[6+j].dstBinding=6+j;writes[6+j].descriptorCount=1;writes[6+j].descriptorType=j==0?vk::DescriptorType::eCombinedImageSampler:vk::DescriptorType::eStorageImage;writes[6+j].pImageInfo=&extras[j];}
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,m_layout,0,9,writes);
            Push p{m_width,m_height,flow.image.width,flow.image.height,float(i+1)/float(outputs.size()+1),0};
            cmd.pushConstants(m_layout,vk::ShaderStageFlagBits::eCompute,0,sizeof(p),&p);
            cmd.dispatch((m_width+7)/8,(m_height+7)/8,1);
            Transit(cmd,dst,vk::ImageLayout::eTransferSrcOptimal,vk::AccessFlagBits2::eTransferRead);
            Barrier(cmd,outputs[i],General,General,vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite,vk::AccessFlagBits2::eTransferWrite);
            Copy(cmd,dst.image.image,vk::ImageLayout::eTransferSrcOptimal,outputs[i],General,m_width,m_height);
            Barrier(cmd,outputs[i],General,vk::ImageLayout::eTransferSrcOptimal,vk::AccessFlagBits2::eTransferWrite,vk::AccessFlagBits2::eTransferRead);
        }
    }
    fb.srcStageMask=vk::PipelineStageFlagBits2::eAllCommands;fb.srcAccessMask=vk::AccessFlagBits2::eShaderWrite|vk::AccessFlagBits2::eTransferWrite;
    fb.dstStageMask=vk::PipelineStageFlagBits2::eHost;fb.dstAccessMask=vk::AccessFlagBits2::eHostRead;cmd.pipelineBarrier2(fd);
    if(diagnostics) {
        DiagnosticCopy(cmd,m_flow_image,flow.image.width,flow.image.height,diagnostics->optical_flow);
        DiagnosticCopy(cmd,m_scene_image,2,1,diagnostics->scene_change);
        DiagnosticCopy(cmd,m_previous.image.image,m_width,m_height,diagnostics->previous);
    }
    // One GPU history update after ALL subframes. m_color is sampled-only/transfer-dst,
    // so a storage copy preserves the legacy color image's exact usage flags.
    Transit(cmd,m_previous,General,vk::AccessFlagBits2::eShaderWrite);
    const vk::DescriptorImageInfo history_images[]={
        {m_color_sampler,m_previous.view,General},{m_color_sampler,m_current_view,General},
        {m_flow_sampler,m_flow_view,General},{nullptr,m_previous.view,General},{m_flow_sampler,m_scene_view,General}};
    vk::WriteDescriptorSet history_writes[9]{};
    for(uint32_t j=0;j<5;++j){history_writes[j].dstBinding=j;history_writes[j].descriptorCount=1;history_writes[j].descriptorType=j==3?vk::DescriptorType::eStorageImage:vk::DescriptorType::eCombinedImageSampler;history_writes[j].pImageInfo=&history_images[j];}
    vk::DescriptorBufferInfo history_feedback{m_feedback->Handle(),0,4};history_writes[5].dstBinding=5;history_writes[5].descriptorCount=1;history_writes[5].descriptorType=vk::DescriptorType::eStorageBuffer;history_writes[5].pBufferInfo=&history_feedback;
    const vk::DescriptorImageInfo history_extras[]={ {m_flow_sampler,m_flow_view,General},{nullptr,m_full_motion.view,General},{nullptr,m_static_layer.view,General} };
    for(uint32_t j=0;j<3;++j){history_writes[6+j].dstBinding=6+j;history_writes[6+j].descriptorCount=1;history_writes[6+j].descriptorType=j==0?vk::DescriptorType::eCombinedImageSampler:vk::DescriptorType::eStorageImage;history_writes[6+j].pImageInfo=&history_extras[j];}
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute,m_pipeline);cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,m_layout,0,9,history_writes);
    const Push history_push{m_width,m_height,flow.image.width,flow.image.height,1.0f,1};cmd.pushConstants(m_layout,vk::ShaderStageFlagBits::eCompute,0,sizeof(history_push),&history_push);
    cmd.dispatch((m_width+7)/8,(m_height+7)/8,1);
    Transit(cmd,m_previous,General,vk::AccessFlagBits2::eShaderRead);m_has_history=true;
    // Existing history-copy dispatch has now sampled current into this TransferSrc-capable
    // target. Capturing it avoids altering the legacy m_color image usage or another dispatch.
    if(diagnostics)DiagnosticCopy(cmd,m_previous.image.image,m_width,m_height,diagnostics->current);
    if(!generate)return {MultiFrameStatus::Seeded,0,0};
    return {MultiFrameStatus::Generated,uint32_t(outputs.size()),(1u<<outputs.size())-1u};
}
bool MultiFrameInterpolator::SceneReadbackPending() const {
    return m_scene_copy_tick&&!m_scheduler.IsFree(m_scene_copy_tick);
}
bool MultiFrameInterpolator::CopySceneForDiagnostics(vk::CommandBuffer command,
    vk::Buffer destination,uint64_t capacity) {
    if(!command||!destination||capacity<8||!m_scene_image||!m_last_tick||
       !m_scheduler.IsFree(m_last_tick)||SceneReadbackPending())return false;
    // The original batch has completed. Read its existing SCD image without another
    // SDK prepare, warp, history update or layout-state transition in the bridge.
    DiagnosticCopy(command,m_scene_image,2,1,{destination,capacity});
    m_scene_copy_tick=m_scheduler.CurrentTick();
    return true;
}
bool MultiFrameInterpolator::CopyFineForDiagnostics(vk::CommandBuffer command,
    const MultiFrameDiagnosticBuffer& full_motion,const MultiFrameDiagnosticBuffer& static_layer) {
    const uint64_t bytes=uint64_t(m_width)*m_height*16;
    if(!command||!m_has_fine||!bytes||!full_motion.buffer||!static_layer.buffer||
       full_motion.buffer==static_layer.buffer||full_motion.capacity<bytes||static_layer.capacity<bytes||
       !m_full_motion.image.image||!m_static_layer.image.image||!m_last_tick||
       !m_scheduler.IsFree(m_last_tick)||SceneReadbackPending())return false;
    // Read the completed pair's existing fields only. GENERAL is restored and the
    // diagnostic tick retains both images and refuses later generation until done.
    DiagnosticCopy(command,m_full_motion.image.image,m_width,m_height,full_motion,16);
    DiagnosticCopy(command,m_static_layer.image.image,m_width,m_height,static_layer,16);
    m_scene_copy_tick=m_scheduler.CurrentTick();
    return true;
}
MultiFrameSceneStatus MultiFrameInterpolator::SceneStatus() const {
    MultiFrameSceneStatus status{};status.recorded_tick=m_last_tick;
    status.completed_tick=m_scheduler.GetMasterSemaphore().KnownGpuTick();
    if(!m_feedback||!m_last_tick)return status;
    if(!m_scheduler.IsFree(m_last_tick)) { status.state=MultiFrameSceneState::Pending;return status; }
    status.completed_tick=m_scheduler.GetMasterSemaphore().KnownGpuTick();
    m_feedback->Invalidate(0,4);std::memcpy(&status.flags,m_feedback->Mapped().data(),4);
    status.state=status.flags?MultiFrameSceneState::Changed:MultiFrameSceneState::Clear;return status;
}

}

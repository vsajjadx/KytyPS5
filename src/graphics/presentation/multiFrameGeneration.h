// Copyright2026 KytyPS5 contributors. SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/frameGeneration.h"
#include "graphics/presentation/ffxMultiFrameFlow.h"
#include <array>
namespace Libs::Graphics {
class Buffer;
class MultiFrameInterpolator final {
public:
    MultiFrameInterpolator(GraphicContext& graphics, CommandScheduler& scheduler);
    ~MultiFrameInterpolator();
    bool Initialize(uint32_t width, uint32_t height);
    MultiFrameSceneStatus SceneStatus() const;
    bool CopySceneForDiagnostics(vk::CommandBuffer command,vk::Buffer destination,uint64_t capacity);
    bool CopyFineForDiagnostics(vk::CommandBuffer command,
        const MultiFrameDiagnosticBuffer& full_motion,const MultiFrameDiagnosticBuffer& static_layer);
    bool SceneReadbackPending() const;
    void Reset() { m_has_history=false;m_has_fine=false; }
    MultiFrameResult Record(vk::CommandBuffer command, vk::Image current,
        const KytyFsrOpticalFlow& flow, std::span<const vk::Image> outputs, bool pair_ready,
        const MultiFrameDiagnostics* diagnostics = nullptr);
private:
    struct Target { VulkanImage image; vk::ImageView view=nullptr; vk::ImageLayout layout=vk::ImageLayout::eUndefined; };
    bool Create(Target& target, vk::Format format=vk::Format::eR8G8B8A8Unorm, uint32_t width=0, uint32_t height=0);
    bool CreateView(vk::Image image, vk::Format format, vk::ImageView& out);
    bool CreatePipeline();
    void Transit(vk::CommandBuffer command, Target& target, vk::ImageLayout layout,
                 vk::AccessFlags2 access);
    GraphicContext& m_graphics;
    CommandScheduler& m_scheduler;
    Target m_previous;
    std::array<Target,5> m_outputs;
    std::array<Target,2> m_refined;
    Target m_full_motion,m_static_layer;
    vk::ImageView m_current_view=nullptr,m_flow_view=nullptr,m_scene_view=nullptr;
    vk::Image m_current_image=nullptr,m_flow_image=nullptr,m_scene_image=nullptr;
    vk::Sampler m_color_sampler=nullptr,m_flow_sampler=nullptr;
    vk::DescriptorSetLayout m_descriptors=nullptr;
    vk::PipelineLayout m_layout=nullptr;
    vk::Pipeline m_pipeline=nullptr,m_refine_pipeline=nullptr,m_validate_pipeline=nullptr;
    std::unique_ptr<Buffer> m_feedback;
    uint32_t m_width=0,m_height=0;
    uint64_t m_last_tick=0,m_scene_copy_tick=0;
    bool m_has_history=false,m_has_fine=false;
};
}

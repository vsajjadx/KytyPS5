// Copyright2026 KytyPS5 contributors. SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>
struct FfxVkFsr3_3_1_6FrameGenerationContext;
struct KytyFsrFlowImage { VkImage image; VkFormat format; uint32_t width,height; VkImageLayout layout; VkImageUsageFlags usage; };
struct KytyFsrOpticalFlow { KytyFsrFlowImage image,scene_change; uint64_t frame_id; };
// Completes an already successful prepare without running FI or advancing its history.
extern "C" bool kytyFsrFinishOpticalFlow(FfxVkFsr3_3_1_6FrameGenerationContext*,KytyFsrOpticalFlow*);

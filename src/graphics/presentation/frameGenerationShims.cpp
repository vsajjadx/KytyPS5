// The FSR frame generation library calls Vulkan commands directly, while the emulator loads
// Vulkan at runtime. These definitions forward its calls to the loaded dispatcher.

#if defined(_WIN32)

#include "graphics/host_gpu/vulkanCommon.h"
extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
                         VkDescriptorSet* pDescriptorSets) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateDescriptorSets(device, pAllocateInfo,
                                                                  pDescriptorSets);
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice device,
                                                const VkMemoryAllocateInfo* pAllocateInfo,
                                                const VkAllocationCallbacks* pAllocator,
                                                VkDeviceMemory* pMemory) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(device, pAllocateInfo, pAllocator,
                                                          pMemory);
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice device, VkBuffer buffer,
                                                  VkDeviceMemory memory,
                                                  VkDeviceSize memoryOffset) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory(device, buffer, memory, memoryOffset);
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice device, VkImage image,
                                                 VkDeviceMemory memory, VkDeviceSize memoryOffset) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory(device, image, memory, memoryOffset);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout,
    uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets,
    uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBindDescriptorSets(
        commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets,
        dynamicOffsetCount, pDynamicOffsets);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer commandBuffer,
                                             VkPipelineBindPoint pipelineBindPoint,
                                             VkPipeline pipeline) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image,
                                                VkImageLayout imageLayout,
                                                const VkClearColorValue* pColor,
                                                uint32_t rangeCount,
                                                const VkImageSubresourceRange* pRanges) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdClearColorImage(commandBuffer, image, imageLayout, pColor,
                                                       rangeCount, pRanges);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer,
                                           VkBuffer dstBuffer, uint32_t regionCount,
                                           const VkBufferCopy* pRegions) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount,
                                                  pRegions);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer,
                                                  VkImage dstImage, VkImageLayout dstImageLayout,
                                                  uint32_t regionCount,
                                                  const VkBufferImageCopy* pRegions) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage,
                                                         dstImageLayout, regionCount, pRegions);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage,
                                          VkImageLayout srcImageLayout, VkImage dstImage,
                                          VkImageLayout dstImageLayout, uint32_t regionCount,
                                          const VkImageCopy* pRegions) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage,
                                                 dstImageLayout, regionCount, pRegions);
}

VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer commandBuffer, uint32_t groupCountX,
                                         uint32_t groupCountY, uint32_t groupCountZ) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdDispatch(commandBuffer, groupCountX, groupCountY,
                                                groupCountZ);
}

VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                                           VkDeviceSize dstOffset, VkDeviceSize size,
                                           uint32_t data) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data);
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPipelineBarrier(
        commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount,
        pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount,
        pImageMemoryBarriers);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice device,
                                              const VkBufferCreateInfo* pCreateInfo,
                                              const VkAllocationCallbacks* pAllocator,
                                              VkBuffer* pBuffer) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer(device, pCreateInfo, pAllocator, pBuffer);
}

VKAPI_ATTR VkResult VKAPI_CALL
vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                         const VkComputePipelineCreateInfo* pCreateInfos,
                         const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines(
        device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice device,
                                                      const VkDescriptorPoolCreateInfo* pCreateInfo,
                                                      const VkAllocationCallbacks* pAllocator,
                                                      VkDescriptorPool* pDescriptorPool) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorPool(device, pCreateInfo, pAllocator,
                                                                pDescriptorPool);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorSetLayout(device, pCreateInfo,
                                                                     pAllocator, pSetLayout);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo,
                                             const VkAllocationCallbacks* pAllocator,
                                             VkImage* pImage) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage(device, pCreateInfo, pAllocator, pImage);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice device,
                                                 const VkImageViewCreateInfo* pCreateInfo,
                                                 const VkAllocationCallbacks* pAllocator,
                                                 VkImageView* pView) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImageView(device, pCreateInfo, pAllocator, pView);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice device,
                                                      const VkPipelineLayoutCreateInfo* pCreateInfo,
                                                      const VkAllocationCallbacks* pAllocator,
                                                      VkPipelineLayout* pPipelineLayout) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreatePipelineLayout(device, pCreateInfo, pAllocator,
                                                                pPipelineLayout);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice device,
                                               const VkSamplerCreateInfo* pCreateInfo,
                                               const VkAllocationCallbacks* pAllocator,
                                               VkSampler* pSampler) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateSampler(device, pCreateInfo, pAllocator, pSampler);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device,
                                                    const VkShaderModuleCreateInfo* pCreateInfo,
                                                    const VkAllocationCallbacks* pAllocator,
                                                    VkShaderModule* pShaderModule) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule(device, pCreateInfo, pAllocator,
                                                              pShaderModule);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice device, VkBuffer buffer,
                                           const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(device, buffer, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                   const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyDescriptorPool(device, descriptorPool, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice device,
                                                        VkDescriptorSetLayout descriptorSetLayout,
                                                        const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyDescriptorSetLayout(device, descriptorSetLayout,
                                                               pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice device, VkImage image,
                                          const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(device, image, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device, VkImageView imageView,
                                              const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImageView(device, imageView, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipeline,
                                             const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipeline(device, pipeline, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout,
                                                   const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipelineLayout(device, pipelineLayout, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice device, VkSampler sampler,
                                            const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroySampler(device, sampler, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                                 const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyShaderModule(device, shaderModule, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice device) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkDeviceWaitIdle(device);
}

VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char* pLayerName,
                                     uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateDeviceExtensionProperties(
        physicalDevice, pLayerName, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                                         const VkMappedMemoryRange* pMemoryRanges) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges(device, memoryRangeCount,
                                                                   pMemoryRanges);
}

VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory memory,
                                        const VkAllocationCallbacks* pAllocator) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(device, memory, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(
    VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements(device, buffer,
                                                                pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice device, VkImage image,
                                                        VkMemoryRequirements* pMemoryRequirements) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements(device, image, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice,
                                                        VkPhysicalDeviceFeatures2* pFeatures) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties* pFormatProperties) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFormatProperties(physicalDevice, format,
                                                                      pFormatProperties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties(physicalDevice,
                                                                      pMemoryProperties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                         VkPhysicalDeviceProperties* pProperties) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties(physicalDevice, pProperties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(physicalDevice, pProperties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount,
    VkQueueFamilyProperties* pQueueFamilyProperties) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice device, VkDeviceMemory memory,
                                           VkDeviceSize offset, VkDeviceSize size,
                                           VkMemoryMapFlags flags, void** ppData) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory(device, memory, offset, size, flags, ppData);
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice device,
                                                     VkDescriptorPool descriptorPool,
                                                     VkDescriptorPoolResetFlags flags) {
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkResetDescriptorPool(device, descriptorPool, flags);
}

VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice device, VkDeviceMemory memory) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory(device, memory);
}

VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                  const VkWriteDescriptorSet* pDescriptorWrites,
                                                  uint32_t descriptorCopyCount,
                                                  const VkCopyDescriptorSet* pDescriptorCopies) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkUpdateDescriptorSets(
        device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}

} // extern "C"

#endif

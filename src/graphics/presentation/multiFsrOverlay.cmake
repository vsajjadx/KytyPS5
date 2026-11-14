# Tracked deterministic overlay; preserve the pinned MIT SDK checkout and legacy shader blobs.
set(kyty_fsr_context_source "${CMAKE_CURRENT_SOURCE_DIR}/fsr-vulkan/src/ffx_vk_fsr3_3_1_6_framegen.cpp")
# These inputs are read at configure time, not compiled as direct includes. Track
# edits so Ninja/CMake regenerates the overlay instead of silently using stale code.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${kyty_fsr_context_source}"
    "${KYTY_SOURCE_DIR}/graphics/presentation/ffxMultiFrameFlow.cpp.inc"
    "${KYTY_SOURCE_DIR}/graphics/presentation/ffxMultiFrameScd.cpp.inc")
file(READ "${kyty_fsr_context_source}" kyty_fsr_context_text)
string(REPLACE "\r\n" "\n" kyty_fsr_context_text "${kyty_fsr_context_text}")
# Accept the pinned unpatched source too; apply the pre-existing lifetime fix in memory.
if(NOT kyty_fsr_context_text MATCHES "ffxVkFsr3_3_1_5BridgeBeginFrame")
    string(REPLACE "    context->frameId = info->frameId;" "    context->frameId = info->frameId;\n    ffxVkFsr3_3_1_5BridgeBeginFrame(context->bridge, info->frameId);" kyty_fsr_context_text "${kyty_fsr_context_text}")
    string(REPLACE "    dispatch.frameID = info->frameId;" "    dispatch.frameID = info->frameId;\n    ffxVkFsr3_3_1_5BridgeBeginFrame(context->bridge, info->frameId);" kyty_fsr_context_text "${kyty_fsr_context_text}")
    string(REPLACE "    if (context->prepared && context->frameId <= completedFrameId) {" "    ffxVkFsr3_3_1_5BridgeRetireFrame(context->bridge, completedFrameId);\n    if (context->prepared && context->frameId <= completedFrameId) {" kyty_fsr_context_text "${kyty_fsr_context_text}")
endif()
string(SHA256 kyty_fsr_context_hash "${kyty_fsr_context_text}")
if(NOT kyty_fsr_context_hash STREQUAL "1b4ee12e9fada75f872a9bd4a0ade545afb37d857f0fc0e8b4b4fd5d3c3b359b")
    message(FATAL_ERROR "Kyty MultiFG: pinned portable FSR context source drift; review overlay before updating")
endif()
# Source hash above is checked BEFORE these reviewed build-tree-only insertions.
file(READ "${KYTY_SOURCE_DIR}/graphics/presentation/ffxMultiFrameScd.cpp.inc" kyty_fsr_scd_hook)
string(REPLACE "#include <vector>" "#include <vector>\n#include <cwchar>\n#include \"gpu_tiler_shaders/multi_frame_scd_sync_spv.h\"" kyty_fsr_context_text "${kyty_fsr_context_text}")
string(REPLACE "static void initialize_backend(" "${kyty_fsr_scd_hook}\nstatic void initialize_backend(" kyty_fsr_context_text "${kyty_fsr_context_text}")
string(REPLACE "backend->fpCreatePipeline = ffxVkFsr3_3_1_5BridgeCreatePipeline;" "backend->fpCreatePipeline = kytyCreatePipelineWithSynchronizedScd;" kyty_fsr_context_text "${kyty_fsr_context_text}")
# Capture the mode in the backend's callback pointer while TLS creation scope is active.
string(REPLACE "backend->fpScheduleGpuJob = ffxVkFsr3_3_1_5BridgeScheduleGpuJob;" "backend->fpScheduleGpuJob = kyty_multi_scd_creation ? kytyScheduleWithSynchronizedScd : ffxVkFsr3_3_1_5BridgeScheduleGpuJob;" kyty_fsr_context_text "${kyty_fsr_context_text}")
file(READ "${KYTY_SOURCE_DIR}/graphics/presentation/ffxMultiFrameFlow.cpp.inc" kyty_fsr_flow_accessor)
set(kyty_fsr_overlay "${CMAKE_CURRENT_BINARY_DIR}/kyty_fsr_context_with_flow.cpp")
file(WRITE "${kyty_fsr_overlay}" "${kyty_fsr_context_text}\n${kyty_fsr_flow_accessor}")
get_target_property(kyty_fsr_sources ffx_vulkan_fsr3_vk_framegeneration_3_1_6 SOURCES)
get_target_property(kyty_fsr_source_dir ffx_vulkan_fsr3_vk_framegeneration_3_1_6 SOURCE_DIR)
set(kyty_fsr_overlay_sources)
set(kyty_fsr_replaced 0)
foreach(kyty_fsr_source IN LISTS kyty_fsr_sources)
    if(kyty_fsr_source MATCHES "(^|/)ffx_vk_fsr3_3_1_6_framegen[.]cpp$")
        math(EXPR kyty_fsr_replaced "${kyty_fsr_replaced}+1")
    else()
        if(IS_ABSOLUTE "${kyty_fsr_source}" OR kyty_fsr_source MATCHES "^[$]<")
            list(APPEND kyty_fsr_overlay_sources "${kyty_fsr_source}")
        else()
            list(APPEND kyty_fsr_overlay_sources "${kyty_fsr_source_dir}/${kyty_fsr_source}")
        endif()
    endif()
endforeach()
if(NOT kyty_fsr_replaced EQUAL 1)
    message(FATAL_ERROR "Kyty MultiFG: expected exactly one portable context source")
endif()
set_property(TARGET ffx_vulkan_fsr3_vk_framegeneration_3_1_6 PROPERTY SOURCES "${kyty_fsr_overlay_sources};${kyty_fsr_overlay}")
target_include_directories(ffx_vulkan_fsr3_vk_framegeneration_3_1_6 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fsr-vulkan/src" "${KYTY_SOURCE_DIR}")

target_include_directories(ffx_vulkan_fsr3_vk_framegeneration_3_1_6 PRIVATE "${PROJECT_BINARY_DIR}")

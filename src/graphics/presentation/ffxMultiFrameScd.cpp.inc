// Copyright 2026 KytyPS5 contributors. SPDX-License-Identifier: MIT
// Inserted inside the pinned context TU's anonymous namespace, before initialize_backend.
// Legacy creation uses the exact original shader blob. No vendor checkout is modified.
static thread_local bool kyty_multi_scd_creation=false;
static FfxErrorCode kytyCreatePipelineWithSynchronizedScd(FfxInterface* backend,
    FfxShaderBlob* blob,const FfxPipelineDescription* description,FfxUInt32 contextId,
    FfxPipelineState* pipeline) {
    if(kyty_multi_scd_creation&&description&&
       std::wcscmp(description->name,L"Opticalflow_SCD_Divergence")==0) {
        FfxShaderBlob replacement=blob?*blob:FfxShaderBlob{};
        replacement.data=reinterpret_cast<const uint8_t*>(MULTI_FRAME_SCD_SYNC_SPV);
        replacement.size=sizeof(MULTI_FRAME_SCD_SYNC_SPV);
        return ffxVkFsr3_3_1_5BridgeCreatePipeline(backend,&replacement,description,contextId,pipeline);
    }
    return ffxVkFsr3_3_1_5BridgeCreatePipeline(backend,blob,description,contextId,pipeline);
}

// This callback pointer is installed in the Multi context during creation. The TLS
// flag has already been restored when RecordPrepare schedules jobs; never consult it here.
static FfxErrorCode kytyScheduleWithSynchronizedScd(FfxInterface* backend,
    const FfxGpuJobDescription* job) {
    if(!job||job->jobType!=FFX_GPU_JOB_COMPUTE||
       std::wcscmp(job->jobLabel,L"OF ComputeSCDDivergence")!=0)
        return ffxVkFsr3_3_1_5BridgeScheduleGpuJob(backend,job);
    if(!job->computeJobDescriptor.pipeline||job->computeJobDescriptor.pipeline->uavTextureCount==0)
        return static_cast<FfxErrorCode>(FFX_ERROR_INVALID_ARGUMENT);
    FfxGpuJobDescription barrier{};barrier.jobType=FFX_GPU_JOB_BARRIER;
    barrier.barrierDescriptor.barrierType=FFX_BARRIER_TYPE_UAV;
    // The existing bridge validates this live resource, then emits a GLOBAL compute
    // shader read/write memory barrier for UAV jobs (no layout/state transition).
    barrier.barrierDescriptor.resource=job->computeJobDescriptor.uavTextures[0].resource;
    auto result=ffxVkFsr3_3_1_5BridgeScheduleGpuJob(backend,&barrier);
    if(result!=FFX_OK)return result;
    result=ffxVkFsr3_3_1_5BridgeScheduleGpuJob(backend,job);
    if(result!=FFX_OK)return result;
    return ffxVkFsr3_3_1_5BridgeScheduleGpuJob(backend,&barrier);
}

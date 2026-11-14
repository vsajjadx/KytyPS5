#ifndef KYTY_SHADER_ACTIVITY_DIAGNOSTICS_H_
#define KYTY_SHADER_ACTIVITY_DIAGNOSTICS_H_
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::IR {
// Observational only. Never enables proof/capture or reads any guest/GPU memory.
inline bool ActivityDiagnosticsEnabled(uint64_t shader) {
    if (shader != 0xb6c04c59ad4fac06ull) return false;
    static const bool enabled = [] {
        const char* text = std::getenv("KYTY_ACTIVITY_PROOF_DIAG");
        return text != nullptr && text[0] == '1' && text[1] == '\0';
    }();
    return enabled;
}
enum class ActivityDiagnosticReason : uint32_t {
    Eligible, Ineligible, RuntimeUnavailable, NoCache, NoSerial, ReadEpochUnavailable, IdentityMismatch, EpochContext, EpochMapping, EpochProtection, EpochBuffers, EpochImages, EpochConfiguration, StrictSourceUnavailable, StrictSourceChanged, TargetsRejected, ProbeEpochChanged, ReuseHit, CaptureDisabled, ObservationsEmptyOrLarge, NoJointSource, NoJointImage, CounterNotObserved, FlatReadNotObserved, PrepareEpochUnavailable, NoOtherObservations, PendingReady, ConfirmInputInvalid, ConfirmInputEpochChanged, Confirmed, NativeNoCache, NativeLayoutRejected, NativeBufferRejected, NativeImageRejected, NativeEpochUnavailable, NativeTargetsRejected, NativeEpochChanged, NativeConfirmRejected, Count
};
// Static metadata only: resource IDs are indices in plan.info vectors, not handles.
struct ActivityDiagnosticSourceMetadata {
    uint32_t source = UINT32_MAX, total_sources = 0, dwords = 0, kind_mask = 0;
    uint32_t buffer_id = UINT32_MAX, buffer_count = 0;
    uint32_t image_id = UINT32_MAX, image_count = 0;
    uint32_t sampler_id = UINT32_MAX, sampler_count = 0;
    uint32_t counter_memory_index = UINT32_MAX;
    bool operator==(const ActivityDiagnosticSourceMetadata&) const = default;
};
struct ActivityDiagnosticAuxCapture {
    uint64_t zero = 0, nonzero = 0, missing = 0, duplicate = 0, capture_matches = 0;
    bool operator==(const ActivityDiagnosticAuxCapture&) const = default;
};
struct ActivityDiagnosticSourceBucket {
    ActivityDiagnosticSourceMetadata first_metadata;
    ActivityDiagnosticAuxCapture aux23, aux86;
    uint64_t failures = 0, count_zero = 0, count_nonzero = 0;
    uint64_t count_missing = 0, count_mixed = 0, capture_matches = 0;
    uint64_t metadata_conflicts = 0;
};
// First failed native-image guard only, from already prepared metadata. This is
// not a mapping/readability verdict: NativeTargetsRejected is a separate guard.
enum class ActivityDiagnosticImageReason : uint32_t {
    Written, Heap, DepthCompare, MipStats, MissingImage, MissingView,
    Unregistered, NativeDepth, RequestedDepth, DepthAssociation, StencilAssociation,
    NativeMetadata, RequestedMetadata, NativeStencilRange, RequestedStencilRange, Count
};
struct ActivityDiagnosticNativeImageMetadata {
    uint64_t module = 0, serial = 0, view_handle = 0;
    uint64_t requested_begin = 0, requested_size = 0, native_begin = 0, native_size = 0;
    uint64_t requested_meta_begin = 0, requested_meta_size = 0;
    uint64_t native_meta_begin = 0, native_meta_size = 0;
    uint64_t requested_stencil_begin = 0, requested_stencil_size = 0;
    uint64_t native_stencil_begin = 0, native_stencil_size = 0;
    uint32_t index = UINT32_MAX, source = UINT32_MAX, descriptor_index = UINT32_MAX, first_use_pc = 0;
    uint32_t resource_class = 0, resource_flags = 0, descriptor_flags = 0;
    uint32_t native_id = UINT32_MAX, binding_type = UINT32_MAX, image_flags = 0;
    uint32_t requested_format = 0, native_format = 0;
    uint32_t requested_metadata = 0, native_metadata = 0;
    uint32_t view_type = 0, view_format = 0, view_aspect = 0;
    uint32_t base_level = 0, level_count = 0, base_layer = 0, layer_count = 0;
    ActivityDiagnosticImageReason reason = ActivityDiagnosticImageReason::Count;
    bool operator==(const ActivityDiagnosticNativeImageMetadata&) const = default;
};
struct ActivityDiagnosticNativeImageBucket {
    ActivityDiagnosticNativeImageMetadata first_metadata;
    uint64_t failures = 0, metadata_changes = 0;
};
enum class ActivityDiagnosticTargetReason : uint32_t {
    Unknown, WrongThread, EmptyTargets, ZeroSize, GpuUnmapped, PhysicalNotUnique, Count
};
enum class ActivityDiagnosticTargetCategory : uint32_t {
    Unknown, RequestedBuffer, OwnerBuffer, RequestedImage, NativeImage, NativeCmask, Count
};
struct ActivityDiagnosticNativeTargetMetadata {
    uint64_t module = 0, target_index = UINT64_MAX, address = 0, size = 0;
    uint32_t resource_index = UINT32_MAX, source = UINT32_MAX, descriptor_index = UINT32_MAX;
    ActivityDiagnosticTargetReason reason = ActivityDiagnosticTargetReason::Unknown;
    ActivityDiagnosticTargetCategory category = ActivityDiagnosticTargetCategory::Unknown;
    bool operator==(const ActivityDiagnosticNativeTargetMetadata&) const = default;
};
struct ActivityDiagnosticNativeTargetBucket {
    ActivityDiagnosticNativeTargetMetadata first_metadata, last_metadata;
    uint64_t failures = 0, metadata_changes = 0;
};
// Capture-only first identity/component and native binding failures.
inline constexpr uint32_t ActivityDiagnosticBindingReasonCount=19,ActivityDiagnosticPreparedReasonCount=8;
// Used ONLY by the enabled diagnostic branch of the original identity comparison.
inline bool ActivityDiagnosticEqualUserdata(const auto& cached,const auto& current,uint32_t& reason,
                                            uint64_t& word_index,uint64_t& cached_word,uint64_t& current_word) {
    if(cached.size()!=current.size()){reason=4;return false;}
    uint64_t index=0;
    return std::ranges::equal(cached,current,[&](uint32_t a,uint32_t b){
        if(a!=b){reason=5;word_index=index;cached_word=a;current_word=b;return false;}
        ++index;return true;
    });
}
struct ActivityDiagnosticIdentityMetadata {
    uint64_t module=0, cached_plan=0, current_plan=0, cached_base=0, current_base=0;
    uint64_t cached_cursor=0, current_cursor=0, cached_words=0, current_words=0;
    uint64_t word_index=UINT64_MAX, cached_word=0, current_word=0, cache_entries=0;
    uint32_t reason=0; // 1 plan,2 shaderbase,3 pushcursor,4 usercount,5 userword
    bool SameGroup(const ActivityDiagnosticIdentityMetadata& o) const {
        return module==o.module && reason==o.reason && word_index==o.word_index;
    }
    bool operator==(const ActivityDiagnosticIdentityMetadata&) const = default;
};
// Capture-only relation of the FIRST image one existing overlap query reported, derived with
// bounded integer arithmetic over already recorded ranges; never queries, walks or maps.
// Classes: 0 unavailable,1 disjoint,2 source,3 native_prefix,4 padding_only,5 unresolved(native unknown).
// Bits: 1 captured,2 intersects_source,4 intersects_native,8 base_eq_source,16 intersects_query,
//       32 image_within_query,64 image_covers_native,128 native_unknown.
// LIMIT: only the FIRST collision is classified; padding_only does not prove all collisions are.
struct ActivityDiagnosticOverlapRelation { uint32_t bits=0,cls=0; };
inline uint64_t ActivityDiagnosticRangeEnd(uint64_t begin,uint64_t size) {
    return size>UINT64_MAX-begin ? UINT64_MAX : begin+size;
}
inline bool ActivityDiagnosticRangesIntersect(uint64_t a,uint64_t as,uint64_t b,uint64_t bs) {
    return as!=0u && bs!=0u && a<ActivityDiagnosticRangeEnd(b,bs) && b<ActivityDiagnosticRangeEnd(a,as);
}
inline bool ActivityDiagnosticRangeContains(uint64_t outer,uint64_t os,uint64_t inner,uint64_t is) {
    return os!=0u && is!=0u && outer<=inner && ActivityDiagnosticRangeEnd(inner,is)<=ActivityDiagnosticRangeEnd(outer,os);
}
inline ActivityDiagnosticOverlapRelation ActivityDiagnosticClassifyOverlap(bool captured,
    uint64_t source_begin,uint64_t source_size,bool native_known,uint64_t native_begin,uint64_t native_size,
    uint64_t query_begin,uint64_t query_size,uint64_t image_begin,uint64_t image_size) {
    ActivityDiagnosticOverlapRelation r;
    if (!captured) return r;
    const bool src=ActivityDiagnosticRangesIntersect(image_begin,image_size,source_begin,source_size);
    const bool nat=native_known && ActivityDiagnosticRangesIntersect(image_begin,image_size,native_begin,native_size);
    const bool qry=ActivityDiagnosticRangesIntersect(image_begin,image_size,query_begin,query_size);
    r.bits=1u|(src?2u:0u)|(nat?4u:0u)|(image_begin==source_begin?8u:0u)|(qry?16u:0u)|
        (ActivityDiagnosticRangeContains(query_begin,query_size,image_begin,image_size)?32u:0u)|
        (native_known && ActivityDiagnosticRangeContains(image_begin,image_size,native_begin,native_size)?64u:0u)|
        (native_known?0u:128u);
    r.cls=src?2u:!qry?1u:!native_known?5u:nat?3u:4u;
    return r;
}
struct ActivityDiagnosticOverlapMetadata {
    uint64_t kind=0,image_id=UINT32_MAX,serial=0,data_begin=0,data_size=0,query_begin=0,query_size=0,relation=0,overlap_class=0;
    bool operator==(const ActivityDiagnosticOverlapMetadata&) const = default;
};
struct ActivityDiagnosticBufferMetadata {
    uint64_t module=0,index=0,source=0,descriptor_index=0,reason=0,binding_reason=0,prepared_reason=0;
    uint64_t address=0,size=0,descriptor_address=0,descriptor_size=0,native_handle=0,native_offset=0,native_range=0;
    uint64_t alignment=0,owner_begin=0,owner_size=0,expected_offset=0,expected_range=0,footprint_begin=0,footprint_size=0;
    uint64_t prepared_offset=0,prepared_sources=0,prepared_buffers=0,resource_flags=0,fields=0,binding_fields=0,prefix=0;
    ActivityDiagnosticOverlapMetadata prepared_overlap,persistent_overlap; // not group keys
    bool SameGroup(const ActivityDiagnosticBufferMetadata& o) const {
        return module==o.module && index==o.index && reason==o.reason && binding_reason==o.binding_reason && prepared_reason==o.prepared_reason;
    }
    bool operator==(const ActivityDiagnosticBufferMetadata&) const = default;
};
template<class Metadata> struct ActivityDiagnosticPairBucket {
    Metadata first,last;
    uint64_t count=0,changes=0;
};

struct ActivityDiagnosticCounters {
    uint64_t calls = 0;
    uint64_t identity_attempts=0,identity_matches=0,identity_calls_matched=0;
    std::array<ActivityDiagnosticPairBucket<ActivityDiagnosticIdentityMetadata>,64> identity_failures {};
    std::array<ActivityDiagnosticPairBucket<ActivityDiagnosticBufferMetadata>,64> buffer_failures {};
    uint64_t identity_overflow=0,buffer_overflow=0;
    std::array<ActivityDiagnosticNativeTargetBucket, 64> native_target_failures {};
    uint64_t native_target_overflow = 0;
    std::array<ActivityDiagnosticNativeImageBucket, 64> native_image_failures {};
    uint64_t native_image_overflow = 0;
    // Exact source indices0..62; category63 aggregates indices>=63. Always64 slots.
    std::array<ActivityDiagnosticSourceBucket, 64> source_failures {};
    std::array<uint64_t, static_cast<size_t>(ActivityDiagnosticReason::Count)> reasons {};
};
inline ActivityDiagnosticCounters& ActivityDiagnosticsCounters() {
    static thread_local ActivityDiagnosticCounters counters;
    return counters;
}
// Keyed by module + compiled image-vector index + exact first failed guard.
// Retain the first metadata and disclose later changes; never allocate or probe.
inline void ActivityDiagnosticNativeImageFailure(uint64_t shader,
                                                const ActivityDiagnosticNativeImageMetadata& metadata) {
    if (!ActivityDiagnosticsEnabled(shader)) return;
    auto& counters = ActivityDiagnosticsCounters();
    for (auto& bucket: counters.native_image_failures) {
        if (bucket.failures != 0u &&
            (bucket.first_metadata.module != metadata.module ||
             bucket.first_metadata.index != metadata.index ||
             bucket.first_metadata.reason != metadata.reason)) continue;
        if (bucket.failures == 0u) bucket.first_metadata = metadata;
        else if (bucket.first_metadata != metadata) ++bucket.metadata_changes;
        ++bucket.failures;
        return;
    }
    ++counters.native_image_overflow;
}
inline void ActivityDiagnosticPrintNativeImages(uint64_t shader,
                                                const ActivityDiagnosticCounters& counters) {
    static constexpr const char* names[] = {"written", "heap", "depth_compare", "mip_stats",
        "missing_image", "missing_view", "unregistered", "native_depth", "requested_depth",
        "depth_association", "stencil_association", "native_metadata", "requested_metadata",
        "native_stencil_range", "requested_stencil_range"};
    static_assert(std::size(names) == static_cast<size_t>(ActivityDiagnosticImageReason::Count));
    for (const auto& bucket: counters.native_image_failures) {
        if (bucket.failures == 0u) continue;
        const auto& m = bucket.first_metadata;
        std::fprintf(stderr, "ACTIVITY_NATIVE_IMAGE v=1 shader=%016llx calls=%llu reason=%s",
            static_cast<unsigned long long>(shader), static_cast<unsigned long long>(counters.calls),
            names[static_cast<size_t>(m.reason)]);
        const auto field = [](const char* key, uint64_t value) {
            std::fprintf(stderr, " %s=%llu", key, static_cast<unsigned long long>(value));
        };
        field("failures", bucket.failures); field("metadata_changes", bucket.metadata_changes);
        field("module", m.module); field("index", m.index); field("source", m.source);
        field("descriptor_index", m.descriptor_index); field("first_use_pc", m.first_use_pc);
        field("resource_class", m.resource_class); field("resource_flags", m.resource_flags);
        field("descriptor_flags", m.descriptor_flags); field("native_id", m.native_id);
        field("serial", m.serial); field("view_handle", m.view_handle);
        field("binding_type", m.binding_type); field("image_flags", m.image_flags);
        field("requested_format", m.requested_format); field("native_format", m.native_format);
        field("requested_metadata", m.requested_metadata); field("native_metadata", m.native_metadata);
        field("requested_begin", m.requested_begin); field("requested_size", m.requested_size);
        field("native_begin", m.native_begin); field("native_size", m.native_size);
        field("requested_meta_begin", m.requested_meta_begin); field("requested_meta_size", m.requested_meta_size);
        field("native_meta_begin", m.native_meta_begin); field("native_meta_size", m.native_meta_size);
        field("requested_stencil_begin", m.requested_stencil_begin); field("requested_stencil_size", m.requested_stencil_size);
        field("native_stencil_begin", m.native_stencil_begin); field("native_stencil_size", m.native_stencil_size);
        field("view_type", m.view_type); field("view_format", m.view_format); field("view_aspect", m.view_aspect);
        field("base_level", m.base_level); field("level_count", m.level_count);
        field("base_layer", m.base_layer); field("layer_count", m.layer_count);
        std::fputc('\n', stderr);
    }
    if (counters.native_image_overflow != 0u)
        std::fprintf(stderr, "ACTIVITY_NATIVE_IMAGE_OVERFLOW v=1 shader=%016llx calls=%llu failures=%llu\n",
            static_cast<unsigned long long>(shader), static_cast<unsigned long long>(counters.calls),
            static_cast<unsigned long long>(counters.native_image_overflow));
}
// The original native target guard already rejected: no query/read/revalidation here.
// Group stable site identity, retaining exact first+last ranges and exposing variation.
inline void ActivityDiagnosticNativeTargetFailure(uint64_t shader,
                                                 const ActivityDiagnosticNativeTargetMetadata& metadata) {
    if (!ActivityDiagnosticsEnabled(shader)) return;
    auto& counters = ActivityDiagnosticsCounters();
    for (auto& bucket : counters.native_target_failures) {
        const auto& first = bucket.first_metadata;
        if (bucket.failures != 0u &&
            (first.module != metadata.module || first.target_index != metadata.target_index ||
             first.reason != metadata.reason || first.category != metadata.category ||
             first.resource_index != metadata.resource_index)) continue;
        if (bucket.failures == 0u) bucket.first_metadata = metadata;
        else if (first != metadata) ++bucket.metadata_changes;
        bucket.last_metadata = metadata;
        ++bucket.failures;
        return;
    }
    ++counters.native_target_overflow;
}
inline void ActivityDiagnosticPrintNativeTargets(uint64_t shader, const ActivityDiagnosticCounters& counters) {
    static constexpr const char* reasons[] = {"unknown", "wrong_thread", "empty_targets", "zero_size", "gpu_unmapped", "physical_not_unique"};
    static constexpr const char* categories[] = {"unknown", "requested_buffer", "owner_buffer", "requested_image", "native_image", "native_cmask"};
    static_assert(std::size(reasons) == static_cast<size_t>(ActivityDiagnosticTargetReason::Count));
    static_assert(std::size(categories) == static_cast<size_t>(ActivityDiagnosticTargetCategory::Count));
    for (const auto& bucket : counters.native_target_failures) {
        if (bucket.failures == 0u) continue;
        const auto& first = bucket.first_metadata; const auto& last = bucket.last_metadata;
        std::fprintf(stderr, "ACTIVITY_NATIVE_TARGET v=1 shader=%016llx calls=%llu reason=%s category=%s module=%llu target_index=%llu resource_index=%u source=%u descriptor_index=%u failures=%llu metadata_changes=%llu first_begin=%llu first_size=%llu last_begin=%llu last_size=%llu\n",
            static_cast<unsigned long long>(shader), static_cast<unsigned long long>(counters.calls),
            reasons[static_cast<size_t>(first.reason)], categories[static_cast<size_t>(first.category)],
            static_cast<unsigned long long>(first.module), static_cast<unsigned long long>(first.target_index),
            first.resource_index, first.source, first.descriptor_index,
            static_cast<unsigned long long>(bucket.failures), static_cast<unsigned long long>(bucket.metadata_changes),
            static_cast<unsigned long long>(first.address), static_cast<unsigned long long>(first.size),
            static_cast<unsigned long long>(last.address), static_cast<unsigned long long>(last.size));
    }
    if (counters.native_target_overflow != 0u)
        std::fprintf(stderr, "ACTIVITY_NATIVE_TARGET_OVERFLOW v=1 shader=%016llx calls=%llu failures=%llu\n",
            static_cast<unsigned long long>(shader), static_cast<unsigned long long>(counters.calls),
            static_cast<unsigned long long>(counters.native_target_overflow));
}
template<class Metadata> inline void ActivityDiagnosticPairRecord(
    std::array<ActivityDiagnosticPairBucket<Metadata>,64>& buckets, uint64_t& overflow, const Metadata& metadata) {
    for (auto& bucket:buckets) {
        if (bucket.count && !bucket.first.SameGroup(metadata)) continue;
        if (!bucket.count) bucket.first=metadata;
        else if (bucket.first!=metadata) ++bucket.changes;
        bucket.last=metadata; ++bucket.count; return;
    }
    ++overflow;
}
inline void ActivityDiagnosticIdentityAttempt(uint64_t shader) {
    if (ActivityDiagnosticsEnabled(shader)) ++ActivityDiagnosticsCounters().identity_attempts;
}
inline void ActivityDiagnosticIdentityMatch(uint64_t shader,bool first) {
    if (!ActivityDiagnosticsEnabled(shader)) return;
    ++ActivityDiagnosticsCounters().identity_matches;
    if (first) ++ActivityDiagnosticsCounters().identity_calls_matched;
}
inline void ActivityDiagnosticIdentityFailure(uint64_t shader,const ActivityDiagnosticIdentityMetadata& m) {
    if (!ActivityDiagnosticsEnabled(shader)) return;
    auto& c=ActivityDiagnosticsCounters(); ActivityDiagnosticPairRecord(c.identity_failures,c.identity_overflow,m);
}
inline void ActivityDiagnosticBufferFailure(uint64_t shader,const ActivityDiagnosticBufferMetadata& m) {
    if (!ActivityDiagnosticsEnabled(shader)) return;
    auto& c=ActivityDiagnosticsCounters(); ActivityDiagnosticPairRecord(c.buffer_failures,c.buffer_overflow,m);
}
inline void ActivityDiagnosticPrintIdentityBuffers(uint64_t shader,const ActivityDiagnosticCounters& c) {
    std::fprintf(stderr,"ACTIVITY_IDENTITY_TOTAL v=1 shader=%016llx calls=%llu candidate_attempts=%llu matched_candidates=%llu calls_with_match=%llu overflow=%llu\n",
        (unsigned long long)shader,(unsigned long long)c.calls,(unsigned long long)c.identity_attempts,
        (unsigned long long)c.identity_matches,(unsigned long long)c.identity_calls_matched,(unsigned long long)c.identity_overflow);
    static constexpr const char* identity_names[]={"unknown","plan_serial","shader_base","push_cursor","user_count","user_word"};
    for(const auto& b:c.identity_failures) if(b.count) {
        const auto& f=b.first;const auto& l=b.last;
        std::fprintf(stderr,"ACTIVITY_IDENTITY v=1 shader=%016llx calls=%llu reason=%s cached_module=%llu word_index=%llu count=%llu metadata_changes=%llu",
            (unsigned long long)shader,(unsigned long long)c.calls,identity_names[f.reason],(unsigned long long)f.module,
            (unsigned long long)f.word_index,(unsigned long long)b.count,(unsigned long long)b.changes);
        const auto field=[](const char* name,uint64_t a,uint64_t z){std::fprintf(stderr," first_%s=%llu last_%s=%llu",name,(unsigned long long)a,name,(unsigned long long)z);};
        field("cached_plan",f.cached_plan,l.cached_plan);
        field("current_plan",f.current_plan,l.current_plan);
        field("cached_base",f.cached_base,l.cached_base);
        field("current_base",f.current_base,l.current_base);
        field("cached_cursor",f.cached_cursor,l.cached_cursor);
        field("current_cursor",f.current_cursor,l.current_cursor);
        field("cached_words",f.cached_words,l.cached_words);
        field("current_words",f.current_words,l.current_words);
        field("cached_word",f.cached_word,l.cached_word);
        field("current_word",f.current_word,l.current_word);
        field("cache_entries",f.cache_entries,l.cache_entries);
        std::fputc('\n',stderr);
    }
    static constexpr const char* buffer_names[]={"unknown","sentinel","source_index","source_address","source_size","native_index","binding"};
    static constexpr const char* binding_names[]={"Unknown","InvalidGuestRange","ZeroSize","NullHandle","ZeroNativeRange","WholeNativeRange","ZeroAlignment","UntrackedRing","NoOwner","OwnerHandle","SourceOutsideOwner","PrefixTooLarge","SizeOverflow","NativeOffset","NativeRange","NativeOutsideOwner","RoundOverflow","FootprintOutsideOwner","FootprintImageOverlap"};
    static constexpr const char* prepared_names[]={"Unknown","NoMatchingSource","NativeOffset","NativeAlignment","NativeRange","NoOwnedBuffer","OwnedBounds","ImageOverlap"};
    static_assert(std::size(binding_names)==ActivityDiagnosticBindingReasonCount);
    static_assert(std::size(prepared_names)==ActivityDiagnosticPreparedReasonCount);
    for(const auto& b:c.buffer_failures) if(b.count) {
        const auto& f=b.first;const auto& l=b.last;
        std::fprintf(stderr,"ACTIVITY_NATIVE_BUFFER v=1 shader=%016llx calls=%llu reason=%s binding_reason=%s prepared_reason=%s module=%llu index=%llu source=%llu descriptor_index=%llu count=%llu metadata_changes=%llu",
            (unsigned long long)shader,(unsigned long long)c.calls,buffer_names[f.reason],binding_names[f.binding_reason],prepared_names[f.prepared_reason],
            (unsigned long long)f.module,(unsigned long long)f.index,(unsigned long long)f.source,(unsigned long long)f.descriptor_index,
            (unsigned long long)b.count,(unsigned long long)b.changes);
        const auto field=[](const char* name,uint64_t a,uint64_t z){std::fprintf(stderr," first_%s=%llu last_%s=%llu",name,(unsigned long long)a,name,(unsigned long long)z);};
        field("address",f.address,l.address);
        field("size",f.size,l.size);
        field("descriptor_address",f.descriptor_address,l.descriptor_address);
        field("descriptor_size",f.descriptor_size,l.descriptor_size);
        field("native_handle",f.native_handle,l.native_handle);
        field("native_offset",f.native_offset,l.native_offset);
        field("native_range",f.native_range,l.native_range);
        field("alignment",f.alignment,l.alignment);
        field("owner_begin",f.owner_begin,l.owner_begin);
        field("owner_size",f.owner_size,l.owner_size);
        field("expected_offset",f.expected_offset,l.expected_offset);
        field("expected_range",f.expected_range,l.expected_range);
        field("footprint_begin",f.footprint_begin,l.footprint_begin);
        field("footprint_size",f.footprint_size,l.footprint_size);
        field("prepared_offset",f.prepared_offset,l.prepared_offset);
        field("prepared_sources",f.prepared_sources,l.prepared_sources);
        field("prepared_buffers",f.prepared_buffers,l.prepared_buffers);
        field("resource_flags",f.resource_flags,l.resource_flags);
        field("fields",f.fields,l.fields);
        field("binding_fields",f.binding_fields,l.binding_fields);
        field("prefix",f.prefix,l.prefix);
        static constexpr const char* overlap_classes[]={"unavailable","disjoint","source","native_prefix","padding_only","unresolved"};
        const auto overlap=[&](const char* tag,const ActivityDiagnosticOverlapMetadata& a,const ActivityDiagnosticOverlapMetadata& z) {
            const auto cls=[](uint64_t v){return v<std::size(overlap_classes)?overlap_classes[v]:"invalid";};
            std::fprintf(stderr," first_%s_class=%s last_%s_class=%s",tag,cls(a.overlap_class),tag,cls(z.overlap_class));
            char name[48];
            const auto sub=[&](const char* n,uint64_t x,uint64_t y){std::snprintf(name,sizeof(name),"%s_%s",tag,n);field(name,x,y);};
            sub("kind",a.kind,z.kind);
            sub("image_id",a.image_id,z.image_id);
            sub("serial",a.serial,z.serial);
            sub("data_begin",a.data_begin,z.data_begin);
            sub("data_size",a.data_size,z.data_size);
            sub("query_begin",a.query_begin,z.query_begin);
            sub("query_size",a.query_size,z.query_size);
            sub("relation",a.relation,z.relation);
        };
        overlap("prepared_overlap",f.prepared_overlap,l.prepared_overlap);
        overlap("persistent_overlap",f.persistent_overlap,l.persistent_overlap);
        if((f.prepared_overlap.kind|f.persistent_overlap.kind|l.prepared_overlap.kind|l.persistent_overlap.kind)!=0u)
            std::fputs(" overlap_scope=first_collision_only",stderr);
        std::fputc('\n',stderr);
    }
    if(c.buffer_overflow) std::fprintf(stderr,"ACTIVITY_NATIVE_BUFFER_OVERFLOW v=1 shader=%016llx calls=%llu count=%llu\n",
        (unsigned long long)shader,(unsigned long long)c.calls,(unsigned long long)c.buffer_overflow);
}

// Called ONLY at the already failed WalkedActive guard. Never walks/probes memory.
inline void ActivityDiagnosticSourceFailure(const auto& plan, uint32_t source) {
    if (!ActivityDiagnosticsEnabled(plan.shader_hash)) return;
    ActivityDiagnosticSourceMetadata metadata;
    metadata.source = source;
    metadata.total_sources = static_cast<uint32_t>(plan.descriptor_sources.size());
    if (source < plan.descriptor_sources.size()) metadata.dwords = plan.descriptor_sources[source].dword_count;
    metadata.counter_memory_index = plan.activity_counter;
    for (uint32_t i=0; i<plan.info.buffers.size(); ++i) if (plan.info.buffers[i].source == source) {
        metadata.kind_mask |= 1u;
        if (metadata.buffer_id == UINT32_MAX) metadata.buffer_id = i;
        ++metadata.buffer_count;
    }
    for (uint32_t i=0; i<plan.info.images.size(); ++i) if (plan.info.images[i].source == source) {
        metadata.kind_mask |= 2u;
        if (metadata.image_id == UINT32_MAX) metadata.image_id = i;
        ++metadata.image_count;
    }
    for (uint32_t i=0; i<plan.info.samplers.size(); ++i) if (plan.info.samplers[i].source == source) {
        metadata.kind_mask |= 4u;
        if (metadata.sampler_id == UINT32_MAX) metadata.sampler_id = i;
        ++metadata.sampler_count;
    }
    auto& bucket = ActivityDiagnosticsCounters().source_failures[source < 63u ? source : 63u];
    if (bucket.failures == 0u) bucket.first_metadata = metadata;
    else if (bucket.first_metadata != metadata) ++bucket.metadata_conflicts;
    ++bucket.failures;
    bool zero = false, nonzero = false;
    uint32_t matches = 0;
    uint32_t aux23_matches = 0, aux86_matches = 0, aux23_value = 0, aux86_value = 0;
    // These address/value/index observations already exist after the ORIGINAL walker.
    // Do not call a walker, raw reader, callback or counter probe here.
    for (const auto& captured: plan.activity_observations) {
        if (captured.memory_index == plan.activity_counter) {
            ++matches;
            zero |= captured.value == 0u;
            nonzero |= captured.value != 0u;
        }
        if (captured.memory_index == 23u) { ++aux23_matches; aux23_value = captured.value; }
        if (captured.memory_index == 86u) { ++aux86_matches; aux86_value = captured.value; }
    }
    bucket.capture_matches += matches;
    if (matches == 0u) ++bucket.count_missing;
    else if (zero && nonzero) ++bucket.count_mixed;
    else if (zero) ++bucket.count_zero;
    else ++bucket.count_nonzero;
    const auto note_aux = [](ActivityDiagnosticAuxCapture& aux, uint32_t count, uint32_t value) {
        aux.capture_matches += count;
        if (count == 0u) ++aux.missing;
        else if (count > 1u) ++aux.duplicate; // Ambiguous even when duplicate values agree.
        else if (value == 0u) ++aux.zero;
        else ++aux.nonzero;
    };
    note_aux(bucket.aux23, aux23_matches, aux23_value);
    note_aux(bucket.aux86, aux86_matches, aux86_value);
}
inline void ActivityDiagnosticNote(uint64_t shader, ActivityDiagnosticReason reason) {
    if (!ActivityDiagnosticsEnabled(shader)) return;
    ++ActivityDiagnosticsCounters().reasons[static_cast<size_t>(reason)];
}
inline void ActivityDiagnosticEpoch(uint64_t shader, const auto& old, const auto& current) {
    if (!ActivityDiagnosticsEnabled(shader)) return;
    if (old.context != current.context) ActivityDiagnosticNote(shader, ActivityDiagnosticReason::EpochContext);
    if (old.mapping != current.mapping) ActivityDiagnosticNote(shader, ActivityDiagnosticReason::EpochMapping);
    if (old.protection != current.protection) ActivityDiagnosticNote(shader, ActivityDiagnosticReason::EpochProtection);
    if (old.buffers != current.buffers) ActivityDiagnosticNote(shader, ActivityDiagnosticReason::EpochBuffers);
    if (old.images != current.images) ActivityDiagnosticNote(shader, ActivityDiagnosticReason::EpochImages);
    if (old.configuration != current.configuration) ActivityDiagnosticNote(shader, ActivityDiagnosticReason::EpochConfiguration);
}
// One cumulative record per1024 materializations on this thread; no clocks, maps,
// allocations or formatting while disabled. Native confirmation belongs to the
// following commit and may lag the latest materialization by one call.
class ActivityDiagnosticMaterialization {
public:
    ActivityDiagnosticMaterialization(uint64_t shader, bool eligible):
        m_shader(shader), m_enabled(ActivityDiagnosticsEnabled(shader)) {
        if (m_enabled) ActivityDiagnosticNote(shader, eligible ? ActivityDiagnosticReason::Eligible : ActivityDiagnosticReason::Ineligible);
    }
    ~ActivityDiagnosticMaterialization() {
        if (!m_enabled) return;
        auto& counters = ActivityDiagnosticsCounters();
        if ((++counters.calls & 1023u) != 0u) return;
        static constexpr const char* names[] = {"eligible","ineligible","runtime_unavailable","no_cache","no_serial","read_epoch_unavailable","identity_mismatch","epoch_context","epoch_mapping","epoch_protection","epoch_buffers","epoch_images","epoch_configuration","strict_source_unavailable","strict_source_changed","targets_rejected","probe_epoch_changed","reuse_hit","capture_disabled","observations_empty_or_large","no_joint_source","no_joint_image","counter_not_observed","flat_read_not_observed","prepare_epoch_unavailable","no_other_observations","pending_ready","confirm_input_invalid","confirm_input_epoch_changed","confirmed","native_no_cache","native_layout_rejected","native_buffer_rejected","native_image_rejected","native_epoch_unavailable","native_targets_rejected","native_epoch_changed","native_confirm_rejected"};
        static_assert(std::size(names) == static_cast<size_t>(ActivityDiagnosticReason::Count));
        std::fprintf(stderr, "ACTIVITY_DIAG v=1 shader=%016llx calls=%llu",
            static_cast<unsigned long long>(m_shader), static_cast<unsigned long long>(counters.calls));
        for (size_t i=0; i<counters.reasons.size(); ++i)
            if (counters.reasons[i]) std::fprintf(stderr, " %s=%llu", names[i], static_cast<unsigned long long>(counters.reasons[i]));
        std::fputc('\n', stderr);
        ActivityDiagnosticPrintNativeImages(m_shader, counters);
        ActivityDiagnosticPrintNativeTargets(m_shader, counters);
        ActivityDiagnosticPrintIdentityBuffers(m_shader,counters);
        // At most64 category lines at the EXISTING1024-call reporting cadence.
        for (size_t i=0; i<counters.source_failures.size(); ++i) {
            const auto& bucket = counters.source_failures[i];
            if (bucket.failures == 0u) continue;
            const auto& m = bucket.first_metadata;
            std::fprintf(stderr,
                "ACTIVITY_SOURCE v=1 shader=%016llx calls=%llu category=%u overflow=%u source=%u sources=%u dwords=%u kind_mask=%u buffer_id=%u buffer_n=%u image_id=%u image_n=%u sampler_id=%u sampler_n=%u counter_index=%u failures=%llu count_zero=%llu count_nonzero=%llu count_missing=%llu count_mixed=%llu capture_matches=%llu metadata_conflicts=%llu aux23_zero=%llu aux23_nonzero=%llu aux23_missing=%llu aux23_duplicate=%llu aux23_matches=%llu aux86_zero=%llu aux86_nonzero=%llu aux86_missing=%llu aux86_duplicate=%llu aux86_matches=%llu\n",
                static_cast<unsigned long long>(m_shader), static_cast<unsigned long long>(counters.calls),
                static_cast<unsigned>(i), static_cast<unsigned>(i == 63u),
                m.source, m.total_sources, m.dwords, m.kind_mask,
                m.buffer_id, m.buffer_count, m.image_id, m.image_count,
                m.sampler_id, m.sampler_count, m.counter_memory_index,
                static_cast<unsigned long long>(bucket.failures),
                static_cast<unsigned long long>(bucket.count_zero),
                static_cast<unsigned long long>(bucket.count_nonzero),
                static_cast<unsigned long long>(bucket.count_missing),
                static_cast<unsigned long long>(bucket.count_mixed),
                static_cast<unsigned long long>(bucket.capture_matches),
                static_cast<unsigned long long>(bucket.metadata_conflicts),
                static_cast<unsigned long long>(bucket.aux23.zero),
                static_cast<unsigned long long>(bucket.aux23.nonzero),
                static_cast<unsigned long long>(bucket.aux23.missing),
                static_cast<unsigned long long>(bucket.aux23.duplicate),
                static_cast<unsigned long long>(bucket.aux23.capture_matches),
                static_cast<unsigned long long>(bucket.aux86.zero),
                static_cast<unsigned long long>(bucket.aux86.nonzero),
                static_cast<unsigned long long>(bucket.aux86.missing),
                static_cast<unsigned long long>(bucket.aux86.duplicate),
                static_cast<unsigned long long>(bucket.aux86.capture_matches));
        }
    }
private:
    uint64_t m_shader;
    bool m_enabled;
};
} // namespace Libs::Graphics::ShaderRecompiler::IR
#endif

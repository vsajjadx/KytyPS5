#ifndef KYTY_READBACK_RECORD_CORRELATION_H_
#define KYTY_READBACK_RECORD_CORRELATION_H_

// Diagnostic only (KYTY_READBACK_ATTRIBUTION). Correlates an existing readback watch's PRE-binding
// mark with the native command that was actually RECORDED afterwards with the same possible
// writer. A record proves a possible write was encoded into the current command buffer at that
// tick: not that any lane executed it, nor when it was submitted or completed. Never changes a
// mark, generation, dirty range or any runtime state; never reads guest/GPU memory.
#include "graphics/host_gpu/renderer/readbackAttribution.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>

namespace Libs::Graphics::ReadbackAttribution {

// Copied from already-final prepared bindings and static resource metadata. source_id is a label
// only: the host buffer ID may be stale and is never dereferenced.
struct RecordedWriter {
    uint64_t shader = 0;
    uint32_t stage = UINT32_MAX, resource = UINT32_MAX, descriptor = UINT32_MAX, source_id = UINT32_MAX;
    uint64_t begin = 0, size = 0;
    uint64_t native_handle = 0, native_offset = 0, native_range = 0;
    uint64_t module = 0; // ShaderModule::id of the bound stage (copied, never resolved)
};
// The record that claimed one pre-mark generation (token). candidates counts every record that
// matched that same token; more than one is ambiguous.
struct PostRecord {
    uint64_t token = 0, sequence = 0, tick = 0;
    RecordedWriter writer;
    uint32_t candidates = 0;
};
struct PreMark {
    uint64_t generation = 0, shader = 0, begin = 0, end = 0;
    uint32_t resource = UINT32_MAX;
};
enum class PostStatus : uint8_t { Untracked, Unrecorded, NewerPremark, Ambiguous, Matched };

inline const char* PostStatusName(PostStatus status) {
    switch (status) {
    case PostStatus::Matched: return "matched";
    case PostStatus::Ambiguous: return "unknown_ambiguous";
    case PostStatus::NewerPremark: return "unknown_newer_premark";
    case PostStatus::Unrecorded: return "unknown_unrecorded";
    default: return "unknown_untracked";
    }
}

// The pre-mark was made by this writer's own binding: same shader and resource, the mark starts at
// the declared guest span and lies inside it (a narrowed written prefix is still inside).
inline bool PreMarkAgrees(const PreMark& mark, const RecordedWriter& writer) {
    return mark.generation != 0 && mark.shader == writer.shader && mark.resource == writer.resource &&
           writer.size != 0 && mark.begin == writer.begin && mark.end > mark.begin &&
           mark.end - writer.begin <= writer.size;
}

inline void NotePostRecord(PostRecord& post, const PreMark& mark, const RecordedWriter& writer,
                           uint64_t sequence, uint64_t tick) {
    if (!PreMarkAgrees(mark, writer)) return; // mismatch: nothing claimed
    if (post.candidates != 0 && post.token == mark.generation) {
        if (post.candidates != UINT32_MAX) ++post.candidates;
        return;
    }
    post = {mark.generation, sequence, tick, writer, 1u};
}

inline PostStatus ClassifyPostRecord(bool tracked, uint64_t mark_generation, const PostRecord& post) {
    if (!tracked) return PostStatus::Untracked;
    if (post.candidates == 0) return PostStatus::Unrecorded;
    if (post.token != mark_generation) return PostStatus::NewerPremark;
    if (post.candidates > 1u) return PostStatus::Ambiguous;
    return PostStatus::Matched;
}

// Walks only the EXISTING watches overlapping the writer's guest span (same bounded walk as the
// pre-mark capture); never inserts, so the watch table stays within its existing cap.
template <class Watch>
void NoteRecordedWriterInWatches(std::map<std::pair<uint64_t, uint64_t>, Watch>& watches, uint64_t max_size,
                                 const RecordedWriter& writer, uint64_t sequence, uint64_t tick) {
    if (!Enabled() || watches.empty() || writer.size == 0 || writer.size > UINT64_MAX - writer.begin) return;
    const uint64_t first = writer.begin > max_size ? writer.begin - max_size : 0;
    const uint64_t end   = writer.begin + writer.size;
    for (auto it = watches.lower_bound({first, 0}); it != watches.end() && it->first.first < end; ++it) {
        const auto [begin, bytes] = it->first;
        if (begin + bytes <= writer.begin) continue;
        auto& watch = it->second;
        NotePostRecord(watch.post,
                       PreMark {watch.mark.generation, watch.mark.shader, watch.mark.begin, watch.mark.end,
                                watch.mark.resource},
                       writer, sequence, tick);
    }
}

// Appended to the existing v=1 readback event. Only a matched record reports values; every other
// status prints explicit unknowns. Owner, mapping and submit time are never known here.
inline int FormatPostRecord(char* out, size_t size, PostStatus status, const PostRecord& post) {
    if (status != PostStatus::Matched) {
        return std::snprintf(out, size,
            " post_status=%s post_seq=unknown post_token=unknown post_record_tick=unknown post_shader=unknown "
            "post_stage=unknown post_resource=unknown post_descriptor=unknown post_begin=unknown "
            "post_size=unknown post_native=unknown post_native_offset=unknown post_native_range=unknown "
            "post_source_id=unknown post_candidates=%u post_module=unknown owner=unknown submit_tick=unknown",
            PostStatusName(status), status == PostStatus::Ambiguous ? post.candidates : 0u);
    }
    const auto& w = post.writer;
    return std::snprintf(out, size,
        " post_status=matched post_seq=%llu post_token=%llu post_record_tick=%llu post_shader=%016llx "
        "post_stage=%u post_resource=%u post_descriptor=%u post_begin=%016llx post_size=%llu "
        "post_native=%016llx post_native_offset=%llu post_native_range=%llu post_source_id=%u "
        "post_candidates=1 post_module=%llu owner=unknown submit_tick=unknown",
        static_cast<unsigned long long>(post.sequence), static_cast<unsigned long long>(post.token),
        static_cast<unsigned long long>(post.tick), static_cast<unsigned long long>(w.shader), w.stage,
        w.resource, w.descriptor, static_cast<unsigned long long>(w.begin),
        static_cast<unsigned long long>(w.size), static_cast<unsigned long long>(w.native_handle),
        static_cast<unsigned long long>(w.native_offset), static_cast<unsigned long long>(w.native_range),
        w.source_id, static_cast<unsigned long long>(w.module));
}

} // namespace Libs::Graphics::ReadbackAttribution
#endif

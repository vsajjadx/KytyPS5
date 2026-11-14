#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ir/ActivityProof.h"
#include "graphics/shader/recompiler/ir/ActivityDiagnostics.h"
#include "graphics/host_gpu/frameStats.h"
#include "graphics/host_gpu/renderer/readbackAttribution.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "common/platform/sysDbg.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <numeric>
#include <map>
#include <mutex>
#include <cstdio>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
uint64_t DownloadWindowSize() {
	static const uint64_t size = [] {
		const char* request = std::getenv("KYTY_DOWNLOAD_WINDOW_KIB");
		const bool wide = request != nullptr && std::strcmp(request, "2048") == 0;
		// Opt-in experiment only: exactly "128" (no numeric/prefix parsing); default stays 512.
		const bool narrow = request != nullptr && std::strcmp(request, "128") == 0;
		const bool valid = wide || narrow || (request != nullptr && std::strcmp(request, "512") == 0);
		const uint64_t effective = wide ? 2 * MiB : narrow ? 128 * 1024 : 512 * 1024;
		if (request != nullptr) {
			std::printf("KYTY_DOWNLOAD_WINDOW_KIB: effective=%llu KiB%s\n",
			            static_cast<unsigned long long>(effective / 1024),
			            valid ? "" : " (invalid value; using default)");
			std::fflush(stdout);
		}
		return effective;
	}();
	return size;
}


// Default-off readback attribution. Tracks only ranges which have actually caused
// a synchronous download. A generation means a conservative GPU-dirty mark, not
// proof that a shader changed the bytes. No diagnostic lock is held across a wait.
struct ReadbackMark {
    uint64_t generation = 0, shader = 0, tick = 0, begin = 0, end = 0;
    uint32_t resource = UINT32_MAX;
    ReadbackAttribution::Phase phase = ReadbackAttribution::Phase::Unknown;
};
struct ReadbackWatch {
    ReadbackMark mark;
    uint64_t previous_generation = 0, previous_tick = 0, marks = 0;
    ReadbackAttribution::PostRecord post; // record that claimed a pre-mark generation (diagnostic)
};
struct ReadbackCapture {
    uint64_t generation = 0, max_size = 0;
    uint64_t record_sequence = 0; // independent of mark generations
    std::map<std::pair<uint64_t, uint64_t>, ReadbackWatch> watches;
};
struct ReadbackCaptures {
    std::mutex mutex;
    std::unordered_map<const BufferCache*, ReadbackCapture> caches;
};
ReadbackCaptures& ReadbackCaptureState() {
    static ReadbackCaptures state;
    return state;
}
void ResetReadbackCapture(const BufferCache* cache) {
    if (!ReadbackAttribution::Enabled()) return;
    auto& state = ReadbackCaptureState();
    std::lock_guard lock(state.mutex);
    state.caches.erase(cache);
}
void CaptureGpuWrite(const BufferCache* cache, uint64_t address, uint64_t size, uint64_t tick) {
    if (!ReadbackAttribution::Enabled()) return;
    auto& state = ReadbackCaptureState();
    std::lock_guard lock(state.mutex);
    auto& capture = state.caches[cache];
    const auto generation = ++capture.generation;
    if (capture.watches.empty()) return;
    const auto first = address > capture.max_size ? address - capture.max_size : 0;
    const auto end = address + size; // ObtainBuffer already validated this guest range.
    for (auto it = capture.watches.lower_bound({first, 0});
         it != capture.watches.end() && it->first.first < end; ++it) {
        const auto [begin, bytes] = it->first;
        if (begin + bytes <= address) continue;
        auto& watch = it->second;
        const auto& context = ReadbackAttribution::current;
        watch.mark = {generation, context.shader, tick, address, end, context.resource, context.phase};
        ++watch.marks;
    }
}
struct ReadbackObservation {
    ReadbackAttribution::Context reader;
    ReadbackMark writer;
    uint64_t previous_generation = 0, previous_tick = 0, marks = 0;
    uint64_t watch_address = 0, watch_size = 0;
    bool tracked = false;
    ReadbackAttribution::PostRecord post;
    ReadbackAttribution::PostStatus post_status = ReadbackAttribution::PostStatus::Untracked;
};
[[maybe_unused]] ReadbackObservation ObserveReadback(const BufferCache* cache, uint64_t tick) {
    ReadbackObservation result;
    if (!ReadbackAttribution::Enabled()) return result;
    result.reader = ReadbackAttribution::current;
    const bool logical_fault = result.reader.fault_size != 0 && !result.reader.request_write;
    const auto address = logical_fault ? result.reader.fault_address : result.reader.request_address;
    const uint64_t size = logical_fault ? result.reader.fault_size : result.reader.request_size;
    result.watch_address = address;
    result.watch_size = size;
    // Keep profiling memory bounded. Large/capped/first-seen ranges remain
    // explicitly unknown; they are still timed and never change runtime behavior.
    if (size == 0 || size > 1024 * 1024 || size > UINT64_MAX - address) return result;
    auto& state = ReadbackCaptureState();
    std::lock_guard lock(state.mutex);
    auto& capture = state.caches[cache];
    auto found = capture.watches.find({address, size});
    if (found == capture.watches.end()) {
        if (capture.watches.size() >= 4096) return result;
        found = capture.watches.emplace(std::pair {address, size}, ReadbackWatch {}).first;
        capture.max_size = std::max(capture.max_size, size);
    }
    auto& watch = found->second;
    result.tracked = true;
    result.writer = watch.mark;
    result.previous_generation = watch.previous_generation;
    result.previous_tick = watch.previous_tick;
    result.marks = watch.marks;
    result.post = watch.post;
    result.post_status = ReadbackAttribution::ClassifyPostRecord(true, watch.mark.generation, watch.post);
    watch.previous_generation = watch.mark.generation;
    watch.previous_tick = tick;
    watch.marks = 0;
    return result;
}
void NoteRecordedWriterCapture(const BufferCache* cache, const ReadbackAttribution::RecordedWriter& writer,
                               uint64_t sequence, uint64_t tick) {
    if (!ReadbackAttribution::Enabled()) return;
    auto& state = ReadbackCaptureState();
    std::lock_guard lock(state.mutex);
    const auto found = state.caches.find(cache);
    if (found == state.caches.end()) return; // no watches yet: nothing to correlate, nothing created
    auto& capture = found->second;
    ReadbackAttribution::NoteRecordedWriterInWatches(capture.watches, capture.max_size, writer, sequence, tick);
}
uint64_t NextRecordedSequence(const BufferCache* cache) {
    if (!ReadbackAttribution::Enabled()) return 0;
    auto& state = ReadbackCaptureState();
    std::lock_guard lock(state.mutex);
    const auto found = state.caches.find(cache);
    return found == state.caches.end() ? 0 : ++found->second.record_sequence;
}
#if defined(TRACY_ENABLE)
[[gnu::noinline]] void TraceReadback(tracy::ScopedZone& zone, const ReadbackObservation& event,
                                   uint64_t window_address, uint64_t window_size,
                                   uint64_t copied_bytes, uint64_t packed_bytes,
                                   uint64_t copy_count, uint64_t tick) {
    if (!zone.IsActive()) return;
    char text[2048];
    const auto& r = event.reader;
    const auto& w = event.writer;
    const int size = std::snprintf(text, sizeof(text),
        "v=1 phase=%s shader=%016llx req=%016llx size=%llu gpu=%u write=%u "
        "faultpc=%016llx faultaddr=%016llx watch=%016llx watch_size=%llu window=%016llx window_size=%llu "
        "copy_bytes=%llu packed_bytes=%llu copies=%llu consumer_tick=%llu tracked=%u "
        "writer_gen=%llu previous_gen=%llu marks=%llu writer=%016llx writer_phase=%s "
        "writer_resource=%u mark_tick=%llu mark_begin=%016llx mark_end=%016llx previous_read_tick=%llu",
        ReadbackAttribution::PhaseName(r.phase), static_cast<unsigned long long>(r.shader),
        static_cast<unsigned long long>(r.request_address), static_cast<unsigned long long>(r.request_size),
        r.origin_gpu ? 1u : 0u, r.request_write ? 1u : 0u,
        static_cast<unsigned long long>(r.fault_pc), static_cast<unsigned long long>(r.fault_address),
        static_cast<unsigned long long>(event.watch_address), static_cast<unsigned long long>(event.watch_size),
        static_cast<unsigned long long>(window_address), static_cast<unsigned long long>(window_size),
        static_cast<unsigned long long>(copied_bytes), static_cast<unsigned long long>(packed_bytes),
        static_cast<unsigned long long>(copy_count),
        static_cast<unsigned long long>(tick), event.tracked ? 1u : 0u,
        static_cast<unsigned long long>(w.generation), static_cast<unsigned long long>(event.previous_generation),
        static_cast<unsigned long long>(event.marks), static_cast<unsigned long long>(w.shader),
        ReadbackAttribution::PhaseName(w.phase), w.resource, static_cast<unsigned long long>(w.tick),
        static_cast<unsigned long long>(w.begin), static_cast<unsigned long long>(w.end),
        static_cast<unsigned long long>(event.previous_tick));
    int total = size;
    if (size > 0 && static_cast<size_t>(size) < sizeof(text)) {
        const int post = ReadbackAttribution::FormatPostRecord(text + size, sizeof(text) - static_cast<size_t>(size),
                                                               event.post_status, event.post);
        total = post > 0 ? size + post : -1;
    }
    if (total > 0 && static_cast<size_t>(total) < sizeof(text)) zone.Text(text, static_cast<size_t>(total));
    else zone.Text("v=1 error=metadata_truncated", 28);
}
#endif

// TEMP DIAG: who marked a range GPU-written that the SRT walker later reads.
struct WriterDiag {
	std::mutex                                                   m;
	std::map<uint64_t, std::pair<uint64_t, std::vector<uint64_t>>> ranges; // start -> (end, stack)
	std::map<std::vector<uint64_t>, uint64_t>                    hits;
	uint64_t                                                     total = 0;
};
inline WriterDiag g_writer_diag;
inline bool WriterDiagOn() {
	static const bool on = std::getenv("KYTY_WRITER_DIAG") != nullptr;
	return on;
}
inline void NoteGpuWriter(uint64_t vaddr, uint64_t size) {
	if (!WriterDiagOn()) return;
	std::vector<uint64_t> frames(12);
	frames.resize(static_cast<size_t>(SysCaptureStackRva(frames.data(), 12)));
	std::scoped_lock lock {g_writer_diag.m};
	g_writer_diag.ranges[vaddr] = {vaddr + size, std::move(frames)};
}
} // namespace
void DiagWalkerSlowRead(uint64_t vaddr) {
	if (!WriterDiagOn()) return;
	std::scoped_lock lock {g_writer_diag.m};
	auto it = g_writer_diag.ranges.upper_bound(vaddr);
	std::vector<uint64_t> key {0xdead};
	uint64_t              span = 0;
	while (it != g_writer_diag.ranges.begin()) {
		--it;
		if (it->second.first > vaddr) { key = it->second.second; span = it->second.first - it->first; break; }
		if (vaddr - it->first > (64ull << 20)) break;
	}
	key.push_back(span);
	g_writer_diag.hits[key]++;
	if ((++g_writer_diag.total % 3000) != 0) return;
	std::vector<std::pair<uint64_t, const std::vector<uint64_t>*>> v;
	for (auto& [k, c]: g_writer_diag.hits) v.emplace_back(c, &k);
	std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
	for (size_t i = 0; i < v.size() && i < 6; i++) {
		std::printf("WRITERDIAG: %llu hits span=0x%llx:", (unsigned long long)v[i].first,
		            (unsigned long long)v[i].second->back());
		for (size_t j = 0; j + 1 < v[i].second->size(); j++)
			std::printf(" %llx", (unsigned long long)(*v[i].second)[j]);
		std::printf("\n");
	}
	std::fflush(stdout);
	g_writer_diag.hits.clear();
}
namespace {
// TEMP DIAG: which guest ranges force synchronous buffer downloads.
struct DlDiag {
	std::mutex                                                               m;
	std::map<std::tuple<uint64_t, int, int>, std::pair<uint64_t, uint64_t>> hits;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
inline DlDiag g_dl_diag;
inline void NoteDl(uint64_t vaddr, uint64_t bytes, int is_write, int gpu_thread) {
	static const bool on = std::getenv("KYTY_DL_DIAG") != nullptr;
	if (!on) return;
	std::scoped_lock lock {g_dl_diag.m};
	auto& h = g_dl_diag.hits[{vaddr >> 12u, is_write, gpu_thread}];
	h.first++;
	h.second += bytes;
	const auto now = std::chrono::steady_clock::now();
	if (now - g_dl_diag.last < std::chrono::seconds(4)) return;
	g_dl_diag.last = now;
	std::vector<std::pair<uint64_t, std::tuple<uint64_t, int, int>>> v;
	uint64_t total = 0;
	for (auto& [k, c]: g_dl_diag.hits) { v.emplace_back(c.first, k); total += c.first; }
	std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
	std::printf("DLDIAG: %llu sync downloads, %zu pages\n", (unsigned long long)total, v.size());
	for (size_t i = 0; i < v.size() && i < 12; i++) {
		auto [pg, w, g] = v[i].second;
		std::printf("DLDIAG:   page 0x%llx size=%d gpu=%d hits=%llu\n",
		            (unsigned long long)(pg << 12u), w, g, (unsigned long long)v[i].first);
	}
	std::fflush(stdout);
	g_dl_diag.hits.clear();
}

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	if (ShaderRecompiler::IR::HostActivityProofEnabled()) ++m_activity_epoch;
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		m_registered_bytes += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		{
			std::lock_guard lock(m_bda_sync_mutex);
			m_dma_accessed.ForEachInRange(
			    buffer.CpuAddress(), buffer.Size(), [&](uint64_t begin, uint64_t end) {
				    for (auto address = begin; address < end; address += CACHING_PAGESIZE) {
					    addresses[buffer.Offset(address) >> CACHING_PAGEBITS] += BDA_ACCESSED;
				    }
			    });
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_registered_bytes -= std::min(m_registered_bytes, buffer.Size());
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_BLOCK(async ? "BufferCache::DownloadBufferMemoryAsync"
	                          : "BufferCache::DownloadBufferMemorySync");
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Start each packed range on its own 64-byte cache line.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
#if defined(TRACY_ENABLE)
	ZoneNamedN(attribution_zone, "ReadbackAttribution::SyncReadback",
	           !async && ReadbackAttribution::Enabled() && tracy::ProfilerAvailable());
#endif
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto download = m_download_buffer.MapDownload(total_size, 64);
	for (auto& copy: copies) {
		copy.dstOffset += download.offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.buffer->Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.buffer->Handle();
	after.offset        = download.offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
#if defined(TRACY_ENABLE)
	[[maybe_unused]] const auto copy_count = copies.size();
	[[maybe_unused]] uint64_t attribution_copy_bytes = 0;
	if (ReadbackAttribution::Enabled()) {
		for (const auto& copy: copies) attribution_copy_bytes += copy.size;
	}
#endif
	auto publish = [total_size, buffer_address, copies = std::move(copies),
	                download = std::move(download)] {
		download.buffer->Invalidate(download.offset, total_size);
		static const bool diff_diag = std::getenv("KYTY_DL_DIAG") != nullptr;
		if (diff_diag) {
			static std::atomic<uint64_t> same_bytes {0}, diff_bytes {0}, calls {0};
			std::vector<uint8_t> old_bytes;
			for (const auto& copy: copies) {
				old_bytes.resize(copy.size);
				if (!Libs::LibKernel::Memory::TryReadBacking(buffer_address + copy.srcOffset,
				                                            old_bytes.data(), copy.size)) continue;
				const auto* fresh = download.mapped + (copy.dstOffset - download.offset);
				uint64_t    d     = 0;
				for (uint64_t i = 0; i < copy.size; i++) d += old_bytes[i] != fresh[i];
				diff_bytes += d;
				same_bytes += copy.size - d;
			}
			if ((++calls % 2000) == 0) {
				std::printf("DLDIFF: %llu downloads, %llu bytes changed, %llu unchanged\n",
				            (unsigned long long)calls.load(), (unsigned long long)diff_bytes.load(),
				            (unsigned long long)same_bytes.load());
				std::fflush(stdout);
			}
		}
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(
			    buffer_address + copy.srcOffset,
			    download.mapped + (copy.dstOffset - download.offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		Libs::Graphics::WaitSiteScope wait_scope(Libs::Graphics::WaitSite::BufferDownloadSync);
		const auto tick = m_scheduler.CurrentTick();
#if defined(TRACY_ENABLE)
		if (attribution_zone.IsActive()) {
			const auto observation = ObserveReadback(this, tick);
			attribution_zone.Value(observation.reader.shader);
			TraceReadback(attribution_zone, observation, vaddr, size, attribution_copy_bytes, total_size, copy_count, tick);
		}
#endif
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	ResetReadbackCapture(this);
	(void)DownloadWindowSize(); // Resolve and report an opt-in experiment once at startup.
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	UpdateGcThresholds(m_graphics.GetTotalMemoryBudget());
}

void BufferCache::UpdateGcThresholds(uint64_t total_budget) {
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget = static_cast<int64_t>(std::min<uint64_t>(total_budget, INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	// Floors must stay reachable when the (possibly learned) budget is small.
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, std::min(GiB, budget / 100 * 50)));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, std::min(2 * GiB, budget / 100 * 80)));
}

BufferCache::~BufferCache() {
	ResetReadbackCapture(this);
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (ShaderRecompiler::IR::HostActivityProofEnabled()) ++m_activity_epoch;
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size, true);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	NoteDl(vaddr, size, static_cast<int>(std::min<uint64_t>(size, 0x7fffffff)), GuestGpu::IsGpuThread() ? 1 : 0);
	const bool guest_thread = !GuestGpu::IsGpuThread();
	const auto guest_begin  = std::chrono::steady_clock::now();
	struct GuestWaitStats {
		~GuestWaitStats() = default;
	};
	auto note_guest_wait = [&](bool downloaded) {
		static const bool on = std::getenv("KYTY_DL_DIAG") != nullptr;
		if (!on || !guest_thread) return;
		static std::mutex m;
		static uint64_t   n = 0, n_dl = 0, us = 0, us_dl = 0;
		static auto       last = std::chrono::steady_clock::now();
		const auto        now  = std::chrono::steady_clock::now();
		const auto        took = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(now - guest_begin).count());
		std::scoped_lock lock {m};
		n++; us += took;
		if (downloaded) { n_dl++; us_dl += took; }
		if (now - last >= std::chrono::seconds(2)) {
			std::printf("GUESTWAIT: %llu guest-thread readbacks, %.1f ms; with a download %llu, %.1f ms (last 2 s)%c",
			            (unsigned long long)n, us / 1000.0, (unsigned long long)n_dl, us_dl / 1000.0, 10);
			std::fflush(stdout);
			n = n_dl = us = us_dl = 0; last = now;
		}
	};
	bool downloaded_any = false;
	auto read_command = [this, vaddr, size, is_write, &downloaded_any] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (IsRegionGpuModified(vaddr, size)) {
			const auto window = DownloadWindowSize();
			const auto start =
			    std::max(Common::AlignDown(vaddr, window), buffer.CpuAddress());
			const auto end = std::min(std::max(start + window, vaddr + size),
			                          buffer.CpuAddress() + buffer.Size());
			// Without GPU-written bytes left (a command-processor store replaced them), the pages
			// must still be unmarked, or the access faults again forever.
			const bool downloaded = DownloadBufferMemory<false>(buffer, start, end - start);
			downloaded_any        = downloaded_any || downloaded;
			if (downloaded ||
			    !m_gpu_modified_ranges.Intersects(start, end - start)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size, true);
		}
	};
	if (ReadbackAttribution::Enabled()) {
		const auto attribution = ReadbackAttribution::Request(vaddr, size, is_write, GuestGpu::IsGpuThread());
		m_scheduler.Context().GetGpu().SendCommandSync([read_command = std::move(read_command), attribution] {
			ReadbackAttribution::Scope attribution_scope(attribution);
			read_command();
		});
	} else {
		m_scheduler.Context().GetGpu().SendCommandSync(std::move(read_command));
	}

	note_guest_wait(downloaded_any);
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept {
		    const BufferUpload upload {&buffer, copies.size()};
		    UploadCopies({&upload, 1}, copies, total_size);
	    });
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

void BufferCache::UploadCopies(std::span<const BufferUpload> uploads,
                               std::span<vk::BufferCopy> copies, uint64_t total_size) {
	if (copies.empty()) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		    size_t first = 0;
		    for (const auto& upload: uploads) {
			    for (auto& copy: copies.subspan(first, upload.end - first)) {
				    const auto address = upload.buffer->CpuAddress() + copy.dstOffset;
				    if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset,
					                                         copy.size)) {
					    EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				    }
				    copy.srcOffset += base_offset;
			    }
			    first = upload.end;
		    }
	    });
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	const vk::MemoryBarrier before {
	    .srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                 vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	    .dstAccessMask = vk::AccessFlagBits::eTransferWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       1, &before, 0, nullptr, 0, nullptr);
	size_t first = 0;
	for (const auto& upload: uploads) {
		native.copyBuffer(source.first->Handle(), upload.buffer->Handle(),
		                  static_cast<uint32_t>(upload.end - first), copies.data() + first);
		first = upload.end;
	}
	const vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                               .dstAccessMask = vk::AccessFlagBits::eMemoryRead |
	                                                vk::AccessFlagBits::eMemoryWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr);
}

namespace {
uintptr_t PreparedUploadThreadToken() {
	static thread_local uint8_t token;
	return reinterpret_cast<uintptr_t>(&token);
}
} // namespace

bool BufferCache::IsPreparedUploadThread() const {
	return m_prepared_upload_thread.load(std::memory_order_relaxed) == PreparedUploadThreadToken();
}

void BufferCache::CheckPreparedUploadThread() {
	const auto thread = PreparedUploadThreadToken();
	auto owner = m_prepared_upload_thread.load(std::memory_order_relaxed);
	if (owner == 0) {
		m_prepared_upload_thread.compare_exchange_strong(owner, thread, std::memory_order_relaxed);
		if (owner == 0) owner = thread;
	}
	// Production uses the GPU command lane. Standalone native fixtures may own an
	// isolated context on their test lane, but never mutate its arena from two lanes.
	EXIT_IF(owner != thread);
}

BufferCache::PreparedUploadScope::PreparedUploadScope(BufferCache& cache)
    : m_cache(cache), m_previous(nullptr) {
	m_cache.CheckPreparedUploadThread();
	m_previous = cache.m_prepared_upload_scope;
	m_cache.m_prepared_upload_scope = this;
}

BufferCache::PreparedUploadScope::~PreparedUploadScope() {
	m_cache.CheckPreparedUploadThread();
	EXIT_IF(m_cache.m_prepared_upload_scope != this);
	m_cache.m_prepared_upload_scope = m_previous;
	// This runs after recording the consumer, including its final command-buffer restart.
	// Early-return paths have no pending consumer but may safely retain the same tick.
	if (!m_uploads.buffers.empty()) {
		m_cache.m_retired_prepared_uploads.push_back(
		    {m_cache.m_scheduler.CurrentTick(), std::move(m_uploads.buffers)});
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ReservePreparedUpload(
    PreparedUpload& uploads, uint64_t size, uint64_t alignment) {
	EXIT_IF(size == 0);
	CheckPreparedUploadThread();
	ReleaseCompletedPreparedUploads();
	const auto atom = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.nonCoherentAtomSize, 1);
	// Isolate every host-written reservation by an atom even on coherent memory. Reusing
	// another part of a chunk must never flush an atom still being read by the GPU.
	const auto divisor = std::gcd(alignment, atom);
	EXIT_IF(alignment != 0 && alignment / divisor > UINT64_MAX / atom);
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	const auto mapped_size = Common::AlignUp(size, atom);
	EXIT_IF(mapped_size < size);
	auto offset = Common::AlignUp(m_prepared_upload_offset, alignment);
	if (!m_prepared_upload_buffer || offset < m_prepared_upload_offset ||
	    offset > m_prepared_upload_buffer->Size() ||
	    mapped_size > m_prepared_upload_buffer->Size() - offset) {
		m_prepared_upload_buffer.reset();
		// A pool-only reference means all preparation owners and GPU-tick callbacks
		// released this backing. Never wait for a consumer which is not recorded yet.
		for (const auto& buffer: m_prepared_upload_pool) {
			if (buffer.use_count() == 1 && buffer->Size() >= mapped_size) {
				m_prepared_upload_buffer = buffer;
				break;
			}
		}
		const auto capacity = std::max<uint64_t>(MiB, mapped_size);
		if (!m_prepared_upload_buffer &&
		    capacity > 64 * MiB - m_prepared_upload_pool_bytes) {
			// Drain only work already recorded. Pending preparations keep their own
			// references, so completing this prefix cannot recycle unconsumed bytes.
			// Extra backing after this retry is bounded by the live preparation itself.
			m_scheduler.Wait(m_scheduler.CurrentTick());
			// Release only upload leases: unrelated deferred cache deletions must
			// retain the scheduler's normal next-operation-boundary timing.
			ReleaseCompletedPreparedUploads();
			for (const auto& buffer: m_prepared_upload_pool) {
				if (buffer.use_count() == 1 && buffer->Size() >= mapped_size) {
					m_prepared_upload_buffer = buffer;
					break;
				}
			}
		}
		if (!m_prepared_upload_buffer) {
			m_prepared_upload_buffer = std::make_shared<Buffer>(
			    m_graphics, m_scheduler, MemoryUsage::Stream, 0,
			    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, capacity);
			if (capacity <= 64 * MiB - m_prepared_upload_pool_bytes) {
				m_prepared_upload_pool.push_back(m_prepared_upload_buffer);
				m_prepared_upload_pool_bytes += capacity;
			}
		}
		m_prepared_upload_offset = 0;
		offset = 0;
	}
	if (std::ranges::find(uploads.buffers, m_prepared_upload_buffer) == uploads.buffers.end()) {
		uploads.buffers.push_back(m_prepared_upload_buffer);
	}
	// A stage may keep its own descriptor preparation as well as the enclosing draw
	// scope. The scope's final tick is sampled only after the actual consumer records.
	if (m_prepared_upload_scope != nullptr && &uploads != &m_prepared_upload_scope->m_uploads) {
		auto& scope = m_prepared_upload_scope->m_uploads.buffers;
		if (std::ranges::find(scope, m_prepared_upload_buffer) == scope.end()) {
			scope.push_back(m_prepared_upload_buffer);
		}
	}
	m_prepared_upload_offset = offset + mapped_size;
	return {m_prepared_upload_buffer.get(), offset};
}

std::pair<Buffer*, uint64_t> BufferCache::UploadPrepared(
    PreparedUpload& uploads, const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	auto result = ReservePreparedUpload(uploads, size, alignment);
	std::memcpy(result.first->Mapped().data() + result.second, source, static_cast<size_t>(size));
	result.first->Flush(result.second, size);
	return result;
}

std::pair<Buffer*, uint64_t> BufferCache::UploadPrepared(
    const void* source, uint64_t size, uint64_t alignment) {
	CheckPreparedUploadThread();
	EXIT_IF(m_prepared_upload_scope == nullptr);
	return UploadPrepared(m_prepared_upload_scope->m_uploads, source, size, alignment);
}

void BufferCache::RetainPreparedUploads(const PreparedUpload& uploads) {
	if (!uploads.buffers.empty()) {
		CheckPreparedUploadThread();
		m_retired_prepared_uploads.push_back({m_scheduler.CurrentTick(), uploads.buffers});
	}
}

void BufferCache::ReleaseCompletedPreparedUploads() {
	while (!m_retired_prepared_uploads.empty() &&
	       m_scheduler.IsFree(m_retired_prepared_uploads.front().tick)) {
		m_retired_prepared_uploads.pop_front();
	}
	// Uncached overflow has no pool reference. Once only the active-page reference
	// remains, all preparations/consumers completed and its memory can be returned.
	if (m_prepared_upload_buffer && m_prepared_upload_buffer.use_count() == 1) {
		m_prepared_upload_buffer.reset();
		m_prepared_upload_offset = 0;
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, bool allow_stream, PreparedUpload* uploads) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	if (allow_stream && !is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		// Only the arena's owner lane may consult the scope pointer. Other existing
		// unscoped cache callers retain their original ring path.
		auto* scope = IsPreparedUploadThread() ? m_prepared_upload_scope : nullptr;
		if (size != 0 && (uploads != nullptr || scope != nullptr)) {
			auto& retained = uploads != nullptr ? *uploads : scope->m_uploads;
			const auto prepared_alignment = std::max<uint64_t>({alignment,
			    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment,
			    m_graphics.physical_device_properties.limits.minTexelBufferOffsetAlignment});
			auto result = ReservePreparedUpload(retained, size, prepared_alignment);
			if (!LibKernel::Memory::CopySparseMemory(
			        vaddr, result.first->Mapped().data() + result.second, size)) {
				EXIT("BufferCache: failed to read sparse prepared buffer memory\n");
			}
			result.first->Flush(result.second, size);
			retained.sources.push_back({vaddr, size, result.first->Handle(), result.second});
			return result;
		}
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		if (ReadbackAttribution::Enabled()) CaptureGpuWrite(this, vaddr, size, m_scheduler.CurrentTick());
		NoteGpuWriter(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// A copy into whole levels of a GPU-written image lands on its GPU contents, not stale guest
	// bytes, while no buffer holds the range.
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		if (!IsRegionGpuModified(src_vaddr, size)) {
			std::memcpy(reinterpret_cast<void*>(dst_vaddr),
			            reinterpret_cast<const void*>(src_vaddr), size);
			return;
		}
		// The source page holds GPU writes, but not these bytes: their guest backing is current.
		// Copying on the GPU instead would mark the destination GPU-written, and the next SRT
		// walk or CPU read of it would force a full GPU sync. UFC 6 menus copy ~500 descriptor
		// dwords a second this way.
		if (size <= 64 * 1024 && !IsRangeGpuWritten(src_vaddr, size)) {
			std::vector<uint8_t> bytes(size);
			if (LibKernel::Memory::TryReadBacking(src_vaddr, bytes.data(), size)) {
				std::memcpy(reinterpret_cast<void*>(dst_vaddr), bytes.data(), size);
				return;
			}
		}
	}

	{
		static const bool on = std::getenv("KYTY_DL_DIAG") != nullptr;
		if (on) {
			static std::atomic<uint64_t> n {0}, n_dst_page {0}, n_dst_bytes {0}, n_src_bytes {0},
			    n_img {0}, bytes {0};
			n++;
			bytes += size;
			if (dst_memory && IsRegionGpuModified(dst_vaddr, size)) n_dst_page++;
			if (dst_memory && IsRangeGpuWritten(dst_vaddr, size)) n_dst_bytes++;
			if (src_memory && IsRangeGpuWritten(src_vaddr, size)) n_src_bytes++;
			if (src_memory && m_texture_cache.FindImageFromRange(src_vaddr, size)) n_img++;
			if ((n % 500) == 0) {
				std::printf("COPYDIAG: gpu copies=%llu bytes=%llu dst_page_gpu=%llu "
				            "dst_bytes_gpu=%llu src_bytes_gpu=%llu src_image=%llu\n",
				            (unsigned long long)n.load(), (unsigned long long)bytes.load(),
				            (unsigned long long)n_dst_page.load(),
				            (unsigned long long)n_dst_bytes.load(),
				            (unsigned long long)n_src_bytes.load(), (unsigned long long)n_img.load());
				std::fflush(stdout);
			}
		}
	}
	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::BindingOwnerRange(uint64_t address, uint64_t size, vk::Buffer handle,
                                   bool written, uint64_t& begin, uint64_t& length,
                                   const PreparedUpload* uploads) const {
	if (!GuestRange {address, size}.Valid() || size == 0u || handle == nullptr) return false;
	if (!written && uploads != nullptr) {
		for (const auto& source: uploads->sources) {
			if (source.buffer == handle && source.address == address && source.size == size) {
				begin = address; length = size;
				return true;
			}
		}
	}
	if (!written && handle == m_stream_buffer.Handle()) {
		// ObtainBuffer's read-only stream copies exactly this source span.
		begin = address; length = size;
		return true;
	}
	auto owner = m_buffers.upper_bound(address);
	if (owner == m_buffers.begin()) return false;
	--owner;
	const auto& buffer = m_slot_buffers[owner->second];
	if (buffer.Handle() != handle || !buffer.IsInBounds(address, size)) return false;
	begin = buffer.CpuAddress(); length = buffer.Size();
	return true;
}

bool BufferCache::BindingActivityRange(uint64_t address, uint64_t size,
                                       const vk::DescriptorBufferInfo& binding, bool written,
                                       uint64_t& begin, uint64_t& length,
                                       const PreparedUpload* uploads,
                                       ActivityBindingFailure* failure) const {
	using Reason = ActivityBindingFailure::Reason;
	using PreparedReason = ActivityBindingFailure::PreparedReason;
	if (failure) *failure = {}; // Fresh diagnostic only; no guard/cache state changes.
	const auto reject = [&](Reason reason) { if (failure) failure->reason = reason; return false; };
	const auto prepared_reject = [&](PreparedReason reason, uint64_t offset = 0) {
		if (failure && static_cast<uint32_t>(reason) > static_cast<uint32_t>(failure->prepared_reason)) {
			failure->prepared_reason = reason; failure->prepared_offset = offset;
			if (reason != PreparedReason::NoMatchingSource) failure->fields |= 2u;
		}
	};
	// Capture-only: keeps the FIRST reported image of the first rejecting query of each kind.
	const auto note_overlap = [&](ActivityBindingFailure::ImageOverlap& out, uint32_t kind,
	                              const TextureCache::RegionOverlapDiagnostic& first_image,
	                              uint64_t query_begin, uint64_t query_size,
	                              uint64_t native_begin, uint64_t native_size) {
		if (!failure || !first_image.valid || out.kind != 0u) return;
		out.kind = kind; out.image_id = first_image.image_id; out.serial = first_image.serial;
		out.data_begin = first_image.data_begin; out.data_size = first_image.data_size;
		out.query_begin = query_begin; out.query_size = query_size;
		const auto relation = ShaderRecompiler::IR::ActivityDiagnosticClassifyOverlap(
		    true, address, size, true, native_begin, native_size, query_begin, query_size,
		    first_image.data_begin, first_image.data_size);
		out.relation = relation.bits; out.overlap_class = relation.cls;
		failure->fields |= kind == 1u ? 64u : 128u;
	};
	// Same original short-circuit predicates, order and number of queries.
	if (!GuestRange {address, size}.Valid()) return reject(Reason::InvalidGuestRange);
	if (size == 0u) return reject(Reason::ZeroSize);
	if (binding.buffer == nullptr) return reject(Reason::NullHandle);
	if (binding.range == 0u) return reject(Reason::ZeroNativeRange);
	if (binding.range == vk::WholeSize) return reject(Reason::WholeNativeRange);
	const auto alignment = m_graphics.StorageMinAlignment();
	if (failure) { failure->alignment = alignment; failure->fields |= 1u; }
	if (alignment == 0u) return reject(Reason::ZeroAlignment);
	if (!written && uploads != nullptr) {
		prepared_reject(PreparedReason::NoMatchingSource);
		for (const auto& source: uploads->sources) {
			if (source.buffer != binding.buffer || source.address != address || source.size != size) continue;
			if (source.offset != binding.offset) { prepared_reject(PreparedReason::NativeOffset, source.offset); continue; }
			if (source.offset % alignment != 0u) { prepared_reject(PreparedReason::NativeAlignment, source.offset); continue; }
			if (binding.range != size) { prepared_reject(PreparedReason::NativeRange, source.offset); continue; }
			prepared_reject(PreparedReason::NoOwnedBuffer, source.offset);
			for (const auto& buffer: uploads->buffers) {
				if (!buffer || buffer->Handle() != binding.buffer) continue;
				if (binding.offset > buffer->Size() || binding.range > buffer->Size() - binding.offset) {
					prepared_reject(PreparedReason::OwnedBounds, source.offset); continue;
				}
				TextureCache::RegionOverlapDiagnostic first_image {};
				if (m_texture_cache.IsRegionRegistered(address, size, failure ? &first_image : nullptr)) {
					// Prepared native view is the exact source.
					if (failure) note_overlap(failure->prepared_overlap, 1u, first_image, address, size, address, size);
					prepared_reject(PreparedReason::ImageOverlap, source.offset); continue;
				}
				begin = address; length = size;
				return true;
			}
		}
	}
	if (binding.buffer == m_stream_buffer.Handle()) return reject(Reason::UntrackedRing);
	auto owner = m_buffers.upper_bound(address);
	if (owner == m_buffers.begin()) return reject(Reason::NoOwner);
	--owner;
	const auto& buffer = m_slot_buffers[owner->second];
	if (failure) { failure->owner_begin = buffer.CpuAddress(); failure->owner_size = buffer.Size(); failure->fields |= 4u; }
	if (buffer.Handle() != binding.buffer) return reject(Reason::OwnerHandle);
	if (!buffer.IsInBounds(address, size)) return reject(Reason::SourceOutsideOwner);
	const auto offset = buffer.Offset(address);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment = offset - aligned_offset;
	if (failure) { failure->expected_offset = aligned_offset; failure->prefix = adjustment; failure->fields |= 8u; }
	if (adjustment >= 256u) return reject(Reason::PrefixTooLarge);
	if (size > UINT64_MAX - adjustment) return reject(Reason::SizeOverflow);
	if (failure) { failure->expected_range = size + adjustment; failure->fields |= 16u; }
	if (binding.offset != aligned_offset) return reject(Reason::NativeOffset);
	if (binding.range != size + adjustment) return reject(Reason::NativeRange);
	if (binding.offset > buffer.Size() || binding.range > buffer.Size() - binding.offset)
		return reject(Reason::NativeOutsideOwner);
	const auto native_begin = address - adjustment;
	const auto native_end = address + size;
	if (native_end > UINT64_MAX - (TRACKER_PAGE_SIZE - 1u)) return reject(Reason::RoundOverflow);
	const auto first = Common::AlignDown(native_begin, TRACKER_PAGE_SIZE);
	const auto last = Common::AlignUp(native_end, TRACKER_PAGE_SIZE);
	if (failure) { failure->footprint_begin = first; failure->footprint_size = last - first; failure->fields |= 32u; }
	if (!buffer.IsInBounds(first, last - first)) return reject(Reason::FootprintOutsideOwner);
	TextureCache::RegionOverlapDiagnostic first_image {};
	if (m_texture_cache.IsRegionRegistered(first, last - first, failure ? &first_image : nullptr)) {
		// Native guest view is [address - prefix, address + size).
		if (failure) note_overlap(failure->persistent_overlap, 2u, first_image, first, last - first, native_begin, native_end - native_begin);
		return reject(Reason::FootprintImageOverlap);
	}
	begin = first; length = last - first;
	return true;
}

uint64_t BufferCache::NextRecordedWorkSequence() const {
	return NextRecordedSequence(this);
}

void BufferCache::NoteRecordedWriter(const ReadbackAttribution::RecordedWriter& writer, uint64_t sequence,
                                     uint64_t tick) const {
	NoteRecordedWriterCapture(this, writer, sequence, tick);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

void BufferCache::QueueReadCheck(uint64_t vaddr, uint32_t expected) {
	m_read_checks.emplace_back(vaddr, expected);
	if (m_read_checks.size() >= 256) {
		FlushReadChecks();
	}
}

void BufferCache::FlushReadChecks() {
	if (m_read_checks.empty()) {
		return;
	}
	struct Check {
		uint64_t vaddr;
		uint32_t expected;
	};
	std::vector<Check>          checks;
	std::vector<vk::BufferCopy> copies;
	std::vector<vk::Buffer>     sources;
	auto download = m_download_buffer.MapDownload(m_read_checks.size() * sizeof(uint32_t), 4);
	for (const auto& [vaddr, expected]: m_read_checks) {
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		if (owner == nullptr || !*owner) continue;
		auto& buffer = m_slot_buffers[*owner];
		if (!buffer.IsInBounds(vaddr, sizeof(uint32_t))) continue;
		copies.push_back({buffer.Offset(vaddr), download.offset + checks.size() * sizeof(uint32_t),
		                  sizeof(uint32_t)});
		sources.push_back(buffer.Handle());
		checks.push_back({vaddr, expected});
	}
	m_read_checks.clear();
	if (checks.empty()) {
		return;
	}
	m_scheduler.EndRendering();
	const auto        native = m_scheduler.Current().Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	for (size_t i = 0; i < copies.size(); i++) {
		native.copyBuffer(sources[i], download.buffer->Handle(), 1, &copies[i]);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &after, 0, nullptr, 0, nullptr);
	auto& context = m_scheduler.Context();
	m_scheduler.DeferPriorityOperation(
	    [checks = std::move(checks), download = std::move(download), &context] {
		    download.buffer->Invalidate(download.offset, checks.size() * sizeof(uint32_t));
		    for (size_t i = 0; i < checks.size(); i++) {
			    uint32_t actual = 0;
			    std::memcpy(&actual, download.mapped + i * sizeof(uint32_t), sizeof(actual));
			    context.NoteReadCheck(checks[i].vaddr, checks[i].expected, actual);
		    }
	    });
}

bool BufferCache::StoreOverGpuDirty(uint64_t vaddr, const void* data, uint64_t size) {
	static const bool disabled = std::getenv("KYTY_NO_STORE_OVER") != nullptr;
	if (disabled) return false;
	if (size == 0 || size > 4096 || (vaddr & 3u) != 0 || (size & 3u) != 0 ||
	    !GuestRange {vaddr, size}.Valid() || !IsRegionRegistered(vaddr, size) ||
	    m_texture_cache.FindImageFromRange(vaddr, size)) {
		return false;
	}
	const auto id     = FindBuffer(vaddr, size);
	auto&      buffer = m_slot_buffers[id];
	if (!buffer.IsInBounds(vaddr, size)) {
		return false;
	}
	if (!LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	// The cached buffer gets the value after the work recorded so far, as the packet would.
	WriteDataBuffer(buffer, vaddr, data, size);
	m_gpu_modified_ranges.Subtract(vaddr, size);
	return true;
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	if (IsPreparedUploadThread()) ReleaseCompletedPreparedUploads();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (const auto oom_events = m_graphics.device_oom_events.load(); oom_events != m_seen_oom_events) {
		// Images ran out of VRAM; shrink our thresholds to the ceiling that was learned.
		m_seen_oom_events = oom_events;
		if (const auto cap = m_graphics.GetTotalMemoryBudget(); cap != 0) {
			UpdateGcThresholds(cap);
		}
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	// m_total_used_memory is total device memory, shared with the texture cache, which cannot
	// evict GPU-modified tiled images. Gating `aggressive` on it alone kept the GC downloading
	// dirty buffers (each a full GPU drain) under pressure it could not relieve. Require this
	// cache to hold a material share of the budget first. KYTY_BUFGC_OWN_SHARE=0 restores the
	// old behaviour. Ported from ReeceKetley/KytyPS5-UFC5 (UFC 5: 1 -> 3-6 fps in fights).
	static const uint64_t own_share_pct = []() -> uint64_t {
		const char* env = std::getenv("KYTY_BUFGC_OWN_SHARE");
		if (env == nullptr) {
			return uint64_t {25};
		}
		char*      end    = nullptr;
		const auto parsed = std::strtoull(env, &end, 10);
		return end == env ? uint64_t {25} : static_cast<uint64_t>(parsed);
	}();
	const bool own_pressure =
	    own_share_pct == 0 || m_registered_bytes >= m_critical_gc_memory * own_share_pct / 100;
	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory && own_pressure;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	Libs::Graphics::WaitSiteScope wait_scope(Libs::Graphics::WaitSite::BufferRetire);
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::PrepareBdaRange(uint64_t vaddr, uint64_t size) {
	const auto id = FindBuffer(vaddr, size);
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	// Enroll the persistent backing in BDA synchronization once. CPU writes queue
	// their dirty subranges thereafter, and ChangeRegister queues a replacement
	// backing in full. PrepareBda consumes those uploads before this dispatch.
	// Keep DMA enrollment aligned to BDA pages, as ChangeRegister uses it to tag
	// whole page-table entries. Existing untagged entries remain valid pointers.
	const auto begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	if (!m_dma_accessed.Contains(begin, end - begin)) {
		m_dma_accessed.Add(begin, end - begin);
		m_bda_sync.Add(begin, end - begin);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	if (!is_write) {
		m_bda_sync.Add(begin, end - begin);
		return;
	}
	m_dma_accessed.ForEachInRange(begin, end - begin, [this](uint64_t first, uint64_t last) {
		m_bda_sync.Add(first, last - first);
	});
}

void BufferCache::MarkDmaAccess(const RangeSet& pages) {
	{
		std::lock_guard lock(m_bda_sync_mutex);
		pages.ForEach([this](uint64_t begin, uint64_t end) {
			m_dma_accessed.Add(begin, end - begin);
			m_bda_sync.Add(begin, end - begin);
		});
	}
	RangeSet                       unbacked;
	std::vector<vk::DeviceAddress> entries;
	pages.ForEach([&](uint64_t begin, uint64_t end) {
		entries.clear();
		for (auto address = begin; address < end; address += CACHING_PAGESIZE) {
			const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
			if (owner == nullptr || !*owner) {
				LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
				unbacked.Add(address, CACHING_PAGESIZE);
				entries.push_back(0);
				continue;
			}
			const auto& buffer = m_slot_buffers[*owner];
			entries.push_back(buffer.BufferDeviceAddress() + buffer.Offset(address) + BDA_ACCESSED);
		}
		if (!unbacked.Contains(begin, end - begin)) {
			WriteDataBuffer(m_bda_pagetable_buffer, PageIndex(begin) * sizeof(vk::DeviceAddress),
			                entries.data(), entries.size() * sizeof(vk::DeviceAddress));
		}
	});
	unbacked.ForEach(
	    [this](uint64_t start, uint64_t end) { (void)FindBuffer(start, end - start); });
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	const auto queued = [this] {
		std::lock_guard lock(m_bda_sync_mutex);
		return std::exchange(m_bda_sync, {});
	}();
	std::vector<vk::BufferCopy> copies;
	std::vector<BufferUpload>   uploads;
	uint64_t                    total_size = 0;
	queued.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
			auto it = m_buffers.upper_bound(start);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < finish; ++it) {
				auto&      buffer = m_slot_buffers[it->second];
				const auto first  = std::max(buffer.CpuAddress(), start);
				const auto last   = std::min(buffer.CpuAddress() + buffer.Size(), finish);
				if (first < last) {
					// Reads follow every range's protection; a write in between re-queues its page.
					m_memory_tracker.ForEachUploadRange(
					    first, last - first, false,
					    [&](uint64_t address, uint64_t bytes) noexcept {
						    if (uploads.empty() || uploads.back().buffer != &buffer) {
							    uploads.push_back({&buffer, 0});
						    }
						    copies.emplace_back(total_size, buffer.Offset(address), bytes);
						    uploads.back().end = copies.size();
						    total_size += bytes;
					    },
					    []() noexcept {});
				}
			}
		});
	});
	UploadCopies(uploads, copies, total_size);
}

} // namespace Libs::Graphics

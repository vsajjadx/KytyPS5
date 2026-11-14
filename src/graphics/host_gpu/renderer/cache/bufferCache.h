#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/readbackRecordCorrelation.h"

#include <atomic>
#include <array>
#include <deque>
#include <map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// Tags the BDA page-table entries of pages a DMA shader accessed; an access to another page
	// records a fault.
	static constexpr vk::DeviceAddress BDA_ACCESSED = 1;

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	// Prepared shader/vertex/index inputs are consumed after allocation-capable discovery.
	// Their backing is monotonic until the actual consumer tick retires, independently of
	// the shared DMA ring's allocation tick.
	struct PreparedUpload {
		struct GuestSource { uint64_t address, size; vk::Buffer buffer; uint64_t offset; };
		std::vector<std::shared_ptr<Buffer>> buffers;
		std::vector<GuestSource> sources;
	};
	class PreparedUploadScope {
	public:
		explicit PreparedUploadScope(BufferCache& cache);
		~PreparedUploadScope();
		KYTY_CLASS_NO_COPY(PreparedUploadScope);
	private:
		friend class BufferCache;
		BufferCache& m_cache;
		PreparedUploadScope* m_previous;
		PreparedUpload m_uploads;
	};
	[[nodiscard]] std::pair<Buffer*, uint64_t> UploadPrepared(
	    PreparedUpload& uploads, const void* source, uint64_t size, uint64_t alignment);
	[[nodiscard]] std::pair<Buffer*, uint64_t> UploadPrepared(
	    const void* source, uint64_t size, uint64_t alignment);
	// Keep a copy through this command tick; the caller may retain its preparation longer.
	void RetainPreparedUploads(const PreparedUpload& uploads);

	[[nodiscard]] uint64_t BindingEpoch() const { return m_activity_epoch.load(); }
	// GPU thread, read-only: validate the actual committed native handle and bound its guest owner.
	[[nodiscard]] bool BindingOwnerRange(uint64_t address, uint64_t size, vk::Buffer handle,
	                                     bool written, uint64_t& begin, uint64_t& length,
	                                     const PreparedUpload* uploads = nullptr) const;
	struct ActivityBindingFailure {
		enum class Reason : uint32_t { Unknown, InvalidGuestRange, ZeroSize, NullHandle, ZeroNativeRange, WholeNativeRange, ZeroAlignment, UntrackedRing, NoOwner, OwnerHandle, SourceOutsideOwner, PrefixTooLarge, SizeOverflow, NativeOffset, NativeRange, NativeOutsideOwner, RoundOverflow, FootprintOutsideOwner, FootprintImageOverlap, Count };
		enum class PreparedReason : uint32_t { Unknown, NoMatchingSource, NativeOffset, NativeAlignment, NativeRange, NoOwnedBuffer, OwnedBounds, ImageOverlap, Count };
		Reason reason = Reason::Unknown; PreparedReason prepared_reason = PreparedReason::Unknown;
		uint64_t alignment = 0, owner_begin = 0, owner_size = 0, expected_offset = 0, expected_range = 0;
		uint64_t footprint_begin = 0, footprint_size = 0, prepared_offset = 0, prefix = 0;
		uint32_t fields = 0; // alignment1,prepared_offset2,owner4,expected_offset/prefix8,range16,footprint32,
		                     // prepared_overlap64,persistent_overlap128
		// Capture-only FIRST image reported by ONE existing overlap query (no second query/walk).
		// A padding-only FIRST collision does not prove every colliding image is padding-only.
		struct ImageOverlap {
			uint32_t kind = 0; // 0 none, 1 prepared exact source, 2 persistent rounded footprint
			uint32_t image_id = UINT32_MAX; // ImageId::index
			uint64_t serial = 0, data_begin = 0, data_size = 0, query_begin = 0, query_size = 0;
			uint32_t relation = 0, overlap_class = 0; // ActivityDiagnosticClassifyOverlap
		};
		ImageOverlap prepared_overlap, persistent_overlap;
	};
	// Certificate footprint: actual native view identity plus every guest page normal
	// preparation may upload. Image-overlapping preparation remains uncertified.
	[[nodiscard]] bool BindingActivityRange(uint64_t address, uint64_t size,
	                                        const vk::DescriptorBufferInfo& binding, bool written,
	                                        uint64_t& begin, uint64_t& length,
	                                        const PreparedUpload* uploads = nullptr,
	                                        ActivityBindingFailure* failure = nullptr) const;
	// Diagnostic only (KYTY_READBACK_ATTRIBUTION, default off; no-ops otherwise): one sequence per
	// recorded native command, then each possible buffer writer it recorded at `tick` (the tick of
	// the command buffer that holds it). Correlates existing readback watches only.
	[[nodiscard]] uint64_t NextRecordedWorkSequence() const;
	void NoteRecordedWriter(const ReadbackAttribution::RecordedWriter& writer, uint64_t sequence,
	                        uint64_t tick) const;
	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {},
	                                                        bool     allow_stream    = true,
	                                                        PreparedUpload* uploads = nullptr);
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Command-processor store of `size` bytes over GPU-written bytes: updates the cached buffer in
	// command order and the guest backing, so the store needs no download or GPU sync. False
	// when the range is not wholly inside one cached buffer or overlaps an image.
	[[nodiscard]] bool StoreOverGpuDirty(uint64_t vaddr, const void* data, uint64_t size);
	// Copies the GPU dword at `vaddr` back after the work recorded so far and reports it to
	// RenderContext::NoteReadCheck; queued checks are recorded together by FlushReadChecks.
	void QueueReadCheck(uint64_t vaddr, uint32_t expected);
	void FlushReadChecks();
	[[nodiscard]] bool HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Byte-precise: whether GPU writes not yet downloaded cover any of these bytes. Pages are
	// protected as a whole, so a page can fault while the bytes read hold CPU-current data.
	[[nodiscard]] bool IsRangeGpuWritten(uint64_t vaddr, uint64_t size) const {
		return m_gpu_modified_ranges.Intersects(vaddr, size);
	}
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	// Persistent, read-only residency for exact GPU-side address reads.
	void PrepareBdaRange(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	// Tags fault-scan pages as DMA-accessed and queues them; creates buffers for unbacked ones.
	void MarkDmaAccess(const RangeSet& pages);
	// Uploads the CPU-dirty bytes of the queued ranges `mapped` covers behind one barrier pair.
	void SynchronizeBda(const RangeSet& mapped);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	// A write queues only the pages a DMA shader accessed.
	void               QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write = false);
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	void UpdateGcThresholds(uint64_t total_budget);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ReservePreparedUpload(
	    PreparedUpload& uploads, uint64_t size, uint64_t alignment);
	void ReleaseCompletedPreparedUploads();
	void CheckPreparedUploadThread();
	[[nodiscard]] bool IsPreparedUploadThread() const;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	// Copies [previous upload's end, end) go to `buffer`.
	struct BufferUpload {
		Buffer* buffer;
		size_t  end;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void UploadCopies(std::span<const BufferUpload> uploads, std::span<vk::BufferCopy> copies,
	                  uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	std::atomic<uint64_t>                              m_activity_epoch {1};
	RangeSet                                          m_gpu_modified_ranges;
	std::vector<std::pair<uint64_t, uint32_t>>        m_read_checks;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	// Pages DMA shaders accessed, guarded by m_bda_sync_mutex.
	RangeSet                                          m_dma_accessed;
	// Cache at most 64 MiB of upload chunks. Additional in-flight chunks remain owned by
	// their scopes/deferred callbacks and are released after their consumer completes.
	std::vector<std::shared_ptr<Buffer>> m_prepared_upload_pool;
	std::shared_ptr<Buffer> m_prepared_upload_buffer;
	uint64_t m_prepared_upload_offset = 0;
	uint64_t m_prepared_upload_pool_bytes = 0;
	PreparedUploadScope* m_prepared_upload_scope = nullptr;
	std::atomic<uintptr_t> m_prepared_upload_thread {0};
	struct RetiredPreparedUpload {
		uint64_t tick;
		std::vector<std::shared_ptr<Buffer>> buffers;
	};
	std::deque<RetiredPreparedUpload> m_retired_prepared_uploads;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	// Bytes of guest memory this cache holds, maintained in ChangeRegister.
	uint64_t m_registered_bytes = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	uint32_t m_seen_oom_events    = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

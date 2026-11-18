#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics {
namespace {

namespace IR = ShaderRecompiler::IR;

// Smaller tables are often written for one draw, so they enter on a second draw of a frame.
constexpr uint64_t MinTableBytes = 64ull * 1024ull;
constexpr uint32_t ElementMask   = (1u << IR::ImageHeapKindShift) - 1u;

template <typename Pred, typename Fn>
void ForEachRun(size_t begin, size_t end, Pred&& pred, Fn&& fn) {
	while (begin < end) {
		if (!pred(begin)) {
			++begin;
			continue;
		}
		auto run = begin + 1u;
		while (run < end && pred(run)) ++run;
		fn(begin, run);
		begin = run;
	}
}

// Grows [begin, end) to cover [first, last).
void Widen(size_t& begin, size_t& end, size_t first, size_t last) {
	const bool empty = begin == end;
	begin            = empty ? first : std::min(begin, first);
	end              = empty ? last : std::max(end, last);
}

// Probes at first + k * step whose first dword lies within a table of size bytes.
uint64_t Probes(uint64_t size, uint64_t first, uint64_t step) {
	return size >= first + 4u ? (size - first - 4u) / step + 1u : 0u;
}

// Tables grow a little on most frames, so their slot words get room to double.
std::unique_ptr<Buffer> SlotBuffer(RenderContext& context, uint64_t size) {
	return std::make_unique<Buffer>(
	    context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer |
	        vk::BufferUsageFlagBits::eShaderDeviceAddress,
	    std::bit_ceil(size + 4u));
}

// Guest bytes of a table that its tracker pages [begin, end) cover.
std::pair<uint64_t, uint64_t> PageRange(uint64_t base, uint64_t size, size_t begin, size_t end) {
	const auto origin = Common::AlignDown(base, TRACKER_PAGE_SIZE);
	return {std::max(base, origin + begin * TRACKER_PAGE_SIZE),
	        std::min(base + size, origin + end * TRACKER_PAGE_SIZE)};
}

} // namespace

size_t ImageHeap::DescriptorHash::operator()(const DescriptorValue& value) const noexcept {
	return XXH3_64bits(value.dwords.data(), sizeof(value.dwords));
}

ImageHeap::ImageHeap(RenderContext& context, PageManager& page_manager)
    : m_context(context), m_page_manager(page_manager) {
	auto& graphics = context.GetGraphics();
	vk::PhysicalDeviceVulkan12Properties limits {};
	vk::PhysicalDeviceProperties2        properties {};
	properties.pNext = &limits;
	graphics.physical_device.getProperties2(&properties);
	EXIT_NOT_IMPLEMENTED(std::min({limits.maxDescriptorSetUpdateAfterBindSampledImages,
	                               limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
	                               limits.maxPerStageUpdateAfterBindResources}) <
	                     IR::ImageHeapCapacity);

	const vk::DescriptorBindingFlags binding_flags =
	    vk::DescriptorBindingFlagBits::ePartiallyBound |
	    vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	    vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = 1;
	flags_info.pBindingFlags = &binding_flags;
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eSampledImage,
	                                              IR::ImageHeapCapacity,
	                                              vk::ShaderStageFlagBits::eAll, nullptr};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout) !=
	        vk::Result::eSuccess);
	const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eSampledImage,
	                                        IR::ImageHeapCapacity};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	EXIT_IF(graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool) !=
	        vk::Result::eSuccess);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &m_layout;
	EXIT_IF(graphics.device.allocateDescriptorSets(&allocate, &m_set) != vk::Result::eSuccess);
	graphics.image_heap_layout = m_layout;
	// Slot word 0 selects element 0, which stays a null image.
	m_elements.push_back({.binding = {.layout = vk::ImageLayout::eShaderReadOnlyOptimal}});
	m_writes.push_back(0u);
	WriteDescriptors();
}

ImageHeap::~ImageHeap() {
	for (const auto& memory: m_memories) Unwatch(*memory);
	auto& graphics             = m_context.GetGraphics();
	graphics.image_heap_layout = nullptr;
	graphics.device.destroyDescriptorPool(m_pool, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
}

bool ImageHeap::Acquire(const IR::ImageHeapRequest& request, IR::ImageHeapView& result) {
	KYTY_PROFILER_FUNCTION();
	const auto size = request.size & ~uint64_t {3};
	if (request.count == 0u || request.step == 0u ||
	    ((request.base | request.first | request.step) & 3u) != 0u) {
		return false;
	}
	auto memory = std::ranges::find_if(
	    m_memories, [&](const auto& candidate) { return candidate->base == request.base; });
	if (memory == m_memories.end() && size < MinTableBytes &&
	    m_sightings.try_emplace(request.base, m_draw).first->second == m_draw) {
		return false;
	}
	// A table follows its descriptor's size, as the first program of a draw acquires it.
	if (memory == m_memories.end() || ((*memory)->size != size && (*memory)->draw != m_draw)) {
		if (!m_context.IsMapped(request.base, size)) return false;
		if (memory == m_memories.end()) {
			std::scoped_lock lock {m_mutex};
			memory = m_memories.insert(m_memories.end(),
			                           std::make_unique<Memory>(Memory {.base = request.base}));
		}
		Resize(**memory, size);
	}
	// A draw refreshes a table and a view takes its changes once, so all programs of the draw see
	// one snapshot and one set of kinds.
	if ((*memory)->draw != m_draw) {
		if (!Refresh(**memory)) return false;
		(*memory)->draw = m_draw;
	}
	auto&      views   = (*memory)->views;
	const auto bytes   = (*memory)->size;
	const auto profile = IR::ImageHeapResource(*request.root, 0u);
	const auto probes  = Probes(size, request.first, request.step);
	const auto limit   = request.count < probes ? request.count : UINT64_MAX;
	auto       view    = std::ranges::find_if(views, [&](const auto& candidate) {
		return candidate->step == request.step && candidate->first == request.first &&
		       candidate->limit == limit && candidate->profile == profile;
	});
	if (view == views.end()) {
		const auto epoch   = Image::heap_epoch.load(std::memory_order_relaxed);
		const auto count   = std::min(limit, Probes(bytes, request.first, request.step));
		auto       created = std::make_unique<View>(View {.memory        = **memory,
		                                                  .step          = request.step,
		                                                  .first         = request.first,
		                                                  .count         = count,
		                                                  .limit         = limit,
		                                                  .profile       = profile,
		                                                  .resolve_epoch = epoch,
		                                                  .layout_epoch  = epoch});
		created->serial = ++m_view_serial;
		created->slots.resize(bytes / 4u + 1u);
		created->buffer = SlotBuffer(m_context, bytes);
		UpdateProbes(*created, 0u, bytes);
		// The first upload also clears the slots no probe selects.
		created->dirty_begin = 0;
		created->dirty_end   = created->slots.size();
		view = views.insert(views.end(), std::move(created));
	} else {
		// Empty after the first acquire in a draw, since the table refreshes once per draw.
		UpdateProbes(**view, (*view)->changed_begin, (*view)->changed_end);
		(*view)->changed_begin = (*view)->changed_end = 0;
	}
	if (std::ranges::find(m_active, view->get()) == m_active.end()) m_active.push_back(view->get());
	const auto& selected = **view;
	result = {.slots      = selected.buffer->BufferDeviceAddress(),
	          .slot_count = static_cast<uint32_t>(selected.slots.size() - 1u),
	          .kinds      = selected.kinds};
	return true;
}

bool ImageHeap::Refresh(Memory& memory) {
	// GPU writes leave no fault; reading the table downloads them.
	const bool gpu_dirty = m_context.GetBufferCache().HasGpuDirtyBytes(memory.base, memory.size);
	const auto pages     = memory.dirty_pages.size();
	m_runs.clear();
	{
		std::scoped_lock lock {m_mutex};
		// Watch before reading, so a write during the read faults the page again.
		ForEachRun(
		    0, pages, [&](size_t page) { return memory.dirty_pages[page] != 0u; },
		    [&](size_t begin, size_t end) {
			    std::fill(memory.dirty_pages.begin() + begin, memory.dirty_pages.begin() + end,
			              uint8_t {0});
			    Watch<true>(memory, begin, end);
			    m_runs.emplace_back(begin, end);
		    });
	}
	if (gpu_dirty) m_runs.assign(1u, {0u, pages});
	for (size_t run = 0; run < m_runs.size(); ++run) {
		const auto [address, end] = PageRange(memory.base, memory.size, m_runs[run].first,
		                                      m_runs[run].second);
		m_scratch.resize((end - address) / 4u);
		if (!LibKernel::Memory::TryReadBufferBacking(address, m_scratch.data(), end - address)) {
			std::scoped_lock lock {m_mutex};
			for (; run < m_runs.size(); ++run) {
				MarkDirty(memory, m_runs[run].first, m_runs[run].second);
			}
			return false;
		}
		const auto words = std::span(memory.words).subspan((address - memory.base) / 4u,
		                                                   m_scratch.size());
		size_t     first = 0;
		size_t     last  = words.size();
		while (first < last && words[first] == m_scratch[first]) ++first;
		while (first < last && words[last - 1u] == m_scratch[last - 1u]) --last;
		if (first == last) continue;
		std::copy(m_scratch.begin() + first, m_scratch.begin() + last, words.begin() + first);
		const auto offset = address - memory.base;
		for (auto& view: memory.views) {
			Widen(view->changed_begin, view->changed_end, offset + first * 4u, offset + last * 4u);
		}
	}
	return true;
}

void ImageHeap::MarkDirty(Memory& memory, size_t begin, size_t end) {
	ForEachRun(
	    begin, end, [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t first, size_t last) {
		    std::fill(memory.dirty_pages.begin() + first, memory.dirty_pages.begin() + last,
		              uint8_t {1});
		    Watch<false>(memory, first, last);
	    });
}

template <bool track>
void ImageHeap::Watch(const Memory& memory, size_t begin, size_t end) {
	const auto [address, limit] = PageRange(memory.base, memory.size, begin, end);
	m_page_manager.UpdatePageWatchers<track>(address, limit - address);
}

void ImageHeap::Unwatch(const Memory& memory) {
	ForEachRun(
	    0, memory.dirty_pages.size(), [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t begin, size_t end) { Watch<false>(memory, begin, end); });
}

void ImageHeap::InvalidateMemory(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_mutex};
	for (auto& memory: m_memories) {
		// Watches cover whole pages, so a write next to the table must unwatch its page too.
		const auto origin = Common::AlignDown(memory->base, TRACKER_PAGE_SIZE);
		const auto begin  = std::max(address, origin);
		const auto end    = std::min(address + size,
		                             origin + memory->dirty_pages.size() * TRACKER_PAGE_SIZE);
		if (begin >= end) continue;
		MarkDirty(*memory, (begin - origin) / TRACKER_PAGE_SIZE,
		          (end - 1u - origin) / TRACKER_PAGE_SIZE + 1u);
	}
}

void ImageHeap::UpdateProbes(View& view, uint64_t begin, uint64_t end) {
	// A probe reads its descriptor from first + probe * step.
	const uint64_t bytes = view.profile.r128 ? 16u : 32u;
	auto probe = begin < view.first + bytes ? 0u : (begin - view.first - bytes) / view.step + 1u;
	for (; probe < view.count && view.first + probe * view.step < end; ++probe) {
		UpdateProbe(view, probe);
	}
}

void ImageHeap::UpdateProbe(View& view, uint64_t probe) {
	const auto&     words  = view.memory.words;
	const auto      offset = (view.first + probe * view.step) / 4u;
	DescriptorValue value {.dword_count = 8u};
	std::copy_n(words.begin() + offset,
	            std::min<size_t>(view.profile.r128 ? 4u : 8u, words.size() - offset),
	            value.dwords.begin());
	// A buried word stays null in every view until its descriptor changes.
	auto& dead = view.memory.dead[offset];
	if (dead != 0u && dead != DescriptorHash {}(value)) dead = 0u;
	const auto kind = dead != 0u ? 0u : IR::ImageHeapKind(view.profile, value);
	Assign(view, offset, kind == 0u ? 0u : Intern(view, kind, value));
}

void ImageHeap::Assign(View& view, size_t slot, uint32_t entry) {
	const auto previous = std::exchange(view.slots[slot], entry);
	Release(view, previous);
	if (previous != entry) Widen(view.dirty_begin, view.dirty_end, slot, slot + 1u);
}

void ImageHeap::Bury() {
	if (m_unmapped.empty()) return;
	for (const auto& memory: m_memories) {
		for (const auto& view: memory->views) {
			bool buried = false;
			for (const auto& [value, index]: view->elements) {
				const auto& binding = m_elements[index].binding;
				if (binding.image_id && std::ranges::any_of(m_unmapped, [&](GuestRange range) {
					    return ImageRangeOverlaps(binding.desc.info.data, range);
				    })) {
					m_elements[index].stale = buried = true;
				}
			}
			if (!buried) continue;
			for (uint64_t probe = 0; probe < view->count; ++probe) {
				const auto  slot    = (view->first + probe * view->step) / 4u;
				const auto& element = m_elements[view->slots[slot] & ElementMask];
				if (!element.stale) continue;
				memory->dead[slot] = DescriptorHash {}(element.value);
				Assign(*view, slot, 0u);
			}
		}
	}
	m_unmapped.clear();
}

uint32_t ImageHeap::Intern(View& view, uint32_t kind, const DescriptorValue& value) {
	const auto arm = static_cast<uint32_t>(std::ranges::find(view.kind_keys, kind) -
	                                       view.kind_keys.begin());
	if (arm == view.kind_keys.size()) {
		view.kind_keys.push_back(kind);
		view.kinds.push_back(value);
	}
	auto [found, inserted] = view.elements.try_emplace(value, 0u);
	if (inserted) {
		found->second    = Allocate();
		auto& element    = m_elements[found->second];
		element.value    = value;
		element.kind     = kind;
		view.pending.push_back(found->second);
	}
	++m_elements[found->second].refs;
	return arm << IR::ImageHeapKindShift | found->second;
}

void ImageHeap::Release(View& view, uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u || --m_elements[index].refs != 0u) return;
	auto& element = m_elements[index];
	if (const auto found = view.elements.find(element.value);
	    found != view.elements.end() && found->second == index) {
		view.elements.erase(found);
	}
	Unreference(element);
	view.released.push_back(index);
}

void ImageHeap::Unreference(Element& element) {
	if (auto* image = ImageOf(element)) {
		image->heap_refs.fetch_sub(1u, std::memory_order_relaxed);
		image->tick_accessed_last = m_context.GetTextureCache().m_gc_tick;
	}
}

Image* ImageHeap::ImageOf(const Element& element) const {
	auto* image = m_context.GetTextureCache().m_slot_images.try_get(element.binding.image_id);
	return image != nullptr && image->serial == element.serial ? image : nullptr;
}

uint32_t ImageHeap::Allocate() {
	auto& scheduler = m_context.GetCommandScheduler();
	while (!m_retired.empty() && scheduler.IsFree(m_retired.front().first)) {
		m_free.push_back(m_retired.front().second);
		m_retired.pop_front();
	}
	if (m_free.empty()) {
		if (m_elements.size() == IR::ImageHeapCapacity) {
			EXIT("image heaps exceed %u resident images\n", IR::ImageHeapCapacity);
		}
		m_elements.emplace_back();
		return static_cast<uint32_t>(m_elements.size() - 1u);
	}
	const auto index = m_free.back();
	m_free.pop_back();
	m_elements[index] = {};
	return index;
}

void ImageHeap::Retire(View& view) {
	if (view.released.empty()) return;
	// A retired index is reused once its tick ends, so no held entry may keep it.
	std::erase_if(view.held, [&](uint32_t index) { return m_elements[index].refs == 0u; });
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	for (const auto index: view.released) m_retired.emplace_back(tick, index);
	view.released.clear();
}

void ImageHeap::Resize(Memory& memory, uint64_t size) {
	{
		std::scoped_lock lock {m_mutex};
		Unwatch(memory);
		memory.size = size;
		// The next refresh rereads the whole table.
		memory.dirty_pages.assign((Common::AlignUp(memory.base + size, TRACKER_PAGE_SIZE) -
		                           Common::AlignDown(memory.base, TRACKER_PAGE_SIZE)) /
		                              TRACKER_PAGE_SIZE,
		                          uint8_t {1});
	}
	memory.words.resize(size / 4u);
	memory.dead.resize(size / 4u);
	auto& scheduler = m_context.GetCommandScheduler();
	for (auto& view: memory.views) {
		const auto count = std::min(view->limit, Probes(size, view->first, view->step));
		for (auto probe = count; probe < view->count; ++probe) {
			Assign(*view, (view->first + probe * view->step) / 4u, 0u);
		}
		view->count       = count;
		const auto slots  = view->slots.size();
		view->slots.resize(size / 4u + 1u);
		view->dirty_end   = std::min(view->dirty_end, view->slots.size());
		view->dirty_begin = std::min(view->dirty_begin, view->dirty_end);
		if (view->slots.size() * sizeof(uint32_t) > view->buffer->Size()) {
			if (scheduler.Active()) {
				scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
			}
			view->buffer = SlotBuffer(m_context, size);
			Widen(view->dirty_begin, view->dirty_end, 0u, view->slots.size());
		} else if (slots < view->slots.size()) {
			// Device words past the old end may hold stale slots.
			Widen(view->dirty_begin, view->dirty_end, slots, view->slots.size());
		}
	}
}

void ImageHeap::Rebind(RenderExecutor& executor) {
	KYTY_PROFILER_FUNCTION();
	auto&      textures = m_context.GetTextureCache();
	const auto epoch    = Image::heap_epoch.load(std::memory_order_relaxed);
	const auto serial   = Image::serials.load(std::memory_order_relaxed);
	for (auto* view: m_active) {
		bool       stale = false;
		const auto mark  = [&](uint32_t index) { m_elements[index].stale = stale = true; };
		if (view->resolve_epoch != epoch) {
			view->resolve_epoch = epoch;
			for (const auto& [value, index]: view->elements) {
				const auto& binding = m_elements[index].binding;
				if (!binding.image_id) continue;
				const auto* image = ImageOf(m_elements[index]);
				if (image == nullptr || !image->registered || image->depth_id ||
				    image->binding.needs_rebind) {
					mark(index);
				} else {
					(void)textures.FindTexture(binding.image_id, binding.desc);
				}
			}
		}
		// Discovery applies the color metadata clears written since the last draw.
		std::erase_if(view->dcc, [&](uint32_t index) {
			const auto& element = m_elements[index];
			if (element.refs == 0u || element.stale) return true;
			if (executor.ResolveTexture(IR::ImageHeapResource(view->profile, element.kind),
			                            element.value).image_id == element.binding.image_id) {
				return false;
			}
			mark(index);
			return true;
		});
		// Held elements retry once an image is created, such as a render target of their extent.
		if (view->held_serial != serial) {
			view->held_serial = serial;
			std::erase_if(view->held, [&](uint32_t index) {
				const auto& element = m_elements[index];
				if (element.refs == 0u || element.stale) return true;
				const auto binding = executor.ResolveTexture(
				    IR::ImageHeapResource(view->profile, element.kind), element.value);
				if (binding.image_id) mark(index);
				// An unmapped texture stays null.
				return binding.image_id || binding.desc.info.data.Empty();
			});
		}
		if (stale) {
			// A changed image needs a new element; pending draws still read the old one. The new
			// element keeps the old value, since a view takes table changes only when acquired.
			std::erase_if(view->elements,
			              [&](const auto& item) { return m_elements[item.second].stale; });
			for (uint64_t probe = 0; probe < view->count; ++probe) {
				const auto  slot    = (view->first + probe * view->step) / 4u;
				const auto& element = m_elements[view->slots[slot] & ElementMask];
				if (!element.stale) continue;
				const auto value = element.value; // Intern may grow m_elements.
				Assign(*view, slot, Intern(*view, element.kind, value));
			}
		}
		for (const auto index: view->pending) {
			if (m_elements[index].refs == 0u) continue;
			auto binding = executor.ResolveTexture(
			    IR::ImageHeapResource(view->profile, m_elements[index].kind),
			    m_elements[index].value);
			if (binding.image_id) {
				auto& image = textures.GetImage(binding.image_id);
				image.heap_refs.fetch_add(1u, std::memory_order_relaxed);
				m_elements[index].serial = image.serial;
				binding.layout     = image.ReadLayout();
				binding.image_view = textures.FindTexture(binding.image_id, binding.desc);
				if (binding.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
					view->dcc.push_back(index);
				}
				view->away.push_back(index);
			} else if (!binding.desc.info.data.Empty()) {
				view->held.push_back(index);
			}
			m_elements[index].binding = std::move(binding);
			m_writes.push_back(index);
		}
		view->pending.clear();
	}
	WriteDescriptors();
}

void ImageHeap::WriteDescriptors() {
	if (m_writes.empty()) return;
	std::vector<vk::DescriptorImageInfo> infos;
	std::vector<vk::WriteDescriptorSet>  writes(m_writes.size());
	infos.reserve(m_writes.size());
	for (size_t write = 0; write < m_writes.size(); ++write) {
		const auto& binding = m_elements[m_writes[write]].binding;
		infos.push_back({nullptr, binding.image_view, binding.layout});
		writes[write].dstSet          = m_set;
		writes[write].dstArrayElement = m_writes[write];
		writes[write].descriptorCount = 1;
		writes[write].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[write].pImageInfo      = &infos.back();
	}
	m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()),
	                                                    writes.data(), 0, nullptr);
	m_writes.clear();
}

void ImageHeap::Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point,
                       vk::PipelineLayout layout) {
	KYTY_PROFILER_FUNCTION();
	const auto epoch = Image::heap_epoch.load(std::memory_order_relaxed);
	for (auto* view: m_active) {
		if (view->dirty_begin != view->dirty_end) {
			const auto slots = std::span(view->slots).subspan(view->dirty_begin,
			                                                  view->dirty_end - view->dirty_begin);
			auto& stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
			const auto offset = stream.Copy(slots.data(), slots.size_bytes(), sizeof(uint32_t));
			view->buffer->CopyFrom(buffer, stream, offset, view->dirty_begin * sizeof(uint32_t),
			                       slots.size_bytes());
			view->dirty_begin = view->dirty_end = 0;
		}
		Retire(*view);
		if (view->layout_epoch != epoch) {
			view->layout_epoch = epoch;
			view->away.clear();
			for (const auto& [value, index]: view->elements) view->away.push_back(index);
		}
		// This draw's own bindings own the layouts of its attachments and storage images. An image
		// this draw's discovery replaced lives until the tick ends, and its slots still select it.
		std::erase_if(view->away, [&](uint32_t index) {
			const auto& element = m_elements[index];
			auto*       image   = ImageOf(element);
			if (element.refs == 0u || image == nullptr) return true;
			if (image->binding.is_target || image->binding.shader_write ||
			    image->binding.force_general) {
				return false;
			}
			const auto& view_info = element.binding.desc.view_info;
			image->Transit(image->ReadLayout(), vk::AccessFlagBits2::eShaderRead,
			               ImageSubresourceRange {view_info.base_level, view_info.level_count,
			                                      view_info.base_layer, view_info.layer_count},
			               buffer.Handle());
			return true;
		});
	}
	m_active.clear();
	buffer.Handle().bindDescriptorSets(bind_point, layout, IR::ImageHeapSet, 1, &m_set, 0,
	                                   nullptr);
}

void ImageHeap::UnmapMemory(uint64_t address, uint64_t size) {
	// Descriptors of textures in the range stay null even once it is mapped again.
	if (!m_unmapped.empty() && m_unmapped.back().End() == address) {
		m_unmapped.back().size += size;
	} else {
		m_unmapped.push_back({address, size});
	}
	std::scoped_lock lock {m_mutex};
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		if (address >= memory->base + memory->size || memory->base >= address + size) {
			return false;
		}
		Drop(*memory);
		return true;
	});
}

void ImageHeap::RunGarbageCollector() {
	// A smaller table rebuilt each frame moves to a new base. Larger ones keep their dead words.
	std::scoped_lock lock {m_mutex};
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		if (memory->size >= MinTableBytes || memory->draw > m_suspend_draw) return false;
		Drop(*memory);
		return true;
	});
	m_suspend_draw = m_draw;
	m_sightings.clear();
}

void ImageHeap::Drop(Memory& memory) {
	auto& scheduler = m_context.GetCommandScheduler();
	Unwatch(memory);
	for (auto& view: memory.views) {
		std::erase(m_active, view.get());
		for (const auto& [value, index]: view->elements) {
			Unreference(m_elements[index]);
			view->released.push_back(index);
		}
		Retire(*view);
		if (scheduler.Active()) {
			scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
		}
	}
}

bool ImageHeap::ReadCallReference(uint32_t image, const IR::ImageResource& resource,
                                  std::span<const uint32_t> flattened_srt,
                                  CallClosureProof::Reference& result,
                                  std::string& reason) const {
	if (!resource.heap || resource.resource_class != IR::ImageResourceClass::Sampled ||
	    !resource.read || resource.written || resource.atomic || resource.atomic64) {
		reason = "image heap closure requires read-only sampled resources";
		return false;
	}
	const auto offset = resource.indirect_mapping_offset;
	if (offset > flattened_srt.size() || flattened_srt.size() - offset < 3u) {
		reason = "image heap closure has no complete slot mapping";
		return false;
	}
	result.image = image;
	result.mapping_offset = offset;
	result.slot_count = flattened_srt[offset];
	result.slots = uint64_t {flattened_srt[offset + 1u]} |
	               (uint64_t {flattened_srt[offset + 2u]} << 32u);
	result.profile = IR::ImageHeapResource(resource, 0u);
	if (result.slot_count == 0u || result.slots == 0u) {
		reason = "image heap closure has an empty slot mapping";
		return false;
	}
	return true;
}

bool ImageHeap::ReadCallView(const View& view, CallClosureProof::View& result,
                             std::string& reason) const {
	const GuestRange table {view.memory.base, view.memory.size};
	if (view.serial == 0u || view.memory.draw != m_draw || !table.Valid() ||
	    view.buffer == nullptr || view.slots.empty() ||
	    view.slots.size() - 1u > UINT32_MAX ||
	    view.memory.size / sizeof(uint32_t) != view.slots.size() - 1u ||
	    view.slots.size() > view.buffer->Size() / sizeof(uint32_t) ||
	    !view.profile.indirect_resources.empty()) {
		reason = "image heap closure has an invalid or stale source view";
		return false;
	}
	result.serial = view.serial;
	result.table = table;
	result.buffer = view.buffer->Handle();
	result.slots = view.buffer->BufferDeviceAddress();
	result.slot_count = static_cast<uint32_t>(view.slots.size() - 1u);
	result.first = view.first;
	result.step = view.step;
	result.count = view.count;
	result.limit = view.limit;
	result.element_count = view.elements.size();
	result.profile = view.profile;
	return true;
}

bool ImageHeap::ReadCallElement(uint32_t index, CallClosureProof::Element& result,
                                std::string& reason) const {
	if (index >= m_elements.size() || std::ranges::find(m_writes, index) != m_writes.end()) {
		reason = "image heap closure has a missing or unpublished sampled descriptor";
		return false;
	}
	const auto& element = m_elements[index];
	if (element.stale || (index != 0u && element.refs == 0u)) {
		reason = "image heap closure has a stale or unreferenced sampled descriptor";
		return false;
	}
	const auto& binding = element.binding;
	const bool image = static_cast<bool>(binding.image_id);
	const bool view = binding.image_view != nullptr;
	if (image != view || (image ? element.serial == 0u : element.serial != 0u) ||
	    (index == 0u && (image || element.kind != 0u))) {
		reason = "image heap closure has an inconsistent null sampled descriptor";
		return false;
	}
	result.index = index;
	result.kind = element.kind;
	result.descriptor = element.value;
	result.image_id = binding.image_id;
	result.serial = element.serial;
	result.view = binding.image_view;
	result.layout = binding.layout;
	// Both element zero and a resolved, intentional nonzero null binding have no image access.
	// Its descriptor identity is still retained and its source table is still checked.
	if (!image) return true;
	if (!binding.desc.heap || binding.desc.type != TextureCache::BindingType::Texture ||
	    !binding.desc.info.data.Valid()) {
		reason = "image heap closure has no bounded sampled texture view";
		return false;
	}
	result.requested_data = binding.desc.info.data;
	if (binding.desc.info.HasStencil()) {
		if (!binding.desc.info.stencil.Valid()) {
			reason = "image heap closure has invalid requested stencil input";
			return false;
		}
		result.requested_stencil = binding.desc.info.stencil;
	}
	if (binding.desc.info.HasMetadata()) {
		if (!binding.desc.info.metadata.range.Valid()) {
			reason = "image heap closure has invalid requested image metadata";
			return false;
		}
		result.requested_metadata = binding.desc.info.metadata.range;
	}
	return true;
}

bool ImageHeap::ReadCallImage(ImageId id, const Image& image,
                              CallClosureProof::ImageInput& result, std::string& reason) {
	// Caller holds TextureCache::m_lock. Even a native-less stencil plane remains a
	// registered cache input: DeleteImage unlinks it before unregistering the record.
	if (!image.registered || image.serial == 0u || !image.info.data.Valid()) {
		reason = "image heap closure has an unavailable or unbounded image input";
		return false;
	}
	result = {};
	result.image_id = id;
	result.depth_id = image.depth_id;
	result.serial = image.serial;
	result.image = image.backing.image;
	result.data = image.info.data;
	if (image.info.HasMetadata()) {
		if (!image.info.metadata.range.Valid()) {
			reason = "image heap closure has invalid native image metadata";
			return false;
		}
		result.metadata = image.info.metadata.range;
	}
	// This range can exist before any association record is created.
	if (image.info.HasStencil()) {
		if (!image.info.stencil.Valid()) {
			reason = "image heap closure has invalid native stencil input";
			return false;
		}
		result.stencil = image.info.stencil;
	}
	if (image.IsTracked()) {
		if (image.track_addr_end <= image.track_addr ||
		    !GuestRange {image.track_addr, image.track_addr_end - image.track_addr}.Valid()) {
			reason = "image heap closure has an invalid tracked image range";
			return false;
		}
		result.tracked = {image.track_addr, image.track_addr_end - image.track_addr};
	}
	result.format = image.info.pixel_format;
	result.guest_format = image.info.guest_format;
	result.type = image.info.type;
	result.extent = image.info.extent;
	result.resources = image.info.resources;
	result.mapping = image.stencil_subresources;
	result.pitch = image.info.pitch;
	result.bytes_per_block = image.info.bytes_per_block;
	result.samples = image.info.samples;
	result.tile = image.info.tile_mode;
	result.metadata_kind = image.info.metadata.kind;
	result.compression = image.info.metadata.compression;
	result.clear_word = image.info.metadata.clear_word;
	result.stencil_compressed = image.info.metadata.stencil_compressed;
	result.clear_register_valid = image.info.metadata.clear_register_valid;
	result.dcc_alpha_msb = image.info.metadata.dcc_alpha_msb;
	result.htile_initial_clear = image.info.htile_initial_clear;
	return true;
}

bool ImageHeap::CheckCallAssociation(const CallClosureProof::ImageInput& owner,
                                     const CallClosureProof::ImageInput& plane,
                                     std::string& reason) {
	// AssociateStencil creates one direct S8 plane mapped into a native depth owner.
	// Do not turn an unexpected graph into an arbitrary recursive dependency walk.
	const auto& mapping = plane.mapping;
	if (plane.image_id == owner.image_id || plane.depth_id != owner.image_id ||
	    DepthAspectTransferFormat(owner.format) == vk::Format::eUndefined ||
	    plane.format != vk::Format::eS8Uint ||
	    plane.guest_format != Prospero::BufferFormat::k8UInt ||
	    plane.bytes_per_block != 1u || plane.samples != 1u ||
	    plane.type != owner.type || plane.extent != owner.extent || plane.tile != owner.tile ||
	    mapping.level_count == 0u || mapping.base_level >= owner.resources.levels ||
	    mapping.level_count > owner.resources.levels - mapping.base_level ||
	    mapping.layer_count == 0u || mapping.base_layer >= owner.resources.layers ||
	    mapping.layer_count > owner.resources.layers - mapping.base_layer ||
	    plane.resources.levels != mapping.level_count ||
	    plane.resources.layers != mapping.layer_count) {
		reason = "image heap closure has an inconsistent stencil association mapping";
		return false;
	}
	return true;
}

bool ImageHeap::CaptureCallOwner(ImageId id, uint64_t serial, CallClosureProof::Owner& result,
                                 std::vector<CallClosureProof::ImageInput>& associations,
                                 std::string& reason) const {
	auto& textures = m_context.GetTextureCache();
	// Capture the complete family under the existing lock. No guest-memory operation,
	// GPU work or retained pointer crosses it; faults may otherwise shrink tracked ranges.
	std::scoped_lock lock {textures.m_lock};
	const auto* owner = textures.m_slot_images.try_get(id);
	if (owner == nullptr || serial == 0u || owner->serial != serial) {
		reason = "image heap closure lost its sampled image owner";
		return false;
	}
	if (owner->depth_id || owner->backing.image == nullptr) {
		reason = "image heap closure has no native sampled image owner";
		return false;
	}
	if (!ReadCallImage(id, *owner, result.input, reason)) return false;
	result.association_begin = associations.size();
	result.association_count = owner->stencil_ids.size();
	std::unordered_set<uint64_t> seen;
	seen.reserve(owner->stencil_ids.size());
	for (const auto child_id: owner->stencil_ids) {
		const auto* child = textures.m_slot_images.try_get(child_id);
		if (child == nullptr || !child->stencil_ids.empty()) {
			reason = "image heap closure has a missing or nested stencil association";
			return false;
		}
		CallClosureProof::ImageInput plane;
		if (!ReadCallImage(child_id, *child, plane, reason) ||
		    !CheckCallAssociation(result.input, plane, reason)) return false;
		if (!seen.insert(plane.serial).second) {
			reason = "image heap closure has a repeated stencil association";
			return false;
		}
		// Existing color records can be reused as planes. Their native backing is optional;
		// UploadImage and descriptor resolution use depth_id, not that unused backing.
		associations.push_back(plane);
	}
	return true;
}

bool ImageHeap::ValidateCallOwner(
    const CallClosureProof::Owner& saved,
    std::span<const CallClosureProof::ImageInput> associations, std::string& reason) const {
	if (saved.association_begin > associations.size() ||
	    saved.association_count > associations.size() - saved.association_begin) {
		reason = "image heap closure has an invalid association certificate";
		return false;
	}
	auto& textures = m_context.GetTextureCache();
	std::scoped_lock lock {textures.m_lock};
	const auto* owner = textures.m_slot_images.try_get(saved.input.image_id);
	if (owner == nullptr || owner->serial != saved.input.serial) {
		reason = "image heap closure lost its sampled image owner";
		return false;
	}
	CallClosureProof::ImageInput current;
	if (!ReadCallImage(saved.input.image_id, *owner, current, reason)) return false;
	if (current != saved.input) {
		reason = "image heap closure native owner changed during commit";
		return false;
	}
	if (owner->stencil_ids.size() != saved.association_count) {
		reason = "image heap closure association graph changed during commit";
		return false;
	}
	// Exact ordered membership also detects reassociation; no allocation is needed here.
	for (size_t i = 0; i < saved.association_count; ++i) {
		const auto& old = associations[saved.association_begin + i];
		if (owner->stencil_ids[i] != old.image_id) {
			reason = "image heap closure association graph changed during commit";
			return false;
		}
		const auto* child = textures.m_slot_images.try_get(old.image_id);
		if (child == nullptr || child->serial != old.serial || !child->stencil_ids.empty()) {
			reason = "image heap closure stencil association retired or changed during commit";
			return false;
		}
		if (!ReadCallImage(old.image_id, *child, current, reason)) return false;
		if (current != old) {
			reason = "image heap closure stencil association inputs changed during commit";
			return false;
		}
	}
	return true;
}


bool ImageHeap::CaptureCallClosure(std::span<const IR::ImageResource> resources,
                                   std::span<const uint32_t> flattened_srt,
                                   std::span<const GuestRange> protected_ranges,
                                   CallClosureProof& proof, std::string& reason) const {
	proof = {};
	reason.clear();
	uint64_t previous_end = 0;
	for (const auto& range: protected_ranges) {
		if (!range.Valid() || range.address < previous_end) {
			reason = "image heap closure has invalid protected intervals";
			return false;
		}
		previous_end = range.End();
	}
	if (m_set == nullptr) {
		reason = "image heap closure has no native sampled descriptor set";
		return false;
	}
	if (!ReadCallElement(0u, proof.null_element, reason)) return false;
	std::unordered_map<uint64_t, ImageId> owners;
	const auto append_inputs = [&](const CallClosureProof::ImageInput& input) {
		for (const auto range: {input.data, input.metadata, input.stencil, input.tracked}) {
			if (!range.Empty()) proof.ranges.push_back(range);
		}
	};
	for (uint32_t index = 0; index < resources.size(); ++index) {
		if (!resources[index].heap) continue;
		CallClosureProof::Reference reference;
		if (!ReadCallReference(index, resources[index], flattened_srt, reference, reason)) return false;
		const auto selected = std::ranges::find_if(m_active, [&](const View* view) {
			return view != nullptr && view->buffer != nullptr &&
			       view->buffer->BufferDeviceAddress() == reference.slots &&
			       view->slots.size() == uint64_t {reference.slot_count} + 1u;
		});
		if (selected == m_active.end() || (*selected)->profile != reference.profile) {
			reason = "image heap closure mapping has no matching active source view";
			return false;
		}
		const auto& view = **selected;
		reference.view_serial = view.serial;
		proof.references.push_back(reference);
		if (std::ranges::any_of(proof.views, [&](const auto& old) { return old.serial == view.serial; })) {
			continue; // Root and numeric kinds can reference the same source mapping.
		}
		CallClosureProof::View captured;
		if (!ReadCallView(view, captured, reason)) return false;
		captured.element_begin = proof.elements.size();
		proof.ranges.push_back(captured.table);
		for (const auto& [value, element_index]: view.elements) {
			CallClosureProof::Element element;
			if (!ReadCallElement(element_index, element, reason)) return false;
			if (element.descriptor != value) {
				reason = "image heap closure source element identity changed";
				return false;
			}
			proof.elements.push_back(element);
			if (!element.image_id) continue; // Stable native null: no guest image input.
			proof.ranges.push_back(element.requested_data);
			if (!element.requested_metadata.Empty()) proof.ranges.push_back(element.requested_metadata);
			if (!element.requested_stencil.Empty()) proof.ranges.push_back(element.requested_stencil);
			const auto [found, inserted] = owners.try_emplace(element.serial, element.image_id);
			if (!inserted) {
				if (found->second != element.image_id) {
					reason = "image heap closure has inconsistent native owner identities";
					return false;
				}
				continue;
			}
			CallClosureProof::Owner owner;
			if (!CaptureCallOwner(element.image_id, element.serial, owner, proof.associations, reason)) {
				return false;
			}
			proof.owners.push_back(owner);
			append_inputs(owner.input);
			for (size_t i = owner.association_begin; i < proof.associations.size(); ++i) {
				append_inputs(proof.associations[i]);
			}
		}
		proof.views.push_back(std::move(captured));
	}
	if (proof.references.empty() || proof.views.empty()) {
		reason = "image heap closure has no bound sampled source mapping";
		return false;
	}
	// Only stage-bound views and their referenced elements are inspected. Deduplicate native
	// owners/ranges, and never scan table words or unrelated resident descriptor slots.
	std::ranges::sort(proof.ranges);
	proof.ranges.erase(std::unique(proof.ranges.begin(), proof.ranges.end()), proof.ranges.end());
	for (const auto& range: proof.ranges) {
		const auto overlap = std::lower_bound(
		    protected_ranges.begin(), protected_ranges.end(), range.address,
		    [](const GuestRange& candidate, uint64_t address) { return candidate.End() <= address; });
		if (overlap != protected_ranges.end() && overlap->address < range.End()) {
			reason = "image heap derived table/image input overlaps protected call inputs";
			return false;
		}
	}
	// Publication is last: a failed capture cannot be reused as a post-commit certificate.
	proof.set = m_set;
	proof.draw = m_draw;
	return true;
}

bool ImageHeap::ValidateCallClosure(std::span<const IR::ImageResource> resources,
                                    std::span<const uint32_t> flattened_srt,
                                    const CallClosureProof& proof, std::string& reason) const {
	reason.clear();
	if (proof.draw == 0u || proof.draw != m_draw || proof.set != m_set ||
	    proof.references.empty() || proof.views.empty() ||
	    static_cast<size_t>(std::ranges::count_if(resources, &IR::ImageResource::heap)) != proof.references.size()) {
		reason = "image heap closure identity changed during commit";
		return false;
	}
	CallClosureProof::Element null_element;
	if (!ReadCallElement(0u, null_element, reason)) return false;
	if (null_element != proof.null_element) {
		reason = "image heap closure null descriptor changed during commit";
		return false;
	}
	for (const auto& saved: proof.references) {
		CallClosureProof::Reference current;
		if (saved.image >= resources.size() ||
		    !ReadCallReference(saved.image, resources[saved.image], flattened_srt, current, reason)) {
			if (reason.empty()) reason = "image heap closure resource disappeared during commit";
			return false;
		}
		current.view_serial = saved.view_serial;
		if (current != saved) {
			reason = "image heap closure slot mapping changed during commit";
			return false;
		}
	}
	for (const auto& saved: proof.views) {
		// Never dereference a retained View*: a wrapped stream can submit and retire it.
		const View* view = nullptr;
		for (const auto& memory: m_memories) {
			if (memory->base != saved.table.address) continue;
			const auto found = std::ranges::find_if(
			    memory->views, [&](const auto& candidate) { return candidate->serial == saved.serial; });
			if (found != memory->views.end()) view = found->get();
			break;
		}
		if (view == nullptr) {
			reason = "image heap closure source view retired during commit";
			return false;
		}
		CallClosureProof::View current;
		if (!ReadCallView(*view, current, reason)) return false;
		current.element_begin = saved.element_begin;
		if (current != saved || saved.element_begin > proof.elements.size() ||
		    saved.element_count > proof.elements.size() - saved.element_begin) {
			reason = "image heap closure source view changed during commit";
			return false;
		}
		for (size_t i = saved.element_begin; i < saved.element_begin + saved.element_count; ++i) {
			const auto& old = proof.elements[i];
			const auto found = view->elements.find(old.descriptor);
			CallClosureProof::Element element;
			if (found == view->elements.end() || found->second != old.index) {
				reason = "image heap closure reachable elements changed during commit";
				return false;
			}
			if (!ReadCallElement(old.index, element, reason)) return false;
			if (element != old) {
				reason = "image heap closure sampled descriptor changed during commit";
				return false;
			}
		}
	}
	// Re-resolve each native owner and its exact direct stencil family once. Child records
	// need not have a VkImage, but every dependency and reciprocal mapping must stay live.
	for (const auto& saved: proof.owners) {
		if (!ValidateCallOwner(saved, proof.associations, reason)) return false;
	}
	return true;
}

} // namespace Libs::Graphics

#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <array>
#include <chrono>
#include <deque>
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/frameStats.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <optional>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
	if (result == vk::Result::eErrorDeviceLost) {
		DumpSubmitHistory();
	}
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active_tick != 0);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	Libs::Graphics::WaitSiteScope wait_scope(Libs::Graphics::WaitSite::Finish, true);
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active_tick = operation.tick;
		}
		{
			Libs::Graphics::WaitSiteScope wait_scope(Libs::Graphics::WaitSite::AsyncPublishWorker);
			m_master.Wait(operation.tick);
		}
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

// TEMP DIAG (KYTY_GPU_TIME): GPU execution time of every command buffer.
namespace {
struct GpuTimer {
	bool                                      enabled = std::getenv("KYTY_GPU_TIME") != nullptr;
	vk::QueryPool                             pool    = nullptr;
	uint32_t                                  next    = 0;
	uint32_t                                  current = UINT32_MAX;
	std::deque<std::pair<uint64_t, uint32_t>> pending;
	uint64_t                                  busy_ns = 0, submits = 0;
	uint64_t                                  first_start = 0, last_end = 0;
	std::chrono::steady_clock::time_point     window = std::chrono::steady_clock::now();
	static constexpr uint32_t                 Slots  = 8192;
	// Per-item stamps: each command buffer owns a 1024-slot range of a 64-range ring.
	static constexpr uint32_t                 ItemRange = 1024, ItemRanges = 64;
	vk::QueryPool                             items = nullptr;
	uint32_t                                  item_range = 0, item_count = 0;
	std::vector<uint64_t>                     item_keys;
	struct Pending { uint32_t range; std::vector<uint64_t> keys; };
	std::deque<Pending>                       item_pending;
	std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> per_key; // ns, count
};
GpuTimer& Timer() {
	static GpuTimer timer;
	return timer;
}
} // namespace

void CommandScheduler::NoteWorkTiming(uint64_t key) {
	auto& timer = Timer();
	if (!timer.enabled || timer.current == UINT32_MAX || timer.item_count >= GpuTimer::ItemRange) {
		return;
	}
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, timer.items,
	                                  timer.item_range * GpuTimer::ItemRange + timer.item_count);
	timer.item_count++;
	timer.item_keys.push_back(key);
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	if (auto& timer = Timer(); timer.enabled) {
		if (!timer.pool) {
			vk::QueryPoolCreateInfo info {};
			info.queryType  = vk::QueryType::eTimestamp;
			info.queryCount = GpuTimer::Slots * 2u;
			timer.pool      = m_graphics.device.createQueryPool(info).value;
			m_graphics.device.resetQueryPool(timer.pool, 0, GpuTimer::Slots * 2u);
		}
		if (!timer.items) {
			vk::QueryPoolCreateInfo info {};
			info.queryType  = vk::QueryType::eTimestamp;
			info.queryCount = GpuTimer::ItemRange * GpuTimer::ItemRanges;
			timer.items     = m_graphics.device.createQueryPool(info).value;
		}
		timer.item_range = (timer.item_range + 1u) % GpuTimer::ItemRanges;
		timer.item_count = 0;
		timer.item_keys.clear();
		m_command.m_buffer.resetQueryPool(timer.items, timer.item_range * GpuTimer::ItemRange,
		                                  GpuTimer::ItemRange);
		timer.current = timer.next;
		timer.next    = (timer.next + 1u) % GpuTimer::Slots;
		m_command.m_buffer.resetQueryPool(timer.pool, timer.current * 2u, 2u);
		m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timer.pool,
		                                  timer.current * 2u);
	}
	m_dynamic_state.Invalidate();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());

	m_context.GetBufferCache().FlushReadChecks();
	auto& timer = Timer();
	if (timer.enabled && timer.current != UINT32_MAX) {
		m_command.EndRendering();
		m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, timer.pool,
		                                  timer.current * 2u + 1u);
	}
	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
		if (result == vk::Result::eSuccess) {
			// CommandBuffer no longer records which operation produced the work, so the history
			// and counters mark the submit Unknown (10) instead of claiming a dispatch.
			RecordSubmitHistory(tick, 10, 0, 0, 0, 0, 0, 0);
		}
	}

	if (result == vk::Result::eSuccess && GpuSyncDebugEnabled()) {
		// Debug mode (KYTY_GPU_SYNC=1): wait for every submit so a hung/lost GPU is blamed on the
		// submit that actually caused it. Slow; use only to locate crashes. The timeout avoids
		// deadlocking on submits that legitimately wait for later host-signalled work.
		const auto           semaphore = m_master.Handle();
		vk::SemaphoreWaitInfo wait_info {};
		wait_info.semaphoreCount = 1;
		wait_info.pSemaphores    = &semaphore;
		wait_info.pValues        = &tick;
		const auto wait_result   = graphics.device.waitSemaphores(&wait_info, 5'000'000'000ull);
		if (wait_result != vk::Result::eSuccess && wait_result != vk::Result::eTimeout) {
			ReportVulkanFatal("GPU sync wait after submit", wait_result, tick,
			                  0, 0,
			                  0, 0,
			                  0, 0,
			                  0);
			if (wait_result != vk::Result::eErrorDeviceLost) {
				DumpSubmitHistory();
			}
			EXIT("GPU failure detected in sync debug mode (see log above)\n");
		}
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(), tick);
	}

	if (timer.enabled && timer.current != UINT32_MAX) {
		timer.pending.emplace_back(tick, timer.current);
		timer.item_pending.push_back({timer.item_range, timer.item_keys});
		timer.current = UINT32_MAX;
		const double period = m_graphics.physical_device_properties.limits.timestampPeriod;
		while (!timer.pending.empty() && IsFree(timer.pending.front().first)) {
			{
				auto& item = timer.item_pending.front();
				std::array<uint64_t, 2> cb {};
				(void)m_graphics.device.getQueryPoolResults(
				    timer.pool, timer.pending.front().second * 2u, 2u, sizeof(cb), cb.data(),
				    sizeof(uint64_t), vk::QueryResultFlagBits::e64);
				if (!item.keys.empty()) {
					std::vector<uint64_t> stamps(item.keys.size());
					if (m_graphics.device.getQueryPoolResults(
					        timer.items, item.range * GpuTimer::ItemRange,
					        static_cast<uint32_t>(stamps.size()), stamps.size() * sizeof(uint64_t),
					        stamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64) ==
					    vk::Result::eSuccess) {
						uint64_t prev = cb[0];
						for (size_t i = 0; i < stamps.size(); i++) {
							if (stamps[i] >= prev) {
								auto& k = timer.per_key[item.keys[i]];
								k.first += static_cast<uint64_t>((stamps[i] - prev) * period);
								k.second++;
								prev = stamps[i];
							}
						}
					}
				}
				timer.item_pending.pop_front();
			}
			std::array<uint64_t, 2> stamps {};
			if (m_graphics.device.getQueryPoolResults(timer.pool, timer.pending.front().second * 2u, 2u,
			                                          sizeof(stamps), stamps.data(), sizeof(uint64_t),
			                                          vk::QueryResultFlagBits::e64) ==
			        vk::Result::eSuccess &&
			    stamps[1] >= stamps[0]) {
				const auto start = static_cast<uint64_t>(stamps[0] * period);
				const auto end   = static_cast<uint64_t>(stamps[1] * period);
				// Count overlapping command buffers once.
				const auto from = std::max(start, timer.last_end);
				if (end > from) timer.busy_ns += end - from;
				if (timer.first_start == 0) timer.first_start = start;
				timer.last_end = std::max(timer.last_end, end);
				timer.submits++;
			}
			timer.pending.pop_front();
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - timer.window >= std::chrono::seconds(2)) {
			const double wall_ms = std::chrono::duration<double, std::milli>(now - timer.window).count();
			std::printf("GPUTIME: GPU busy %.0f ms of %.0f ms (%.0f%%), %llu command buffers\n",
			            timer.busy_ns / 1e6, wall_ms, 100.0 * timer.busy_ns / 1e6 / wall_ms,
			            static_cast<unsigned long long>(timer.submits));
			std::fflush(stdout);
			std::vector<std::pair<uint64_t, uint64_t>> top;
			for (auto& [k, v]: timer.per_key) top.emplace_back(v.first, k);
			std::sort(top.rbegin(), top.rend());
			for (size_t i = 0; i < top.size() && i < 20; i++) {
				const auto& v = timer.per_key[top[i].second];
				std::printf("GPUTIME:   %7.2f ms  %5llu x  %s %016llx\n", v.first / 1e6,
				            static_cast<unsigned long long>(v.second),
				            (top[i].second >> 63u) ? "draw(ps/vs)" : "dispatch   ",
				            static_cast<unsigned long long>(top[i].second & ~(1ull << 63u)));
			}
			timer.per_key.clear();
			timer.window  = now;
			timer.busy_ns = 0;
			timer.submits = 0;
		}
	}
	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	m_submit_work_count = 0;
	BeginCommand();
}

} // namespace Libs::Graphics

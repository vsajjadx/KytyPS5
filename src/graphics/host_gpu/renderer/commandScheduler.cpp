#include <vector>
#include <cstdlib>
#include <array>
#include <chrono>
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuSubmitTiming.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/frameStats.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <optional>
#include <cmath>
#include <cinttypes>
#include <new>
#include <cstdio>

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

struct CommandScheduler::GpuTimingV2 {
	static constexpr size_t Capacity = 128;
	using Leases = GpuSubmitTiming::LeasePool<Capacity>;
	struct Sample { uint64_t tick = 0, ns = 0; uint32_t work = 0; };
	GpuTimingV2(uintptr_t owner, uint32_t bits, double period)
	    : domain(owner), leases(owner, true), valid_bits(bits), period_ns(period) {}
	uintptr_t domain;
	Leases leases;
	vk::QueryPool pool = nullptr;
	uint32_t valid_bits;
	double period_ns;
	size_t active = Leases::Invalid, cursor = 0;
	bool faulted = false;
	uint64_t submitted = 0, measured = 0, skipped = 0, not_ready = 0, errors = 0, ambiguous = 0;
	std::array<std::chrono::steady_clock::time_point, Capacity> begun {};
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
	uint64_t window_count = 0, minimum = UINT64_MAX, maximum = 0, report_sequence = 0;
	double sum_ns = 0;
	std::array<Sample, 8> largest {};
	size_t largest_count = 0;

	void SampleSpan(uint64_t tick, uint32_t work, uint64_t ns) {
		++measured; ++window_count;
		sum_ns += static_cast<double>(ns);
		minimum = std::min(minimum, ns); maximum = std::max(maximum, ns);
		if (largest_count == largest.size() && ns <= largest.back().ns) return;
		auto index = std::min(largest_count, largest.size() - 1u);
		while (index != 0u && largest[index - 1u].ns < ns) {
			largest[index] = largest[index - 1u]; --index;
		}
		largest[index] = {tick, ns, work};
		if (largest_count < largest.size()) ++largest_count;
	}
	void Report(bool final) {
		const auto now = std::chrono::steady_clock::now();
		if (!final && now - window < std::chrono::seconds(2)) return;
		std::fprintf(stderr, "GPUSUBMIT_V2 v=2 domain=%" PRIxPTR " report=%" PRIu64
		    " metric=command_buffer_span aggregation=harvest_window final=%u valid_bits=%u"
		    " period_ns=%.9g measured=%" PRIu64 " measured_total=%" PRIu64
		    " submitted_total=%" PRIu64 " skipped_total=%" PRIu64 " pending=%zu"
		    " dropped_capacity_total=%" PRIu64 " not_ready_attempts_total=%" PRIu64
		    " query_errors_total=%" PRIu64 " ambiguous_total=%" PRIu64
		    " span_sum_ns=%.0f min_ns=%" PRIu64 " max_ns=%" PRIu64
		    " shader_cost=unknown utilization=unknown\n",
		    domain, ++report_sequence, final ? 1u : 0u, valid_bits, period_ns,
		    window_count, measured, submitted, skipped, leases.Pending(), leases.Dropped(),
		    not_ready, errors, ambiguous, sum_ns, window_count ? minimum : 0u, maximum);
		for (size_t i = 0; i < largest_count; ++i) {
			const auto& sample = largest[i];
			std::fprintf(stderr, "GPUSUBMIT_V2_SAMPLE v=2 domain=%" PRIxPTR
			    " report=%" PRIu64 " sample=largest_window_span rank=%zu tick=%" PRIu64
			    " span_ns=%" PRIu64 " work_items=%u\n",
			    domain, report_sequence, i, sample.tick, sample.ns, sample.work);
		}
		window = now; window_count = 0; minimum = UINT64_MAX; maximum = 0;
		sum_ns = 0; largest_count = 0;
	}
};

void CommandScheduler::BeginGpuTimingV2() {
	if (!m_gpu_time_v2_checked) {
		m_gpu_time_v2_checked = true;
		// Immutable opt-in only. The legacy KYTY_GPU_TIME flag is never consulted.
		if (!GpuSubmitTiming::EnabledFor(std::getenv("KYTY_GPU_TIME_V2"),
		                                this == &m_context.GetCommandScheduler())) return;
		if (!m_graphics.physical_device || !m_graphics.device) return;
		const double period = m_graphics.physical_device_properties.limits.timestampPeriod;
		uint32_t family_count = 0;
		m_graphics.physical_device.getQueueFamilyProperties(&family_count, nullptr);
		std::array<vk::QueueFamilyProperties, 256> families {};
		if (family_count == 0u || family_count > families.size() ||
		    m_graphics.queue_family >= family_count || !std::isfinite(period) || period <= 0) {
			std::fputs("GPUSUBMIT_V2 v=2 status=disabled reason=unsupported_queue_or_period\n", stderr);
			return;
		}
		m_graphics.physical_device.getQueueFamilyProperties(&family_count, families.data());
		if (m_graphics.queue_family >= family_count ||
		    !GpuSubmitTiming::ValidTimestampBits(families[m_graphics.queue_family].timestampValidBits)) {
			std::fputs("GPUSUBMIT_V2 v=2 status=disabled reason=timestamp_valid_bits\n", stderr);
			return;
		}
		auto timing = std::unique_ptr<GpuTimingV2>(new (std::nothrow) GpuTimingV2(
		    reinterpret_cast<uintptr_t>(this), families[m_graphics.queue_family].timestampValidBits, period));
		if (!timing) return;
		vk::QueryPoolCreateInfo info {};
		info.queryType = vk::QueryType::eTimestamp;
		info.queryCount = static_cast<uint32_t>(GpuTimingV2::Capacity * 2u);
		const auto result = m_graphics.device.createQueryPool(&info, nullptr, &timing->pool);
		if (result != vk::Result::eSuccess || !timing->pool) {
			std::fprintf(stderr, "GPUSUBMIT_V2 v=2 status=disabled reason=create_query_pool result=%d\n",
			             static_cast<int>(result));
			return;
		}
		m_gpu_time_v2 = std::move(timing);
		std::fputs("GPUSUBMIT_V2 v=2 status=enabled role=renderer old_flag_ignored=1\n", stderr);
	}
	if (!m_gpu_time_v2) return;
	HarvestGpuTimingV2();
	auto& timing = *m_gpu_time_v2;
	if (timing.faulted) return;
	if (timing.active != GpuTimingV2::Leases::Invalid) {
		timing.faulted = true; ++timing.errors; return;
	}
	timing.active = timing.leases.Acquire(timing.domain);
	if (timing.active == GpuTimingV2::Leases::Invalid) return;
	timing.begun[timing.active] = std::chrono::steady_clock::now();
	const auto first = static_cast<uint32_t>(timing.active * 2u);
	// BeginCommand has begun a new command buffer outside rendering. No host reset.
	m_command.m_buffer.resetQueryPool(timing.pool, first, 2u);
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timing.pool, first);
}

void CommandScheduler::EndGpuTimingV2() {
	if (!m_gpu_time_v2 || m_gpu_time_v2->active == GpuTimingV2::Leases::Invalid) return;
	auto& timing = *m_gpu_time_v2;
	// End() already closes rendering at this point; doing so before the final stamp
	// includes attachment stores in the span. No work/submit counter is changed.
	m_command.EndRendering();
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, timing.pool,
	                                  static_cast<uint32_t>(timing.active * 2u + 1u));
}

void CommandScheduler::SubmittedGpuTimingV2(uint64_t tick) {
	if (!m_gpu_time_v2) return;
	auto& timing = *m_gpu_time_v2;
	++timing.submitted;
	if (timing.active == GpuTimingV2::Leases::Invalid) ++timing.skipped;
	else if (!timing.leases.Submitted(timing.active, timing.domain, tick, m_submit_work_count)) {
		timing.faulted = true; ++timing.errors;
	}
	timing.active = GpuTimingV2::Leases::Invalid;
	HarvestGpuTimingV2();
}

void CommandScheduler::HarvestGpuTimingV2(bool final) {
	if (!m_gpu_time_v2) return;
	auto& timing = *m_gpu_time_v2;
	// Cached completed ticks only: no Refresh, wait, flush, or extra submission.
	const auto completed = m_master.KnownGpuTick();
	const size_t budget = final ? GpuTimingV2::Capacity : 8u;
	size_t attempted = 0;
	for (size_t visited = 0; !timing.faulted && visited < GpuTimingV2::Capacity && attempted < budget; ++visited) {
		const auto index = timing.cursor;
		timing.cursor = (timing.cursor + 1u) % GpuTimingV2::Capacity;
		if (!timing.leases.Ready(index, timing.domain, completed)) continue;
		++attempted;
		struct QueryResult { uint64_t value = 0, available = 0; };
		std::array<QueryResult, 2> results {};
		const auto result = m_graphics.device.getQueryPoolResults(timing.pool,
		    static_cast<uint32_t>(index * 2u), 2u, sizeof(results), results.data(), sizeof(QueryResult),
		    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
		if (result == vk::Result::eNotReady ||
		    (result == vk::Result::eSuccess && (!results[0].available || !results[1].available))) {
			++timing.not_ready; continue; // Keep the lease; never read/reset another use's data.
		}
		if (result != vk::Result::eSuccess) {
			timing.faulted = true; ++timing.errors; break;
		}
		const auto* slot = timing.leases.Get(index);
		const double host_age_ns = std::chrono::duration<double, std::nano>(
		    std::chrono::steady_clock::now() - timing.begun[index]).count();
		const auto ns = GpuSubmitTiming::SpanNanoseconds(results[0].value, results[1].value,
		    timing.valid_bits, timing.period_ns, host_age_ns);
		if (ns) timing.SampleSpan(slot->tick, slot->work_count, *ns);
		else ++timing.ambiguous;
		(void)timing.leases.Harvested(index, timing.domain, completed, true);
	}
	timing.Report(final);
}

void CommandScheduler::DestroyGpuTimingV2AfterDrain() {
	if (!m_gpu_time_v2) return;
	HarvestGpuTimingV2(true);
	// Shutdown has already waited for every successful submit and joined completion
	// work. There is no diagnostic wait, and no presenter/device-global pool.
	m_graphics.device.destroyQueryPool(m_gpu_time_v2->pool, nullptr);
	m_gpu_time_v2.reset();
}

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
	DestroyGpuTimingV2AfterDrain();
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

void CommandScheduler::NoteWorkTiming(uint64_t key) {
	// Kept for caller compatibility only. Legacy keys sacrifice hash bit63 and
	// item-to-item intervals are not shader costs. Neither legacy flag nor key
	// enables a query. V2 command-buffer timing has its own exact-one flag.
	(void)key;
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
	BeginGpuTimingV2();
	m_dynamic_state.Invalidate();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());

	m_context.GetBufferCache().FlushReadChecks();
	EndGpuTimingV2();
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

	SubmittedGpuTimingV2(tick);
	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	m_submit_work_count = 0;
	BeginCommand();
}

} // namespace Libs::Graphics

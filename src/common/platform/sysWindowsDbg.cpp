#include "common/common.h"

// IWYU pragma: no_include <basetsd.h>
// IWYU pragma: no_include <memoryapi.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <winbase.h>

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// #error "KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS"
#else

#include <windows.h> // IWYU pragma: keep

#include "common/assert.h"
#include "common/platform/sysDbg.h"

void SysRaiseProcessPriority() {
	SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
#ifdef PROCESS_POWER_THROTTLING_CURRENT_VERSION
	// Windows' EcoQoS slows down and coarsens the timers of processes it thinks are idle.
	PROCESS_POWER_THROTTLING_STATE state {};
	state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
	state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
#ifdef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
	state.ControlMask |= PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
#endif
	state.StateMask = 0;
	SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
#endif
}



#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>
#include <array>
#include <thread>

void SysStartSamplingCurrentThread(const char* name) {
	if (std::getenv("KYTY_SAMPLE") == nullptr) return;
	HANDLE thread = nullptr;
	DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread,
	                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
	std::thread([thread, name] {
		const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
		std::map<std::array<uint64_t, 24>, uint64_t> stacks;
		auto last = std::chrono::steady_clock::now();
		uint64_t samples = 0;
		for (;;) {
			Sleep(1);
			std::array<uint64_t, 24> frames {};
			if (SuspendThread(thread) == static_cast<DWORD>(-1)) return;
			CONTEXT ctx {};
			ctx.ContextFlags = CONTEXT_FULL;
			if (GetThreadContext(thread, &ctx)) {
				for (int i = 0; i < 24 && ctx.Rip != 0; i++) {
					frames[i] = ctx.Rip;
					DWORD64 image_base = 0;
					auto*   fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
					if (fn == nullptr) {
						if (ctx.Rsp == 0) break;
						ctx.Rip = *reinterpret_cast<DWORD64*>(ctx.Rsp);
						ctx.Rsp += 8;
						continue;
					}
					void*   handler_data = nullptr;
					DWORD64 establisher  = 0;
					RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx,
					                 &handler_data, &establisher, nullptr);
				}
			}
			ResumeThread(thread);
			for (auto& f: frames) if (f != 0) f -= base;
			stacks[frames]++;
			samples++;
			const auto now = std::chrono::steady_clock::now();
			if (now - last >= std::chrono::seconds(10)) {
				last = now;
				std::printf("SAMPLE: thread=%s samples=%llu\n", name, (unsigned long long)samples);
				for (auto& [k, c]: stacks) {
					std::printf("SAMPLE: %llu hits:", (unsigned long long)c);
					for (auto f: k) if (f != 0) std::printf(" %llx", (unsigned long long)f);
					std::printf("\n");
				}
				std::fflush(stdout);
				stacks.clear();
				samples = 0;
			}
		}
	}).detach();
}

int SysCaptureStackRva(uint64_t* frames, int max_frames) {
	void*      raw[32] {};
	const auto n    = RtlCaptureStackBackTrace(1, static_cast<DWORD>(std::min(max_frames, 32)), raw, nullptr);
	const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	for (int i = 0; i < n; i++) frames[i] = reinterpret_cast<uint64_t>(raw[i]) - base;
	return n;
}

void SysFaultDiag(void* native_context, uint64_t fault_vaddr) {
	static const bool on = std::getenv("KYTY_FAULT_DIAG") != nullptr;
	if (!on || native_context == nullptr) return;
	CONTEXT ctx = *static_cast<CONTEXT*>(native_context);
	const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	std::vector<uint64_t> frames;
	for (int i = 0; i < 10 && ctx.Rip != 0; i++) {
		frames.push_back(ctx.Rip - base);
		DWORD64 image_base = 0;
		auto*   fn         = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
		if (fn == nullptr) {
			ctx.Rip = *reinterpret_cast<DWORD64*>(ctx.Rsp);
			ctx.Rsp += 8;
			continue;
		}
		void*   handler_data = nullptr;
		DWORD64 establisher  = 0;
		RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data,
		                 &establisher, nullptr);
	}
	static std::mutex                                 m;
	static std::map<std::vector<uint64_t>, uint64_t> hits;
	static auto                                       last = std::chrono::steady_clock::now();
	std::scoped_lock                                  lock {m};
	hits[frames]++;
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	last = now;
	std::vector<std::pair<uint64_t, const std::vector<uint64_t>*>> v;
	for (auto& [k, c]: hits) v.emplace_back(c, &k);
	std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
	std::printf("FAULTDIAG: base=0x%llx stacks=%zu last_vaddr=0x%llx\n",
	            static_cast<unsigned long long>(base), v.size(),
	            static_cast<unsigned long long>(fault_vaddr));
	for (size_t i = 0; i < v.size() && i < 6; i++) {
		std::printf("FAULTDIAG: %llu hits:", static_cast<unsigned long long>(v[i].first));
		for (auto f: *v[i].second) std::printf(" %llx", static_cast<unsigned long long>(f));
		std::printf("\n");
	}
	std::fflush(stdout);
	hits.clear();
}

sys_dbg_stack_info_t SysStackUsage() {
	MEMORY_BASIC_INFORMATION mbi {};
	[[maybe_unused]] size_t  ss = VirtualQuery(&mbi, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	PVOID reserved = mbi.AllocationBase;
	ss             = VirtualQuery(reserved, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	ss = VirtualQuery(static_cast<char*>(reserved) + mbi.RegionSize, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	return {.reserved_addr = reinterpret_cast<uintptr_t>(reserved),
	        .guard_addr    = reinterpret_cast<uintptr_t>(mbi.BaseAddress),
	        .guard_size    = mbi.RegionSize};
}

#endif

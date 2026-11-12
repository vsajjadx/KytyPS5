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

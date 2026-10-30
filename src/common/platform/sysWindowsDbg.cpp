#include "common/common.h"

// IWYU pragma: no_include <basetsd.h>
// IWYU pragma: no_include <memoryapi.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <processthreadsapi.h>
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

void SysStackUsage(sys_dbg_stack_info_t& s) {
	MEMORY_BASIC_INFORMATION mbi {};
	[[maybe_unused]] size_t  ss = VirtualQuery(&mbi, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	PVOID reserved = mbi.AllocationBase;
	ss             = VirtualQuery(reserved, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	size_t reserved_size = mbi.RegionSize;
	ss = VirtualQuery(static_cast<char*>(reserved) + reserved_size, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	void*  guard_page      = mbi.BaseAddress;
	size_t guard_page_size = mbi.RegionSize;
	ss = VirtualQuery(static_cast<char*>(guard_page) + guard_page_size, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	void*  commited      = mbi.BaseAddress;
	size_t commited_size = mbi.RegionSize;
	s.reserved_addr      = reinterpret_cast<uintptr_t>(reserved);
	s.reserved_size      = reserved_size;
	s.guard_addr         = reinterpret_cast<uintptr_t>(guard_page);
	s.guard_size         = guard_page_size;
	s.commited_addr      = reinterpret_cast<uintptr_t>(commited);
	s.commited_size      = commited_size;

	s.addr       = s.reserved_addr;
	s.total_size = s.reserved_size + s.guard_size + s.commited_size;
}

#endif

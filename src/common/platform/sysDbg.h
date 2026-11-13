#ifndef KYTY_COMMON_PLATFORM_SYSDBG_H_
#define KYTY_COMMON_PLATFORM_SYSDBG_H_

#include "common/common.h"

// NOLINTNEXTLINE(readability-identifier-naming)
struct sys_dbg_stack_info_t {
	uintptr_t reserved_addr; // lowest address of the calling thread's stack reservation
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t guard_addr; // guard page just below the committed stack
	size_t    guard_size;
#endif
};

sys_dbg_stack_info_t SysStackUsage();

// Above-normal priority, never power throttled, so the emulator's threads keep their clocks.
// Windows only; elsewhere it does nothing.
void SysRaiseProcessPriority();
// TEMP DIAG: aggregate call stacks of handled page faults (KYTY_FAULT_DIAG=1).
void SysFaultDiag(void* native_context, uint64_t fault_vaddr);
int  SysCaptureStackRva(uint64_t* frames, int max_frames);
// TEMP DIAG: sample the calling thread's stacks every ms when KYTY_SAMPLE is set.
void SysStartSamplingCurrentThread(const char* name);

#endif /* KYTY_COMMON_PLATFORM_SYSDBG_H_ */

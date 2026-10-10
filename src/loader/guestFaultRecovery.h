#ifndef KYTY_LOADER_GUEST_FAULT_RECOVERY_H_
#define KYTY_LOADER_GUEST_FAULT_RECOVERY_H_

#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Loader::GuestFaultRecovery {

// Where a recoverable fault may land. A null/near-null dereference is the signature of the
// failure this exists for: an unresolved import returns 0 (or the invalid-memory sentinel the
// relocator installs) and the guest then dereferences it. Anything else keeps the previous
// behaviour, so genuine emulator faults are still reported and stopped.
// The relocator points an unresolved import at this reserved, never-mapped sentinel page when it
// cannot install an executable thunk; it must stay inside the reserved window tested below.
constexpr uint64_t INVALID_MEMORY_SENTINEL = 0x0000000840000000ull;
constexpr uint64_t RECOVERABLE_LOW_ADDRESS = 0x0000000000010000ull;
constexpr uint64_t RESERVED_RANGE_BEGIN    = 0x0000000800000000ull;
constexpr uint64_t RESERVED_RANGE_END      = 0x0000000900000000ull;

inline bool IsRecoverableFault(uint64_t address) {
	return address < RECOVERABLE_LOW_ADDRESS ||
	       (address >= RESERVED_RANGE_BEGIN && address < RESERVED_RANGE_END);
}

// Set KYTY_NO_GUEST_FAULT_RECOVERY to restore the old fatal behaviour while diagnosing.
inline bool RecoveryEnabled() {
	static const bool enabled = std::getenv("KYTY_NO_GUEST_FAULT_RECOVERY") == nullptr;
	return enabled;
}

struct RecoveryPoint {
	jmp_buf env {};
};

inline thread_local RecoveryPoint*  g_active          = nullptr;
inline thread_local volatile bool   g_unwind_requested = false;

// Counted so a recovered fault stays visible in the log without flooding it.
inline thread_local uint64_t g_recovered_faults = 0;

// Arms a recovery point for the calling guest thread. The setjmp() must sit directly in the
// frame of the host function that calls into guest code, so that frame is still live when the
// handler unwinds to it:
//
//   RecoveryPoint point {};
//   auto* previous = g_active;
//   g_active = &point;
//   if (std::setjmp(point.env) != 0) {
//       g_active = previous;
//       ...report and return...
//   }
//   ...enter guest code...
//   g_active = previous;
//
// True only on a guest thread with a live recovery point that faulted at a recoverable
// address, so the caller can report the fault before unwinding.
inline bool CanRecover(uint64_t address) {
	return RecoveryEnabled() && g_active != nullptr && IsRecoverableFault(address);
}

// Unwind the faulting guest thread to its recovery point. Never returns.
[[noreturn]] inline void Unwind() {
	auto* active = g_active;
	if (active == nullptr) {
		std::abort();
	}
	g_unwind_requested = true;
	g_recovered_faults++;
	longjmp(active->env, 1);
}

} // namespace Loader::GuestFaultRecovery

#endif /* KYTY_LOADER_GUEST_FAULT_RECOVERY_H_ */

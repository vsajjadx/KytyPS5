#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstddef>
#include <cstdint>
#include <cstddef>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);
// Completes a faulting load into EAX as a zero read and resumes after it.
void SkipLoad(void* native_context, size_t length);
// Completes a faulting `mov eax, [mem]` with `value` (upper rax bits cleared).
void CompleteLoad32(void* native_context, size_t length, uint32_t value);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */

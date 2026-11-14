#ifndef KYTY_SHADER_ACTIVITY_PROOF_H_
#define KYTY_SHADER_ACTIVITY_PROOF_H_

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::IR {

// This experiment reuses validated binding metadata, never a GPU counter value.
inline bool HostActivityProofEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_HOST_ACTIVITY_PROOF");
		return text != nullptr && text[0] == '1' && text[1] == '\0';
	}();
	return enabled;
}

struct ActivityEpoch {
	uint64_t context = 0;
	uint64_t mapping = 0;
	uint64_t protection = 0;
	uint64_t buffers = 0;
	uint64_t images = 0;
	uint64_t configuration = 0;
	bool operator==(const ActivityEpoch&) const = default;
};

struct ActivityObservation {
	uint64_t address = 0;
	uint32_t value = 0;
	uint32_t memory_index = UINT32_MAX;
};

struct ActivityCache;

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif

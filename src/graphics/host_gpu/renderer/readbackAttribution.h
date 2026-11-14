#ifndef KYTY_READBACK_ATTRIBUTION_H_
#define KYTY_READBACK_ATTRIBUTION_H_

// Diagnostic context only: no memory/coherence state or speculative values.
#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::ReadbackAttribution {
enum class Phase : uint8_t { Unknown, Program, Materialize, WalkSrt, Compute, Bindings };
inline const char* PhaseName(Phase value) {
    switch (value) {
    case Phase::Program: return "program";
    case Phase::Materialize: return "materialize";
    case Phase::WalkSrt: return "srt";
    case Phase::Compute: return "compute";
    case Phase::Bindings: return "bindings";
    default: return "unknown";
    }
}
inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("KYTY_READBACK_ATTRIBUTION");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}
struct Context {
    uint64_t shader = 0, fault_pc = 0, fault_address = 0;
    uint64_t request_address = 0, request_size = 0;
    uint32_t resource = UINT32_MAX, fault_size = 0;
    Phase phase = Phase::Unknown;
    bool origin_gpu = false, request_write = false;
};
inline thread_local Context current;
class Scope {
public:
    Scope(uint64_t shader, Phase phase): m_active(Enabled()) {
        if (!m_active) return;
        m_previous = current;
        current.shader = shader;
        current.phase = phase;
    }
    explicit Scope(const Context& context): m_active(Enabled()) {
        if (!m_active) return;
        m_previous = current;
        current = context;
    }
    ~Scope() { if (m_active) current = m_previous; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    Context m_previous {};
    bool m_active;
};
inline Context Request(uint64_t address, uint64_t size, bool write, bool gpu) {
    Context context = Enabled() ? current : Context {};
    context.request_address = address;
    context.request_size = size;
    context.request_write = write;
    context.origin_gpu = gpu;
    return context;
}
inline Context Fault(uint64_t pc, uint64_t address, uint32_t size = 0) {
    Context context = Enabled() ? current : Context {};
    context.fault_pc = pc;
    context.fault_address = address;
    context.fault_size = size;
    return context;
}
inline Context Resource(uint32_t index) {
    Context context = Enabled() ? current : Context {};
    context.resource = index;
    return context;
}
} // namespace Libs::Graphics::ReadbackAttribution
#endif

#ifndef KYTY_GPU_SUBMIT_TIMING_H_
#define KYTY_GPU_SUBMIT_TIMING_H_

// CPU-only lease bookkeeping for opt-in GPU command-buffer timing. This header
// does not read an environment, issue Vulkan commands, or wait for completion.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>

namespace Libs::Graphics::GpuSubmitTiming {

inline bool ExactOne(const char* value) noexcept {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}
inline bool EnabledFor(const char* value, bool renderer_scheduler) noexcept {
    return renderer_scheduler && ExactOne(value);
}
inline bool ValidTimestampBits(uint32_t bits) noexcept {
    return bits >= 36u && bits <= 64u;
}
inline std::optional<uint64_t> Delta(uint64_t begin, uint64_t end, uint32_t bits) noexcept {
    if (!ValidTimestampBits(bits)) return {};
    // Unsigned subtraction and the valid-bit mask handle one device-clock wrap.
    // A command span lasting a whole clock period is inherently ambiguous.
    const auto mask = bits == 64u ? UINT64_MAX : (uint64_t {1} << bits) - 1u;
    return (end - begin) & mask;
}
inline std::optional<uint64_t> SpanNanoseconds(uint64_t begin, uint64_t end, uint32_t bits,
                                              double period_ns, double host_age_ns) noexcept {
    const auto delta = Delta(begin, end, bits);
    if (!delta || !std::isfinite(period_ns) || period_ns <= 0 ||
        !std::isfinite(host_age_ns) || host_age_ns < 0) return {};
    // The host interval begins before query recording/submission and ends after
    // completed-tick harvest. If it permits a whole device-clock period, the
    // number of wraps cannot be established, so do not publish a duration.
    const auto wrap_ns = std::ldexp(period_ns, static_cast<int>(bits));
    const auto ns = static_cast<double>(*delta) * period_ns;
    if (host_age_ns >= wrap_ns || !std::isfinite(ns) || ns < 0 ||
        ns >= static_cast<double>(UINT64_MAX)) return {};
    return static_cast<uint64_t>(ns);
}

template <size_t Capacity>
class LeasePool {
public:
    static_assert(Capacity != 0u);
    static constexpr size_t Invalid = Capacity;
    enum class State : uint8_t { Free, Recording, Submitted };
    struct Slot {
        State state = State::Free;
        uint64_t tick = 0;
        uint32_t work_count = 0;
        bool operator==(const Slot&) const = default;
    };
    LeasePool(uintptr_t domain, bool enabled): m_domain(domain), m_enabled(enabled) {}

    size_t Acquire(uintptr_t domain) noexcept {
        if (!m_enabled || domain != m_domain) return Invalid;
        for (size_t i = 0; i < Capacity; ++i) {
            if (m_slots[i].state != State::Free) continue;
            m_slots[i] = {State::Recording, 0, 0};
            return i;
        }
        ++m_dropped;
        return Invalid; // Telemetry drops; callers must never wait or grow.
    }
    bool Submitted(size_t index, uintptr_t domain, uint64_t tick, uint32_t work_count) noexcept {
        if (!m_enabled || domain != m_domain || index >= Capacity ||
            m_slots[index].state != State::Recording || tick == 0u || tick <= m_last_tick) return false;
        m_slots[index] = {State::Submitted, tick, work_count};
        m_last_tick = tick;
        return true;
    }
    bool Ready(size_t index, uintptr_t domain, uint64_t completed_tick) const noexcept {
        return m_enabled && domain == m_domain && index < Capacity &&
               m_slots[index].state == State::Submitted && m_slots[index].tick <= completed_tick;
    }
    bool Harvested(size_t index, uintptr_t domain, uint64_t completed_tick,
                   bool results_available) noexcept {
        if (!results_available || !Ready(index, domain, completed_tick)) return false;
        m_slots[index] = {};
        return true;
    }
    const Slot* Get(size_t index) const noexcept {
        return index < Capacity ? &m_slots[index] : nullptr;
    }
    size_t Pending() const noexcept {
        size_t count = 0;
        for (const auto& slot: m_slots) count += slot.state != State::Free;
        return count;
    }
    uint64_t Dropped() const noexcept { return m_dropped; }

private:
    uintptr_t m_domain;
    bool m_enabled;
    std::array<Slot, Capacity> m_slots {};
    uint64_t m_last_tick = 0;
    uint64_t m_dropped = 0;
};

} // namespace Libs::Graphics::GpuSubmitTiming
#endif

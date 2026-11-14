// Copyright2026 KytyPS5 contributors. SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace Libs::Graphics::MultiFramePresentPolicy {
constexpr uint32_t InputCapacity = 3;
constexpr uint32_t OutputSets = 2;
constexpr uint32_t MaxGenerated = 5;
constexpr uint64_t RetainTick(uint64_t current, uint64_t next) { return current > next ? current : next; }

// A committed ticket may finish after cancellation; cancellation prevents subsequent tickets.
class Cadence {
public:
    enum class Kind : uint8_t { None, Generated, Real };
    struct Ticket { Kind kind = Kind::None; uint32_t index = 0; uint64_t due = 0, epoch = 0; };
    struct Cancelled { uint32_t generated = 0, real = 0; };
    struct Counts { uint32_t late = 0, superseded = 0; };
    bool Begin(uint32_t factor, uint64_t start, uint64_t interval, uint64_t epoch) {
        if (Pending() || factor < 2 || factor > 6 || interval < factor ||
            start > std::numeric_limits<uint64_t>::max() - interval) return false;
        m_factor = factor; m_cursor = 0; m_start = start; m_step = interval / factor;
        m_epoch = epoch; m_counts = {}; return true;
    }
    bool Pending() const { return m_cursor < m_factor; }
    uint64_t NextDue() const { return Pending() ? m_start + m_step * m_cursor : 0; }
    Ticket Take(uint64_t now, uint64_t epoch, bool newer_waiting = false) {
        if (!Pending()) return {};
        const bool force_real = newer_waiting || epoch != m_epoch;
        if (force_real && m_cursor + 1 < m_factor) {
            m_counts.superseded += m_factor - 1 - m_cursor; m_cursor = m_factor - 1;
        }
        while (m_cursor + 1 < m_factor && now >= NextDue() && now - NextDue() >= m_step) {
            ++m_counts.late; ++m_cursor;
        }
        if (!force_real && now < NextDue()) return {};
        const auto index = m_cursor++;
        return {index + 1 == m_factor ? Kind::Real : Kind::Generated, index,
                m_start + m_step * index, m_epoch};
    }
    static bool Current(const Ticket& ticket, uint64_t epoch) {
        return ticket.kind != Kind::Generated || ticket.epoch == epoch;
    }
    Cancelled Cancel() {
        if (!Pending()) return {};
        const Cancelled result {m_factor - 1 - m_cursor, 1}; m_cursor = m_factor; return result;
    }
    Counts GetCounts() const { return m_counts; }
private:
    uint32_t m_factor = 0, m_cursor = 0;
    uint64_t m_start = 0, m_step = 0, m_epoch = 0;
    Counts m_counts {};
};

// Caller owns the mutex. An evicted or rejected value is returned for exactly-once release.
template<class T, uint32_t Capacity = InputCapacity> class Queue {
    static_assert(Capacity > 0);
public:
    std::optional<T> Push(T value) {
        if (m_closed) return value;
        std::optional<T> evicted;
        if (m_size == Capacity) { evicted = std::move(m_jobs[m_head]); m_head = (m_head + 1) % Capacity; --m_size; }
        m_jobs[(m_head + m_size) % Capacity] = std::move(value); ++m_size; return evicted;
    }
    std::optional<T> Pop() {
        if (!m_size) return {};
        auto value = std::move(m_jobs[m_head]); m_head = (m_head + 1) % Capacity; --m_size; return value;
    }
    uint32_t Size() const { return m_size; }
    void Close() { m_closed = true; }
private:
    std::array<T, Capacity> m_jobs {};
    uint32_t m_head = 0, m_size = 0;
    bool m_closed = false;
};
// VK_SUBOPTIMAL_KHR accepted an image; only the new Multi path changes legacy retry policy.
constexpr bool AcceptedPresent(bool success, bool suboptimal, bool multi) {
    return success || (suboptimal && multi);
}

enum class PresentedKind : uint8_t { Real, Generated, Repeat, Blank };
struct SuccessCounts {
    uint64_t real = 0, generated = 0, repeat = 0, blank = 0;
    void Presented(PresentedKind kind) {
        switch (kind) {
            case PresentedKind::Real: ++real; break;
            case PresentedKind::Generated: ++generated; break;
            case PresentedKind::Repeat: ++repeat; break;
            case PresentedKind::Blank: ++blank; break;
        }
    }
};
} // namespace Libs::Graphics::MultiFramePresentPolicy

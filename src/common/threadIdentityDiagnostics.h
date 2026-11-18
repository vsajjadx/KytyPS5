#ifndef KYTY_THREAD_IDENTITY_DIAGNOSTICS_H_
#define KYTY_THREAD_IDENTITY_DIAGNOSTICS_H_

// Diagnostic identity only: independent from CAUSAL event families and never a
// scheduling token. Native creation times are raw integer FILETIMEs, not seconds.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace Common::ThreadIdentityDiagnostics {
inline constexpr uint32_t MaxRecords = 256;
inline constexpr size_t MaxLineBytes = 512;
inline constexpr char LimitLine[] =
    "THREAD_IDENTITY_LIMIT v=1 max_records=256 exhausted=1\n";

enum class Phase : uint8_t { Main = 0, Begin = 1 };
struct Record {
    Phase phase = Phase::Begin;
    uint64_t process_id = 0;
    uint64_t process_creation_time = 0;
    uint64_t os_thread_id = 0;
    uint64_t thread_creation_time = 0;
    int32_t internal_unique_id = 0;
    int32_t guest_thread_id = 0;
    uint64_t object_token = 0;
    uint64_t entry = 0;
    bool process_creation_valid = false;
    uint32_t process_creation_error = 0;
    bool thread_creation_valid = false;
    uint32_t thread_creation_error = 0;
};

constexpr bool EnabledFor(const char* value) noexcept {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

enum class Permit : uint8_t { Record, Limit, Suppressed };
enum class Result : uint8_t { Disabled, Emitted, LimitNotice, Suppressed, FormatError };

class Budget {
public:
    constexpr Budget() noexcept = default;
    Permit Acquire() noexcept {
        auto claimed = m_claimed.load(std::memory_order_relaxed);
        // Strong CAS cannot fail spuriously. Every failed claim observes forward
        // progress toward 257, so this loop has a fixed bound and never wraps.
        while (claimed <= MaxRecords) {
            if (m_claimed.compare_exchange_strong(claimed, claimed + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed))
                return claimed < MaxRecords ? Permit::Record : Permit::Limit;
        }
        return Permit::Suppressed;
    }
    uint32_t Claimed() const noexcept { return m_claimed.load(std::memory_order_relaxed); }
private:
    std::atomic<uint32_t> m_claimed{0};
};
static_assert(std::atomic<uint32_t>::is_always_lock_free);
inline constinit Budget process_budget{};
inline Budget& ProcessBudget() noexcept { return process_budget; }

// Returns one complete newline-terminated numeric line, excluding its NUL, or
// zero with an empty buffer. Invalid creation witnesses serialize time as zero;
// their validity/error fields remain explicit. IDs are never derived or equated.
inline size_t FormatRecord(const Record& record, char* output, size_t capacity) noexcept {
    if (output == nullptr || capacity == 0) return 0;
    const auto process_creation = record.process_creation_valid ? record.process_creation_time : 0;
    const auto thread_creation = record.thread_creation_valid ? record.thread_creation_time : 0;
    const int count = std::snprintf(output, capacity,
        "THREAD_IDENTITY v=1 phase=%u process_id=%llu process_creation_time=%llu "
        "process_creation_valid=%u process_creation_error=%u os_tid=%llu "
        "thread_creation_time=%llu thread_creation_valid=%u thread_creation_error=%u "
        "internal_unique_id=%d guest_thread_id=%d object=0x%016llx entry=0x%016llx\n",
        static_cast<unsigned>(record.phase),
        static_cast<unsigned long long>(record.process_id),
        static_cast<unsigned long long>(process_creation),
        record.process_creation_valid ? 1u : 0u, static_cast<unsigned>(record.process_creation_error),
        static_cast<unsigned long long>(record.os_thread_id),
        static_cast<unsigned long long>(thread_creation),
        record.thread_creation_valid ? 1u : 0u, static_cast<unsigned>(record.thread_creation_error),
        static_cast<int>(record.internal_unique_id), static_cast<int>(record.guest_thread_id),
        static_cast<unsigned long long>(record.object_token),
        static_cast<unsigned long long>(record.entry));
    if (count <= 0 || static_cast<size_t>(count) >= capacity) {
        output[0] = '\0';
        return 0;
    }
    return static_cast<size_t>(count);
}

// Caller caches the exact environment flag, and guards any TLS/object capture.
// Supplier returns Record; sink receives exactly one bounded (char*, size_t)
// line. A disabled call must not even consult the budget. The generic budget
// parameter permits a test spy to verify that default-off contract.
template <typename BudgetType, typename Supplier, typename Sink>
Result Emit(bool enabled, BudgetType& budget, Supplier&& supplier, Sink&& sink) {
    if (!enabled) return Result::Disabled;
    const auto permit = budget.Acquire();
    if (permit == Permit::Suppressed) return Result::Suppressed;
    if (permit == Permit::Limit) {
        std::forward<Sink>(sink)(LimitLine, sizeof(LimitLine) - 1);
        return Result::LimitNotice;
    }
    const Record record = std::forward<Supplier>(supplier)();
    char line[MaxLineBytes]{};
    const auto length = FormatRecord(record, line, sizeof(line));
    if (length == 0) return Result::FormatError;
    std::forward<Sink>(sink)(line, length);
    return Result::Emitted;
}
} // namespace Common::ThreadIdentityDiagnostics
#endif

#ifndef KYTY_CAUSAL_OPERATION_WITNESS_H_
#define KYTY_CAUSAL_OPERATION_WITNESS_H_

#include "graphics/host_gpu/renderer/causalDiagnostics.h"

namespace Libs::Graphics::CausalDiagnostics {
// These are diagnostic witnesses, never scheduling tokens. Their selected tick
// and enqueue thread/parent come from the actual accepted PendingOperation.
struct OperationWitness {
    uint64_t native_domain = 0;
    uint64_t lifetime_sequence = 0;
    uint64_t operation_id = 0;
    uint64_t selected_tick = 0;
    ParentRef queued_parent{};
    uint32_t enqueue_thread = 0; // Internal emulator surrogate, not an OS TID.
    bool priority = false;
};
enum class EnqueueOutcome : uint8_t { NotAccepted, Queued, Inline };
struct EnqueueReceipt {
    EnqueueOutcome outcome = EnqueueOutcome::NotAccepted;
    OperationWitness operation{};
};
using AcceptedSink = void (*)(void*, const OperationWitness&) noexcept;

struct CallbackExecution {
    bool valid = false;
    bool inline_call = false;
    OperationWitness operation{};
    ParentRef callback_begin{};
};
inline CallbackExecution QueuedExecution(OperationWitness operation, ParentRef begin) noexcept {
    return {true, false, operation, begin};
}
inline CallbackExecution InlineExecution() noexcept {
    return {false, true, {}, {}};
}

namespace OperationWitnessDetail {
struct EnqueueBinding {
    EnqueueReceipt* receipt = nullptr;
    AcceptedSink sink = nullptr;
    void* state = nullptr;
};
inline thread_local EnqueueBinding* current_enqueue = nullptr;
inline thread_local CallbackExecution current_execution{};
}

class EnqueueCapture;
inline EnqueueCapture TakeEnqueueWitness(bool enabled = true) noexcept;

// Stack scope around exactly one DeferOperation. Its metadata-only sink must not
// enqueue, wait, query the GPU or throw. The scheduler consumes this binding at
// QueueOperation entry and invokes it synchronously only after push and parent
// commit, while its operation mutex still excludes dequeue. No pointer is kept
// in a queued callback. The caller must not publish pending state after return.
class EnqueueWitnessScope {
public:
    EnqueueWitnessScope(EnqueueReceipt& receipt, AcceptedSink sink = nullptr, void* state = nullptr,
                        bool enabled = true) noexcept : m_enabled(enabled) {
        if (!m_enabled) return;
        m_binding = {&receipt, sink, state};
        m_previous = OperationWitnessDetail::current_enqueue;
        receipt = {};
        OperationWitnessDetail::current_enqueue = &m_binding;
    }
    ~EnqueueWitnessScope() {
        if (m_enabled) OperationWitnessDetail::current_enqueue = m_previous;
    }
    EnqueueWitnessScope(const EnqueueWitnessScope&) = delete;
    EnqueueWitnessScope& operator=(const EnqueueWitnessScope&) = delete;
private:
    bool m_enabled = false;
    OperationWitnessDetail::EnqueueBinding m_binding{};
    OperationWitnessDetail::EnqueueBinding* m_previous = nullptr;
};

// Local synchronous capability; intentionally noncopyable and nonmovable so it
// cannot be captured by the deferred callable. Publication is one-shot.
class EnqueueCapture {
public:
    EnqueueCapture() noexcept = default;
    EnqueueCapture(const EnqueueCapture&) = delete;
    EnqueueCapture& operator=(const EnqueueCapture&) = delete;
    EnqueueCapture(EnqueueCapture&&) = delete;
    EnqueueCapture& operator=(EnqueueCapture&&) = delete;
    void PublishAccepted(const OperationWitness& operation) noexcept {
        const auto binding = m_binding;
        m_binding = {}; // Consume before a reentrant metadata sink can observe it.
        if (binding.receipt == nullptr) return;
        binding.receipt->operation = operation;
        binding.receipt->outcome = EnqueueOutcome::Queued;
        if (binding.sink != nullptr) binding.sink(binding.state, operation);
    }
    void PublishInline() noexcept {
        const auto binding = m_binding;
        m_binding = {};
        if (binding.receipt != nullptr) *binding.receipt = {EnqueueOutcome::Inline, {}};
    }
private:
    friend EnqueueCapture TakeEnqueueWitness(bool) noexcept;
    explicit EnqueueCapture(OperationWitnessDetail::EnqueueBinding binding) noexcept : m_binding(binding) {}
    OperationWitnessDetail::EnqueueBinding m_binding{};
};
inline EnqueueCapture TakeEnqueueWitness(bool enabled) noexcept {
    if (!enabled) return {};
    auto* binding = OperationWitnessDetail::current_enqueue;
    OperationWitnessDetail::current_enqueue = nullptr;
    return EnqueueCapture(binding != nullptr ? *binding : OperationWitnessDetail::EnqueueBinding{});
}

// Separate execution TLS: Context and inline Flip request storage do not grow.
// Inline calls explicitly mask an outer accepted callback. Disabled scopes do
// not consult or change TLS; nested and exceptional exits restore the caller.
class CallbackExecutionScope {
public:
    explicit CallbackExecutionScope(CallbackExecution execution, bool enabled = true) noexcept
        : m_enabled(enabled) {
        if (!m_enabled) return;
        m_previous = OperationWitnessDetail::current_execution;
        if (execution.inline_call) execution.valid = false;
        if (!execution.valid) {
            execution.operation = {};
            execution.callback_begin = {};
        }
        OperationWitnessDetail::current_execution = execution;
    }
    ~CallbackExecutionScope() {
        if (m_enabled) OperationWitnessDetail::current_execution = m_previous;
    }
    CallbackExecutionScope(const CallbackExecutionScope&) = delete;
    CallbackExecutionScope& operator=(const CallbackExecutionScope&) = delete;
private:
    bool m_enabled = false;
    CallbackExecution m_previous{};
};
inline CallbackExecution CurrentCallbackExecution(bool enabled = true) noexcept {
    return enabled ? OperationWitnessDetail::current_execution : CallbackExecution{};
}
} // namespace Libs::Graphics::CausalDiagnostics
#endif

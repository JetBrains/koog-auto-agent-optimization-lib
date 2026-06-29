package ai.koog.agents.optimization.common.abort


import ai.koog.agents.optimization.consumption.LLMConsumption

/**
 * A condition that may abort an execution based on the running cumulative LLM consumption.
 *
 * Implementations are injected into a training session / evaluation run and consulted from a hook
 * that sees the running cumulative — typically the root scope's `onUpdate`. They abort via a shared
 * [AbortController], which guarantees at most one [ExecutionAbortException] escapes.
 */
public fun interface AbortPolicy {
    /** Inspect the running [cumulative] consumption and abort if this policy's condition is met. */
    public fun checkAndAbortIfExceeded(cumulative: LLMConsumption?)
}

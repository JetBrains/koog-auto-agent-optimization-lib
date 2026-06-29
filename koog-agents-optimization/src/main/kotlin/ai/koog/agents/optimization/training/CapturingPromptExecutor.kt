package ai.koog.agents.optimization.training


import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.prompt.executor.model.PromptExecutor

/**
 * A [PromptExecutor] that accumulates the [LLMConsumption] of the calls it intercepts and
 * hands it back via [collectAndClear]. The training infrastructure creates one per leaf prompt
 * call (via the session's `capturingExecutorFactory`) and reads its consumption between retry
 * attempts; the concrete provider-aware extraction lives in the app.
 */
public abstract class CapturingPromptExecutor : PromptExecutor() {
    /**
     * Returns the consumption accumulated since the previous call (or since construction),
     * then resets the accumulator. Returns `null` when nothing was captured.
     */
    public abstract fun collectAndClear(): LLMConsumption?
}

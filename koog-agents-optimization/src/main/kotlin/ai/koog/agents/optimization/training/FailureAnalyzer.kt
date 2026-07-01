package ai.koog.agents.optimization.training


import ai.koog.agents.optimization.common.AnalyzedFailure

/**
 * Converts a caught exception into an [AnalyzedFailure]. This is the injection point for
 * provider-specific failure recognition: the library calls it from the retry / stage-failure
 * paths, the app supplies the recognizers (provider-specific and network heuristics).
 *
 * ### Expectations
 *
 * - **The two methods are independent recognizers for different paths — they should not delegate to
 *   each other.** [analyzeAgentRunFailure] classifies exceptions thrown while *running the tracked
 *   agent*; [analyzeTrainingFailure] classifies exceptions thrown while *executing a leaf prompt or
 *   an optimizer's own stage block*. Each is called on its own path with its own fallback, so an
 *   implementation may reasonably classify the same exception type differently in each.
 * - **Neither method sees [ai.koog.agents.optimization.common.abort.ExecutionAbortException].** Aborts
 *   are short-circuited before the analyzer is reached — the retry loop catches them in a dedicated
 *   arm and `resolveStageFailure` maps them directly — so implementations need not handle it. (This
 *   is why [NoOpFailureAnalyzer] doesn't.)
 * - **[analyzeTrainingFailure] never sees the framework's own already-analyzed wrappers**
 *   ([ai.koog.agents.optimization.training.structures.AgentRunFailureException],
 *   [ai.koog.agents.optimization.training.structures.PromptExecutionFailureException]). Those already
 *   carry an [AnalyzedFailure] and are unwrapped before the analyzer is reached.
 * - **Classify, don't throw.** An implementation must always return an [AnalyzedFailure], falling
 *   back to [ai.koog.agents.optimization.common.FailureKind.UNKNOWN] /
 *   [ai.koog.agents.optimization.common.TransiencyLevel.UNKNOWN] for exceptions it doesn't recognize
 *   (see [NoOpFailureAnalyzer]), rather than throwing out of the recognizer.
 */
public interface FailureAnalyzer {
    /**
     * Recognizes an exception thrown while running the tracked agent (the `runAgent` path). The
     * returned classification drives the retry policy's transient-vs-terminal decision.
     */
    public fun analyzeAgentRunFailure(exception: Exception): AnalyzedFailure

    /**
     * Recognizes an exception thrown while executing a leaf prompt or inside an optimizer's own
     * training-stage block. Only reached for exceptions that are neither an abort nor an
     * already-analyzed wrapper (those are resolved before this is called).
     */
    public fun analyzeTrainingFailure(exception: Exception): AnalyzedFailure
}

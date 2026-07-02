package ai.koog.agents.optimization.training


import ai.koog.agents.optimization.common.AnalyzedFailure

/**
 * Converts a caught exception into an [AnalyzedFailure]. This is the injection point for
 * provider-specific failure recognition: the library calls it from the retry / stage-failure
 * paths, the app supplies the recognizers (provider-specific and network heuristics).
 *
 * The two methods are independent recognizers for their respective paths and should not delegate to
 * each other — each may reasonably classify the same exception type differently. Two rules for
 * implementers:
 * - Aborts ([ai.koog.agents.optimization.common.abort.ExecutionAbortException]) and the framework's
 *   own already-analyzed wrappers are resolved before the analyzer is reached, so neither method ever
 *   sees them and needn't handle them.
 * - Classify, don't throw: always return an [AnalyzedFailure], falling back to
 *   [ai.koog.agents.optimization.common.FailureKind.UNKNOWN] /
 *   [ai.koog.agents.optimization.common.TransiencyLevel.UNKNOWN] for the unrecognized (see [NoOpFailureAnalyzer]).
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

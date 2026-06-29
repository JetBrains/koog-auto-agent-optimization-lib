package ai.koog.agents.optimization.training


import ai.koog.agents.optimization.common.AnalyzedFailure

/**
 * Converts a caught exception into an [AnalyzedFailure]. This is the injection seam for
 * provider-specific failure recognition: the library calls it from the retry / stage-failure
 * paths, the app supplies the recognizers (Grazie / LiteLLM / network heuristics).
 */
public interface FailureAnalyzer {
    /** Recognizes a failure thrown while running an agent. */
    public fun analyzeAgentRunFailure(exception: Exception): AnalyzedFailure

    /** Recognizes a failure thrown inside a training-stage block or prompt execution. */
    public fun analyzeTrainingFailure(exception: Exception): AnalyzedFailure
}

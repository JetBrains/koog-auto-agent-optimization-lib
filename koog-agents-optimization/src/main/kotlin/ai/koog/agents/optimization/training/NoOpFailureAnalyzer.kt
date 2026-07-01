package ai.koog.agents.optimization.training


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.FailureKind
import ai.koog.agents.optimization.common.TransiencyLevel

/**
 * A provider-agnostic [FailureAnalyzer] that classifies every exception as [FailureKind.UNKNOWN] /
 * [TransiencyLevel.UNKNOWN] — i.e. it makes no retryable-vs-terminal judgement.
 *
 * This is the default when no provider-specific recognizers are wired in: the retry loop then
 * treats failures according to its [ai.koog.agents.optimization.common.retries.RetryPolicy]
 * (conservative by default). Supply a richer [FailureAnalyzer] to get smart transient-error retries.
 */
public object NoOpFailureAnalyzer : FailureAnalyzer {
    override fun analyzeAgentRunFailure(exception: Exception): AnalyzedFailure = unknown(exception)

    override fun analyzeTrainingFailure(exception: Exception): AnalyzedFailure = unknown(exception)

    private fun unknown(exception: Exception): AnalyzedFailure = AnalyzedFailure(
        resolvedId = "UnrecognizedError:${exception::class.simpleName}",
        description = exception.message ?: exception.toString(),
        kind = FailureKind.UNKNOWN,
        transiency = TransiencyLevel.UNKNOWN,
        exception = exception,
    )
}

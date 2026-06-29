package ai.koog.agents.optimization.training.structures


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.FailureKind
import ai.koog.agents.optimization.common.abort.ExecutionAbortException
import ai.koog.agents.optimization.training.FailureAnalyzer
import ai.koog.agents.optimization.training.records.AgentRunRecord
import ai.koog.agents.optimization.training.records.PromptExecutionRecord
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.utils.common.Ratio

/**
 * Thrown by `runAgent().getOrThrow()` / `runAgentOrThrow()` to propagate a recorded agent run failure.
 * The [failedRecord] is already added to the stage's substages before this exception is constructed.
 */
public class AgentRunFailureException(
    /** The recorded failed agent run, already attached to the stage's substages. */
    public val failedRecord: AgentRunRecord.Failed,
    cause: Throwable,
) : Exception("Agent run failed", cause) {
    /** The analyzed failure carried by [failedRecord]. */
    public val analyzedFailure: AnalyzedFailure get() = failedRecord.failure
}

/**
 * Thrown when a prompt execution fails. Wraps the already-recorded [PromptExecutionRecord.Failed]
 * so that the enclosing [resolveStageFailure] can extract the analyzed failure directly.
 */
public class PromptExecutionFailureException(
    /** The recorded failed prompt execution, already attached to the stage's substages. */
    public val failedRecord: PromptExecutionRecord.Failed,
    cause: Throwable,
) : Exception("Prompt execution '${failedRecord.promptName}' failed", cause) {
    /** The analyzed failure carried by [failedRecord]. */
    public val analyzedFailure: AnalyzedFailure get() = failedRecord.failure
}

/**
 * Thrown by `runStage(...).getOrThrow()` / `runStageOrThrow()` to propagate a recorded stage failure.
 * The [failedRecord] is already added to the parent's substages before this exception is constructed.
 */
public class StageFailedException(
    /** The recorded failed stage, already attached to the parent's substages. */
    public val failedRecord: StageRecord,
    cause: Throwable,
) : Exception("Stage '${failedRecord.name}' failed", cause)

/**
 * Thrown synchronously by `StageScope.iterateDataset` after the loop completes, when the
 * observed per-item failure rate exceeds the configured threshold. Carries the structured
 * cause (rate + threshold) so the recognizer can describe the abort without parsing a string.
 *
 * Recognized via a dedicated branch in `analyzeTrainingFailure` that maps it to [FailureKind.EXECUTION_ABORTED].
 * TODO: Make it an [ExecutionAbortException] once the behaviour of this failure is standardized and
 *       can reuse the abort logic.
 */
public class DatasetMaxFailureRateExceededException(
    /** The observed per-item failure rate that tripped the limit. */
    public val failedRatio: Ratio,
    /** The configured maximum allowed failure rate. */
    public val threshold: Double,
) : RuntimeException(
    "Dataset iteration has failureRate = $failedRatio, which is over the allowed limit $threshold"
)

/**
 * Returns the [AnalyzedFailure] for an exception caught inside a stage block.
 *
 * Already-analyzed failures ([AgentRunFailureException], [PromptExecutionFailureException],
 * [ExecutionAbortException]) are unwrapped via the value they carry; anything else is run
 * through [FailureAnalyzer.analyzeTrainingFailure].
 */
public fun resolveStageFailure(exception: Exception, analyzer: FailureAnalyzer): AnalyzedFailure {
    return when (exception) {
        is AgentRunFailureException -> exception.analyzedFailure
        is PromptExecutionFailureException -> exception.analyzedFailure
        is ExecutionAbortException -> exception.toAnalyzedFailure()
        else -> analyzer.analyzeTrainingFailure(exception)
    }
}

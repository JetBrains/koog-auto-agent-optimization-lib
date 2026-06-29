package ai.koog.agents.optimization.common.retries


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.training.records.AgentRunRecord
import ai.koog.agents.optimization.training.records.PromptExecutionRecord
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import ai.koog.agents.optimization.utils.serialization.PrettyRoundedDurationSerializer
import kotlinx.serialization.Serializable
import kotlin.time.Duration

/**
 * A single failed attempt that preceded the final attempt of a retried operation.
 *
 * Lives in [AgentRunRecord.previousAttempts] / [PromptExecutionRecord.previousAttempts]
 * / `DatasetItemEvaluation.previousAttempts`. The list order matches execution order.
 */
@Serializable
public data class FailedAttempt(
    /**
     * 0-indexed position in the retry sequence.
     * `attemptIndex == 0` is the original attempt (whose failure triggered the first retry);
     * `attemptIndex == 1` is the first retry attempt that itself failed; and so on.
     */
    val attemptIndex: Int,

    /** The classified failure this attempt ended with. */
    val failure: AnalyzedFailure,

    /** Wall-clock time this attempt itself took (excludes the delay before it). */
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    val elapsedTime: Duration,

    /** LLM consumption recorded for this attempt, or N/A when unavailable. */
    val consumption: LLMConsumptionOrNA,

    /**
     * Wall-clock delay applied *before* this attempt started, per the retry policy.
     * `Duration.ZERO` for the original attempt (`attemptIndex == 0`).
     */
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    val delayBeforeAttempt: Duration,
)

package ai.koog.agents.optimization.training.records


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.retries.FailedAttempt
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import ai.koog.agents.optimization.utils.serialization.PrettyRoundedDurationSerializer
import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlin.time.Duration

/** Records the outcome of a single prompt execution (possibly across retries). Leaf record — no substages. */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
public sealed class PromptExecutionRecord : TrainingRecord(), LeafRecordWithRetries {
    /** Name of the executed prompt. */
    public abstract val promptName: String

    /**
     * Wall-clock elapsed time of the **final** attempt. For an aggregate, see [aggregatedElapsedTime].
     */
    abstract override val elapsedTime: Duration

    /**
     * LLM consumption attributable to the **final** attempt. For an aggregate, see [aggregatedConsumption].
     */
    abstract override val consumption: LLMConsumptionOrNA

    /**
     * Failed attempts preceding the final attempt; empty if no retries were performed.
     * Order matches execution order.
     */
    abstract override val previousAttempts: List<FailedAttempt>

    /** Prompt execution completed successfully. */
    @SerialName("Prompt execution: completed")
    @Serializable
    public data class Completed(
        /** Name of the executed prompt. */
        override val promptName: String,
        @Serializable(with = PrettyRoundedDurationSerializer::class)
        override val elapsedTime: Duration,
        override val consumption: LLMConsumptionOrNA,
        /** Free-form structured log of actions taken during the execution, serialized only when present. */
        @EncodeDefault(EncodeDefault.Mode.NEVER)
        val actionLog: JsonElement? = null,
        @EncodeDefault(EncodeDefault.Mode.NEVER)
        override val previousAttempts: List<FailedAttempt> = emptyList(),
    ) : PromptExecutionRecord()

    /** Prompt execution failed (exception thrown). */
    @SerialName("Prompt execution: failed")
    @Serializable
    public data class Failed(
        /** Name of the executed prompt. */
        override val promptName: String,
        /** Failure analysis of the final failed attempt. */
        val failure: AnalyzedFailure,
        @Serializable(with = PrettyRoundedDurationSerializer::class)
        override val elapsedTime: Duration,
        override val consumption: LLMConsumptionOrNA,
        @EncodeDefault(EncodeDefault.Mode.NEVER)
        override val previousAttempts: List<FailedAttempt> = emptyList(),
    ) : PromptExecutionRecord()
}

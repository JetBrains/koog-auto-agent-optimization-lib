package ai.koog.agents.optimization.training.records


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.retries.FailedAttempt
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import ai.koog.agents.optimization.utils.serialization.PrettyRoundedDurationSerializer
import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.time.Duration

/** Records the outcome of a single agent run (possibly across retries). Leaf record — no substages. */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
public sealed class AgentRunRecord : TrainingRecord(), LeafRecordWithRetries {
    /**
     * Wall-clock elapsed time of the **final** attempt (the successful one, or the last failed
     * attempt if no attempt succeeded). For an aggregate including all prior failed attempts and
     * the wait delays between them, see [aggregatedElapsedTime].
     */
    abstract override val elapsedTime: Duration

    /**
     * LLM consumption attributable to the **final** attempt. For an aggregate over all attempts,
     * see [aggregatedConsumption].
     */
    abstract override val consumption: LLMConsumptionOrNA

    /**
     * Failed attempts preceding the final attempt; empty if no retries were performed.
     * Order matches execution order (first failed attempt first).
     */
    abstract override val previousAttempts: List<FailedAttempt>

    /** Agent ran to completion without infrastructure failure. */
    @SerialName("Agent run: completed")
    @Serializable
    public data class Completed(
        /** Final output produced by the agent. */
        val agentOutput: String,
        /** Evaluation score assigned to this run. */
        val score: Double,
        /** Whether the run is considered to have solved the task. */
        val solved: Boolean,
        @Serializable(with = PrettyRoundedDurationSerializer::class)
        override val elapsedTime: Duration,
        override val consumption: LLMConsumptionOrNA,
        @EncodeDefault(EncodeDefault.Mode.NEVER)
        override val previousAttempts: List<FailedAttempt> = emptyList(),
    ) : AgentRunRecord()

    /** Agent run failed with an infrastructure error. */
    @SerialName("Agent run: failed")
    @Serializable
    public data class Failed(
        /** Failure analysis of the **final** failed attempt. */
        val failure: AnalyzedFailure,
        @Serializable(with = PrettyRoundedDurationSerializer::class)
        override val elapsedTime: Duration,
        override val consumption: LLMConsumptionOrNA,
        @EncodeDefault(EncodeDefault.Mode.NEVER)
        override val previousAttempts: List<FailedAttempt> = emptyList(),
    ) : AgentRunRecord()
}

package ai.koog.agents.optimization.common.results


import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import ai.koog.agents.optimization.utils.serialization.PrettyRoundedDurationSerializer
import kotlinx.serialization.Serializable
import kotlin.time.Duration

/** Retries-aware consumption totals, attached to a consumption metric when any retry fired below. */
@Serializable
public data class ConsumptionWithRetries(
    /** Final attempt + every preceding retry attempt. */
    val totalConsumptionWithRetries: LLMConsumptionOrNA = LLMConsumptionOrNA.NA,
    /** Retry attempts only (the final attempt — whether successful or failed — is excluded). */
    val totalConsumptionInRetries: LLMConsumptionOrNA = LLMConsumptionOrNA.NA,
)

/** Retries-aware elapsed-time totals, attached to an elapsed-time metric when any retry fired below. */
@Serializable
public data class ElapsedWithRetries(
    /** Final attempt + retry attempts + retry wait delays. */
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    val totalElapsedTimeWithRetries: Duration = Duration.ZERO,
    /** Retry attempts + retry wait delays (the final attempt's elapsed is excluded). */
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    val totalElapsedTimeInRetries: Duration = Duration.ZERO,
)

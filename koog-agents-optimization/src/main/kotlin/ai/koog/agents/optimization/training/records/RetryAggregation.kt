package ai.koog.agents.optimization.training.records


import ai.koog.agents.optimization.common.retries.FailedAttempt
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import ai.koog.agents.optimization.utils.serialization.sumConsumptions
import kotlin.time.Duration

/**
 * Leaf record that may carry retry history. The [elapsedTime] and [consumption] fields are the
 * **final** attempt's values; the extension helpers below combine them with [previousAttempts].
 */
public interface LeafRecordWithRetries {
    /** Wall-clock elapsed time of the final attempt. See [aggregatedElapsedTime] for the all-attempts total. */
    public val elapsedTime: Duration

    /** LLM consumption of the final attempt. See [aggregatedConsumption] for the all-attempts total. */
    public val consumption: LLMConsumptionOrNA

    /** Failed attempts preceding the final attempt, in execution order; empty if no retries occurred. */
    public val previousAttempts: List<FailedAttempt>
}

/** Final attempt + retry attempts. Returns `NA` only if every contributing value is `NA`. */
public fun LeafRecordWithRetries.aggregatedConsumption(): LLMConsumptionOrNA {
    if (previousAttempts.isEmpty()) return consumption
    val parts = buildList(previousAttempts.size + 1) {
        add(consumption)
        previousAttempts.forEach { add(it.consumption) }
    }
    return sumConsumptions(parts)
}

/** Final attempt + retry attempts + retry wait delays. */
public fun LeafRecordWithRetries.aggregatedElapsedTime(): Duration =
    previousAttempts.fold(elapsedTime) { acc, attempt ->
        acc + attempt.elapsedTime + attempt.delayBeforeAttempt
    }

/** Retry attempts only (excludes the final attempt). `NA` if all retries had `NA` consumption. */
public fun LeafRecordWithRetries.consumptionInRetries(): LLMConsumptionOrNA =
    sumConsumptions(previousAttempts.map { it.consumption })

/** Retry-attempt elapsed + retry wait delays (excludes the final attempt). */
public fun LeafRecordWithRetries.elapsedTimeInRetries(): Duration =
    previousAttempts.fold(Duration.ZERO) { acc, attempt ->
        acc + attempt.elapsedTime + attempt.delayBeforeAttempt
    }

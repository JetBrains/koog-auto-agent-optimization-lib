package ai.koog.agents.optimization.training.metrics


import ai.koog.agents.optimization.training.metrics.impl.ConsumptionMetric
import ai.koog.agents.optimization.training.metrics.impl.ElapsedTimeMetric
import ai.koog.agents.optimization.training.metrics.impl.Metric
import ai.koog.agents.optimization.training.metrics.impl.RetryStatsMetric
import ai.koog.agents.optimization.training.records.*
import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import ai.koog.agents.optimization.utils.serialization.sumConsumptions
import kotlin.time.Duration

// ======================================================================
// Elapsed time extraction
// ======================================================================

/**
 * Wall-clock per finished substage. Stage substages contribute [StageRecord.realElapsed] (only
 * once finished, since a running stage's live value would skew avg / max); leaf substages
 * contribute [aggregatedElapsedTime] (final attempt + retries + retry wait delays) — the real
 * cost a downstream `avgPerFinishedSubstage` / ETC projection should reflect.
 */
public fun List<TrainingRecord>.finishedElapsedTimes(): List<Duration> = mapNotNull { record ->
    when (record) {
        is StageRecord -> record.realElapsed.takeIf { record.isFinished }
        is AgentRunRecord -> record.aggregatedElapsedTime()
        is PromptExecutionRecord -> record.aggregatedElapsedTime()
    }
}

// ======================================================================
// Consumption extraction
// ======================================================================

/** Get the consumption of any record. */
public fun TrainingRecord.consumption(): LLMConsumptionOrNA = when (this) {
    is StageRecord -> metrics[ConsumptionMetric.KEY]?.totalConsumption ?: LLMConsumptionOrNA.NA
    is AgentRunRecord.Completed -> consumption
    is AgentRunRecord.Failed -> consumption
    is PromptExecutionRecord.Completed -> consumption
    is PromptExecutionRecord.Failed -> consumption
}

/** Sum consumption across all substages, skipping NA values. */
public fun List<TrainingRecord>.totalConsumption(): LLMConsumption? {
    var total: LLMConsumption? = null
    for (record in this) {
        val c = record.consumption()
        if (c is LLMConsumptionOrNA.Value) {
            total = if (total == null) c.consumption else total + c.consumption
        }
    }
    return total
}

// ======================================================================
// Counting utilities
// ======================================================================

/** Count substages that have finished, regardless of failures. */
public fun List<TrainingRecord>.countFinished(): Int = count { record ->
    when (record) {
        is StageRecord -> record.isFinished
        else -> true
    }
}

/**
 * Keep only finished substages: leaves are atomic so they always pass, stages pass only
 * when [StageRecord.isFinished]. Intended for aggregate metrics that would otherwise
 * include an in-flight stage's partial state (consumption/solved counts/etc.) in the roll-up.
 */
public fun List<TrainingRecord>.filterFinished(): List<TrainingRecord> = filter { record ->
    when (record) {
        is StageRecord -> record.isFinished
        else -> true
    }
}

/** Count substages that are failed (stages with failure, or failed leaf records). */
public fun List<TrainingRecord>.countFailed(): Int = count { record ->
    when (record) {
        is StageRecord -> record.failure != null
        is AgentRunRecord.Failed -> true
        is AgentRunRecord.Completed -> false
        is PromptExecutionRecord.Failed -> true
        is PromptExecutionRecord.Completed -> false
    }
}

/** Count substages that are completed successfully (no failure). */
public fun List<TrainingRecord>.countCompleted(): Int = count { record ->
    when (record) {
        is StageRecord -> record.isFinished && record.failure == null
        is AgentRunRecord.Failed -> false
        is AgentRunRecord.Completed -> true
        is PromptExecutionRecord.Failed -> false
        is PromptExecutionRecord.Completed -> true
    }
}

// ======================================================================
// Metric extraction from substages
// ======================================================================

/** Collect a specific metric from all substage StageRecords that have it. */
public fun <T : Metric> List<TrainingRecord>.substageMetrics(key: MetricKey<T>): List<T> =
    filterIsInstance<StageRecord>().mapNotNull { it.metrics[key] }

/** Get a double value from a specific metric in substages, for min/max/avg aggregation. */
public fun <T : Metric> List<TrainingRecord>.substageMetricValues(
    key: MetricKey<T>,
    extract: (T) -> Double?,
): List<Double> = substageMetrics(key).mapNotNull(extract)

// ======================================================================
// Agent run utilities
// ======================================================================

/** Collect all AgentRunRecords (both completed and failed), recursively through stages. */
public fun List<TrainingRecord>.allAgentRunsRecursively(): List<AgentRunRecord> =
    flatMap { record ->
        when (record) {
            is AgentRunRecord -> listOf(record)
            is StageRecord -> record.substages.allAgentRunsRecursively()
            else -> emptyList()
        }
    }

// ======================================================================
// Solved detection
// ======================================================================

/** Check if a record (or its substages recursively) contains any solved agent run. */
public fun TrainingRecord.hasAnySolvedAgentRun(): Boolean = when (this) {
    is AgentRunRecord.Completed -> solved
    is StageRecord -> substages.any { it.hasAnySolvedAgentRun() }
    else -> false
}

// ======================================================================
// Numeric aggregation helpers
// ======================================================================

/** Return the average of the list, or null if the list is empty. **/
public fun List<Double>.averageOrNull(): Double? = if (isEmpty()) null else average()

// ======================================================================
// Retries-aware aggregation
// ======================================================================

/**
 * Sum a value across substages, dispatching by record type. `forStage` reads from a stage's
 * cached sub-metric; `forLeaf` reads from a leaf's per-attempt fields.
 */
private inline fun <T> List<TrainingRecord>.mapByRecord(
    forStage: (StageRecord) -> T,
    forLeaf: (LeafRecordWithRetries) -> T,
): List<T> = map { record ->
    when (record) {
        is StageRecord -> forStage(record)
        is LeafRecordWithRetries -> forLeaf(record)
    }
}

/** Final attempt + retry attempts, summed across substages. */
public fun List<TrainingRecord>.totalConsumptionWithRetries(): LLMConsumptionOrNA = sumConsumptions(
    mapByRecord(
        forStage = { it.metrics[ConsumptionMetric.KEY]?.withRetries?.totalConsumptionWithRetries ?: it.consumption() },
        forLeaf = { it.aggregatedConsumption() },
    )
)

/** Retry-attempt consumption only (excludes the final attempt), summed across substages. */
public fun List<TrainingRecord>.totalConsumptionInRetries(): LLMConsumptionOrNA = sumConsumptions(
    mapByRecord(
        forStage = { it.metrics[ConsumptionMetric.KEY]?.withRetries?.totalConsumptionInRetries ?: LLMConsumptionOrNA.NA },
        forLeaf = { it.consumptionInRetries() },
    )
)

/** Final attempt + retry attempts + retry wait delays, summed across substages. */
public fun List<TrainingRecord>.totalElapsedTimeWithRetries(): Duration = mapByRecord(
    forStage = {
        it.metrics[ElapsedTimeMetric.KEY]?.withRetries?.totalElapsedTimeWithRetries
            ?: if (it.isFinished) it.realElapsed else Duration.ZERO
    },
    forLeaf = { it.aggregatedElapsedTime() },
).fold(Duration.ZERO, Duration::plus)

/** Retry-attempt elapsed + retry wait delays (excludes the final attempt), summed across substages. */
public fun List<TrainingRecord>.totalElapsedTimeInRetries(): Duration = mapByRecord(
    forStage = { it.metrics[ElapsedTimeMetric.KEY]?.withRetries?.totalElapsedTimeInRetries ?: Duration.ZERO },
    forLeaf = { it.elapsedTimeInRetries() },
).fold(Duration.ZERO, Duration::plus)

/** `true` if any leaf in this subtree performed at least one retry. */
public fun List<TrainingRecord>.anyRetriesPerformed(): Boolean = any { record ->
    when (record) {
        is StageRecord -> {
            val cached = record.metrics[RetryStatsMetric.KEY]?.totalRetriesPerformed
            if (cached != null) cached > 0 else record.substages.anyRetriesPerformed()
        }
        is LeafRecordWithRetries -> record.previousAttempts.isNotEmpty()
    }
}

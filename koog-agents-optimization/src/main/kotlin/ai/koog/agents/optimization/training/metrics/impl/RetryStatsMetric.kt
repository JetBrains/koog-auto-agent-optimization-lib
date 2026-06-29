package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.training.metrics.MetricKey
import ai.koog.agents.optimization.training.records.AgentRunRecord
import ai.koog.agents.optimization.training.records.PromptExecutionRecord
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.common.Ratio
import kotlinx.serialization.*

/**
 * Counts retries performed below this stage and how often they resolve a failure.
 *
 * Present on every stage. All fields are omitted when at their default, so a retries-free stage
 * serializes to an empty `"retryStats": {}` object.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
@SerialName("retryStats")
public data class RetryStatsMetric(
    @Transient
    override val key: MetricKey<RetryStatsMetric> = KEY,

    /** Total retry attempts performed by all leaves below this stage. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val totalRetriesPerformed: Int = 0,

    /** Number of leaves that retried at least once. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val leavesWithRetries: Int = 0,
    /** [leavesWithRetries] over the number of finished leaves. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val leavesWithRetriesRatioOfFinished: Ratio = Ratio.ZERO,
    /** Leaves that retried and ultimately completed successfully. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val leavesResolvedAfterRetries: Int = 0,
    /** [leavesResolvedAfterRetries] over [leavesWithRetries]. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val leavesResolvedAfterRetriesRatioOfLeavesWithRetries: Ratio = Ratio.ZERO,
    /** Mean retries per leaf that retried at least once, or `null` when none retried. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val avgRetriesPerLeafWithRetries: Double? = null,

    /** Histogram of finished-Completed leaves keyed by how many retries each needed (0 included). */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val completedLeavesByRetryCount: Map<Int, Int> = emptyMap(),
    /** Leaves that retried at least once and ended Failed. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val leavesUnresolvedAfterRetries: Int = 0,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): RetryStatsMetric {
        val acc = substages.fold(LeafAccumulator()) { acc, record -> acc + record.contribution() }
        val avgRetries = if (acc.leavesWithRetries > 0) acc.totalRetries.toDouble() / acc.leavesWithRetries else null
        val anyRetried = acc.totalRetries > 0
        return RetryStatsMetric(
            totalRetriesPerformed = acc.totalRetries,
            leavesWithRetries = acc.leavesWithRetries,
            leavesWithRetriesRatioOfFinished = if (anyRetried) Ratio(
                acc.leavesWithRetries,
                acc.finishedLeaves
            ) else Ratio.ZERO,
            leavesResolvedAfterRetries = acc.leavesResolvedAfterRetries,
            leavesResolvedAfterRetriesRatioOfLeavesWithRetries = Ratio(
                acc.leavesResolvedAfterRetries,
                acc.leavesWithRetries,
            ),
            avgRetriesPerLeafWithRetries = avgRetries,

            completedLeavesByRetryCount = if (anyRetried) acc.completedLeavesByRetryCount.toSortedMap() else emptyMap(),
            leavesUnresolvedAfterRetries = acc.leavesUnresolvedAfterRetries,
        )
    }

    /** Key holder for [RetryStatsMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<RetryStatsMetric> = MetricKey<RetryStatsMetric>("retryStats")
    }
}

private data class LeafAccumulator(
    val totalRetries: Int = 0,
    val leavesWithRetries: Int = 0,
    val leavesResolvedAfterRetries: Int = 0,
    val leavesUnresolvedAfterRetries: Int = 0,
    val finishedLeaves: Int = 0,
    val completedLeavesByRetryCount: Map<Int, Int> = emptyMap(),
) {
    operator fun plus(other: LeafAccumulator): LeafAccumulator {
        val mergedHistogram = (completedLeavesByRetryCount.keys + other.completedLeavesByRetryCount.keys)
            .associateWith { (completedLeavesByRetryCount[it] ?: 0) + (other.completedLeavesByRetryCount[it] ?: 0) }
        return LeafAccumulator(
            totalRetries = totalRetries + other.totalRetries,
            leavesWithRetries = leavesWithRetries + other.leavesWithRetries,
            leavesResolvedAfterRetries = leavesResolvedAfterRetries + other.leavesResolvedAfterRetries,
            leavesUnresolvedAfterRetries = leavesUnresolvedAfterRetries + other.leavesUnresolvedAfterRetries,
            finishedLeaves = finishedLeaves + other.finishedLeaves,
            completedLeavesByRetryCount = mergedHistogram,
        )
    }
}

private fun TrainingRecord.contribution(): LeafAccumulator = when (this) {
    is StageRecord -> metrics[RetryStatsMetric.KEY]?.toAccumulator()
        ?: substages.fold(LeafAccumulator()) { acc, record -> acc + record.contribution() }
    is AgentRunRecord.Completed -> oneCompletedLeaf(previousAttempts.size)
    is AgentRunRecord.Failed -> oneFailedLeaf(previousAttempts.size)
    is PromptExecutionRecord.Completed -> oneCompletedLeaf(previousAttempts.size)
    is PromptExecutionRecord.Failed -> oneFailedLeaf(previousAttempts.size)
}

private fun RetryStatsMetric.toAccumulator(): LeafAccumulator = LeafAccumulator(
    totalRetries = totalRetriesPerformed,
    leavesWithRetries = leavesWithRetries,
    leavesResolvedAfterRetries = leavesResolvedAfterRetries,
    leavesUnresolvedAfterRetries = leavesUnresolvedAfterRetries,
    finishedLeaves = leavesWithRetriesRatioOfFinished.bottom,
    completedLeavesByRetryCount = completedLeavesByRetryCount,
)

private fun oneCompletedLeaf(retryCount: Int): LeafAccumulator = LeafAccumulator(
    totalRetries = retryCount,
    leavesWithRetries = if (retryCount > 0) 1 else 0,
    leavesResolvedAfterRetries = if (retryCount > 0) 1 else 0,
    leavesUnresolvedAfterRetries = 0,
    finishedLeaves = 1,
    completedLeavesByRetryCount = mapOf(retryCount to 1),
)

private fun oneFailedLeaf(retryCount: Int): LeafAccumulator = LeafAccumulator(
    totalRetries = retryCount,
    leavesWithRetries = if (retryCount > 0) 1 else 0,
    leavesResolvedAfterRetries = 0,
    leavesUnresolvedAfterRetries = if (retryCount > 0) 1 else 0,
    finishedLeaves = 1,
    completedLeavesByRetryCount = emptyMap(),
)

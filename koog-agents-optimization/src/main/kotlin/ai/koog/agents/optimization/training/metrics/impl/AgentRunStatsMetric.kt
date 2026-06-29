package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.training.metrics.MetricKey
import ai.koog.agents.optimization.training.metrics.allAgentRunsRecursively
import ai.koog.agents.optimization.training.records.AgentRunRecord
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.common.Ratio
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

/**
 * Aggregated statistics for all agent runs within a stage _**recursively**_ through substages.
 */
@Serializable
@SerialName("agentRunStats")
public data class AgentRunStatsMetric(
    @Transient
    override val key: MetricKey<AgentRunStatsMetric> = KEY,
    /** Total number of agent runs found recursively under the stage. */
    val totalRuns: Int = 0,
    /** Agent runs that completed (regardless of whether they solved). */
    val completedRuns: Int = 0,
    /** Agent runs that ended in failure. */
    val failedRuns: Int = 0,
    /** Agent runs that completed with `solved == true`. */
    val solvedRuns: Int = 0,
    /** [completedRuns] over [totalRuns]. */
    val completionRatio: Ratio = Ratio.ZERO,
    /** [solvedRuns] over [totalRuns]. */
    val solvedRatio: Ratio = Ratio.ZERO,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): AgentRunStatsMetric {
        val allRuns = substages.allAgentRunsRecursively()
        val total = allRuns.size
        val completed = allRuns.count { it is AgentRunRecord.Completed }
        val failed = allRuns.count { it is AgentRunRecord.Failed }
        val solved = allRuns.count { it is AgentRunRecord.Completed && it.solved }

        return AgentRunStatsMetric(
            totalRuns = total,
            completedRuns = completed,
            failedRuns = failed,
            solvedRuns = solved,
            completionRatio = Ratio(completed, total),
            solvedRatio = Ratio(solved, total),
        )
    }

    /** Key holder for [AgentRunStatsMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<AgentRunStatsMetric> = MetricKey<AgentRunStatsMetric>("agentRunStats")
    }
}

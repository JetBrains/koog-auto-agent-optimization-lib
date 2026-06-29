package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.training.metrics.MetricKey
import ai.koog.agents.optimization.training.metrics.filterFinished
import ai.koog.agents.optimization.training.metrics.substageMetricValues
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

/**
 * Tracks the minimum solved rate across substages that have a [DatasetSolvedRateMetric].
 *
 * Only recalculates when [isStageUpdate] is `true`, to avoid alarming intermediate values
 * (e.g. 0% after only one item processed).
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
@SerialName("minSolvedRate")
public data class MinSolvedRateMetric(
    @Transient
    override val key: MetricKey<MinSolvedRateMetric> = KEY,
    /** Lowest solved ratio observed across finished children that expose a [DatasetSolvedRateMetric], or `null` if none. */
    val minSolvedRate: Double? = null,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): MinSolvedRateMetric {
        if (!isStageUpdate) return this

        // Only finished direct children expose a settled solvedRatio; a running iteration's
        // running ratio reflects a tiny prefix of items and would report an alarming 0% / 100%.
        val solvedRates = substages.filterFinished()
            .substageMetricValues(DatasetSolvedRateMetric.KEY) { it.solvedRatio.fraction }
        return MinSolvedRateMetric(minSolvedRate = solvedRates.minOrNull())
    }

    /** Key holder for [MinSolvedRateMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<MinSolvedRateMetric> = MetricKey<MinSolvedRateMetric>("minSolvedRate")
    }
}

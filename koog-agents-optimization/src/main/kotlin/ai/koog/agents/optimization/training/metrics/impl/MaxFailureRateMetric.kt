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
 * Tracks the maximum failure rate across substages.
 *
 * Only recalculates when [isStageUpdate] is `true`, to avoid alarming intermediate values
 * (e.g. 100% after only one item processed).
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
@SerialName("maxFailureRate")
public data class MaxFailureRateMetric(
    @Transient
    override val key: MetricKey<MaxFailureRateMetric> = KEY,
    /** Highest failed-substage ratio observed across finished direct children, or `null` if none. */
    val maxFailureRate: Double? = null,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): MaxFailureRateMetric {
        if (!isStageUpdate) return this

        // Only finished direct children contribute: a running child's failedRatio reflects
        // a tiny prefix of its own substages — one early failure would pin this at 100% for
        // the entire session.
        val failureRates = substages.filterFinished()
            .substageMetricValues(SubstageCountMetric.KEY) { it.failedRatio.fraction }
        return MaxFailureRateMetric(maxFailureRate = failureRates.maxOrNull())
    }

    /** Key holder for [MaxFailureRateMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<MaxFailureRateMetric> = MetricKey<MaxFailureRateMetric>("maxFailureRate")
    }
}

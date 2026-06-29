package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.training.metrics.MetricKey
import ai.koog.agents.optimization.training.metrics.countCompleted
import ai.koog.agents.optimization.training.metrics.countFailed
import ai.koog.agents.optimization.training.metrics.countFinished
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.common.Ratio
import kotlinx.serialization.*

/** Tracks processed / completed / failed substage counts and ratios, and the currently running substage name. */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
@SerialName("substageCount")
public data class SubstageCountMetric(
    @Transient
    override val key: MetricKey<SubstageCountMetric> = KEY,
    /** Number of substages that have finished (the denominator for the ratios). */
    val finished: Int = 0,
    /** Substages that completed successfully (no failure). */
    val completed: Int = 0,
    /** Substages that ended in failure. */
    val failed: Int = 0,
    /** [completed] over [finished]. */
    val completedRatio: Ratio = Ratio.ZERO,
    /** [failed] over [finished]. */
    val failedRatio: Ratio = Ratio.ZERO,
    /** Name of the substage currently running, or `null` when none is in flight. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val currentSubstageName: String? = null,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): SubstageCountMetric {
        val total = substages.countFinished()
        val ok = substages.countCompleted()
        val bad = substages.countFailed()

        val currentName = run {
            val lastStage = substages.lastOrNull() as? StageRecord ?: return@run null
            if (lastStage.isFinished) null else lastStage.name
        }

        return SubstageCountMetric(
            finished = total,
            completed = ok,
            failed = bad,
            completedRatio = Ratio(ok, total),
            failedRatio = Ratio(bad, total),
            currentSubstageName = currentName,
        )
    }

    /** Key holder for [SubstageCountMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<SubstageCountMetric> = MetricKey<SubstageCountMetric>("substageCount")
    }
}

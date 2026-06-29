package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.training.metrics.MetricKey
import ai.koog.agents.optimization.training.metrics.hasAnySolvedAgentRun
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.common.Ratio
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

/**
 * Dataset-specific metric: tracks solved / not-solved / failed dataset items.
 *
 * Only attached to stages created by `iterateDataset`. Each direct substage
 * represents one dataset item. A dataset item is "solved" if ANY of its agent runs
 * completed with `solved == true`.
 *
 * Entirely inside the [ai.koog.agents.optimization.training.metrics.MetricsMap] — never leaks as a top-level field.
 */
@Serializable
@SerialName("datasetSolvedRate")
public data class DatasetSolvedRateMetric(
    @Transient
    override val key: MetricKey<DatasetSolvedRateMetric> = KEY,
    /** Number of dataset items that have finished (the denominator for the ratios). */
    val itemsFinished: Int = 0,
    /** Finished items that had at least one solved agent run and did not fail. */
    val itemsSolved: Int = 0,
    /** Finished items that completed without being solved. */
    val itemsNotSolved: Int = 0,
    /** Finished items that ended in failure. */
    val itemsFailed: Int = 0,
    /** [itemsSolved] over [itemsFinished]. */
    val solvedRatio: Ratio = Ratio.ZERO,
    /** [itemsNotSolved] over [itemsFinished]. */
    val notSolvedRatio: Ratio = Ratio.ZERO,
    /** [itemsFailed] over [itemsFinished]. */
    val failedRatio: Ratio = Ratio.ZERO,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): DatasetSolvedRateMetric {
        // Each dataset item is a StageRecord that is appended to `substages` BEFORE its body
        // runs. An in-flight item therefore sits in this list with `isFinished=false`, and may
        // already contain an agent-run leaf record that makes `hasAnySolvedAgentRun()` true —
        // which would inflate both the denominator and the numerator the moment a new item
        // begins, before it has truly been evaluated. Restrict the ratio to truly-finished items.
        val finishedItems = substages.filterIsInstance<StageRecord>().filter { it.isFinished }
        val total = finishedItems.size
        var solved = 0
        var notSolved = 0
        var failed = 0

        for (item in finishedItems) {
            when {
                item.failure != null -> failed++
                item.hasAnySolvedAgentRun() -> solved++
                else -> notSolved++
            }
        }

        return DatasetSolvedRateMetric(
            itemsFinished = total,
            itemsSolved = solved,
            itemsNotSolved = notSolved,
            itemsFailed = failed,
            solvedRatio = Ratio(solved, total),
            notSolvedRatio = Ratio(notSolved, total),
            failedRatio = Ratio(failed, total),
        )
    }

    /** Key holder for [DatasetSolvedRateMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<DatasetSolvedRateMetric> = MetricKey<DatasetSolvedRateMetric>("datasetSolvedRate")
    }
}


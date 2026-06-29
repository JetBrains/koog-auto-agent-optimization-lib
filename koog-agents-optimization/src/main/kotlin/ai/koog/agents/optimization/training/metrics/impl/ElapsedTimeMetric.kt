package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.common.results.ElapsedWithRetries
import ai.koog.agents.optimization.training.metrics.*
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.serialization.PrettyRoundedDurationSerializer
import kotlinx.serialization.*
import kotlin.time.Duration

/**
 * Derived timing aggregates for a stage.
 *
 * The stage's own wall-clock duration lives on [StageRecord.realElapsed] — this metric only
 * exposes values that are genuinely *derived* from substages or from the stage's own elapsed:
 *
 * - [avgPerFinishedSubstage]: mean elapsed time across finished substages. In-progress substages
 *   are excluded — including their partial elapsed would skew the average.
 * - [maxPerFinishedSubstage]: longest finished substage. Useful for spotting outlier items
 *   in a dataset iteration.
 * - [etc]: projected time-to-completion. Prefers the running direct child's own ETC when
 *   available (plus projected time for siblings not yet started) — this covers the common
 *   "single `iterateDataset` child" case that the naive `realElapsed / processed × remaining`
 *   formula can't handle. Falls back to that formula when no child ETC exists.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
@SerialName("elapsedTime")
public data class ElapsedTimeMetric(
    @Transient
    override val key: MetricKey<ElapsedTimeMetric> = KEY,

    /** Mean elapsed time across finished substages (in-progress ones excluded), or `null` if none. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    val avgPerFinishedSubstage: Duration? = null,

    /** Longest finished substage, useful for spotting outlier items, or `null` if none. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    val maxPerFinishedSubstage: Duration? = null,

    /** Projected estimated time to completion for the stage, or `null` when it can't be derived. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    val etc: Duration? = null,

    /** Retries-aware totals; `null` when no retries fired below this stage. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val withRetries: ElapsedWithRetries? = null,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): ElapsedTimeMetric {
        val withRetries = if (substages.anyRetriesPerformed()) {
            ElapsedWithRetries(
                totalElapsedTimeWithRetries = substages.totalElapsedTimeWithRetries(),
                totalElapsedTimeInRetries = substages.totalElapsedTimeInRetries(),
            )
        } else null

        // avg/max depend on whether substages are finished; that only changes on stage-level events.
        // Skip the scan on leaf updates (agent runs, prompt executions) and reuse the stored avg.
        if (!isStageUpdate) {
            return copy(
                etc = computeEtc(currentStage, substages, avgPerFinishedSubstage),
                withRetries = withRetries,
            )
        }
        val finished = substages.finishedElapsedTimes()
        val avg = if (finished.isNotEmpty()) finished.reduce(Duration::plus) / finished.size else null
        val max = finished.maxOrNull()
        return copy(
            avgPerFinishedSubstage = avg,
            maxPerFinishedSubstage = max,
            etc = computeEtc(currentStage, substages, avg),
            withRetries = withRetries,
        )
    }

    private fun computeEtc(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        avgPerFinished: Duration?,
    ): Duration? {
        val total = currentStage.substagesTotal ?: return null
        if (total <= 0) return null

        // Preferred: running direct child's own ETC, plus projected time for siblings not yet started.
        // This covers single-iterateDataset stages where `substagesTotal == 1` puts the fallback
        // formula out of its `1..<total` range, so we'd otherwise report no ETC at all.
        val runningChildEtc = (substages.lastOrNull { it is StageRecord && !it.isFinished } as? StageRecord)
            ?.metrics?.get(KEY)?.etc
        if (runningChildEtc != null) {
            val notStarted = total - substages.size
            val siblingsPart = if (avgPerFinished != null) avgPerFinished * notStarted else Duration.ZERO
            return runningChildEtc + siblingsPart
        }

        // Fallback: `realElapsed / finished × remaining`. Works when no child publishes its own ETC
        // (e.g. dataset iteration where item stages don't set `substagesTotal`).
        val finished = substages.countFinished()
        if (finished == 0 || finished == total) return null
        val elapsed = currentStage.realElapsed
        if (elapsed == Duration.ZERO) return null
        return elapsed / finished * (total - finished)
    }

    /** Key holder for [ElapsedTimeMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<ElapsedTimeMetric> = MetricKey<ElapsedTimeMetric>("elapsedTime")
    }
}

package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.common.results.ConsumptionWithRetries
import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.agents.optimization.training.metrics.*
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import ai.koog.agents.optimization.utils.serialization.SolvedAwareConsumptionMetricSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

/**
 * Solved-aware consumption metric for dataset iterations.
 *
 * Tracks total LLM consumption split by solved/not-solved, plus per-substage
 * and per-solved aggregations (min, max, avg). Also computes [averageSolvedCost]:
 * total consumption divided by the number of solved items.
 *
 * **Important:** tracks the consumption of all substages (including optimizer
 * substages), not just agent runs. Consider [AgentRunStatsMetric] if you need agent-specific stats.
 *
 * Only meaningful on stages created by `iterateDataset`, where each direct substage
 * represents a dataset item whose solved/not-solved status is known.
 *
 * The canonical fields above carry the **final-attempt** view (research signal). When any leaf
 * below has performed retries, a [withRetries] sub-object is also emitted carrying the
 * `*WithRetries` / `*InRetries` totals — auxiliary metrics that quantify the cluster-cost
 * overhead of retries.
 */
@Serializable(with = SolvedAwareConsumptionMetricSerializer::class)
@SerialName("solvedAwareConsumption")
public data class SolvedAwareConsumptionMetric(
    @Transient
    override val key: MetricKey<SolvedAwareConsumptionMetric> = KEY,
    /** Total LLM consumption across solved (and not failed) finished items, or NA if none. */
    val totalSolvedConsumption: LLMConsumptionOrNA = LLMConsumptionOrNA.NA,
    /** Mean per-item total cost across solved items, or `null` if none. */
    val avgPerSolved: Double? = null,
    /** Smallest per-item total cost across solved items, or `null` if none. */
    val minPerSolved: Double? = null,
    /** Largest per-item total cost across solved items, or `null` if none. */
    val maxPerSolved: Double? = null,
    /** Total consumption across all finished items divided by the number of solved items, or `null` if none solved. */
    val averageSolvedCost: Double? = null,
    /**
     * Retries-aware totals. `null` when no retries fired anywhere under this dataset iteration —
     * the serializer omits the sub-object in that case.
     */
    val withRetries: ConsumptionWithRetries? = null,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): SolvedAwareConsumptionMetric {
        // Each dataset item stage is appended BEFORE it runs. An in-flight item may already
        // have an inner AgentRunRecord.Completed(solved=true), so `hasAnySolvedAgentRun()` would
        // classify it as solved and pull its partial consumption into the aggregates — before
        // the item has actually finished. Restrict classification to finished substages.
        val finishedSubstages = substages.filterFinished()

        val solvedTotals = mutableListOf<Double>()
        var solvedCount = 0
        var totalSolvedAccum: LLMConsumption? = null

        for (substage in finishedSubstages) {
            val consumption = (substage.consumption() as? LLMConsumptionOrNA.Value)?.consumption
            val total = consumption?.total

            val isSolved = substage.hasAnySolvedAgentRun()
            val isFailed = substage is StageRecord && substage.failure != null
            if (isSolved && !isFailed) {
                solvedCount++
                if (consumption != null) {
                    totalSolvedAccum = totalSolvedAccum?.plus(consumption) ?: consumption
                    solvedTotals.add(total!!)
                }
            }
        }

        // Use the same finished-substage base for [averageSolvedCost] so the numerator and the
        // denominator are over the same item set (total spent across finished items / solved items).
        val finishedTotal = finishedSubstages.totalConsumption()
        val totalSolvedOrNA = totalSolvedAccum
            ?.let { LLMConsumptionOrNA.Value(it) }
            ?: LLMConsumptionOrNA.NA

        val finishedTotalValue = finishedTotal?.total

        val withRetries = if (substages.anyRetriesPerformed()) {
            ConsumptionWithRetries(
                totalConsumptionWithRetries = substages.totalConsumptionWithRetries(),
                totalConsumptionInRetries = substages.totalConsumptionInRetries(),
            )
        } else null

        return SolvedAwareConsumptionMetric(
            totalSolvedConsumption = totalSolvedOrNA,
            avgPerSolved = solvedTotals.averageOrNull(),
            minPerSolved = solvedTotals.minOrNull(),
            maxPerSolved = solvedTotals.maxOrNull(),
            averageSolvedCost = if (solvedCount > 0 && finishedTotalValue != null) finishedTotalValue / solvedCount else null,
            withRetries = withRetries,
        )
    }

    /** Key holder for [SolvedAwareConsumptionMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<SolvedAwareConsumptionMetric> = MetricKey<SolvedAwareConsumptionMetric>("solvedAwareConsumption")
    }
}

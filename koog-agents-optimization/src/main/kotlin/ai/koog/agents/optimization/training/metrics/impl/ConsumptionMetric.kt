package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.common.results.ConsumptionWithRetries
import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.agents.optimization.training.metrics.*
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.serialization.ConsumptionMetricSerializer
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

/**
 * Total / min / max / avg LLM consumption across substages. Canonical totals are final-attempt
 * only; [withRetries], when non-null, exposes the retries-aware counterparts.
 */
@Serializable(with = ConsumptionMetricSerializer::class)
@SerialName("consumption")
public data class ConsumptionMetric(
    @Transient
    override val key: MetricKey<ConsumptionMetric> = KEY,
    /** Running sum of LLM consumption across substages (final-attempt only), or NA if none observed. */
    val totalConsumption: LLMConsumptionOrNA = LLMConsumptionOrNA.NA,
    /** Smallest per-substage total cost across finished substages, or `null` if none. */
    val minPerSubstage: Double? = null,
    /** Largest per-substage total cost across finished substages, or `null` if none. */
    val maxPerSubstage: Double? = null,
    /** Mean per-substage total cost across finished substages, or `null` if none. */
    val avgPerSubstage: Double? = null,
    /** Retries-aware totals; `null` when no retries fired below this stage. */
    val withRetries: ConsumptionWithRetries? = null,
) : Metric {
    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): ConsumptionMetric {
        // Total is a live running sum: in-flight child stages contribute their partial total,
        // so the session keeps growing as LLM calls happen.
        val total = substages.totalConsumption()
        val totalOrNA = if (total != null) LLMConsumptionOrNA.Value(total) else LLMConsumptionOrNA.NA

        // Per-substage aggregates only make sense over finished substages: an in-flight stage's
        // partial consumption would drag min down and skew the average below each item's real
        // cost. Leaves are atomic, so they always count.
        val finishedSubstages = substages.filterFinished()
        val substageTotals = finishedSubstages.mapNotNull { record ->
            (record.consumption() as? LLMConsumptionOrNA.Value)?.consumption?.total
        }

        val withRetries = if (substages.anyRetriesPerformed()) {
            ConsumptionWithRetries(
                totalConsumptionWithRetries = substages.totalConsumptionWithRetries(),
                totalConsumptionInRetries = substages.totalConsumptionInRetries(),
            )
        } else null

        return ConsumptionMetric(
            totalConsumption = totalOrNA,
            minPerSubstage = substageTotals.minOrNull(),
            maxPerSubstage = substageTotals.maxOrNull(),
            // NA substages count in the denominator (substage with no observed LLM call is still
            // one substage); sum is over Values only. Same shape as eval-side averagePerElement.
            avgPerSubstage = if (finishedSubstages.isEmpty()) null else substageTotals.sum() / finishedSubstages.size,
            withRetries = withRetries,
        )
    }

    /** Get the total as an [LLMConsumption] or null. */
    public fun totalOrNull(): LLMConsumption? = (totalConsumption as? LLMConsumptionOrNA.Value)?.consumption

    /** Key holder for [ConsumptionMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<ConsumptionMetric> = MetricKey<ConsumptionMetric>("consumption")
    }
}


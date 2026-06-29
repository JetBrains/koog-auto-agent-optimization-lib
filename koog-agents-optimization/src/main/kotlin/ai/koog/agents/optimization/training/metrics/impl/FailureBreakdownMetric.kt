package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.TransiencyLevel
import ai.koog.agents.optimization.common.results.FailureKindEntry
import ai.koog.agents.optimization.common.results.FinalFailures
import ai.koog.agents.optimization.common.results.TransientEncounters
import ai.koog.agents.optimization.common.retries.FailedAttempt
import ai.koog.agents.optimization.training.metrics.MetricKey
import ai.koog.agents.optimization.training.metrics.countFinished
import ai.koog.agents.optimization.training.records.AgentRunRecord
import ai.koog.agents.optimization.training.records.PromptExecutionRecord
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.utils.common.Ratio
import kotlinx.serialization.*

/**
 * Failure breakdown for finished substages, split into two sections: [finalFailures] (the run's
 * final outcomes, bucketed by kind) and [transientEncounters] (transient blips seen across all
 * attempts — the environment's noise profile). Empty stages — no final failure and no transient
 * encounter below — serialize to `{}` so the JSON stays clean.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
@SerialName("failureBreakdown")
public data class FailureBreakdownMetric(
    @Transient
    override val key: MetricKey<FailureBreakdownMetric> = KEY,

    /** Final failure outcomes of finished substages, bucketed by failure kind. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val finalFailures: FinalFailures = FinalFailures(),
    /** Transient failures seen across all attempts below this stage (the environment's noise profile). */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val transientEncounters: TransientEncounters = TransientEncounters(),
) : Metric {

    override fun recompute(
        currentStage: StageRecord,
        substages: List<TrainingRecord>,
        isStageUpdate: Boolean
    ): FailureBreakdownMetric {
        // Failures are detectable only on finished records (StageRecord.failure is null for
        // in-flight stages; failed leaves are atomic). Anchor the "ratio of" denominator on
        // [countFinished] so it's consistent with the numerator's population, matching the
        // finished-based semantics of [SubstageCountMetric].
        val finalFailureList = substages.mapNotNull { it.extractFailure() }
        val totalFinished = substages.countFinished()
        val totalFinalFailed = finalFailureList.size

        val byKind = finalFailureList.groupBy { it.kind }.mapValues { (_, list) ->
            val perResolved = mutableMapOf<String, Int>()
            for (f in list) {
                perResolved[f.resolvedId] = (perResolved[f.resolvedId] ?: 0) + 1
            }
            FailureKindEntry(
                count = list.size,
                ratioOfFinished = Ratio(list.size, totalFinished),
                ratioOfAllFinal = Ratio(list.size, totalFinalFailed),
                byId = perResolved.entries
                    .sortedByDescending { it.value }
                    .associate { it.key to it.value },
            )
        }

        // Transient-encounter counts: every TRANSIENT previousAttempts entry across every leaf,
        // plus every TRANSIENT final failure. byId tracks the per-resolvedId frequency of the
        // retry-attempt transients only (the final transient failure adds to total but not byId).
        var transientTotal = 0
        val perId = mutableMapOf<String, Int>()
        for (record in substages) {
            // Roll up per-leaf transient stats, including substages recursively via cached metrics
            // when available (consistent with the design of other rolled-up retry helpers).
            val acc = record.collectTransientEncounters()
            transientTotal += acc.total
            for ((id, count) in acc.byId) {
                perId[id] = (perId[id] ?: 0) + count
            }
        }

        return FailureBreakdownMetric(
            finalFailures = FinalFailures(total = totalFinalFailed, byKind = byKind),
            transientEncounters = TransientEncounters(
                total = transientTotal,
                byId = perId.entries
                    .sortedByDescending { it.value }
                    .associate { it.key to it.value },
            ),
        )
    }

    /** Key holder for [FailureBreakdownMetric]. */
    public companion object {
        /** The [MetricKey] under which this metric is stored. */
        public val KEY: MetricKey<FailureBreakdownMetric> = MetricKey<FailureBreakdownMetric>("failureBreakdown")
    }
}

private data class TransientAcc(
    val total: Int,
    val byId: Map<String, Int>,
)

/**
 * Walks each leaf (recursing into stage substages) and accumulates transient-failure encounters.
 * Stages contribute via their `FailureBreakdownMetric` sub-metric when populated, so an ancestor's
 * recompute reads the descendant's already-computed counts instead of re-walking the full subtree.
 */
private fun TrainingRecord.collectTransientEncounters(): TransientAcc = when (this) {
    is StageRecord -> {
        val cached = metrics[FailureBreakdownMetric.KEY]
        if (cached != null) {
            TransientAcc(cached.transientEncounters.total, cached.transientEncounters.byId)
        } else {
            TransientAcc(0, emptyMap())
        }
    }
    is AgentRunRecord.Completed -> transientAcc(previousAttempts, finalFailure = null)
    is AgentRunRecord.Failed -> transientAcc(previousAttempts, finalFailure = failure)
    is PromptExecutionRecord.Completed -> transientAcc(previousAttempts, finalFailure = null)
    is PromptExecutionRecord.Failed -> transientAcc(previousAttempts, finalFailure = failure)
}

private fun transientAcc(previousAttempts: List<FailedAttempt>, finalFailure: AnalyzedFailure?): TransientAcc {
    val transientPrev = previousAttempts.filter { it.failure.transiency == TransiencyLevel.TRANSIENT }
    val byId = transientPrev.groupingBy { it.failure.resolvedId }.eachCount()
    // A transient final failure is also an encounter (the policy gave up on a still-transient
    // failure). Non-transient finals aren't transient encounters, so don't count them.
    val finalTransient = if (finalFailure?.transiency == TransiencyLevel.TRANSIENT) 1 else 0
    return TransientAcc(total = transientPrev.size + finalTransient, byId = byId)
}

private fun TrainingRecord.extractFailure(): AnalyzedFailure? = when (this) {
    is StageRecord -> failure
    is AgentRunRecord.Failed -> failure
    is PromptExecutionRecord.Failed -> failure
    else -> null
}

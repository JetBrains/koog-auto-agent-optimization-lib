package ai.koog.agents.optimization.common.results


import ai.koog.agents.optimization.utils.common.Ratio
import kotlinx.serialization.Serializable

/**
 * Per-kind breakdown of final failures: count, ratios against the population, per-`resolvedId`
 * frequencies. Shared between training (`FailureBreakdownMetric`) and eval (`FailuresSummary`) so
 * the two sides expose the same JSON shape under `finalFailures.byKind`.
 */
@Serializable
public data class FailureKindEntry(
    /** Count of **final** failures of this kind. */
    val count: Int = 0,

    /** Ratio of finished substages (or processed items, on the eval side) that ended with a final failure of this kind. */
    val ratioOfFinished: Ratio = Ratio.ZERO,

    /** Ratio of this kind within all final failures. */
    val ratioOfAllFinal: Ratio = Ratio.ZERO,

    /** Per-`resolvedId` frequency of final failures within this kind, sorted by count descending. */
    val byId: Map<String, Int> = emptyMap(),
)

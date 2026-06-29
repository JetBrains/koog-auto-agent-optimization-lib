package ai.koog.agents.optimization.common.results


import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable

/**
 * The two sections of a failure breakdown, shared between training (`FailureBreakdownMetric`) and
 * eval (`FailuresSummary`) so both sides emit the same JSON. [FinalFailures] reports the run's final
 * outcomes; [TransientEncounters] reports transient blips seen across all attempts. Each section's
 * fields default out (`@EncodeDefault(NEVER)`), so an empty section serializes to `{}` and an
 * untouched breakdown stays clean.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
public data class FinalFailures(
    /** Number of leaves (or items, on the eval side) whose final outcome was Failed. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val total: Int = 0,
    /** Final failures bucketed by `FailureKind`; entries carry per-kind counts, ratios, and per-`resolvedId` frequencies. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val byKind: Map<ai.koog.agents.optimization.common.FailureKind, FailureKindEntry> = emptyMap(),
)

/**
 * Transient-failure events seen across all attempts — the environment's noise profile. Counts
 * events: a leaf that retried several times contributes several events here. The per-leaf retry
 * outcome (resolved or unresolved, and at which attempt) lives in `retryStats`.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
public data class TransientEncounters(
    /** Every transient event: each `TRANSIENT` retry attempt across all leaves, plus each `TRANSIENT` final failure. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val total: Int = 0,
    /** Per-`resolvedId` frequency of transient failures seen during retry attempts, sorted by count descending. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val byId: Map<String, Int> = emptyMap(),
)

package ai.koog.agents.optimization.training


/**
 * Outcome of an early-stop check, e.g. the `earlyStop` callback of
 * [ai.koog.agents.optimization.training.dsl.StageScope.iterateDataset].
 *
 * @property conditionMet `true` when execution should stop before processing the next item.
 * @property conditionMetReason Lazily evaluated human-readable reason, logged only when
 *   [conditionMet] is `true`.
 */
public data class PrematureExecutionStopDecision(
    val conditionMet: Boolean,
    val conditionMetReason: () -> String,
)

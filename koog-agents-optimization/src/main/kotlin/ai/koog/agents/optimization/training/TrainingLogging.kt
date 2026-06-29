package ai.koog.agents.optimization.training


import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.training.metrics.impl.*
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.common.toFilePathLog
import ai.koog.agents.optimization.utils.serialization.pretty

/**
 * Log formatting utilities for the training infrastructure.
 *
 * Reads metrics from [StageRecord.metrics] to build console log strings.
 */
public object TrainingLogging {

    // ===================================================================
    // Session
    // ===================================================================

    /**
     * One-line message announcing the start of a training session and where its records will be
     * written ([recordsFilePath] may be `null`, rendered as a "not persisted" hint).
     */
    public fun buildSessionBeginConsoleLog(trainingName: ExperimentName, recordsFilePath: ResilientPath?): String =
        "Starting training session '$trainingName'. Records will be saved to ${recordsFilePath.toFilePathLog()}."

    /**
     * Multi-line summary of a finished session: total elapsed time, consumption, per-stage substage
     * counts, the overall session summary (stages finished/completed/failed, failure and solved
     * rates), and any root failure. Reads aggregated values from [StageRecord.metrics].
     */
    public fun buildSessionFinishedConsoleLog(rootRecord: StageRecord): String = buildString {
        appendLine("Training session '${rootRecord.name}' finished")

        val consumption = rootRecord.metrics[ConsumptionMetric.KEY]
        val substageCount = rootRecord.metrics[SubstageCountMetric.KEY]
        val minSolved = rootRecord.metrics[MinSolvedRateMetric.KEY]
        val maxFailure = rootRecord.metrics[MaxFailureRateMetric.KEY]

        appendLine("- Total elapsed: ${rootRecord.realElapsed.pretty()}")
        if (consumption != null) {
            val consumptionStr = consumption.totalOrNull()?.toPrettyString() ?: "n/a"
            appendLine("- Total consumption: $consumptionStr")
        }

        for (substage in rootRecord.substages.filterIsInstance<StageRecord>()) {
            val sc = substage.metrics[SubstageCountMetric.KEY]
            appendLine("- Stage '${substage.name}': ${sc?.finished ?: 0} substages, ${sc?.completed ?: 0} completed")
        }

        if (substageCount != null) {
            val total = rootRecord.substagesTotal
            val finishedDisplay = if (total != null) "${substageCount.finished}/$total" else "${substageCount.finished}"
            appendLine("- Session summary:")
            appendLine("  > Stages finished: $finishedDisplay")
            appendLine("  > Completed stages: ${substageCount.completed}, failed stages: ${substageCount.failed}")
            if (maxFailure?.maxFailureRate != null) {
                appendLine("  > Max failure rate: ${maxFailure.maxFailureRate}")
            }
            if (minSolved?.minSolvedRate != null) {
                appendLine("  > Min dataset solved rate: ${minSolved.minSolvedRate}")
            }
        }

        rootRecord.failure?.let {
            appendLine("- FAILURE: ${it.resolvedId} — ${it.description}")
        }
    }

    // ===================================================================
    // Stage
    // ===================================================================

    /**
     * One-line message announcing the start of a stage, prefixed with its ancestor path
     * (derived from [stagePath]) and optionally a parent ETC hint read from [parentRecord].
     */
    public fun buildStageBeginConsoleLog(
        name: String,
        stagePath: List<String>,
        parentRecord: StageRecord,
    ): String = buildString {
        append(stagePrefix(stagePath))
        append("Starting stage '$name'")
        parentEtcHint(parentRecord)?.let { append(" ").append(it) }
    }

    /**
     * Multi-line summary of a finished stage: substage counts, timing (elapsed, averages, ETC),
     * dataset results when present, consumption, any failure, and an optional parent ETC hint.
     * Reads aggregated values from [StageRecord.metrics]; [parentRecord] is `null` for top-level stages.
     */
    public fun buildStageFinishedConsoleLog(
        stageRecord: StageRecord,
        stagePath: List<String>,
        parentRecord: StageRecord?,
    ): String = buildString {
        val prefix = stagePrefix(stagePath)
        appendLine("${prefix}Stage '${stageRecord.name}' finished")

        val substageCount = stageRecord.metrics[SubstageCountMetric.KEY]
        val elapsed = stageRecord.metrics[ElapsedTimeMetric.KEY]
        val consumption = stageRecord.metrics[ConsumptionMetric.KEY]
        val datasetSolved = stageRecord.metrics[DatasetSolvedRateMetric.KEY]

        if (substageCount != null) {
            appendLine("- Substages: ${substageCount.finished} total, ${substageCount.completed} completed, ${substageCount.failed} failed")
        }

        appendLine("- Timing:")
        appendLine("  > Elapsed: ${stageRecord.realElapsed.pretty()}")
        if (elapsed != null) {
            elapsed.avgPerFinishedSubstage?.let { appendLine("  > Avg per finished substage: ${it.pretty()}") }
            elapsed.maxPerFinishedSubstage?.let { appendLine("  > Max per finished substage: ${it.pretty()}") }
            elapsed.etc?.let { etc ->
                val remaining = (stageRecord.substagesTotal ?: 0) - (substageCount?.finished ?: 0)
                appendLine("  > ETC: ${etc.pretty()} (~${remaining.coerceAtLeast(0)} stages remaining)")
            }
        }

        if (datasetSolved != null) {
            appendLine("- Results:")
            appendLine("  > Solved: ${datasetSolved.solvedRatio}")
            appendLine("  > Not solved: ${datasetSolved.notSolvedRatio}")
            appendLine("  > Failed: ${datasetSolved.failedRatio}")
        } else if (substageCount != null) {
            appendLine("- Failed: ${substageCount.failedRatio}")
        }

        if (consumption != null) {
            val consumptionStr = consumption.totalOrNull()?.toPrettyString() ?: "n/a"
            appendLine("- Consumption: $consumptionStr")
        }

        if (stageRecord.failure != null) {
            appendLine("- FAILURE: ${stageRecord.failure!!.resolvedId} — ${stageRecord.failure!!.description}")
        }

        if (parentRecord != null) {
            parentEtcHint(parentRecord)?.let { appendLine("- $it") }
        }
    }

    /**
     * Builds a `[ancestor > ancestor]` prefix from [stagePath], skipping the session/root name
     * (index 0) and trimming a single pair of surrounding brackets from each component to avoid
     * nested bracket noise (item names like `[Dataset iteration | Item #5 / 10]`).
     *
     * Returns an empty string when there is nothing meaningful to show (root or top-level stage).
     */
    private fun stagePrefix(stagePath: List<String>): String {
        val ancestors = stagePath.drop(1)
        if (ancestors.isEmpty()) return ""
        val parts = ancestors.map { it.trim().removeSurrounding("[", "]") }
        return "[${parts.joinToString(" > ")}] "
    }

    /**
     * Reads ETC from [parentRecord]'s metrics and formats a "parent ETC: X (~N substages left)" hint.
     * Returns null when the parent has no meaningful ETC (e.g., no `substagesTotal` set, first item
     * of an iteration, or the iteration is already complete).
     */
    private fun parentEtcHint(parentRecord: StageRecord): String? {
        val elapsed = parentRecord.metrics[ElapsedTimeMetric.KEY] ?: return null
        val etc = elapsed.etc ?: return null
        val total = parentRecord.substagesTotal ?: return null
        val finished = parentRecord.metrics[SubstageCountMetric.KEY]?.finished ?: return null
        val remaining = (total - finished).coerceAtLeast(0)
        return "(parent '${parentRecord.name}' ETC: ${etc.pretty()}, ~$remaining substages left)"
    }
}

package ai.koog.agents.optimization.optimizers.mipro

import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.putEntryToActionLog
import io.github.oshai.kotlinlogging.KotlinLogging

/*
 * Logs MIPROv2's optimization: this file owns every action-log entry key a MIPROv2 stage records, and
 * the run-log message that reports the same facts in detail.
 *
 * Entries are clipped by the session's action-log truncation, so a function carrying a text artifact
 * writes it to the run log as well, where it survives in full.
 *
 * Lines that only announce what a step is about to do stay where that step runs.
 */

private val logger = KotlinLogging.logger {}

// ---------- Run setup ----------

/**
 * Records the modules MIPROv2 discovered, the budget it resolved and the size of each set it splits
 * the dataset into.
 *
 * The budget is the values the run used. [overriddenBudgetValues] names the ones set explicitly,
 * so the rest read as [autoMode]'s own; it is recorded only when something was overridden. Every
 * dataset item appears by name under the stage that runs it, so this entry keeps the sizes only.
 */
internal fun StageScope<*, *, *>.logMiproRunSetup(
    modules: List<OptimizableModule>,
    autoMode: AutoRunMode,
    numCandidates: Int,
    numTrials: Int,
    overriddenBudgetValues: List<String>,
    bootstrapSetSize: Int,
    validationSetSize: Int,
) {
    appendToActionLog {
        put("modules", modules.map { module -> module.name })
        put("autoRunMode", autoMode.name.lowercase())
        put("numCandidates", numCandidates)
        put("numTrials", numTrials)
        if (overriddenBudgetValues.isNotEmpty()) {
            put("budgetOverrides", overriddenBudgetValues)
        }
        put("bootstrapSetSize", bootstrapSetSize)
        put("validationSetSize", validationSetSize)
    }
    logger.info {
        val budgetSource = if (overriddenBudgetValues.isEmpty()) {
            "Auto mode ${autoMode.name.lowercase()} sets the budget"
        } else {
            "Auto mode ${autoMode.name.lowercase()}, with ${overriddenBudgetValues.joinToString()} " +
                    "overridden, sets the budget"
        }
        "MIPRO found ${modules.size} module(s): ${modules.map { module -> module.name }}. " +
                "$budgetSource: $numCandidates candidate(s) per module, $numTrials trial(s). " +
                "Bootstrap set: $bootstrapSetSize item(s), validation set: $validationSetSize item(s)."
    }
}

// ---------- Demo generation ----------

/**
 * Records how many demonstrations each generated demo set holds, per module. Zero-shot mode records an
 * empty map.
 *
 * A trial refers to a demo set by its position in the module's list. The demonstrations themselves are
 * too large to record: the winning set is in the saved artifact, and the bootstrap stages of this step
 * hold the agent runs every other set was built from.
 */
internal fun StageScope<*, *, *>.logMiproDemoSets(demoCandidates: Map<String, List<List<Demonstration>>>?) {
    val sizes = demoCandidates.orEmpty().mapValues { (_, sets) -> sets.map { set -> set.size } }
    putEntryToActionLog("demoSetSizes", sizes)
    logger.info {
        if (sizes.isEmpty()) {
            "MIPRO generated no demo sets; the run is zero-shot."
        } else {
            "MIPRO generated demo sets: " +
                    sizes.entries.joinToString(", ") { (module, setSizes) ->
                        "module '$module': ${setSizes.size} set(s) of $setSizes demo(s)"
                    }
        }
    }
}

// ---------- Instruction proposal ----------

/**
 * Records the instructions the meta-LLM proposed for each module, and repeats them in full in the run log.
 *
 * A trial refers to an instruction by its position in the module's list.
 */
internal fun StageScope<*, *, *>.logMiproInstructionCandidates(candidates: Map<String, List<String>>) {
    putEntryToActionLog("instructionCandidates", candidates)
    logger.info {
        "MIPRO proposed ${candidates.mapValues { (_, instructions) -> instructions.size }} instruction " +
                "candidate(s):\n" +
                candidates.entries.joinToString("\n") { (module, instructions) ->
                    instructions.withIndex().joinToString("\n") { (index, instruction) ->
                        "module '$module' candidate $index (${instruction.length} chars):\n$instruction"
                    }
                }
    }
}

// ---------- Grid search ----------

/** Records the score the unoptimized agent reached on the validation set, which a trial has to beat. */
internal fun StageScope<*, *, *>.logMiproBaselineScore(averageValidationScore: Double) {
    putEntryToActionLog("baselineAverageValidationScore", averageValidationScore)
    logger.info { "MIPRO baseline score: ${"%.4f".format(averageValidationScore)}" }
}

/** Records the candidate positions one trial drew, per module. */
internal fun StageScope<*, *, *>.logMiproTrialCombination(
    instructionIndices: Map<String, Int>,
    demoSetIndices: Map<String, Int>,
) {
    appendToActionLog {
        put("instructionIndices", instructionIndices)
        put("demoSetIndices", demoSetIndices)
    }
}

/**
 * Records one trial's score and whether it took the lead.
 *
 * @param bestAverageValidationScore The best score of the run so far, this trial included.
 */
internal fun StageScope<*, *, *>.logMiproTrialOutcome(
    trialNumber: Int,
    trialCount: Int,
    averageValidationScore: Double,
    isNewBest: Boolean,
    bestAverageValidationScore: Double,
) {
    appendToActionLog {
        put("trialAverageValidationScore", averageValidationScore)
        put("isNewBest", isNewBest)
        put("bestAverageValidationScore", bestAverageValidationScore)
    }
    val messagePrefix = "MIPRO trial $trialNumber/$trialCount: score=${"%.4f".format(averageValidationScore)}"
    logger.info {
        if (isNewBest) {
            "$messagePrefix *** new best ***"
        } else {
            "$messagePrefix, best=${"%.4f".format(bestAverageValidationScore)}"
        }
    }
}

/**
 * What a finished grid search kept.
 *
 * @property label The value recorded as the grid search's `outcome`.
 */
internal enum class MIPROGridSearchOutcome(val label: String) {
    /** No trial reached the baseline's score, so the run keeps the unoptimized agent. */
    BASELINE_KEPT("baseline kept"),

    /** A trial reached or beat the baseline, and its combination is what the run saves. */
    TRIAL_KEPT("trial kept"),
}

/**
 * Records what the grid search kept and the score that ranks it.
 *
 * @param bestTrialNumber The one-based number of the trial that last took the lead, or null when no
 *   trial reached the baseline. A null records the [MIPROGridSearchOutcome.BASELINE_KEPT] outcome and
 *   keeps both combinations out of the entry, since the run then has no combination to name.
 */
internal fun StageScope<*, *, *>.logMiproGridSearchResult(
    bestAverageValidationScore: Double,
    bestTrialNumber: Int?,
    bestInstructionIndices: Map<String, Int>,
    bestDemoSetIndices: Map<String, Int>,
) {
    val outcome = if (bestTrialNumber == null) {
        MIPROGridSearchOutcome.BASELINE_KEPT
    } else {
        MIPROGridSearchOutcome.TRIAL_KEPT
    }
    appendToActionLog {
        put("outcome", outcome.label)
        put("bestAverageValidationScore", bestAverageValidationScore)
        bestTrialNumber?.let { trialNumber ->
            put("bestTrialNumber", trialNumber)
            put("bestInstructionIndices", bestInstructionIndices)
            put("bestDemoSetIndices", bestDemoSetIndices)
        }
    }
    logger.info {
        val score = "%.4f".format(bestAverageValidationScore)
        if (bestTrialNumber == null) {
            "MIPRO grid search kept the baseline: no trial reached its score of $score."
        } else {
            "MIPRO grid search kept trial $bestTrialNumber with an average validation score of $score."
        }
    }
}

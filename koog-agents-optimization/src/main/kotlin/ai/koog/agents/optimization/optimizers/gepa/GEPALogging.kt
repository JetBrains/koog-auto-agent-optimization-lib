package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.putEntryToActionLog
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.common.toFilePathLog
import io.github.oshai.kotlinlogging.KotlinLogging

/*
 * Logs GEPA's optimization: this file owns every action-log entry key a GEPA stage records, and the
 * run-log message that reports the same facts in detail.
 *
 * Entries are clipped by the session's action-log truncation, so a function carrying a text artifact
 * writes it to the run log as well, where it survives in full.
 *
 * Lines that explain a branch the algorithm takes stay where that branch is.
 */

private val logger = KotlinLogging.logger {}

// ---------- Run setup ----------

/**
 * Records the modules GEPA discovered and the size of each set of the split it trains on.
 *
 * Set membership stays with the split function GEPA was given, and each item appears by name under the
 * rollout stage that runs it, so this entry keeps the sizes only.
 */
internal fun StageScope<*, *, *>.logGepaRunSetup(
    modules: List<OptimizableModule>,
    feedbackSetSize: Int,
    validationSetSize: Int,
) {
    appendToActionLog {
        put("modules", modules.map { module -> module.name })
        put("feedbackSetSize", feedbackSetSize)
        put("validationSetSize", validationSetSize)
    }
}

// ---------- Candidate pool seed ----------

/**
 * Records the seed candidate's id, instructions and validation score, and repeats the instructions in
 * full in the run log.
 */
internal fun StageScope<*, *, *>.logGepaSeedCandidate(
    seedCandidateId: String,
    seedArtifact: OptimizationArtifact,
    averageValidationScore: Double,
) {
    appendToActionLog {
        put("seedCandidateId", seedCandidateId)
        put("instructions", seedArtifact.modulePrompts().toMap())
        put("averageValidationScore", averageValidationScore)
    }
    logger.info {
        "GEPA seed candidate $seedCandidateId averaged $averageValidationScore on the validation set. " +
                "Its instructions:\n" +
                seedArtifact.modulePrompts().joinToString("\n") { (module, instruction) ->
                    "module '$module' (${instruction.length} chars):\n$instruction"
                }
    }
}

// ---------- Rollouts ----------

/** Records the scores one candidate reached over a rollout set, one entry per item in dataset order. */
internal fun StageScope<*, *, *>.logGepaRolloutScores(scores: List<Double>) {
    putEntryToActionLog("rolloutScores", scores)
}

/**
 * Records the score GEPA put in the rollout vector for one dataset item.
 *
 * A rollout that completed carries its score on its own agent-run record too; a rollout that failed
 * terminally carries none, and this entry states the failure score substituted for it.
 */
internal fun StageScope<*, *, *>.logGepaAssignedScore(score: Double) {
    putEntryToActionLog("assignedScore", score)
}

// ---------- Feedback batches ----------

/** Records the parent a feedback batch drew, and the frontier weights it was drawn from. */
internal fun StageScope<*, *, *>.logGepaParentSelection(
    parentId: String,
    frontierWeights: Map<String, Int>,
) {
    appendToActionLog {
        put("parentId", parentId)
        put("parentSelectionWeights", frontierWeights)
    }
}

/** Records the parent's average score over the feedback batch, which a candidate has to beat. */
internal fun StageScope<*, *, *>.logGepaParentBatchScore(averageScore: Double) {
    putEntryToActionLog("parentBatchAverageScore", averageScore)
}

/** Records which module a feedback batch updates; a null [selectedModule] means every discovered module. */
internal fun StageScope<*, *, *>.logGepaSelectedModule(selectedModule: OptimizableModule?) {
    putEntryToActionLog("selectedModule", selectedModule?.name ?: ALL_MODULES_LABEL)
}

private const val ALL_MODULES_LABEL = "all modules"

/** Records the instruction reflection proposed for one module together with the inputs it read. */
internal fun StageScope<*, *, *>.logGepaProposedInstruction(
    moduleName: String,
    parentInstruction: String,
    feedbackTexts: List<String>,
    proposedInstruction: String,
) {
    appendToActionLog {
        put("module", moduleName)
        put("parentInstruction", parentInstruction)
        put("feedbackTexts", feedbackTexts)
        put("proposedInstruction", proposedInstruction)
    }
    logger.info {
        "GEPA reflection proposed an instruction for module '$moduleName' " +
                "(${proposedInstruction.length} chars):\n$proposedInstruction"
    }
}

/** Records the proposed candidate's average score over the feedback batch. */
internal fun StageScope<*, *, *>.logGepaCandidateBatchScore(averageScore: Double) {
    putEntryToActionLog("candidateBatchAverageScore", averageScore)
}

/** Records a candidate that entered the pool, with the score that ranks it there. */
internal fun StageScope<*, *, *>.logGepaAcceptedCandidate(
    candidateId: String,
    averageValidationScore: Double,
) {
    appendToActionLog {
        put("newCandidateId", candidateId)
        put("averageValidationScore", averageValidationScore)
    }
    logger.info { "GEPA accepted $candidateId with an average validation score of $averageValidationScore." }
}

// ---------- Merge attempts ----------

/** Records the pair a merge attempt recombined, and the shared ancestor it recombined them over. */
internal fun StageScope<*, *, *>.logGepaMergeProposal(
    parentIds: List<String>,
    ancestorId: String,
) {
    appendToActionLog {
        put("parentIds", parentIds)
        put("ancestorId", ancestorId)
    }
}

/** Records the validation subsample a merge acceptance was decided on, and both sides of that decision. */
internal fun StageScope<*, *, *>.logGepaMergeSubsample(
    validationSubsampleIndices: List<Int>,
    subsampleScore: Double,
    bestParentSubsampleScore: Double,
) {
    appendToActionLog {
        put("validationSubsampleIndices", validationSubsampleIndices)
        put("subsampleScore", subsampleScore)
        put("bestParentSubsampleScore", bestParentSubsampleScore)
    }
}

// ---------- Step outcome ----------

/**
 * How one GEPA step ended, where a step is one feedback batch or one merge attempt.
 *
 * @property label The value recorded as the step stage's `outcome`.
 */
internal enum class GEPAStepOutcome(val label: String) {
    PARENT_ROLLOUTS_FAILED("parent rollouts failed"),
    PARENT_ALREADY_PERFECT("parent already perfect"),
    NO_MODULE_FEEDBACK("no module feedback"),
    CANDIDATE_ALREADY_IN_POOL("candidate already in pool"),

    NO_MERGE_PROPOSAL("no merge proposal"),

    CANDIDATE_REJECTED("candidate rejected"),
    CANDIDATE_ACCEPTED("candidate accepted"),
}

/** Records how a feedback batch or a merge attempt ended. */
internal fun StageScope<*, *, *>.logGepaStepOutcome(outcome: GEPAStepOutcome) {
    putEntryToActionLog("outcome", outcome.label)
}

// ---------- Result ----------

/** Records the candidate GEPA kept, with the pool it was chosen from. */
internal fun StageScope<*, *, *>.logGepaBestCandidate(
    bestCandidateId: String,
    bestAverageValidationScore: Double,
    candidates: List<GEPACandidateSummary>,
    storagePath: ResilientPath,
) {
    appendToActionLog {
        put("bestCandidateId", bestCandidateId)
        put("bestAverageValidationScore", bestAverageValidationScore)
        put("candidates", candidates)
    }
    logger.info {
        "GEPA chose $bestCandidateId with an average validation score of $bestAverageValidationScore. " +
                "Artifact saved to ${storagePath.toFilePathLog()}."
    }
}

/**
 * Records the counters of a finished run: what it spent, how many steps it took, and what it kept.
 *
 * Only counters land here. Each candidate, batch, merge attempt and rollout carries its own record on
 * the stage that produced it.
 *
 * @param mergeAttempts Pairs with [batchesRun] to account for every iteration of the optimization loop,
 *   since a merge attempt spends rollouts of its own. Null when merging is disabled, which keeps the
 *   entry out of the record.
 */
internal fun StageScope<*, *, *>.logGepaRunTotals(
    rolloutsUsed: Int,
    batchesRun: Int,
    mergeAttempts: Int?,
    candidatesAdded: Int,
    bestCandidateId: String,
) {
    appendToActionLog {
        put("rolloutsUsed", rolloutsUsed)
        put("batchesRun", batchesRun)
        mergeAttempts?.let { attempts -> put("mergeAttempts", attempts) }
        put("candidatesAdded", candidatesAdded)
        put("bestCandidateId", bestCandidateId)
    }
}

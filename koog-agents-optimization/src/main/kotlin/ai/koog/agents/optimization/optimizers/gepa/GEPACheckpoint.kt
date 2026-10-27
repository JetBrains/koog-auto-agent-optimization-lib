package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.utils.common.ResilientPath
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.security.MessageDigest

private val logger = KotlinLogging.logger {}

/**
 * GEPA's search state after a completed step, enough to continue the search in a new run.
 *
 * Saving: GEPA overwrites one checkpoint file after the seed validation, after every feedback batch and after every
 * merge attempt. The file holds the candidate pool (artifacts, parents, validation scores, module cursors), the spent
 * budget, the step counters and the merge state.
 *
 * Resuming: a new run reads the file, checks it was saved for the same [GEPARunSetup], rebuilds the pool and the
 * merge state from it, deducts the spent budget, and continues the main loop with the next step numbers. The seed
 * validation is not repeated, since its scores are part of the saved pool.
 *
 * The random generator isn't saved; see [GEPAOptimizer] for how a resumed run seeds it.
 */
@Serializable
internal data class GEPACheckpoint(
    val formatVersion: Int,
    val runSetup: GEPARunSetup,
    val candidates: List<GEPACheckpointCandidate>,
    val rolloutsSpent: Int,
    val batchesRun: Int,
    val mergeAttempts: Int,
    val merge: GEPAMergeState? = null,
) {
    /** Checks that a run with [runSetup] can resume from this checkpoint; fails naming what differs. */
    fun checkMatches(runSetup: GEPARunSetup) {
        check(formatVersion == CURRENT_FORMAT_VERSION) {
            "GEPA checkpoint has format version $formatVersion; this GEPA reads version $CURRENT_FORMAT_VERSION"
        }
        check(this.runSetup.modules == runSetup.modules) {
            "GEPA checkpoint was written for modules ${this.runSetup.modules}, not ${runSetup.modules}"
        }
        check(this.runSetup.validationItemHashes == runSetup.validationItemHashes) {
            "GEPA checkpoint was written for a different validation set"
        }
        check(this.runSetup.feedbackItemHashes == runSetup.feedbackItemHashes) {
            "GEPA checkpoint was written for a different feedback set"
        }
        check(candidates.isNotEmpty()) { "GEPA checkpoint holds no candidates" }
    }

    companion object {
        const val CURRENT_FORMAT_VERSION: Int = 1
    }
}

/**
 * For what setting a checkpoint was saved: the names of the optimized modules, and the hash of each validation
 * and feedback item in order ([itemHashesOf]; the checkpoint doesn't store the items). A run resumes only
 * from a checkpoint saved for an equal setup.
 */
@Serializable
internal data class GEPARunSetup(
    val modules: List<String>,
    val validationItemHashes: List<String>,
    val feedbackItemHashes: List<String>,
) {
    companion object {
        fun <Input, InputLabel> of(
            modules: List<OptimizableModule>,
            validationSet: TrainSet<Input, InputLabel>,
            feedbackSet: TrainSet<Input, InputLabel>,
        ): GEPARunSetup = GEPARunSetup(
            modules = modules.map { module -> module.name },
            validationItemHashes = itemHashesOf(validationSet),
            feedbackItemHashes = itemHashesOf(feedbackSet),
        )
    }
}

/** One candidate pool entry, with its parents named by id and its scores in validation-set order. */
@Serializable
internal data class GEPACheckpointCandidate(
    val id: String,
    val artifact: OptimizationArtifact,
    val parentIds: List<String>,
    val validationScores: List<Double>,
    val nextModuleIndex: Int,
)

/** GEPA merge's own counters and the parent/ancestor triplets it has already tried, by candidate id. */
@Serializable
internal data class GEPAMergeState(
    val mergesDue: Int,
    val acceptedMergeCount: Int,
    val lastIterFoundNewPrompt: Boolean,
    val triedTriplets: List<GEPAMergeTripletIds>,
)

/**
 * A tried merge triplet by candidate id: the two merged candidates and their shared ancestor. The merged pair is
 * unordered, so [firstCandidateId] is the smaller of the two ids.
 */
@Serializable
internal data class GEPAMergeTripletIds(
    val firstCandidateId: String,
    val secondCandidateId: String,
    val ancestorId: String,
)

/**
 * The SHA-256 hash of each item of [items], in order, over the item's query and label rendered with `toString()`.
 * Data classes render their content, so the same item gets the same hash in every process; a type without a
 * content-based `toString` gets a new one each time, which makes a resume refuse the checkpoint.
 */
internal fun <Input, InputLabel> itemHashesOf(items: TrainSet<Input, InputLabel>): List<String> =
    items.map { item ->
        MessageDigest.getInstance("SHA-256")
            .digest("${item.userQuery}\u0000${item.itemLabel}".toByteArray())
            .joinToString("") { byte -> "%02x".format(byte) }
    }

private val checkpointJson = Json { ignoreUnknownKeys = true }

/** Reads the checkpoint at [path]; fails when the file is missing or unreadable. */
internal fun readGepaCheckpoint(path: ResilientPath): GEPACheckpoint {
    require(path.exists()) { "Cannot resume GEPA from $path: file does not exist" }
    return checkpointJson.decodeFromString(GEPACheckpoint.serializer(), path.readText())
}

/** Reads the checkpoint at [path] for a run with [runSetup] to resume from; fails when it doesn't match. */
internal fun readGepaCheckpointToResume(path: ResilientPath, runSetup: GEPARunSetup): GEPACheckpoint {
    val checkpoint = readGepaCheckpoint(path)
    checkpoint.checkMatches(runSetup)
    return checkpoint
}

/**
 * Writes [checkpoint] to [path]. A failed write is logged and otherwise ignored: losing a checkpoint costs only the
 * ability to resume from this step, while failing would stop the run it protects.
 */
internal fun writeGepaCheckpoint(path: ResilientPath, checkpoint: GEPACheckpoint) {
    try {
        path.createParentDirectories()
        path.writeText(checkpointJson.encodeToString(GEPACheckpoint.serializer(), checkpoint))
    } catch (exception: Exception) {
        logger.warn(exception) { "Could not write the GEPA checkpoint to $path; the run continues without it." }
    }
}

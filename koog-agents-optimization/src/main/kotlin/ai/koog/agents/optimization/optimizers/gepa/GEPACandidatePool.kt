package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizationArtifact
import kotlinx.serialization.Serializable
import kotlin.math.ceil
import kotlin.math.max
import kotlin.random.Random

/** A loggable summary of one candidate pool entry. */
@Serializable
internal data class GEPACandidateSummary(
    val id: String,
    val averageValidationScore: Double,
    val parentIds: List<String>,
)

/** A parent draw together with the frontier weights it was sampled from, keyed by candidate id. */
internal data class GEPAParentSelection(
    val parent: OptimizationArtifact,
    val frontierWeights: Map<String, Int>,
)

/**
 * Stores GEPA candidates with their validation scores, ancestry, and round-robin module cursor.
 */
internal class GEPACandidatePool(
    val evaluationDatasetSize: Int,
    private val random: Random,
) {
    private val candidateScores = mutableMapOf<OptimizationArtifact, List<Double>>()
    private val candidateParents = mutableMapOf<OptimizationArtifact, List<OptimizationArtifact>>()
    private val candidateNextModuleIndex = mutableMapOf<OptimizationArtifact, Int>()
    private val candidateIds = mutableMapOf<OptimizationArtifact, String>()

    val size: Int
        get() = candidateScores.size

    operator fun contains(candidate: OptimizationArtifact): Boolean =
        candidate in candidateScores

    fun candidates(): Set<OptimizationArtifact> =
        candidateScores.keys.toSet()

    fun scoresOf(candidate: OptimizationArtifact): List<Double> =
        candidateScores.getValue(candidate)

    fun averageScoreOf(candidate: OptimizationArtifact): Double =
        scoresOf(candidate).average()

    fun add(cand: OptimizationArtifact, parents: List<OptimizationArtifact>, scores: List<Double>) {
        require(cand !in candidateScores) { "Candidate already exists" }
        require(parents.all { it in candidateScores }) { "All candidate parents must already exist in the pool" }
        require(scores.size == evaluationDatasetSize) {
            "Expected $evaluationDatasetSize scores, got ${scores.size}"
        }
        candidateScores[cand] = scores
        candidateParents[cand] = parents
        candidateNextModuleIndex[cand] = 0
        candidateIds[cand] = "candidate-${candidateIds.size}"
    }

    /** Returns the id [candidate] was given when it entered the pool. */
    fun idOf(candidate: OptimizationArtifact): String = candidateIds.getValue(candidate)

    /** Returns every entry with its average validation score and its parents, in insertion order. */
    fun summarizeCandidates(): List<GEPACandidateSummary> =
        candidateScores.keys.map { candidate ->
            GEPACandidateSummary(
                id = idOf(candidate),
                averageValidationScore = averageScoreOf(candidate),
                parentIds = candidateParents.getValue(candidate).map { parent -> idOf(parent) },
            )
        }

    internal fun advanceNextModuleIndex(candidate: OptimizationArtifact, moduleCount: Int): Int {
        check(candidate in candidateNextModuleIndex) { "Candidate not found" }
        check(moduleCount > 0) { "Cannot select module from an empty module list" }

        val moduleIndex = candidateNextModuleIndex.getValue(candidate)
        candidateNextModuleIndex[candidate] = (moduleIndex + 1) % moduleCount
        return moduleIndex
    }

    fun findFrontierRepresentatives(): Pair<Set<OptimizationArtifact>, List<Set<OptimizationArtifact>>> {
        val bestScores: List<Double> =
            (0 until evaluationDatasetSize).map { i ->
                candidateScores.values.maxOf { scores ->
                    scores[i]
                }
            }

        val candidatesByDatasetItem: List<Set<OptimizationArtifact>> =
            (0 until evaluationDatasetSize).map { i ->
                candidateScores.entries
                    .filter { (_, scores) ->
                        scores[i] == bestScores[i]
                    }
                    .map { (params, _) -> params }
                    .toSet()
            }
        val paretoCandidates: Set<OptimizationArtifact> =
            candidatesByDatasetItem.flatten().toSet()

        val redundantCandidates = findRedundantFrontierCandidates(
            paretoCandidates = paretoCandidates,
            candidatesByDatasetItem = candidatesByDatasetItem,
        )

        val frontierRepresentatives = paretoCandidates - redundantCandidates
        val representativesByDatasetItem = (0 until evaluationDatasetSize).map { i ->
            candidatesByDatasetItem[i] - redundantCandidates
        }
        return Pair(frontierRepresentatives, representativesByDatasetItem)
    }

    private fun findRedundantFrontierCandidates(
        paretoCandidates: Set<OptimizationArtifact>,
        candidatesByDatasetItem: List<Set<OptimizationArtifact>>,
    ): Set<OptimizationArtifact> {
        val candidatesByAscendingScore = paretoCandidates.sortedBy { candidate -> averageScoreOf(candidate) }
        val allCandidates = candidatesByAscendingScore.toSet()
        val redundant = mutableSetOf<OptimizationArtifact>()

        fun isRedundant(candidate: OptimizationArtifact): Boolean {
            val remainingCandidates = allCandidates - candidate - redundant
            return candidatesByDatasetItem
                .filter { candidate in it }
                .all { front -> front.any { other -> other in remainingCandidates } }
        }

        while (true) {
            val candidateToRemove = candidatesByAscendingScore.firstOrNull { candidate ->
                candidate !in redundant && isRedundant(candidate)
            } ?: break
            redundant.add(candidateToRemove)
        }

        val frontierRepresentatives = allCandidates - redundant
        for (front in candidatesByDatasetItem) {
            check(front.isEmpty() || front.any { candidate -> candidate in frontierRepresentatives }) {
                "Redundant-candidate pruning removed an entire Pareto front"
            }
        }
        return redundant
    }

    /**
     * Samples from the Pareto front, weighted by how many validation items each candidate is best on.
     */
    fun selectParent(): GEPAParentSelection {
        check(candidateScores.isNotEmpty()) { "Cannot select parent from an empty candidate pool" }

        val (frontierRepresentatives, representativesByDatasetItem) = findFrontierRepresentatives()

        val selectionWeights: Map<OptimizationArtifact, Int> =
            frontierRepresentatives.associateWith { candidate ->
                representativesByDatasetItem.count { candidate in it }
            }

        return GEPAParentSelection(
            parent = selectionWeights.weightedRandom(),
            frontierWeights = selectionWeights.mapKeys { (candidate, _) -> idOf(candidate) },
        )
    }

    private fun <T> Map<T, Int>.weightedRandom(): T {
        check(isNotEmpty()) { "Cannot sample from empty map" }

        val totalWeight = values.sum()
        check(totalWeight > 0) { "Total weight must be positive" }

        var r = random.nextInt(totalWeight)

        for ((item, weight) in this) {
            r -= weight
            if (r < 0) return item
        }

        error("Unreachable")
    }

    fun getBestCandidate(): Pair<OptimizationArtifact, Double> {
        check(candidateScores.isNotEmpty()) { "Cannot get best candidate from an empty candidate pool" }

        val (params, scores) = candidateScores.maxBy { (_, scores) -> scores.average() }
        return params to scores.average()
    }

    internal fun ancestorsOf(candidate: OptimizationArtifact): Set<OptimizationArtifact> {
        check(candidate in candidateParents) { "Candidate not found" }
        val ancestors = mutableSetOf<OptimizationArtifact>()
        val todo = ArrayDeque(candidateParents.getValue(candidate))
        while (todo.isNotEmpty()) {
            val next = todo.removeFirst()
            if (next !in ancestors) {
                ancestors.add(next)
                todo.addAll(candidateParents.getValue(next))
            }
        }
        return ancestors
    }

    fun selectMergeValidationSubsample(
        parent: OptimizationArtifact,
        otherParent: OptimizationArtifact,
        valSetSubsampleSize: Int,
    ): List<Int> {
        val scores = candidateScores.getValue(parent)
        val otherScores = candidateScores.getValue(otherParent)
        val indices = scores.indices.toList()

        val parentBetterIndices = indices.filter { index -> scores[index] > otherScores[index] }
        val otherParentBetterIndices = indices.filter { index -> otherScores[index] > scores[index] }
        val nonNeutralIndices = (parentBetterIndices + otherParentBetterIndices).toSet()
        val neutralIndices = indices.filter { index -> index !in nonNeutralIndices }
        val samplesPerBucket = max(1, ceil(valSetSubsampleSize.toDouble() / 3).toInt())

        val selected = mutableListOf<Int>()
        val selectedSet = mutableSetOf<Int>()

        for (bucket in listOf(parentBetterIndices, otherParentBetterIndices, neutralIndices)) {
            val remainingSampleSize = valSetSubsampleSize - selected.size
            if (remainingSampleSize <= 0) break

            val available = bucket.filter { index -> index !in selectedSet }
            val sampled = available.shuffled(random).take(minOf(available.size, samplesPerBucket, remainingSampleSize))
            selected.addAll(sampled)
            selectedSet.addAll(sampled)
        }

        val remainingSampleSize = valSetSubsampleSize - selected.size
        if (remainingSampleSize > 0) {
            val unusedIndices = indices.filter { index -> index !in selectedSet }
            val sampledUnusedIndices = unusedIndices.shuffled(random).take(remainingSampleSize)
            selected.addAll(sampledUnusedIndices)

            val stillRemainingSampleSize = valSetSubsampleSize - selected.size
            if (stillRemainingSampleSize > 0) {
                // The reference implementation switches to replacement here without first exhausting unused items.
                selected.addAll(List(stillRemainingSampleSize) { indices.random(random) })
            }
        }

        require(selected.size == valSetSubsampleSize)
        return selected
    }

    fun sumValidationScores(candidate: OptimizationArtifact, idxs: List<Int>): Double {
        return candidateScores.getValue(candidate).withIndex().filter { idxs.contains(it.index) }.sumOf { it.value }
    }
}

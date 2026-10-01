package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizationArtifact
import kotlin.random.Random

/**
 * Runtime controls for GEPA's optional merge step.
 *
 * _Merging is experimental:_ tests reach its proposal step and no further, it has never run end to end,
 * and it may diverge from GEPA's original implementation more significantly.
 *
 * A merge combines the module one lineage changed with the module the other changed, so it needs an
 * agent with at least two optimizable modules. Enabling it for a single-module agent does nothing.
 *
 * @property enabled Whether the optimizer should attempt candidate merges.
 * @property maxMergeInvocations Maximum number of accepted merges during one optimization run.
 * @property maxMergeAttempts Maximum number of merge proposals to try for one due merge.
 * @property maxFindSharedAncestorAttempts Maximum number of candidate-pair samples used while searching
 *   for parents with a shared ancestor.
 * @property valSetSubsampleSize Number of validation examples used to decide whether to accept a merge.
 */
public data class GEPAMergeConfig(
    val enabled: Boolean = false,
    val maxMergeInvocations: Int = 5,
    val maxMergeAttempts: Int = 10,
    val maxFindSharedAncestorAttempts: Int = 10,
    val valSetSubsampleSize: Int = 5
) {
    init {
        require(maxMergeInvocations >= 0) { "maxMergeInvocations must be non-negative" }
        require(maxMergeAttempts > 0) { "maxMergeAttempts must be positive" }
        require(maxFindSharedAncestorAttempts > 0) { "maxFindSharedAncestorAttempts must be positive" }
        require(valSetSubsampleSize > 0) { "valSetSubsampleSize must be positive" }
    }
}

// TODO: merge is barely covered. GEPAMergeTest reaches propose() only, so the optimizer's merge stage
//  has never run: nothing exercises the validation subsample, the acceptance comparison or
//  notifyAcceptedMerge, and no scheduling path is covered. Before relying on merge, cover those paths
//  and settle the points where this differs from the reference implementation: a merged candidate
//  already in the pool is rejected here, where the reference dedups merged texts across triplets; and
//  shouldAttemptMerge stops once maxMergeInvocations merges are accepted, where the reference stops
//  scheduling new ones and still runs those already due.
internal class GEPAMerge(
    private val candidatePool: GEPACandidatePool,
    private val config: GEPAMergeConfig = GEPAMergeConfig(),
    private val random: Random = Random,
) {
    private var mergesDue = 0
    private var acceptedMergeCount = 0
    private var lastIterFoundNewPrompt = false

    init {
        check(config.enabled)
    }

    data class MergeProposal(
        val candidate: OptimizationArtifact,
        val parents: List<OptimizationArtifact>,
        val ancestor: OptimizationArtifact,
    )

    private data class MergeTriplet(
        val cand: OptimizationArtifact,
        val other: OptimizationArtifact,
        val ancestor: OptimizationArtifact,
    )

    private data class MergeTripletKey(
        val candidates: Set<OptimizationArtifact>,
        val ancestor: OptimizationArtifact,
    ) {
        init {
            check(candidates.size == 2) { "MergeTripletKey requires two distinct candidates" }
        }
    }

    private val triedMergeTriplets = mutableSetOf<MergeTripletKey>()

    fun shouldAttemptMerge(): Boolean {
        return config.enabled && acceptedMergeCount < config.maxMergeInvocations && mergesDue > 0 && lastIterFoundNewPrompt
    }

    fun notifyAcceptedMerge() {
        acceptedMergeCount++
        mergesDue -= 1
    }

    fun notifySuccessfulReflection() {
        if (acceptedMergeCount < config.maxMergeInvocations) {
            mergesDue++
        }
        lastIterFoundNewPrompt = true
    }

    fun notifyMergeAttemptFinished() {
        lastIterFoundNewPrompt = false
    }

    fun propose(): MergeProposal? {
        val (frontierRepresentatives, _) = candidatePool.findFrontierRepresentatives()
        if (frontierRepresentatives.size < 2) {
            return null
        }

        repeat(config.maxMergeAttempts) {
            val triplet = findSharedAncestors(
                frontierRepresentatives,
                maxAttempts = config.maxFindSharedAncestorAttempts,
            ) ?: return@repeat

            val (cand, other, ancestor) = triplet
            triedMergeTriplets.add(MergeTripletKey(setOf(cand, other), ancestor))

            val merged = mergeCandidates(ancestor, cand, other)
            if (merged in candidatePool) {
                return@repeat
            }
            return MergeProposal(
                candidate = merged,
                parents = listOf(cand, other),
                ancestor = ancestor,
            )
        }
        return null
    }

    private fun isDesirable(
        ancestor: OptimizationArtifact,
        cand: OptimizationArtifact,
        other: OptimizationArtifact,
    ): Boolean {
        for ((moduleName, ancestorPrompt) in ancestor.modulePrompts()) {
            val candPrompt = checkNotNull(cand.getModulePrompt(moduleName)) {
                "Candidate is missing prompt for module '$moduleName'"
            }
            val otherPrompt = checkNotNull(other.getModulePrompt(moduleName)) {
                "Other candidate is missing prompt for module '$moduleName'"
            }
            if ((ancestorPrompt == candPrompt && candPrompt != otherPrompt) ||
                (ancestorPrompt == otherPrompt && candPrompt != otherPrompt)
            ) {
                return true
            }
        }
        return false
    }

    private fun findSharedAncestors(
        frontierRepresentatives: Set<OptimizationArtifact>,
        maxAttempts: Int,
    ): MergeTriplet? {
        repeat(maxAttempts) {
            check(frontierRepresentatives.size >= 2)

            val (cand, other) = frontierRepresentatives.shuffled(random).take(2)

            val candAncestors = candidatePool.ancestorsOf(cand)
            val otherAncestors = candidatePool.ancestorsOf(other)
            if (candAncestors.contains(other) || otherAncestors.contains(cand)) {
                return@repeat
            }

            val commonAncestors = candAncestors
                .intersect(otherAncestors)
                .filter { ancestor ->
                    MergeTripletKey(setOf(cand, other), ancestor) !in triedMergeTriplets &&
                            candidatePool.averageScoreOf(ancestor) <=
                            minOf(
                                candidatePool.averageScoreOf(cand),
                                candidatePool.averageScoreOf(other),
                            ) &&
                            isDesirable(ancestor, cand, other)
                }

            val ancestorWeights = commonAncestors.associateWith { candidatePool.averageScoreOf(it) }

            if (commonAncestors.isNotEmpty()) {
                return MergeTriplet(
                    cand = cand,
                    other = other,
                    ancestor = ancestorWeights.weightedRandom(),
                )
            }
        }
        return null
    }

    private fun <T> Map<T, Double>.weightedRandom(): T {
        check(isNotEmpty()) { "Cannot sample from empty map" }

        val totalWeight = values.sum()
        check(totalWeight > 0) { "Total weight must be positive" }

        var r = random.nextDouble(totalWeight)

        for ((item, weight) in this) {
            r -= weight
            if (r < 0) return item
        }

        error("Unreachable")
    }

    private fun mergeCandidates(
        ancestor: OptimizationArtifact,
        cand: OptimizationArtifact,
        other: OptimizationArtifact,
    ): OptimizationArtifact {
        var merged = ancestor.copy()
        val candScore = candidatePool.averageScoreOf(cand)
        val otherScore = candidatePool.averageScoreOf(other)

        for ((moduleName, ancestorPrompt) in ancestor.modulePrompts()) {
            val candPrompt = checkNotNull(cand.getModulePrompt(moduleName)) {
                "Candidate is missing prompt for module '$moduleName'"
            }
            val otherPrompt = checkNotNull(other.getModulePrompt(moduleName)) {
                "Other candidate is missing prompt for module '$moduleName'"
            }

            val mergedPrompt = when {
                candPrompt == otherPrompt -> candPrompt
                ancestorPrompt == candPrompt -> otherPrompt
                ancestorPrompt == otherPrompt -> candPrompt
                candScore > otherScore -> candPrompt
                otherScore > candScore -> otherPrompt
                else -> listOf(candPrompt, otherPrompt).random(random)
            }
            merged = merged.withModulePrompt(moduleName, mergedPrompt)
        }

        return merged
    }
}

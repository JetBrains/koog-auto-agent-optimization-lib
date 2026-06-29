package ai.koog.agents.optimization.optimizers.gepa


import ai.koog.agents.optimization.core.OptimizationArtifact
import kotlin.random.Random

/**
 * A candidate in the Pareto frontier: an [OptimizationArtifact] with per-instance validation scores.
 *
 * @property artifact The optimization artifact (instructions only for GEPA).
 * @property perInstanceScores Map from the dataset item index to score on that item.
 * @property parentIndex Index of the parent candidate in the frontier (null for the seed).
 */
public data class ParetoCandidate(
    val artifact: OptimizationArtifact,
    val perInstanceScores: Map<Int, Double> = emptyMap(),
    val parentIndex: Int? = null,
) {
    /** Mean of [perInstanceScores], or `0.0` when no scores are recorded. */
    val aggregateScore: Double
        get() = if (perInstanceScores.isEmpty()) 0.0
        else perInstanceScores.values.sum() / perInstanceScores.size
}

/**
 * Candidate pool with Pareto-inspired selection for GEPA's evolutionary search.
 *
 * All candidates are kept (never pruned). The Pareto property is enforced implicitly
 * through [selectCandidate]: selection is weighted by how many validation instances
 * each candidate is the best performer on ("coverage"). Dominated candidates (leading
 * on zero instances) have zero selection probability and are effectively ignored.
 *
 * This avoids explicit pruning while still biasing selection toward the Pareto frontier.
 * [getBestCandidate] returns the candidate with the highest aggregate score regardless
 * of coverage.
 */
public class ParetoFrontier(
    private val random: Random,
) {
    private val candidates = mutableListOf<ParetoCandidate>()

    /** Per-instance best score and the candidate index that achieved it. */
    private val bestPerInstance = mutableMapOf<Int, Pair<Int, Double>>() // instanceIdx -> (candidateIdx, score)

    /**
     * Adds [candidate] to the pool, updating per-instance best-score bookkeeping.
     *
     * @return the index assigned to the added candidate.
     */
    public fun addCandidate(candidate: ParetoCandidate): Int {
        val idx = candidates.size
        candidates.add(candidate)
        for ((instanceIdx, score) in candidate.perInstanceScores) {
            val current = bestPerInstance[instanceIdx]
            if (current == null || score > current.second) {
                bestPerInstance[instanceIdx] = idx to score
            }
        }
        return idx
    }

    /**
     * Selects a candidate probabilistically from the Pareto frontier.
     *
     * Weight is proportional to the number of instances where the candidate is the best performer.
     * This encourages selecting underexplored candidates that cover unique instance niches.
     * Falls back to uniform random if no instance-level data exists.
     */
    public fun selectCandidate(): ParetoCandidate {
        require(candidates.isNotEmpty()) { "Cannot select from empty frontier" }
        if (candidates.size == 1) return candidates.first()

        // Count how many instances each candidate leads on
        val coverage = IntArray(candidates.size)
        for ((_, bestInfo) in bestPerInstance) {
            coverage[bestInfo.first]++
        }

        // If no coverage data yet, uniform random
        val totalCoverage = coverage.sum()
        if (totalCoverage == 0) return candidates[random.nextInt(candidates.size)]

        // Weighted selection
        val target = random.nextInt(totalCoverage)
        var cumulative = 0
        for (i in coverage.indices) {
            cumulative += coverage[i]
            if (target < cumulative) return candidates[i]
        }
        return candidates.last()
    }

    /**
     * Returns the candidate with the highest [ParetoCandidate.aggregateScore], ignoring coverage.
     *
     * @throws IllegalArgumentException if the frontier is empty.
     */
    public fun getBestCandidate(): ParetoCandidate {
        require(candidates.isNotEmpty()) { "No candidates in frontier" }
        return candidates.maxBy { it.aggregateScore }
    }

    /** Number of candidates currently held in the pool. */
    public fun size(): Int = candidates.size

    /** Snapshot of all candidates in insertion order. */
    public fun getAllCandidates(): List<ParetoCandidate> = candidates.toList()
}

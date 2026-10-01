package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizationArtifact
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * Pins which pools `GEPAMerge.propose` can draw a merge from.
 *
 * A merge combines the module one lineage changed with the module the other changed, so a single-module
 * agent yields none. The second case is the control that keeps the first from passing for another reason.
 */
// TODO: these two cases cover propose() only, which is the desirability filter, the ancestor search and
//  the per-module merge rules. Untested: the optimizer's merge stage, meaning the validation subsample,
//  the acceptance comparison and notifyAcceptedMerge; every scheduling path through shouldAttemptMerge,
//  notifySuccessfulReflection and notifyMergeAttemptFinished; and the triplet dedup that keeps a pair
//  from being proposed twice. Merge also carries unchecked divergences from the reference
//  implementation, named in the TODO on GEPAMerge.
internal class GEPAMergeTest {
    @Test
    fun `a single-module pool never yields a merge proposal`() {
        val pool = GEPACandidatePool(evaluationDatasetSize = 2, random = Random(0))
        val seed = OptimizationArtifact(strategyInstruction = "Seed instruction.")
        val firstChild = OptimizationArtifact(strategyInstruction = "First instruction.")
        val secondChild = OptimizationArtifact(strategyInstruction = "Second instruction.")

        pool.add(seed, parents = emptyList(), scores = listOf(0.2, 0.2))
        pool.add(firstChild, parents = listOf(seed), scores = listOf(1.0, 0.0))
        pool.add(secondChild, parents = listOf(seed), scores = listOf(0.0, 1.0))

        assertNull(merge(pool).propose())
    }

    @Test
    fun `two lineages that changed different modules yield a merge of both changes`() {
        val pool = GEPACandidatePool(evaluationDatasetSize = 2, random = Random(0))
        val seed = OptimizationArtifact(
            strategyInstruction = "Seed strategy.",
            subgraphInstructions = mapOf(TASK_MODULE to "Seed task."),
        )
        val strategyChild = seed.withStrategyInstruction("Reworked strategy.")
        val taskChild = seed.withSubgraphInstruction(TASK_MODULE, "Reworked task.")

        pool.add(seed, parents = emptyList(), scores = listOf(0.2, 0.2))
        pool.add(strategyChild, parents = listOf(seed), scores = listOf(1.0, 0.0))
        pool.add(taskChild, parents = listOf(seed), scores = listOf(0.0, 1.0))

        val proposal = assertNotNull(merge(pool).propose())

        assertEquals(setOf(strategyChild, taskChild), proposal.parents.toSet())
        assertEquals(seed, proposal.ancestor)
        assertEquals(
            OptimizationArtifact(
                strategyInstruction = "Reworked strategy.",
                subgraphInstructions = mapOf(TASK_MODULE to "Reworked task."),
            ),
            proposal.candidate,
        )
    }

    private fun merge(pool: GEPACandidatePool): GEPAMerge =
        GEPAMerge(pool, GEPAMergeConfig(enabled = true), Random(0))

    private companion object {
        const val TASK_MODULE = "task"
    }
}

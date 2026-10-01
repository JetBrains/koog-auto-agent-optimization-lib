package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizationArtifact
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class GEPACandidatePoolTest {
    @Test
    fun `findFrontierRepresentatives prunes redundant frontier coverage like reference implementation`() {
        val pool = GEPACandidatePool(evaluationDatasetSize = 4, random = Random(0))
        val redundant = artifact("redundant")
        val first = artifact("first")
        val second = artifact("second")

        pool.add(redundant, parents = emptyList(), scores = listOf(1.0, 1.0, 0.0, 0.0))
        pool.add(first, parents = emptyList(), scores = listOf(1.0, 0.0, 1.0, 1.0))
        pool.add(second, parents = emptyList(), scores = listOf(0.0, 1.0, 1.0, 1.0))

        val (frontierRepresentatives, representativesByDatasetItem) = pool.findFrontierRepresentatives()

        assertEquals(setOf(first, second), frontierRepresentatives)
        assertFalse(redundant in frontierRepresentatives)
        assertEquals(setOf(first), representativesByDatasetItem[0])
        assertEquals(setOf(second), representativesByDatasetItem[1])
        assertEquals(setOf(first, second), representativesByDatasetItem[2])
        assertEquals(setOf(first, second), representativesByDatasetItem[3])
        assertTrue(representativesByDatasetItem.all { it.isNotEmpty() })
    }

    @Test
    fun `selectMergeValidationSubsample fills requested size with replacement when there are not enough unused indices`() {
        val pool = GEPACandidatePool(evaluationDatasetSize = 2, random = Random(0))
        val parent = artifact("parent")
        val otherParent = artifact("other")

        pool.add(parent, parents = emptyList(), scores = listOf(1.0, 0.0))
        pool.add(otherParent, parents = emptyList(), scores = listOf(0.0, 1.0))

        val subsample = pool.selectMergeValidationSubsample(parent, otherParent, valSetSubsampleSize = 5)

        assertEquals(5, subsample.size)
        assertTrue(subsample.all { it in 0..1 })
        assertEquals(setOf(0, 1), subsample.toSet())
    }

    private fun artifact(instruction: String): OptimizationArtifact =
        OptimizationArtifact(strategyInstruction = instruction)
}

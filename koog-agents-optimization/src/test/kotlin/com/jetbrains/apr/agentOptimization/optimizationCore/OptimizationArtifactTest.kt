package com.jetbrains.apr.agentOptimization.optimizationCore

import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.OptimizationArtifact
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class OptimizationArtifactTest {

    private val json = Json { prettyPrint = true }

    @Test
    fun testGetInstructionReturnsConfiguredValue() {
        val config = OptimizationArtifact(
            subgraphInstructions = mapOf("classify" to "Classify the sentiment.")
        )
        assertEquals("Classify the sentiment.", config.getInstruction("classify"))
    }

    @Test
    fun testGetInstructionReturnsNullForMissingKey() {
        val config = OptimizationArtifact()
        assertNull(config.getInstruction("nonexistent"))
    }

    @Test
    fun testGetDemonstrationsReturnsEmptyForMissingKey() {
        val config = OptimizationArtifact()
        assertEquals(emptyList(), config.getDemonstrations("nonexistent"))
    }

    @Test
    fun testWithSubgraphInstructionCreatesNewConfigWithoutMutatingOriginal() {
        val original = OptimizationArtifact()
        val updated = original.withSubgraphInstruction("classify", "New instruction")
        assertEquals("New instruction", updated.getInstruction("classify"))
        assertNull(original.getInstruction("classify"), "Original config must not be mutated")
    }

    @Test
    fun testMergeWithOtherTakesPrecedence() {
        val base = OptimizationArtifact(
            strategyInstruction = "base strategy",
            subgraphInstructions = mapOf("a" to "base-a", "b" to "base-b"),
        )
        val override = OptimizationArtifact(
            subgraphInstructions = mapOf("a" to "override-a"),
        )
        val merged = base.mergeWith(override)
        assertEquals("override-a", merged.getInstruction("a"))
        assertEquals("base-b", merged.getInstruction("b"))
        assertEquals("base strategy", merged.strategyInstruction)
    }

    @Test
    fun testSerializationRoundTrip() {
        val config = OptimizationArtifact(
            strategyInstruction = "Be helpful",
            strategyDemonstrations = listOf(Demonstration("hello", "Hi there!")),
            subgraphInstructions = mapOf("classify" to "Classify sentiment."),
            subgraphDemonstrations = mapOf(
                "classify" to listOf(
                    Demonstration("great movie", "positive"),
                    Demonstration("terrible", "negative"),
                ),
            ),
        )
        val encoded = json.encodeToString(OptimizationArtifact.serializer(), config)
        val decoded = json.decodeFromString(OptimizationArtifact.serializer(), encoded)
        assertEquals(config, decoded)
    }

    @Test
    fun testImmutableConfigSafeForParallelTrials() {
        val base = OptimizationArtifact(
            subgraphInstructions = mapOf("classify" to "base instruction"),
        )
        val trial1 = base.withSubgraphInstruction("classify", "trial 1 instruction")
        val trial2 = base.withSubgraphInstruction("classify", "trial 2 instruction")

        assertEquals("base instruction", base.getInstruction("classify"))
        assertEquals("trial 1 instruction", trial1.getInstruction("classify"))
        assertEquals("trial 2 instruction", trial2.getInstruction("classify"))
    }
}

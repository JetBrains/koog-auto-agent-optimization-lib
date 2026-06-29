package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.agent.entity.createStorageKey
import kotlinx.serialization.Serializable

/**
 * The serializable output of an optimization run, holding optimized instructions and demonstrations.
 *
 * Produced by optimizers, consumed by optimizable subgraphs at runtime, and persisted
 * to the filesystem between training and inference.
 *
 * Immutability simplifies parallel evaluation: each optimization trial can safely hold
 * its own artifact instance without synchronization.
 *
 * Subgraph entries use the subgraph name as key.
 * The strategy-level instruction and demonstrations are stored separately from subgraph entries.
 *
 * @property strategyInstruction Optimized instruction for the strategy's system prompt, or null to keep the original.
 * @property strategyDemonstrations Demonstrations for the strategy level.
 * @property subgraphInstructions Map from subgraph name to optimized instruction text.
 * @property subgraphDemonstrations Map from subgraph name to demonstration list.
 */
@Serializable
public data class OptimizationArtifact(
    val strategyInstruction: String? = null,
    val strategyDemonstrations: List<Demonstration> = emptyList(),
    val subgraphInstructions: Map<String, String> = emptyMap(),
    val subgraphDemonstrations: Map<String, List<Demonstration>> = emptyMap(),
) {
    /**
     * Returns the optimized instruction for the given subgraph, or null if not present.
     */
    public fun getInstruction(subgraphName: String): String? = subgraphInstructions[subgraphName]

    /**
     * Returns the demonstrations for the given subgraph, or an empty list if not present.
     */
    public fun getDemonstrations(subgraphName: String): List<Demonstration> =
        subgraphDemonstrations[subgraphName].orEmpty()

    /**
     * Creates a new artifact with an additional or updated subgraph instruction.
     */
    public fun withSubgraphInstruction(subgraphName: String, instruction: String): OptimizationArtifact =
        copy(subgraphInstructions = subgraphInstructions + (subgraphName to instruction))

    /**
     * Creates a new artifact with additional or updated subgraph demonstrations.
     */
    public fun withSubgraphDemonstrations(
        subgraphName: String,
        demonstrations: List<Demonstration>,
    ): OptimizationArtifact =
        copy(subgraphDemonstrations = subgraphDemonstrations + (subgraphName to demonstrations))

    /**
     * Creates a new artifact with an updated strategy-level instruction.
     */
    public fun withStrategyInstruction(instruction: String): OptimizationArtifact =
        copy(strategyInstruction = instruction)

    /**
     * Creates a new artifact with updated strategy-level demonstrations.
     */
    public fun withStrategyDemonstrations(demonstrations: List<Demonstration>): OptimizationArtifact =
        copy(strategyDemonstrations = demonstrations)

    /**
     * Merges this artifact with another, with the [other] artifact's entries taking precedence.
     */
    public fun mergeWith(other: OptimizationArtifact): OptimizationArtifact = OptimizationArtifact(
        strategyInstruction = other.strategyInstruction ?: strategyInstruction,
        strategyDemonstrations = other.strategyDemonstrations.ifEmpty { strategyDemonstrations },
        subgraphInstructions = subgraphInstructions + other.subgraphInstructions,
        subgraphDemonstrations = subgraphDemonstrations + other.subgraphDemonstrations,
    )

    /** Holds the agent-storage key under which the artifact is published at runtime. */
    public companion object {
        /** Storage key for injecting the artifact into agent storage at runtime. */
        public val STORAGE_KEY: AIAgentStorageKey<OptimizationArtifact> =
            createStorageKey("optimization-artifact")
    }
}

package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.prompt.message.Message

/**
 * Strategy-level module key used in [OptimizationArtifact] maps and [OptimizableModule] lists
 * to refer to the agent's top-level system prompt (as opposed to a named subgraph).
 *
 * Shared between MIPRO, GEPA, and any future optimizer.
 */
public const val STRATEGY_MODULE_KEY: String = "__strategy__"

/**
 * Represents an optimizable module discovered in an agent.
 *
 * A module is either the strategy-level system prompt (identified by [STRATEGY_MODULE_KEY])
 * or an optimizable subgraph registered via [optimizableSubgraphWithTask].
 *
 * @property name Module identifier — subgraph name or [STRATEGY_MODULE_KEY].
 * @property currentInstruction The user-supplied baseline instruction to seed optimizer candidates with.
 * @property description Optional human description of the module's purpose (used in meta-LLM prompts).
 */
public data class OptimizableModule(
    val name: String,
    val currentInstruction: String,
    val description: String? = null,
)

/**
 * Discovers optimizable modules in an agent: the strategy-level system prompt plus any
 * subgraphs created via [optimizableSubgraphWithTask].
 *
 * Non-optimizable subgraphs — those created via plain `subgraph { }` or `subgraphWithTask` —
 * are filtered out, since their runtime does not consume an [OptimizationArtifact] and any
 * candidate the optimizer produces for them would be a no-op.
 *
 * Each subgraph module is seeded with the user's `optimizableInstruction`. Lookup is identity-keyed
 * via [OptimizableSubgraphDelegate.lookupBaseline], populated at strategy-construction
 * time by [OptimizableSubgraphDelegate.getValue].
 *
 * @return The strategy-level module first, followed by one module per optimizable subgraph
 *   in graph-traversal order.
 */
public fun <Input, Output> discoverModules(agent: GraphAIAgent<Input, Output>): List<OptimizableModule> {
    val modules = mutableListOf<OptimizableModule>()

    val systemPrompt = agent.agentConfig.prompt.messages
        .filterIsInstance<Message.System>()
        .firstOrNull()?.content ?: ""
    modules.add(OptimizableModule(
        name = STRATEGY_MODULE_KEY,
        currentInstruction = systemPrompt,
        description = "Strategy-level system prompt for the agent.",
    ))

    for (subgraph in agent.strategy.findAllSubgraphs()) {
        val baseline = OptimizableSubgraphDelegate.lookupBaseline(subgraph) ?: continue
        modules.add(OptimizableModule(
            name = subgraph.name,
            currentInstruction = baseline,
        ))
    }

    return modules
}

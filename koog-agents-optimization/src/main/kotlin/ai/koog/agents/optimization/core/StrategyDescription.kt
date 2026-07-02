package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.entity.*

/**
 * Creates a text description of the strategy structure for use in MIPRO instruction proposal.
 *
 * Traverses the strategy graph and describes:
 * - Strategy name
 * - All subgraphs (potential optimization targets) with their names
 * - All nodes with their names and types
 * - Graph connectivity (edges between nodes)
 *
 * @return A human-readable description of the strategy structure.
 */
public fun <TInput, TOutput> AIAgentGraphStrategy<TInput, TOutput>.describeForOptimization(): String =
    buildString {
        appendLine("Strategy: $name")
        appendLine()

        val subgraphs = findAllSubgraphs()
        val nodes = findAllNodes()

        if (subgraphs.isNotEmpty()) {
            appendLine("Subgraphs (${subgraphs.size}):")
            for (subgraph in subgraphs) {
                appendLine("  - ${subgraph.name}")
                val innerNodes = countInnerNodes(subgraph)
                appendLine("    Inner nodes: $innerNodes")
            }
            appendLine()
        }

        if (nodes.isNotEmpty()) {
            appendLine("Nodes (${nodes.size}):")
            for (node in nodes) {
                val edgeTargets = node.edges.map { it.toNode.name }
                val edgeDesc = if (edgeTargets.isNotEmpty()) " → ${edgeTargets.joinToString(", ")}" else ""
                appendLine("  - ${node.name}$edgeDesc")
            }
        }
    }

/**
 * Finds all [AIAgentSubgraph] instances in the strategy, excluding the strategy itself.
 */
public fun <TInput, TOutput> AIAgentGraphStrategy<TInput, TOutput>.findAllSubgraphs(): List<AIAgentSubgraph<*, *>> {
    val subgraphs = mutableListOf<AIAgentSubgraph<*, *>>()
    val visited = mutableSetOf<AIAgentNodeBase<*, *>>()

    fun visit(node: AIAgentNodeBase<*, *>) {
        if (node in visited) return
        visited.add(node)

        if (node is AIAgentSubgraph<*, *>) {
            subgraphs.add(node)
        }

        if (node is AIAgentSubgraph<*, *>) {
            visit(node.start)
        }

        for (edge in node.edges) {
            visit(edge.toNode)
        }
    }

    visit(start)
    return subgraphs
}

/**
 * Finds all [AIAgentNode] instances in the strategy, excluding start/finish nodes and subgraphs.
 */
public fun <TInput, TOutput> AIAgentGraphStrategy<TInput, TOutput>.findAllNodes(): List<AIAgentNode<*, *>> {
    val nodes = mutableListOf<AIAgentNode<*, *>>()
    val visited = mutableSetOf<AIAgentNodeBase<*, *>>()

    fun visit(node: AIAgentNodeBase<*, *>) {
        if (node in visited) return
        visited.add(node)

        if (node is AIAgentNode<*, *> && node !is StartNode<*> && node !is FinishNode<*>) {
            nodes.add(node)
        }

        if (node is AIAgentSubgraph<*, *>) {
            visit(node.start)
        }

        for (edge in node.edges) {
            visit(edge.toNode)
        }
    }

    visit(start)
    return nodes
}

private fun countInnerNodes(subgraph: AIAgentSubgraph<*, *>): Int {
    val visited = mutableSetOf<AIAgentNodeBase<*, *>>()

    fun visit(node: AIAgentNodeBase<*, *>) {
        if (node in visited) return
        visited.add(node)
        for (edge in node.edges) {
            visit(edge.toNode)
        }
    }

    visit(subgraph.start)
    // Exclude start and finish
    return (visited.size - 2).coerceAtLeast(0)
}

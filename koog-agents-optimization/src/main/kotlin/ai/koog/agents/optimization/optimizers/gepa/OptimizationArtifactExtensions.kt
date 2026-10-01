package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.STRATEGY_MODULE_KEY

/**
 * Iterates over strategy and subgraph instructions as one module-keyed prompt map.
 * The strategy instruction is keyed by [STRATEGY_MODULE_KEY].
 */
internal fun OptimizationArtifact.modulePrompts(): Sequence<Pair<String, String>> = sequence {
    strategyInstruction?.let { yield(STRATEGY_MODULE_KEY to it) }
    yieldAll(subgraphInstructions.asSequence().map { (moduleName, prompt) -> moduleName to prompt })
}

/** Returns the prompt for [moduleName], or null if the artifact does not override it. */
internal fun OptimizationArtifact.getModulePrompt(moduleName: String): String? =
    if (moduleName == STRATEGY_MODULE_KEY) {
        strategyInstruction
    } else {
        getInstruction(moduleName)
    }

/** Returns a copy with [prompt] set for [moduleName]. */
internal fun OptimizationArtifact.withModulePrompt(
    moduleName: String,
    prompt: String,
): OptimizationArtifact =
    if (moduleName == STRATEGY_MODULE_KEY) {
        withStrategyInstruction(prompt)
    } else {
        withSubgraphInstruction(moduleName, prompt)
    }

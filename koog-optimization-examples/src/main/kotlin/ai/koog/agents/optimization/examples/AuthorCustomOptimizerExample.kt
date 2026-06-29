// Authoring a custom optimizer touches the @OptimizationExtensionApi SPI (the StageScope training DSL),
// so this file opts in.
@file:OptIn(OptimizationExtensionApi::class)

package ai.koog.agents.optimization.examples

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.annotations.OptimizationExtensionApi
import ai.koog.agents.optimization.optimizers.AgentOptimizer
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.records.TrainingResult

/**
 * Smallest possible custom [AgentOptimizer]: runs the tracked agent once over each dataset item
 * (recording metrics) and applies no transformation. Exists to demonstrate the optimizer-authoring
 * SPI
 */
class RunOnceOptimizer<Input, Output, InputLabel> : AgentOptimizer<Input, Output, InputLabel> {
    override suspend fun train(
        session: TrainingSession<Input, Output, InputLabel>,
    ): TrainingResult = session.use(stagesTotal = 1) {
        dataset.forEach { runAgent(it) }
    }

    override fun loadOptimizedAgent(
        baseAgent: GraphAIAgent<Input, Output>,
    ): GraphAIAgent<Input, Output> = baseAgent
}

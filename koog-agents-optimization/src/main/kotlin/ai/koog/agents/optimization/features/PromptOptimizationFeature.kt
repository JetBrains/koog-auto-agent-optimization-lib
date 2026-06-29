package ai.koog.agents.optimization.features

import ai.koog.agents.core.agent.GraphAIAgent.FeatureContext
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.feature.AIAgentGraphFeature
import ai.koog.agents.core.feature.config.FeatureConfig
import ai.koog.agents.core.feature.pipeline.AIAgentGraphPipeline
import ai.koog.agents.optimization.core.DemonstrationRenderer
import ai.koog.agents.optimization.core.FewShotPromptType
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.PromptInsertionDefaults
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.message.Message

/**
 * Configuration for the [PromptOptimizationFeature].
 *
 * @property artifact The optimization artifact to apply at runtime.
 * @property promptInsertionDefaults Default settings for how demos are inserted into prompts.
 *   Individual subgraphs can override these.
 */
public class PromptPromptOptimizationFeatureConfig : FeatureConfig() {
    /** The optimization artifact with tuned instructions and bootstrapped demonstrations. */
    public var artifact: OptimizationArtifact = OptimizationArtifact()

    /** Default settings for demo insertion, propagated to subgraphs that don't override them. */
    public var promptInsertionDefaults: PromptInsertionDefaults = PromptInsertionDefaults()
}

/**
 * Feature that applies an [OptimizationArtifact] to an agent at strategy start.
 *
 * At strategy start, this feature:
 * 1. Stores the [OptimizationArtifact] in agent storage for optimizable subgraphs to read.
 * 2. Stores [PromptInsertionDefaults] in agent storage for subgraphs to inherit.
 * 3. If [OptimizationArtifact.strategyInstruction] is set, replaces the system message in the
 *    prompt with the optimized instruction.
 * 4. If [OptimizationArtifact.strategyDemonstrations] is non-empty, appends demonstrations
 *    after the system message.
 *
 * Without this feature installed, optimizable subgraphs fall back to their default
 * instructions and empty demonstrations.
 *
 * Usage:
 * ```kotlin
 * val agent = AIAgent(...) {
 *     installPromptOptimization {
 *         artifact = loadedArtifact
 *     }
 * }
 * ```
 */
public object PromptOptimizationFeature :
    AIAgentGraphFeature<PromptPromptOptimizationFeatureConfig, OptimizationArtifact> {

    /** Storage key under which the applied [OptimizationArtifact] is published. */
    override val key: AIAgentStorageKey<OptimizationArtifact> = OptimizationArtifact.STORAGE_KEY

    /** Creates the default, empty [PromptPromptOptimizationFeatureConfig]. */
    override fun createInitialConfig(agentConfig: AIAgentConfig): PromptPromptOptimizationFeatureConfig =
        PromptPromptOptimizationFeatureConfig()

    /** Registers the strategy-start interceptor that publishes the artifact and applies strategy-level prompt edits. */
    override fun install(
        config: PromptPromptOptimizationFeatureConfig,
        pipeline: AIAgentGraphPipeline,
    ): OptimizationArtifact {
        val artifact = config.artifact
        val defaults = config.promptInsertionDefaults

        // Intercept every time the strategy starts and substitute the system prompt
        // and the artifacts required by the subgraphs
        pipeline.interceptStrategyStarting(this) { eventContext ->
            // Store artifact and defaults for optimizable subgraphs
            eventContext.context.storage.set(OptimizationArtifact.STORAGE_KEY, artifact)
            eventContext.context.storage.set(PromptInsertionDefaults.STORAGE_KEY, defaults)

            // Apply strategy-level instruction and demonstrations
            val strategyInstruction = artifact.strategyInstruction
            val strategyDemos = artifact.strategyDemonstrations

            if (strategyInstruction != null || strategyDemos.isNotEmpty()) {
                eventContext.context.llm.writeSession {
                    if (strategyInstruction != null) {
                        rewritePrompt { currentPrompt ->
                            replaceSystemMessage(currentPrompt, strategyInstruction)
                        }
                    }

                    if (strategyDemos.isNotEmpty()) {
                        when (defaults.fewShotPromptType) {
                            FewShotPromptType.AS_STRING -> {
                                val rendered = DemonstrationRenderer.renderAsString(
                                    strategyDemos, defaults.demonstrationFormat
                                )
                                if (rendered != null) {
                                    appendPrompt { user(rendered) }
                                }
                            }

                            FewShotPromptType.AS_MESSAGE_HISTORY -> {
                                val demoMessages = DemonstrationRenderer.renderAsMessages(
                                    strategyDemos, defaults.demonstrationFormat
                                )
                                if (demoMessages.isNotEmpty()) {
                                    appendPrompt { messages(demoMessages) }
                                }
                            }
                        }
                    }
                }
            }
        }

        return artifact
    }

    /**
     * Replaces the first system message in the prompt with the given instruction.
     * If no system message exists, the prompt is returned unchanged.
     */
    private fun replaceSystemMessage(prompt: Prompt, newInstruction: String): Prompt {
        val messages = prompt.messages
        val systemIndex = messages.indexOfFirst { it is Message.System }
        if (systemIndex < 0) return prompt // Probably should not happen

        val original = messages[systemIndex] as Message.System
        val replacement = Message.System(newInstruction, original.metaInfo)
        val updatedMessages = messages.toMutableList().apply { set(systemIndex, replacement) }
        return Prompt(updatedMessages, prompt.id, prompt.params)
    }
}

/**
 * Installs the [PromptOptimizationFeature] with the given configuration.
 *
 * Makes the [OptimizationArtifact] available to all optimizable subgraphs via agent storage.
 * If the artifact includes strategy-level instruction or demonstrations, those are applied
 * to the prompt at strategy start.
 *
 * @param configure Lambda to set the artifact and prompt insertion defaults.
 */
public fun FeatureContext.installPromptOptimization(configure: PromptPromptOptimizationFeatureConfig.() -> Unit = {}) {
    install(PromptOptimizationFeature) {
        configure()
    }
}

package ai.koog.agents.optimization.optimizers.ace


import ai.koog.agents.core.agent.GraphAIAgent.FeatureContext
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.feature.AIAgentGraphFeature
import ai.koog.agents.core.feature.config.FeatureConfig
import ai.koog.agents.core.feature.pipeline.AIAgentGraphPipeline
import ai.koog.agents.optimization.utils.agentic.appendToSystemMessage

/**
 * Configuration for [ACEPlaybookFeature].
 *
 * @property playbook The ACE playbook to inject into the system prompt at strategy start.
 */
public class ACEPlaybookFeatureConfig : FeatureConfig() {
    public var playbook: ACEPlaybook? = null
}

/**
 * ACE-specific feature that appends playbook content to the agent's system prompt at strategy start.
 *
 * Unlike [ai.koog.agents.optimization.features.PromptOptimizationFeature] which replaces the system
 * message, this feature preserves the original system prompt and appends the playbook after it.
 * This is necessary because the original system prompt contains task-specific instructions that
 * must be preserved.
 *
 * Per-subgraph playbook content is handled separately via [ai.koog.agents.optimization.core.OptimizationArtifact.subgraphInstructions],
 * baked in at save time and applied by [ai.koog.agents.optimization.features.PromptOptimizationFeature].
 *
 * Composable with other features:
 * ```kotlin
 * agent.copyWith(installFeatures = {
 *     agent.installFeatures(this)
 *     installACEPlaybook { playbook = loadedPlaybook }
 *     installPromptOptimization { artifact = perSubgraphArtifact }
 * })
 * ```
 */
public object ACEPlaybookFeature : AIAgentGraphFeature<ACEPlaybookFeatureConfig, Unit> {

    /** Storage key identifying this feature within the agent pipeline. */
    override val key: AIAgentStorageKey<Unit> = createStorageKey<Unit>("ace-playbook-feature")

    /** Creates an empty [ACEPlaybookFeatureConfig] (no playbook set). */
    override fun createInitialConfig(agentConfig: AIAgentConfig): ACEPlaybookFeatureConfig = ACEPlaybookFeatureConfig()

    /** Registers a strategy-start interceptor that appends the configured playbook to the system prompt. */
    override fun install(
        config: ACEPlaybookFeatureConfig,
        pipeline: AIAgentGraphPipeline,
    ) {
        val playbook = config.playbook ?: return

        pipeline.interceptStrategyStarting(this) { eventContext ->
            val content = ACEPrompts.constructPlaybookContext(playbook)
            if (content.isNotBlank()) {
                eventContext.context.llm.writeSession {
                    rewritePrompt { it.appendToSystemMessage(content) }
                }
            }
        }
    }
}

/**
 * Installs the [ACEPlaybookFeature] with the given configuration.
 *
 * @param configure Lambda to set the playbook.
 */
public fun FeatureContext.installACEPlaybook(configure: ACEPlaybookFeatureConfig.() -> Unit) {
    install(ACEPlaybookFeature) {
        configure()
    }
}

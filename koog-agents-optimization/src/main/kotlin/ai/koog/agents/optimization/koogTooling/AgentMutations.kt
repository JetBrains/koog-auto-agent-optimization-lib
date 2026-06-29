package ai.koog.agents.optimization.koogTooling


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.config.MissingToolsConversionStrategy
import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.serialization.TypeToken
import kotlin.time.Clock

/**
 * Copies this agent and runs the copy on [input].
 *
 * [GraphAIAgent] is a single-use object: calling [GraphAIAgent.run] on an instance that has
 * already been started throws "Agent was already started". Use [runFresh] wherever the same
 * agent configuration must serve more than one run — evaluation loops, training loops, retries.
 * The copy preserves all configuration (prompts, tools, installed features such as consumption
 * tracking), so metrics and tracing continue to work correctly across runs.
 */
// TODO: as of Koog 0.7, this function may no longer be needed, as the agent can be re-run.
public suspend fun <Input, Output> GraphAIAgent<Input, Output>.runFresh(input: Input): Output =
    copyWith().run(input)

/**
 * Returns a copy of this agent whose [AIAgentConfig.prompt] is replaced with [newPrompt],
 * leaving all other configuration (model, tools, installed features) intact.
 */
public fun <Input, Output> GraphAIAgent<Input, Output>.copyWithPrompt(
    newPrompt: Prompt
): GraphAIAgent<Input, Output> =
    copyWith(
        agentConfig = agentConfig.copyWith(prompt = newPrompt)
    )

/**
 * Method to copy the Agent's config with updated parameters
 * (since the original class is immutable and is not a data class)
 */
public fun AIAgentConfig.copyWith(
    prompt: Prompt = this.prompt,
    model: LLModel = this.model,
    maxAgentIterations: Int = this.maxAgentIterations,
    missingToolsConversionStrategy: MissingToolsConversionStrategy = this.missingToolsConversionStrategy
): AIAgentConfig = AIAgentConfig(
    prompt = prompt,
    model = model,
    maxAgentIterations = maxAgentIterations,
    missingToolsConversionStrategy = missingToolsConversionStrategy
)

/**
 * To be able to optimize the agent, we need to be able to automatically
 * evolve it by modifying or creating copies of the AIAgent object.
 */
@OptIn(InternalAgentsApi::class)
@Suppress("UNCHECKED_CAST")
public fun <Input, Output> GraphAIAgent<Input, Output>.copyWith(
    inputType: TypeToken = this.inputType,
    outputType: TypeToken = this.outputType,
    promptExecutor: PromptExecutor = this.promptExecutor,
    agentConfig: AIAgentConfig = this.agentConfig,
    id: String = this.id,
    toolRegistry: ToolRegistry = this.toolRegistry,
    clock: Clock = this.clock,
    installFeatures: GraphAIAgent.FeatureContext.() -> Unit = this.installFeatures,
    strategy: AIAgentGraphStrategy<Input, Output> = this.strategy,
): GraphAIAgent<Input, Output> {

    return GraphAIAgent(
        inputType = inputType,
        outputType = outputType,
        promptExecutor = promptExecutor,
        strategy = strategy,
        agentConfig = agentConfig,
        id = id,
        toolRegistry = toolRegistry,
        clock = clock,
        installFeatures = installFeatures,
    )
}

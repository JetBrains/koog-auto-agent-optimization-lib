package ai.koog.agents.optimization.koogTooling


import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.feature.AIAgentGraphFeature
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.optimization.features.CollectedSubgraphTraces
import ai.koog.agents.optimization.features.SubgraphTraceCollectionFeature
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.params.LLMParams
import kotlin.reflect.KClass
import kotlin.reflect.typeOf
import kotlin.time.Clock
import kotlin.uuid.ExperimentalUuidApi

/**
 * Retrieves an installed feature instance from the agent's pipeline.
 *
 * Shorthand for the `agent.createSession().pipeline()?.feature(...)` ceremony.
 * `createSession()` is lightweight — it wraps the agent's existing pipeline without
 * creating a new one, so calling this repeatedly is safe.
 *
 * @param featureClass The KClass of the feature implementation type.
 * @param feature The feature object (singleton) to look up.
 * @return The feature instance, or null if not installed.
 */
public fun <T : Any> AIAgent<*, *>.getFeature(
    featureClass: KClass<T>,
    feature: AIAgentGraphFeature<*, T>,
): T? = createSession().pipeline()?.feature(featureClass, feature)

/**
 * Retrieves the [CollectedSubgraphTraces] from an agent that has
 * [SubgraphTraceCollectionFeature] installed.
 *
 * @return The collected traces, or null if the feature is not installed.
 */
public fun AIAgent<*, *>.getCollectedTraces(): CollectedSubgraphTraces? =
    getFeature(CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature)

/**
 * Creates a [GraphAIAgent] with the given parameters.
 */
@OptIn(ExperimentalUuidApi::class)
public inline fun <reified Input, reified Output> AIAgent.Companion.invokeGraphAgent(
    promptExecutor: PromptExecutor,
    llmModel: LLModel,
    strategy: AIAgentGraphStrategy<Input, Output>,
    toolRegistry: ToolRegistry = ToolRegistry.EMPTY,
    id: String? = null,
    clock: Clock = Clock.System,
    systemPrompt: String = "",
    temperature: Double = 1.0,
    numberOfChoices: Int = 1,
    maxIterations: Int = 50,
    noinline installFeatures: GraphAIAgent.FeatureContext.() -> Unit = {},
): GraphAIAgent<Input, Output> {
    // Koog splits AIAgent into FunctionalAgent and GraphAgent implementations, with the shared
    // properties living on each implementation rather than the interface, so this helper targets
    // GraphAIAgent concretely.
    return GraphAIAgent(
        inputType = typeOf<Input>(),
        outputType = typeOf<Output>(),
        id = id,
        promptExecutor = promptExecutor,
        strategy = strategy,
        agentConfig = AIAgentConfig(
            prompt = prompt(
                id = "chat",
                params = LLMParams(
                    temperature = temperature,
                    numberOfChoices = numberOfChoices
                )
            ) {
                system(systemPrompt)
            },
            model = llmModel,
            maxAgentIterations = maxIterations,
        ),
        clock = clock,
        toolRegistry = toolRegistry,
        installFeatures = installFeatures,
    )
}

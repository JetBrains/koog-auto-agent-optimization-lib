package ai.koog.agents.optimization.examples

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.dsl.builder.node
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.common.retries.RetryPolicy
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.optimization.koogTooling.invokeGraphAgent
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.optimizers.gepa.ComponentSelection
import ai.koog.agents.optimization.optimizers.gepa.GEPAOptimizer
import ai.koog.agents.optimization.training.ActionLogTruncation
import ai.koog.agents.optimization.training.LiteLLMConsumptionCapturingPromptExecutor
import ai.koog.agents.optimization.training.NoOpFailureAnalyzer
import ai.koog.agents.optimization.training.TrainingResources
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.metrics.impl.ConsumptionMetric
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking

/**
 * The agent is a two-stage support-ticket router with **two optimizable subgraphs**:
 *   nodeStart(message) -> [analyze] -> pack -> [route] -> nodeFinish(code)
 *
 * Both subgraphs start with deliberately **ambiguous instructions** that say nothing about the
 * allowed routing codes or the required output format. So the baseline emits prose ("This should
 * go to billing.") and fails the exact-match metric. GEPA reflects on those failure traces, which
 * include the expected labels, and evolves each subgraph's instruction until the agent emits the
 * bare codes BILLING / TECH / OTHER. The saved artifact then contains the learned instructions.
 *
 * Run:
 * ```
 * LITELLM_BASE_URL=https://your-litellm-host LITELLM_API_KEY=sk-... \
 *   ./gradlew :koog-optimization-examples:runSupportRouter
 * ```
 */

private val origMessageKey = createStorageKey<String>("orig-support-message")

/** Allowed routing codes — intentionally NOT mentioned in the baseline instructions below. */
@Suppress("unused")
private const val ROUTING_CODES = "BILLING (payments/invoices), TECH (bugs/errors/login), OTHER"

private fun supportRouterStrategy(): AIAgentGraphStrategy<String, String> =
    strategy("support-router") {
        val remember by node<String, String>("remember") { message ->
            storage.set(origMessageKey, message)
            message
        }

        // Optimizable subgraph #1: ambiguous; doesn't say what to extract or why.
        val analyze by optimizableSubgraphWithTask<String, String>(
            optimizableInstruction = "Look at the customer message.",
            freshHistory = true,
        ) { _, message -> "Customer message:\n$message" }

        val pack by node<String, String>("pack") { analysis ->
            val message = storage.getValue(origMessageKey)
            "Customer message:\n$message\n\nNotes:\n$analysis"
        }

        // Optimizable subgraph #2: ambiguous; no codes, no output-format requirement.
        val route by optimizableSubgraphWithTask<String, String>(
            optimizableInstruction = "Decide how this request should be handled.",
            freshHistory = true,
        ) { _, packed -> packed }

        edge(nodeStart forwardTo remember)
        edge(remember forwardTo analyze)
        edge(analyze forwardTo pack)
        edge(pack forwardTo route)
        edge(route forwardTo nodeFinish)
    }

fun main() = runBlocking {
    val baseUrl = System.getenv("LITELLM_BASE_URL")
        ?: error("Set LITELLM_BASE_URL to your LiteLLM proxy URL.")
    val apiKey = System.getenv("LITELLM_API_KEY")
        ?: error("Set LITELLM_API_KEY to your LiteLLM API key.")

    val executor = MultiLLMPromptExecutor(
        OpenAILLMClient(apiKey = apiKey, settings = OpenAIClientSettings(baseUrl = baseUrl)),
    )
    val capturingExecutor = LiteLLMConsumptionCapturingPromptExecutor(executor)

    val agent: GraphAIAgent<String, String> = AIAgent.invokeGraphAgent(
        id = "support-router",
        promptExecutor = capturingExecutor,
        systemPrompt = "You are a customer-support assistant.",
        llmModel = OpenAIModels.Chat.GPT4oMini,
        temperature = 0.0,
        strategy = supportRouterStrategy(),
    )

    // Labelled tickets. Exact-match metric on the routing code: the vague baseline can't satisfy it.
    val dataset: TrainSet<String, String> = listOf(
        TrainSetItem("I was charged twice for my subscription this month.", "BILLING"),
        TrainSetItem("The app crashes every time I open the settings page.", "TECH"),
        TrainSetItem("Do you have a referral program for existing customers?", "OTHER"),
        TrainSetItem("My invoice shows the wrong amount, can you fix it?", "BILLING"),
        TrainSetItem("I can't log in — it keeps returning a 500 server error.", "TECH"),
        TrainSetItem("What are your customer support office hours?", "OTHER"),
    )
    val metric: (TrainSetItem<String, String>, String) -> Double = { item, answer ->
        if (answer.trim() == item.itemLabel) 1.0 else 0.0
    }
    val serializers = DatasetExecutionSerializers<String, String, String>(
        serializeItem = { it.userQuery },
        serializeOutput = { it },
    )

    val resources = TrainingResources(
        trackedAgent = agent,
        dataset = dataset,
        substepPromptExecutor = executor,
        metric = metric,
        threshold = 0.9,
        serializers = serializers,
        actionLogTruncation = ActionLogTruncation.DEFAULT,
        retriesPolicy = RetryPolicy(maxAttempts = 1),
        failureAnalyzer = NoOpFailureAnalyzer,
        capturingExecutorFactory = ::LiteLLMConsumptionCapturingPromptExecutor,
        consumptionCollector = { capturingExecutor.collectAndClear() },
    )
    val session = TrainingSession(
        resources = resources,
        logger = KotlinLogging.logger {},
        trainingName = ExperimentName(
            runId = "support-router-gepa-example",
            optimizerName = "GEPA",
            agentName = "support-router",
        ),
        recordsFilePath = null,
        throwIfRecordsFileExists = false,
        podName = null,
        spendLimit = null,
        retries = null,
    )

    val storagePath = ResilientPath("build/example-artifacts/support_router_gepa.json")
    // ALL_AT_ONCE so every iteration proposes new instructions for *both* subgraphs together.
    val optimizer = GEPAOptimizer<String, String, String>(
        reflectionModel = OpenAIModels.Chat.GPT4oMini,
        storagePath = storagePath,
        maxIterations = 3,
        minibatchSize = 6,
        componentSelection = ComponentSelection.ALL_AT_ONCE,
        enableCrossover = false,
        randomSeed = 42,
        labelExtractor = { it.itemLabel },
    )

    // Before: the unoptimized agent on a held-out ticket
    val heldOut = "Your billing system double-charged my credit card."
    println("=== BEFORE optimization ===")
    println("Held-out ticket: $heldOut")
    println("Baseline agent answered: ${agent.run(heldOut)}")

    val result = optimizer.train(session)
    val tokens = result.rootStage.metrics[ConsumptionMetric.KEY]?.totalOrNull()
    println()
    println("=== Optimization summary ===")
    println("Tokens used: ${tokens?.toPrettyString() ?: "n/a"}")
    println("Learned artifact (${storagePath.absolutePathString}):")
    println(storagePath.readText())

    // After: the optimized agent should now emit the bare routing code.
    val optimized = optimizer.loadOptimizedAgent(agent)
    println("=== AFTER optimization ===")
    println("Optimized agent answered: ${optimized.run(heldOut)}")
}

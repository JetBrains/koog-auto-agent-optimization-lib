package ai.koog.agents.optimization.examples

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.singleRunStrategy
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.annotations.Tool
import ai.koog.agents.core.tools.reflect.ToolSet
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.koogTooling.invokeGraphAgent
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.optimizers.fewShot.BootstrapFewShotOptimizer
import ai.koog.agents.optimization.training.trainingSession
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import kotlinx.coroutines.runBlocking

/**
 * Optimize a tiny Koog "weather" agent with [BootstrapFewShotOptimizer] against a
 * LiteLLM endpoint, then run the optimized agent.
 *
 * Run it with the LiteLLM proxy URL and key in the environment:
 * ```
 * LITELLM_BASE_URL=https://your-litellm-host LITELLM_API_KEY=sk-... \
 *   ./gradlew :koog-optimization-examples:run
 * ```
 */
fun main() = runBlocking {
    val baseUrl = System.getenv("LITELLM_BASE_URL")
        ?: error("Set LITELLM_BASE_URL to your LiteLLM proxy URL.")
    val apiKey = System.getenv("LITELLM_API_KEY")
        ?: error("Set LITELLM_API_KEY to your LiteLLM API key.")

    // A PromptExecutor backed by LiteLLM (an OpenAI-compatible client pointed at the proxy).
    val executor = MultiLLMPromptExecutor(
        OpenAILLMClient(apiKey = apiKey, settings = OpenAIClientSettings(baseUrl = baseUrl)),
    )

    // A simple agent to optimize: answers weather questions using a mock tool.
    val agent: GraphAIAgent<String, String> = AIAgent.invokeGraphAgent(
        id = "weather-agent",
        promptExecutor = executor,
        systemPrompt = "You are a helpful assistant that reports the weather using the provided tools.",
        llmModel = OpenAIModels.Chat.GPT4oMini,
        temperature = 0.0,
        toolRegistry = ToolRegistry { tools(WeatherTools()) },
        strategy = singleRunStrategy(),
    )

    // A labeled training set and a metric scoring the agent's answer against the golden temperature.
    val outputHint = "Answer only with a single floating number -- the temperature."
    val dataset: TrainSet<String, Double> = listOf(
        TrainSetItem("What is the weather like at 9:00 am? $outputHint", 12.0),
        TrainSetItem("What is the weather like at 14:00 pm? $outputHint", 19.0),
        TrainSetItem("What is the weather like at 19:00 pm? $outputHint", 14.0),
    )
    val metric: (TrainSetItem<String, Double>, String) -> Double = { item, answer ->
        if (answer.trim().toDoubleOrNull() == item.itemLabel) 1.0 else 0.0
    }
    val serializers = DatasetExecutionSerializers<String, String, Double>(
        serializeItem = { it.userQuery },
        serializeOutput = { it },
    )

    // The training session wires library defaults (NoOp failure analysis, single-attempt runs, no
    // records file, no consumption tracking). Pass `consumptionCollector` to count tokens; see
    // OptimizeSupportRouterExample.
    val session = trainingSession(
        experimentName = ExperimentName(runId = "weather-fewshot-example", optimizerName = "BootstrapFewShot", agentName = "weather-agent"),
        trackedAgent = agent,
        dataset = dataset,
        substepPromptExecutor = executor,
        metric = metric,
        serializers = serializers,
    )

    // BootstrapFewShot turns successful runs into few-shot demonstrations baked into the optimized
    // agent's prompt.
    val storagePath = ResilientPath("build/example-artifacts/weather_fewshot.json")
    val optimizer = BootstrapFewShotOptimizer<String, String, Double>(
        maxBootstrappedDemos = 4,
        maxRounds = 1,
        maxTotalDemos = 4,
        includeLabeledExamples = true,
        storagePath = storagePath,
        randomSeed = 42,
    )

    // Before: the unoptimized agent on a held-out question.
    val heldOut = "What is the weather like right now? $outputHint"
    println("=== BEFORE optimization ===")
    println("Held-out question: $heldOut")
    println("Baseline agent answered: ${agent.run(heldOut)}")

    optimizer.train(session)
    println()
    println("Learned artifact saved to ${storagePath.absolutePathString}")

    // After: the optimized agent on the same question.
    val optimized = optimizer.loadOptimizedAgent(agent)
    println("=== AFTER optimization ===")
    println("Optimized agent answered: ${optimized.run(heldOut)}")
}

/** Mock weather toolset — returns a deterministic temperature inferred from the hour in the query. */
@LLMDescription("Utility toolset providing mock weather info.")
class WeatherTools : ToolSet {
    @Tool
    @LLMDescription("Returns mock weather (temperature in Celsius) for the given time string.")
    @Suppress("unused")
    fun getWeather(
        @LLMDescription("Time string (any format, ideally containing an hour).") time: String,
    ): String {
        val hour = Regex("""\b([01]?\d|2[0-3])\b""").find(time)?.groupValues?.get(1)?.toInt()
        val temperature = when (hour) {
            in 0..5 -> 7
            in 6..11 -> 12
            in 12..17 -> 19
            in 18..21 -> 14
            in 22..23 -> 9
            else -> 15
        }
        return "Temperature at time '$time': $temperature°C"
    }
}

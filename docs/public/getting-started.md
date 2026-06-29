# Getting started

This walkthrough takes a small Koog "weather" agent and optimizes it end-to-end with
`BootstrapFewShotOptimizer`: you build the agent, describe a dataset and a metric, wrap them in a
training session, train, then load and run the optimized agent. It mirrors the runnable
`OptimizeWeatherAgentExample` shipped in `:koog-optimization-examples`.

This page is the canonical reference for the **dataset**, the **metric**, the **serializers**, and
the **`trainingSession(...)`** helper. Other pages link back here instead of re-explaining them.

## Prerequisites

The library is provider-agnostic: you bring any Koog `PromptExecutor`. It can be backed by OpenAI,
Anthropic, or any OpenAI-compatible proxy. The bundled examples read their endpoint and key from
environment variables and construct an executor pointed at an OpenAI-compatible client — never
hardcode secrets.

```kotlin
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor

// Any Koog PromptExecutor works; here an OpenAI-compatible client read from the environment.
val executor = MultiLLMPromptExecutor(
    OpenAILLMClient(
        apiKey = System.getenv("LITELLM_API_KEY"),
        settings = OpenAIClientSettings(baseUrl = System.getenv("LITELLM_BASE_URL")),
    ),
)
```

### Install

!!! warning "Install the Koog fork first"

    This library depends on a **fork of Koog** with optimizable-subgraph support
    (`ai.koog:koog-agents:0.8.0-SNAPSHOT`), not the upstream release. Publish it to your local Maven
    repository before building:

    ```bash
    git clone https://github.com/valemore/koog.git
    cd koog
    git checkout tags/stable-agent-optimization
    ./gradlew publishToMavenLocal
    ```

    (This is expected to merge into [JetBrains/koog](https://github.com/JetBrains/koog) later, after
    which the fork won't be needed.)

Then add the dependencies — both resolve from `mavenLocal()`:

```kotlin
repositories { mavenLocal(); mavenCentral() }
dependencies {
    implementation("ai.koog:koog-agents-optimization:0.1.0-SNAPSHOT")
    implementation("ai.koog:koog-agents:0.8.0-SNAPSHOT") // the Koog fork
}
```

The optimization library itself is published to your local Maven repository via
`./gradlew :koog-agents-optimization:publishToMavenLocal`. See [Overview](index.md) for more.

## Step 1 — Build the agent

The optimizer works on a `GraphAIAgent<Input, Output>`. Build one with the library's
`AIAgent.invokeGraphAgent` helper. Here it answers weather questions using a mock `WeatherTools`
toolset.

```kotlin
import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.singleRunStrategy
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.optimization.koogTooling.invokeGraphAgent
import ai.koog.prompt.executor.clients.openai.OpenAIModels

val agent: GraphAIAgent<String, String> = AIAgent.invokeGraphAgent(
    id = "weather-agent",
    promptExecutor = executor,
    systemPrompt = "You are a helpful assistant that reports the weather using the provided tools.",
    llmModel = OpenAIModels.Chat.GPT4oMini,
    temperature = 0.0,
    toolRegistry = ToolRegistry { tools(WeatherTools()) }, // optional
    strategy = singleRunStrategy(),
)
```

`WeatherTools` is an ordinary Koog `ToolSet` — a single `@Tool` that returns a deterministic
temperature inferred from the hour in the query. The agent has no idea how to phrase its answer yet;
that is what we will optimize.

```kotlin
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.annotations.Tool
import ai.koog.agents.core.tools.reflect.ToolSet

@LLMDescription("Utility toolset providing mock weather info.")
class WeatherTools : ToolSet {
    @Tool
    @LLMDescription("Returns mock weather (temperature in Celsius) for the given time string.")
    fun getWeather(
        @LLMDescription("Time string (any format, ideally containing an hour).") time: String,
    ): String {
        val hour = Regex("""\b([01]?\d|2[0-3])\b""").find(time)?.groupValues?.get(1)?.toInt()
        val temperature = when (hour) {
            in 0..5 -> 7; in 6..11 -> 12; in 12..17 -> 19; in 18..21 -> 14; in 22..23 -> 9; else -> 15
        }
        return "Temperature at time '$time': $temperature°C"
    }
}
```

## Step 2 — The dataset

A dataset is a `TrainSet<Input, InputLabel>`, which is just a `List<TrainSetItem<Input, InputLabel>>`.
Each item pairs an agent input with a label:

```kotlin
// data class TrainSetItem<AgentInput, ItemLabel>(val userQuery: AgentInput, val itemLabel: ItemLabel)
```

For the weather agent, `Input` is `String` (the user query) and the label is the golden temperature,
a `Double`:

```kotlin
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem

val outputHint = "Answer only with a single floating number -- the temperature."
val dataset: TrainSet<String, Double> = listOf(
    TrainSetItem("What is the weather like at 9:00 am? $outputHint", 12.0),
    TrainSetItem("What is the weather like at 14:00 pm? $outputHint", 19.0),
    TrainSetItem("What is the weather like at 19:00 pm? $outputHint", 14.0),
)
```

`itemLabel` is ground truth — whatever your metric needs to judge a run. When success can be judged
from the agent's output alone (no ground-truth label), set `InputLabel` to `Unit` and use
`TrainSetItem(query, Unit)`.

## Step 3 — The metric

A metric is a function `(TrainSetItem<Input, InputLabel>, Output) -> Double` returning a score in
`[0.0, 1.0]`. It receives the dataset item (so you can read `item.itemLabel`) and the agent's output.

```kotlin
val metric: (TrainSetItem<String, Double>, String) -> Double = { item, answer ->
    if (answer.trim().toDoubleOrNull() == item.itemLabel) 1.0 else 0.0
}
```

The training session's `threshold` (default `0.9`) defines what counts as **solved**: a run with
`score >= threshold` is a success. Optimizers like BootstrapFewShot mine those successful runs.

## Step 4 — Serializers

`DatasetExecutionSerializers` turn your inputs and outputs into strings for human-readable logs and
the persisted artifacts. They do not affect scoring — they only make traces and stored runs
readable.

```kotlin
import ai.koog.agents.optimization.common.DatasetExecutionSerializers

val serializers = DatasetExecutionSerializers<String, String, Double>(
    serializeItem = { it.userQuery },   // how to render a TrainSetItem
    serializeOutput = { it },           // how to render an agent Output
)
```

## Step 5 — The training session

`trainingSession(...)` is the easy entry point. It wraps a `TrainingSession` with sensible library
defaults so you only supply what matters. (For full control, the `TrainingResources` / `TrainingSession`
constructors are available directly.)

```kotlin
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.training.trainingSession

val session = trainingSession(
    experimentName = ExperimentName(
        runId = "weather-fewshot-example",
        optimizerName = "BootstrapFewShot",
        agentName = "weather-agent",
    ),
    trackedAgent = agent,
    dataset = dataset,
    substepPromptExecutor = executor, // executor used for the optimizer's own LLM substeps
    metric = metric,
    serializers = serializers,
    // threshold = 0.9 by default
)
```

Arguments:

- `experimentName` — identifies the run (`runId`, `optimizerName`, `agentName`); used in logs and artifacts.
- `trackedAgent` — the `GraphAIAgent` being optimized.
- `dataset`, `metric`, `serializers` — from the previous steps.
- `substepPromptExecutor` — the executor the optimizer uses for its own LLM calls (proposing demos, reflection, etc.).
- `threshold` — score at or above which a run is "solved" (default `0.9`).

It fills in defaults you can otherwise tune: no failure analysis, single-attempt runs, default
truncation, no records file, and no token accounting. To count tokens, build the agent on a
consumption-capturing executor and pass a `consumptionCollector` (see
[optimizable-agents.md](optimizable-agents.md) and the support-router example for depth).

## Step 6 — Run the optimizer

Every optimizer follows the same two-phase contract: **train** to learn and persist an artifact, then
**load** to rebuild the agent with that artifact applied.

```kotlin
import ai.koog.agents.optimization.optimizers.fewShot.BootstrapFewShotOptimizer
import ai.koog.agents.optimization.utils.common.ResilientPath

val optimizer = BootstrapFewShotOptimizer<String, String, Double>(
    maxBootstrappedDemos = 4,
    maxRounds = 1,
    maxTotalDemos = 4,
    includeLabeledExamples = true,
    storagePath = ResilientPath("build/example-artifacts/weather_fewshot.json"),
    randomSeed = 42,
)

optimizer.train(session)                            // learns + saves an artifact JSON to storagePath
val optimized = optimizer.loadOptimizedAgent(agent) // rebuilds the agent with the artifact applied
val answer = optimized.run("What is the weather like right now? $outputHint")
println("Optimized agent answered: $answer")
```

`train` runs the agent over the dataset, scores each run with your metric, and turns the successful
runs into few-shot demonstrations — saved as a JSON artifact at `storagePath`. `loadOptimizedAgent`
reads that artifact and returns a new agent with the demonstrations baked into its prompt. The
original `agent` is left untouched.

## Run it

The runnable examples (`OptimizeWeatherAgentExample`, `OptimizeSupportRouterExample`,
`AuthorCustomOptimizerExample`) live in the `:koog-optimization-examples` module **of the
[source repository](https://github.com/JetBrains/koog-auto-agent-optimization)**; clone the repo to run them:

```bash
git clone https://github.com/JetBrains/koog-auto-agent-optimization.git
cd koog-auto-agent-optimization
```

The bundled example wires all of the above together. Run it with an LLM endpoint configured via
environment variables (see the example source under
`koog-optimization-examples/src/main/kotlin/ai/koog/agents/optimization/examples/` for the exact
variable names):

```bash
./gradlew :koog-optimization-examples:run
```

This executes `OptimizeWeatherAgentExample`, which builds the weather agent, optimizes it with
`BootstrapFewShotOptimizer`, and prints the optimized agent's answer.

## Next

- [Making agents optimizable](optimizable-agents.md) — `optimizableSubgraphWithTask`, strategy- vs
  subgraph-level optimization, and the multi-subgraph support-router walkthrough.
- [Optimizers](optimizers.md) — reference for the other three optimizers (GEPA, MIPROv2, ACE): what
  each learns, when to use it, and its constructor.
- [Writing a custom optimizer](custom-optimizers.md) — the `AgentOptimizer` SPI and the `StageScope`
  training DSL.
- [API reference](api/index.html) — full Dokka API docs.

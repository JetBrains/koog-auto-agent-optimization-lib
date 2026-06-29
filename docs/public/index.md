# koog-agents-optimization

This package allows you to automatically optimize [Koog](https://docs.koog.ai) agents. Give it a Koog `GraphAIAgent`, a labeled dataset, and a metric, and it improves the agent's prompts, instructions, and few-shot demonstrations using one of four optimizers — **BootstrapFewShot**, **MIPROv2**, **GEPA**, or **ACE**. BootstrapFewShot, MIPROv2, and GEPA optimizers are non-official Kotlin implementations of the the optimizers, implemented in [DSPy](https://dspy.ai). ACE is a non-official implementation of the _"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"_ [paper](https://arxiv.org/abs/2510.04618). The library is provider-agnostic: you bring any Koog `PromptExecutor` (OpenAI, Anthropic, an OpenAI-compatible proxy, etc.).

## Why

- **No prompt-tuning by hand.** Describe what "good" means with a metric; the optimizer searches for
  better prompts so you don't have to guess.
- **Works on any `GraphAIAgent`.** Optimize an agent you already built — no special framework, no
  rewrite.
- **Four complementary algorithms.** Bootstrap few-shot demos, jointly search instructions + demos,
  evolve instructions from failure traces, or curate an insight playbook. Pick what fits your task.
- **Learned artifacts are saved and reloadable.** Each `train()` persists a JSON artifact you can
  reload later — no need to re-optimize on every run.
- **The result is still a normal Koog agent.** `loadOptimizedAgent()` hands you back a plain
  `GraphAIAgent` you `run()` like any other.

## Install

!!! warning "Prerequisite: install the Koog fork"

    This library currently depends on a **fork of Koog** that adds optimizable-subgraph support
    (`ai.koog:koog-agents:0.8.0-SNAPSHOT`), not the upstream release. Publish the fork to your local
    Maven repository **before** building:

    ```bash
    git clone https://github.com/valemore/koog.git
    cd koog
    git checkout tags/stable-agent-optimization
    ./gradlew publishToMavenLocal
    ```

    This support is expected to land upstream in [JetBrains/koog](https://github.com/JetBrains/koog)
    eventually, after which the fork won't be needed.

Then add the dependencies. Both this library and the Koog fork resolve from `mavenLocal()`:

```kotlin
repositories { mavenLocal(); mavenCentral() }

dependencies {
    implementation("ai.koog:koog-agents-optimization:0.1.0-SNAPSHOT")
    implementation("ai.koog:koog-agents:0.8.0-SNAPSHOT") // the Koog fork (the agent you optimize is a Koog agent)
}
```

## 30-second example

Build an agent, optimize it with BootstrapFewShot, then run the optimized agent:

```kotlin
// 1. Build a Koog agent to optimize (any PromptExecutor works).
val agent: GraphAIAgent<String, String> = AIAgent.invokeGraphAgent(
    id = "weather-agent",
    promptExecutor = executor,
    systemPrompt = "You are a helpful assistant that reports the weather using the provided tools.",
    llmModel = OpenAIModels.Chat.GPT4oMini,
    temperature = 0.0,
    toolRegistry = ToolRegistry { tools(WeatherTools()) },
    strategy = singleRunStrategy(),
)

// 2. A labeled dataset + a metric in [0.0, 1.0] that scores each answer.
val dataset: TrainSet<String, Double> = listOf(
    TrainSetItem("What is the weather at 9:00? Answer with a single number.", 12.0),
    TrainSetItem("What is the weather at 14:00? Answer with a single number.", 19.0),
)
val metric: (TrainSetItem<String, Double>, String) -> Double = { item, answer ->
    if (answer.trim().toDoubleOrNull() == item.itemLabel) 1.0 else 0.0
}
val serializers = DatasetExecutionSerializers<String, String, Double>(
    serializeItem = { it.userQuery },
    serializeOutput = { it },
)

// 3. Wrap everything in a training session.
val session = trainingSession(
    experimentName = ExperimentName(
        runId = "weather-fewshot",
        optimizerName = "BootstrapFewShot",
        agentName = "weather-agent",
    ),
    trackedAgent = agent,
    dataset = dataset,
    substepPromptExecutor = executor,
    metric = metric,
    serializers = serializers,
)

// 4. Train, reload the optimized agent, and run it.
val optimizer = BootstrapFewShotOptimizer<String, String, Double>(
    maxBootstrappedDemos = 4,
    maxRounds = 1,
    maxTotalDemos = 4,
    includeLabeledExamples = true,
    storagePath = ResilientPath("build/artifacts/weather_fewshot.json"),
    randomSeed = 42,
)

optimizer.train(session)                              // learns + saves an artifact
val optimized = optimizer.loadOptimizedAgent(agent)   // rebuilds the agent with it applied
val answer = optimized.run("What is the weather right now? Answer with a single number.")
```

The full walkthrough — building `executor`, the dataset/metric/serializers, and `trainingSession`
explained step by step — is in [getting-started.md](getting-started.md).

## Next steps

- [getting-started.md](getting-started.md) — minimal end-to-end: BootstrapFewShot over the weather agent.
- [optimizable-agents.md](optimizable-agents.md) — make an agent optimizable with `optimizableSubgraphWithTask`; strategy- vs subgraph-level optimization.
- [optimizers.md](optimizers.md) — reference for all four optimizers: what each learns, when to use it, and constructors.
- [custom-optimizers.md](custom-optimizers.md) — author your own optimizer with the `@OptimizationExtensionApi` SPI.
- [API reference](api/index.html) — generated API docs.
- **Runnable examples** — live in the `:koog-optimization-examples` module of the
  [source repository](https://github.com/JetBrains/koog-auto-agent-optimization); clone it to run them.

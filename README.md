# koog-agents-optimization

Automatically optimize [Koog](https://docs.koog.ai) agents. Give it a Koog `GraphAIAgent`, a
labeled dataset, and a metric, and it improves the agent's prompts, instructions, and few-shot
demonstrations using one of four optimizers — **BootstrapFewShot**, **MIPROv2**, **GEPA**, or
**ACE**. BootstrapFewShot, MIPROv2, and GEPA are non-official Kotlin implementations of the
optimizers from [DSPy](https://dspy.ai); ACE is a non-official implementation of
_"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"_
([paper](https://arxiv.org/abs/2510.04618)). The library is provider-agnostic: you bring any Koog
`PromptExecutor` (OpenAI, Anthropic, an OpenAI-compatible proxy, etc.).

A project of the APR team from JetBrains Research.

## Why

- **No prompt-tuning by hand.** Describe what "good" means with a metric; the optimizer searches for
  better prompts so you don't have to guess.
- **Works on any `GraphAIAgent`.** Optimize an agent you already built — no special framework, no rewrite.
- **Four complementary algorithms.** Bootstrap few-shot demos, jointly search instructions + demos,
  evolve instructions from failure traces, or curate an insight playbook.
- **Learned artifacts are saved and reloadable.** Each `train()` persists a JSON artifact you can
  reload later — no need to re-optimize on every run.
- **The result is still a normal Koog agent.** `loadOptimizedAgent()` hands you back a plain
  `GraphAIAgent` you `run()` like any other.

## Install

> [!WARNING]
> **Prerequisite: install the Koog fork.** This library currently depends on a **fork of Koog** that
> adds optimizable-subgraph support (`ai.koog:koog-agents:0.8.0-SNAPSHOT`), not the upstream release.
> It is **not yet published to Maven Central.** Publish the fork — and this library — to your local
> Maven repository before building. This support is expected to land upstream in
> [JetBrains/koog](https://github.com/JetBrains/koog) eventually, after which the fork won't be needed.

```bash
# 1. The Koog fork
git clone https://github.com/valemore/koog.git
cd koog && git checkout tags/stable-agent-optimization && ./gradlew publishToMavenLocal && cd ..

# 2. This library
git clone https://github.com/JetBrains/koog-auto-agent-optimization-lib.git
cd koog-auto-agent-optimization-lib && ./gradlew :koog-agents-optimization:publishToMavenLocal
```

Then depend on it — both resolve from `mavenLocal()`:

```kotlin
repositories { mavenLocal(); mavenCentral() }

dependencies {
    implementation("ai.koog:koog-agents-optimization:0.1.0-SNAPSHOT")
    implementation("ai.koog:koog-agents:0.8.0-SNAPSHOT") // the Koog fork (the agent you optimize is a Koog agent)
}
```

## 30-second example

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

// 2. Wrap a dataset + metric in a training session, train, then reload and run the optimized agent.
val optimizer = BootstrapFewShotOptimizer<String, String, Double>(
    maxBootstrappedDemos = 4, maxRounds = 1, maxTotalDemos = 4,
    includeLabeledExamples = true,
    storagePath = ResilientPath("build/artifacts/weather_fewshot.json"),
    randomSeed = 42,
)
optimizer.train(session)                              // learns + saves an artifact
val optimized = optimizer.loadOptimizedAgent(agent)   // rebuilds the agent with it applied
val answer = optimized.run("What is the weather right now? Answer with a single number.")
```

The full step-by-step walkthrough (building `executor`, the dataset/metric/serializers, and
`trainingSession`) is in the
[Getting started guide](https://jetbrains.github.io/koog-auto-agent-optimization-lib/getting-started/).

## Runnable examples

Self-contained examples of optimizing agents and authoring custom optimizers live in the
[`koog-optimization-examples`](https://github.com/JetBrains/koog-auto-agent-optimization-lib/tree/main/koog-optimization-examples)
module:

```bash
./gradlew :koog-optimization-examples:run
```

## Documentation

- [Overview](https://jetbrains.github.io/koog-auto-agent-optimization-lib/)
- [Getting started](https://jetbrains.github.io/koog-auto-agent-optimization-lib/getting-started/)
- [Making your agent optimizable](https://jetbrains.github.io/koog-auto-agent-optimization-lib/optimizable-agents/)
- [Optimizers](https://jetbrains.github.io/koog-auto-agent-optimization-lib/optimizers/)
- [Custom optimizers](https://jetbrains.github.io/koog-auto-agent-optimization-lib/custom-optimizers/)
- [API reference](https://jetbrains.github.io/koog-auto-agent-optimization-lib/api/)

## License

[Apache License 2.0](LICENSE).

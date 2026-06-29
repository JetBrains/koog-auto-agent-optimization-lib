# Authoring a custom optimizer

The four built-in optimizers — [BootstrapFewShot, MIPROv2, GEPA, and ACE](optimizers.md) — cover most
prompt-optimization needs. However, you may easily write your own optimizer when the built-ins don't express your intent: a different search algorithm, a custom artifact you persist and reload, a
novel use of failure traces, or a learning loop that installs your own Koog feature. API for writing a new optimizer will later on become more stable and will not require opt-in.

A custom optimizer plugs into the same two-phase flow as the built-ins, so the rest of your code —
dataset, metric, [`trainingSession(...)`](getting-started.md), `train` / `loadOptimizedAgent` — stays
identical.

## The SPI

An optimizer implements `AgentOptimizer<Input, Output, InputLabel>` (package
`ai.koog.agents.optimization.optimizers`):

```kotlin
public interface AgentOptimizer<Input, Output, InputLabel> {
    public suspend fun train(session: TrainingSession<Input, Output, InputLabel>): TrainingResult
    public fun loadOptimizedAgent(baseAgent: GraphAIAgent<Input, Output>): GraphAIAgent<Input, Output>
}
```

The contract is two-phase:

- **`train(session)`** runs the agent over your dataset *through the session* and saves whatever it
  learns (typically a JSON artifact on disk). It returns a `TrainingResult` describing the run.
- **`loadOptimizedAgent(baseAgent)`** rebuilds the agent with that learning applied — usually by
  installing a Koog feature that injects the learned instructions, demonstrations, or playbook — and
  returns a plain `GraphAIAgent` you `run()` like any other.

The caller constructs and passes the `TrainingSession` (see [getting-started.md](getting-started.md)
for how `trainingSession(...)` builds one). Your optimizer never creates the session itself — it only
*uses* the one it's handed.

```kotlin
val session = trainingSession(/* experimentName, agent, dataset, metric, serializers, ... */)
optimizer.train(session)                              // learns + persists
val optimized = optimizer.loadOptimizedAgent(agent)   // rebuilds with learning applied
val answer = optimized.run(input)
```

## The opt-in marker

Authoring an optimizer touches the training DSL, which is gated behind `@OptimizationExtensionApi`
(package `ai.koog.agents.optimization.annotations`). This marker covers `AgentOptimizer`, the
`TrainingSession`, and the `StageScope` DSL — the advanced *extension* surface for people building
optimizers, not the everyday optimize-with-a-built-in flow.

Opt in at the top of your file:

```kotlin
@file:OptIn(OptimizationExtensionApi::class)
```

The everyday flow — building an optimizable agent and optimizing it with a built-in — uses only
plain-public API and needs no opt-in. You reach for `@OptimizationExtensionApi` only when you write
the optimizer.

## The smallest example

`RunOnceOptimizer` is the minimal complete implementation: it runs the agent once over each dataset
item (recording metrics) and applies no transformation. It exists to show the SPI shape.

```kotlin
@file:OptIn(OptimizationExtensionApi::class)

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.annotations.OptimizationExtensionApi
import ai.koog.agents.optimization.optimizers.AgentOptimizer
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.records.TrainingResult

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
```

### Inside `session.use { }`

`session.use(stagesTotal = N) { ... }` opens the session and runs your training body. `stagesTotal` is
how many stages your algorithm has (one pass over the data, or several search rounds). Inside the
block `this` is a `StageScope`, which exposes:

- **`dataset`** — the `TrainSet` you were configured with, ready to iterate.
- **`runAgent(item)`** — run the tracked agent on a dataset item, recording its metric; the
  throwing variant is **`runAgentOrThrow(item, agent)`** (pass a specific agent, e.g. one with extra
  features installed).
- **`runStage("name") { }`** / **`runStageOrThrow("name") { }`** — group work into a named stage for
  logging and progress reporting.
- **`executePromptOrThrow { }`** / **`executePromptStructuredOrThrow<T> { }`** — call the optimizer's
  substep LLM (a meta-LLM for reflection, proposal, scoring, etc.); the structured variant decodes
  the response into `T`.
- **`logAction { }`** — emit a structured log entry for the training timeline.

`runAgent` returns enough to inspect each run's outcome, which is where richer optimizers collect what
they learn.

## A slightly richer optimizer (sketch)

The pattern every prompt-learning optimizer follows: run each item, collect something from the
*successful* runs, fold that into an artifact you save in `train`, then install a feature that applies
the artifact in `loadOptimizedAgent`.

> **Sketch, not copy-paste code.** This illustrates the shape only. Artifact types, feature installers,
> and trace-collection helpers are covered in the deep-dive guide linked below — don't rely on helper
> names from this sketch.

```kotlin
class MyOptimizer<Input, Output, InputLabel>(
    private val storagePath: ResilientPath,
) : AgentOptimizer<Input, Output, InputLabel> {

    private var learned: MyArtifact? = null

    override suspend fun train(
        session: TrainingSession<Input, Output, InputLabel>,
    ): TrainingResult = session.use(stagesTotal = 1) {
        runStage("collect") {
            val collected = mutableListOf<Something>()
            dataset.forEach { item ->
                val result = runAgent(item)
                if (result.isSuccessful) {
                    // pull what you want to learn from a successful run
                    collected += extractSomething(result)
                }
            }
            // optionally refine with a meta-LLM substep:
            // val refined = executePromptStructuredOrThrow<MyArtifact> { /* build prompt */ }
            val artifact = buildArtifact(collected)
            saveArtifact(storagePath, artifact)   // persist for reload
            learned = artifact
            logAction { "collected ${collected.size} examples" }
        }
    }

    override fun loadOptimizedAgent(
        baseAgent: GraphAIAgent<Input, Output>,
    ): GraphAIAgent<Input, Output> {
        val artifact = learned ?: loadArtifact(storagePath)  // reload across processes
        return baseAgent.copyWith(installFeatures = {
            baseAgent.installFeatures(this)
            // install YOUR feature that applies `artifact` at runtime
        })
    }
}
```

The built-in optimizers are the best worked reference — read their source under
`koog-agents-optimization/src/main/kotlin/ai/koog/agents/optimization/optimizers/`
(`fewShot/`, `mipro/`, `gepa/`, `ace/`). For a full treatment of `OptimizationArtifact`, the feature
mechanism (`copyWith` + `installFeatures`), and trace collection from successful runs, see the
[optimizer-authoring guide](https://github.com/JetBrains/koog-auto-agent-optimization/blob/main/docs/dev/OPTIMIZER_AUTHORING_GUIDE.md)
in the repository.

## Next

- [Optimizers](optimizers.md) — the four built-ins you'll usually reach for first.
- [Getting started](getting-started.md) — the dataset / metric / serializers / `trainingSession`
  setup your optimizer consumes.
- [API reference](api/index.html) — full signatures.

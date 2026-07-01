# Optimizers reference

The library ships four optimizers. They all implement the same interface, so once
you know how to drive one, you know how to drive all of them:

```kotlin
public interface AgentOptimizer<Input, Output, InputLabel> {
    public suspend fun train(session: TrainingSession<Input, Output, InputLabel>): TrainingResult
    public fun loadOptimizedAgent(baseAgent: GraphAIAgent<Input, Output>): GraphAIAgent<Input, Output>
}
```

Each optimizer learns from your training runs and writes a JSON artifact to its
`storagePath` (a `ResilientPath`). Usage is always two-phase: `train` learns and
persists the artifact, then `loadOptimizedAgent` rebuilds your agent with that
artifact applied.

```kotlin
optimizer.train(session)                              // learns + saves an artifact to storagePath
val optimized = optimizer.loadOptimizedAgent(agent)   // rebuilds the agent with the artifact applied
val answer = optimized.run(input)
```

This page assumes you already have a `TrainingSession`. Set one up as shown in
[Getting Started](getting-started.md) (build the agent, dataset, metric,
serializers, and call `trainingSession(...)`), then pick an optimizer below.

!!! note "Attribution & provenance"

    None of these optimizers are original work. They are **non-official re-implementations** of
    algorithms published by others, ported to Koog. Our implementations **may diverge** from the
    original references — due to technical constraints of the Koog runtime, differences in the agent
    model, or pragmatic simplifications — but we have tried to preserve the core semantics of each.
    For the authoritative description of any algorithm, consult its original source below.

    **ACE optimizer deviates the most.** Its current implementation differs from the original
    algorithm in ways that can affect results. Treat it, and other optimizers, as
    early, and validate on your own task before relying on them. Closing the gap is future work.

    | Optimizer | Original authors | Reference |
    |---|---|---|
    | **BootstrapFewShot** | Omar Khattab et al. (DSPy) | [DSPy `BootstrapFewShot`](https://dspy.ai/api/optimizers/BootstrapFewShot/) |
    | **MIPROv2** | Krista Opsahl-Ong et al. | [DSPy `MIPROv2`](https://dspy.ai/api/optimizers/MIPROv2/) · [arXiv:2406.11695](https://arxiv.org/abs/2406.11695) |
    | **GEPA** | Lakshya A. Agrawal et al. | [DSPy GEPA tutorial](https://dspy.ai/tutorials/gepa_ai_program/) · [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) |
    | **ACE** | Qizheng Zhang et al. | [arXiv:2510.04618](https://arxiv.org/abs/2510.04618) |

## Choosing an optimizer

| Optimizer | What it learns | Best for | Needs ground-truth label? |
|---|---|---|---|
| **BootstrapFewShot** | few-shot demonstrations mined from successful runs | quick wins when you already have good runs to mine | optional — judged by the metric |
| **MIPROv2** | instructions + demos, via proposal and search | when you want both instruction tuning and demos | optional — judged by the metric |
| **GEPA** | module instructions, via reflective evolution over a Pareto frontier | ambiguous or underspecified instructions; instruction-only | yes — via a `labelExtractor` (string label) |
| **ACE** | an evolving "playbook" of insights appended to the system prompt | accumulating reusable strategy and knowledge | yes — via a `labelExtractor` (string label) |

For models, the examples below use `OpenAIModels.Chat.GPT4oMini` as the example
`LLModel`; substitute any Koog model. `storagePath` is always a `ResilientPath`,
e.g. `ResilientPath("build/artifacts/foo.json")`.

## BootstrapFewShot

Runs each training item up to `maxRounds` times, mines the successful runs into
few-shot demonstrations, and applies them to your agent. The simplest optimizer:
no extra LLM calls beyond running your own agent.

```kotlin
BootstrapFewShotOptimizer<Input, Output, InputLabel>(
    maxBootstrappedDemos: Int,
    maxRounds: Int,
    maxTotalDemos: Int,
    includeLabeledExamples: Boolean,
    storagePath: ResilientPath,
    randomSeed: Long,
)
```

Key params:

- `maxBootstrappedDemos` — how many successful traces to collect and use as demos.
- `maxRounds` — retry budget per training item while hunting for a success.
- `maxTotalDemos` — total demo slots. If bootstrapped demos don't fill them and
  `includeLabeledExamples` is `true`, the remaining slots are filled from labeled
  dataset items.
- `includeLabeledExamples` — whether to backfill empty demo slots with labeled
  examples.

```kotlin
import ai.koog.agents.optimization.optimizers.fewShot.BootstrapFewShotOptimizer
import ai.koog.agents.optimization.utils.common.ResilientPath

val optimizer = BootstrapFewShotOptimizer<String, String, Double>(
    maxBootstrappedDemos = 4,
    maxRounds = 3,
    maxTotalDemos = 8,
    includeLabeledExamples = true,
    storagePath = ResilientPath("build/artifacts/fewshot.json"),
    randomSeed = 42L,
)

optimizer.train(session)
val optimized = optimizer.loadOptimizedAgent(agent)
val answer = optimized.run("What is the weather at 9:00? Answer with a single number.")
```

## MIPROv2

Jointly proposes and searches over instructions and demos. A meta-LLM proposes
candidate instructions; demos are bootstrapped; candidates are scored and the best
combination is kept.

```kotlin
MIPROv2Optimizer<Input, Output, InputLabel>(
    metaModel: LLModel,
    autoMode: AutoRunMode = AutoRunMode.LIGHT,    // LIGHT | MEDIUM | HEAVY
    numCandidatesOverride: Int? = null,
    numTrialsOverride: Int? = null,
    maxBootstrappedDemos: Int = 4,
    maxTotalDemos: Int = 8,
    includeLabeledExamples: Boolean = true,
    proposerConfig: InstructionProposerConfig = InstructionProposerConfig(),
    storagePath: ResilientPath,                   // named arg, has no default
    randomSeed: Long = 42,
    parallelism: Int = 1,
)
```

Key params:

- `metaModel` — the LLM used for instruction proposal and dataset summarization.
- `autoMode` — a preset budget: `LIGHT`, `MEDIUM`, or `HEAVY`. Higher modes try
  more candidates and trials (and cost more). Start with `LIGHT`.
- `numCandidatesOverride` / `numTrialsOverride` — manual overrides for the search
  budget when the preset doesn't fit.
- `storagePath` — note this has **no default**, so pass it as a named argument
  even though it sits after defaulted params.
- `parallelism` — concurrent runs during bootstrapping.

```kotlin
import ai.koog.agents.optimization.optimizers.mipro.MIPROv2Optimizer
import ai.koog.agents.optimization.optimizers.mipro.AutoRunMode
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.prompt.executor.clients.openai.OpenAIModels

val optimizer = MIPROv2Optimizer<String, String, Double>(
    metaModel = OpenAIModels.Chat.GPT4oMini,
    autoMode = AutoRunMode.LIGHT,
    storagePath = ResilientPath("build/artifacts/mipro.json"),
)

optimizer.train(session)
val optimized = optimizer.loadOptimizedAgent(agent)
val answer = optimized.run("What is the weather at 14:00? Answer with a single number.")
```

## GEPA

Instruction-only. GEPA evolves module instructions with reflective LLM feedback
over failure traces, maintaining a Pareto frontier of candidates and optionally
merging complementary ones (crossover).

```kotlin
GEPAOptimizer<Input, Output, InputLabel>(
    reflectionModel: LLModel,
    storagePath: ResilientPath,
    maxIterations: Int,
    minibatchSize: Int,
    componentSelection: ComponentSelection,   // ROUND_ROBIN | ALL_AT_ONCE
    enableCrossover: Boolean,
    maxMergeInvocations: Int = 5,
    randomSeed: Long = 42,
    labelExtractor: (TrainSetItem<Input, InputLabel>) -> String,
)
```

Key params:

- `reflectionModel` — the LLM that reflects on traces and proposes new instructions.
- `maxIterations` / `minibatchSize` — the evolution budget: how many iterations,
  and how many items are sampled per iteration for reflection.
- `componentSelection` — which optimizable modules to update each iteration.
  `ROUND_ROBIN` updates one module at a time in rotation; `ALL_AT_ONCE` updates
  every module each iteration.
- `enableCrossover` / `maxMergeInvocations` — whether to merge complementary
  candidates, and the cap on merge attempts per run.
- `labelExtractor` — converts a dataset item's label into the string the
  reflection LM sees as the expected answer. Required.

```kotlin
import ai.koog.agents.optimization.optimizers.gepa.GEPAOptimizer
import ai.koog.agents.optimization.optimizers.gepa.ComponentSelection
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.prompt.executor.clients.openai.OpenAIModels

val optimizer = GEPAOptimizer<String, String, Double>(
    reflectionModel = OpenAIModels.Chat.GPT4oMini,
    storagePath = ResilientPath("build/artifacts/gepa.json"),
    maxIterations = 10,
    minibatchSize = 5,
    componentSelection = ComponentSelection.ROUND_ROBIN,
    enableCrossover = true,
    labelExtractor = { item -> item.itemLabel.toString() },
)

optimizer.train(session)
val optimized = optimizer.loadOptimizedAgent(agent)
val answer = optimized.run("Route this support ticket: my invoice is wrong")
```

## ACE

Curates an evolving "playbook" of bullet-point insights that is appended to the
system prompt. Useful when you want the agent to accumulate reusable strategy and
knowledge across runs rather than just demos or a reworded instruction.

```kotlin
ACEOptimizer<Input, Output, InputLabel>(
    playbookStoragePath: ResilientPath,
    onExistingPlaybook: OnExistingPlaybookAction,  // OVERWRITE | OPTIMIZE_FURTHER | THROW
    reflectorModel: LLModel,
    curatorModel: LLModel,
    labelExtractor: (datasetItem: TrainSetItem<Input, InputLabel>) -> String,
)
```

Key params:

- `playbookStoragePath` — where the playbook artifact is read from and written to.
- `onExistingPlaybook` — what to do if a playbook already exists at that path:
  `OVERWRITE` deletes it and starts fresh, `OPTIMIZE_FURTHER` loads it and keeps
  refining, `THROW` refuses to start to avoid clobbering prior results.
- `reflectorModel` — diagnoses trajectories into insights.
- `curatorModel` — turns those insights into playbook deltas (additions/edits).
- `labelExtractor` — extracts the ground-truth label string for an item, fed to
  the reflector. Required.

```kotlin
import ai.koog.agents.optimization.optimizers.ace.ACEOptimizer
import ai.koog.agents.optimization.optimizers.ace.OnExistingPlaybookAction
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.prompt.executor.clients.openai.OpenAIModels

val optimizer = ACEOptimizer<String, String, Double>(
    playbookStoragePath = ResilientPath("build/artifacts/ace-playbook.json"),
    onExistingPlaybook = OnExistingPlaybookAction.OVERWRITE,
    reflectorModel = OpenAIModels.Chat.GPT4oMini,
    curatorModel = OpenAIModels.Chat.GPT4oMini,
    labelExtractor = { item -> item.itemLabel.toString() },
)

optimizer.train(session)
val optimized = optimizer.loadOptimizedAgent(agent)
val answer = optimized.run("Route this support ticket: my invoice is wrong")
```

## Artifacts & reuse

`train` writes the learned artifact to the optimizer's `storagePath`
(`playbookStoragePath` for ACE). `loadOptimizedAgent` reads that artifact back and
applies it to your base agent.

The two steps are decoupled by the file on disk. If the artifact already exists —
for example from a previous training run, or one you committed — you can skip
`train` entirely and just call `loadOptimizedAgent(agent)` to get the optimized
agent. This is how you deploy an optimized agent without re-running optimization.

## Next

- [Making agents optimizable](optimizable-agents.md) — declare optimizable
  modules with `optimizableSubgraphWithTask`, and strategy- vs subgraph-level
  optimization.
- [Writing a custom optimizer](custom-optimizers.md) — implement the
  `AgentOptimizer` interface with the `StageScope` training DSL.
- [API reference](api/index.html) — full Dokka docs.

# Making your agent optimizable

Optimizers can only tune the parts of your agent that you expose to them. This page shows
how to declare that surface in your Koog strategy so the optimizers have something to work on.

## What gets optimized

An optimizer sees your agent as a set of tunable **modules** — the same idea as a
[DSPy module](https://dspy.ai): a named unit with an instruction (and optional demonstrations)
that the optimizer is free to rewrite. There are two levels:

- **The strategy system prompt** — the `__strategy__` module. This is the system prompt you
  pass when you build the agent. Every agent has it, so even a plain `singleRunStrategy()` agent
  (like the one in [getting-started.md](getting-started.md)) is already optimizable: an optimizer
  can rewrite its system prompt, attach few-shot demonstrations, or append a learned playbook.

- **Optimizable subgraphs** — finer-grained, per-step modules inside your strategy (the closest
  analogue to a DSPy module). When a single system prompt is too coarse — your agent reasons in
  stages, and each stage needs its own focused instruction — you declare those stages as optimizable
  subgraphs. The optimizer then tunes each one independently.

GEPA, MIPROv2, and BootstrapFewShot optimize the strategy system prompt **plus** every optimizable
subgraph you declare. **ACE currently optimizes only the strategy-level context** (it curates a
playbook for the `__strategy__` module); per-subgraph optimization for ACE is a work in progress.

You don't need subgraphs to start. Reach for them when one prompt can't capture everything the
agent must get right.

## `optimizableSubgraphWithTask`

Declare an optimizable subgraph inside a `strategy { }` block with property delegation (`by`).
It is the optimization-aware counterpart of Koog's `subgraphWithTask`: a subgraph that runs a
single task whose **instruction** and **few-shot demonstrations** an optimizer can replace without
you touching the graph.

```kotlin
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask

val analyze by optimizableSubgraphWithTask<String, String>(
    optimizableInstruction = "Look at the customer message.",
    freshHistory = true,
) { instruction, message -> "Customer message:\n$message" }
```

The key parameters:

- **`optimizableInstruction`** — the starting instruction. This is what runs before any
  optimization; an optimizer may replace it with a better one. Start vague or start with your best
  guess — either way the optimizer evolves it from here.
- **`name`** — the module's name, defaulting to the delegated property name (`analyze` above).
  It must be **unique within the strategy**: the optimization artifact uses it as a lookup key.
- **`freshHistory`** — when `true`, the subgraph starts with an empty conversation history. The
  resolved instruction is placed as a fresh system message and the task query follows it. This is a
  new parameter we added to Koog's `subgraph(...)` in the [fork](index.md) (expected to land
  upstream); it lets you (a) **isolate** a subgraph's prompt from the rest of the conversation,
  (b) give each subgraph its **own clean, independently optimizable system message**, and (c) keep
  prompts focused and cheaper for classification/routing-style steps that shouldn't inherit prior
  chatter.
- **`defineTask { instruction, input -> ... }`** — builds the **user message** for this subgraph
  from the (resolved) instruction and the subgraph input, returning a `String`. The `instruction`
  argument already reflects any optimization; you usually compose your query from `input` and let
  the system message carry the instruction.

Few-shot knobs `fewShotPromptType` and `demonstrationFormat` control how learned demonstrations are
inserted into the prompt (as message history vs. a single string, compact vs. detailed); leave them
`null` to inherit sensible defaults. As a brief aside, the tool/LLM knobs — `toolSelectionStrategy`,
`llmModel`, `llmParams`, `runMode`, `assistantResponseRepeatMax`, `responseProcessor` — let a
subgraph use its own tools or model; they behave exactly as in a regular `subgraphWithTask`.

## A worked example: a support-ticket router

This mirrors the runnable `OptimizeSupportRouterExample`. The agent is a two-stage router with two
optimizable subgraphs:

```
nodeStart(message) -> remember -> [analyze] -> pack -> [route] -> nodeFinish(code)
```

Both subgraphs start from **deliberately vague** instructions that say nothing about the allowed
routing codes or the required output format.

```kotlin
import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.dsl.builder.node
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask

private val origMessageKey = createStorageKey<String>("orig-support-message")

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
```

The dataset labels each ticket with the routing code it should produce, and the metric is an exact
match on that code:

```kotlin
val dataset: TrainSet<String, String> = listOf(
    TrainSetItem("I was charged twice for my subscription this month.", "BILLING"),
    TrainSetItem("The app crashes every time I open the settings page.", "TECH"),
    TrainSetItem("Do you have a referral program for existing customers?", "OTHER"),
    // ...
)
val metric: (TrainSetItem<String, String>, String) -> Double = { item, answer ->
    if (answer.trim() == item.itemLabel) 1.0 else 0.0
}
```

**The payoff.** With the vague baseline instructions the agent has no idea the answer must be a bare
code, so it replies with prose like "This should go to billing." and scores ~0 on the exact-match
metric. An optimizer — GEPA here — reflects on the failure traces (which carry the expected labels),
and evolves **both** subgraph instructions until `analyze` extracts the right signals and `route`
emits exactly `BILLING` / `TECH` / `OTHER`. The learned instructions are saved to the optimization
artifact and reapplied when you load the optimized agent.

You wire the optimizer to this agent exactly as in [getting-started.md](getting-started.md): build a
`trainingSession`, call `optimizer.train(session)`, then `optimizer.loadOptimizedAgent(agent)`. See
[optimizers.md](optimizers.md) for how to pick and configure GEPA (or any other optimizer); we don't
repeat that here.

## Tips and gotchas

- **Names are artifact keys.** Each optimizable subgraph must have a unique name within its
  strategy. The artifact maps instructions and demonstrations to subgraphs by name, so duplicates
  silently cause the wrong instruction to land on the wrong subgraph. Let names default to the
  property name and keep those distinct, or pass an explicit `name`.
- **`freshHistory = true` suits classification-style subtasks.** When a subgraph should decide
  something from its input alone — classify, route, extract — a fresh history keeps it focused and
  makes its prompt cleanly optimizable. Leave it `false` when the step genuinely needs the prior
  conversation.
- **The two levels compose.** Strategy-level and subgraph-level optimization are not exclusive —
  an optimizer tunes the `__strategy__` system prompt and every subgraph in the same run. Start with
  just the strategy prompt; add optimizable subgraphs as you find steps that need their own
  instructions.

## Next

- [optimizers.md](optimizers.md) — choose and configure an optimizer (GEPA, MIPROv2,
  BootstrapFewShot, ACE).
- [getting-started.md](getting-started.md) — the minimal end-to-end loop: build agent, dataset,
  metric, `trainingSession`, train, and run the optimized agent.

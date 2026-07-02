@file:Suppress("unused")

package ai.koog.agents.optimization.training.dsl

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.annotations.OptimizationExtensionApi
import ai.koog.agents.optimization.common.defaultExperimentsJson
import ai.koog.agents.optimization.common.retries.RetryPolicy
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.CapturingPromptExecutor
import ai.koog.agents.optimization.training.PrematureExecutionStopDecision
import ai.koog.agents.optimization.training.metrics.MetricsMap
import ai.koog.agents.optimization.training.metrics.impl.Metric
import ai.koog.agents.optimization.training.metrics.standardStageMetrics
import ai.koog.agents.optimization.training.records.PromptExecutionRecord
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.structures.StageFailedException
import ai.koog.agents.optimization.utils.llm.executeStructuredOrThrow
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.encodeToJsonElement

/**
 * The single unified scope for the training DSL.
 *
 * Provides all operations: nested stages, agent runs, prompt executions, action logging,
 * and dataset iteration.
 *
 * Stages can be nested to arbitrary depth via [runStage].
 * Leaf operations ([runAgent], [executePrompt]) create leaf records.
 *
 * **When to use stages:** stages are primarily for exception handling, metric aggregation,
 * and structured logging. If a piece of code doesn't need any of these, it doesn't need
 * to be wrapped in a stage. A good "smell" that you're using something wrong is if your
 * stage only has a single substage inside: while technically okay, you will have unnecessary
 * metrics aggregation in the records tree, polluting it.
 */
@OptimizationExtensionApi
public interface StageScope<Input, Output, InputLabel> {
    /** The agent being trained, available as the default target of [runAgent]. */
    public val trackedAgent: GraphAIAgent<Input, Output>

    /** The full training dataset, used as the default for [iterateDataset]. */
    public val dataset: TrainSet<Input, InputLabel>

    // ===================================================================
    // Nested stage (tree node)
    // ===================================================================

    /**
     * Runs a named substage. Creates a [StageRecord] as a substage of the current stage.
     *
     * Exceptions from the block are caught, recorded as a stage failure, and wrapped in
     * [StageFailedException]. Use `.getOrThrow()` to re-throw.
     *
     * @param name The display name of this substage.
     * @param substagesTotal How many substages does this stage have. Used to calculate ETC and progress.
     * @param metrics The metrics preset for the substage. Defaults to [standardStageMetrics];
     *   pass a different preset (e.g. [ai.koog.agents.optimization.training.metrics.datasetIterationMetrics])
     *   when the stage needs an extended set. Follow the `standardStageMetrics() + extras`
     *   convention so cluster progress and consumption tracking keep working. Pass a fresh
     *   [MetricsMap] per call (factory or `+`); the map is mutated in place during the
     *   stage's lifetime, so sharing instances across stages is unsafe.
     */
    public suspend fun <T> runStage(
        name: String,
        substagesTotal: Int? = null,
        metrics: MetricsMap = standardStageMetrics(),
        block: suspend StageScope<Input, Output, InputLabel>.() -> T,
    ): Result<T>

    // ===================================================================
    // Agent run (leaf)
    // ===================================================================

    /**
     * Runs the agent on the given item.
     *
     * Transient-failure retries are applied automatically per the session's retry configuration —
     * each call returns with either a successful [CompletedAgentRun] or a *non-transient* failure
     * (the inner retry layer has already absorbed transient blips before the result is returned).
     * Per-attempt details for retried operations live in the resulting record's `previousAttempts`.
     *
     * @param retryPolicy override for the session-default retry policy. `null` (default) =
     *   use `resources.retriesPolicy`. [RetryPolicy.None] = disable transient retries for this call
     *   (useful in tests or where deterministic single-attempt behavior is required).
     */
    public suspend fun runAgent(
        item: TrainSetItem<Input, InputLabel>,
        agentToRun: GraphAIAgent<Input, Output> = trackedAgent,
        retryPolicy: RetryPolicy? = null,
    ): Result<CompletedAgentRun<Output>>

    /**
     * Re-samples the agent: runs it up to [maxAttempts] times, returning the first attempt whose
     * outcome satisfies [until] (default [RunUntil.SOLVED]). The intended use is intentional
     * re-sampling — e.g. bootstrap demonstration generation — not transient-failure recovery.
     *
     * Transient-failure retries are applied automatically *inside* each call to [runAgent] per
     * the session's retry configuration. Each re-sample attempt therefore completes with either a
     * success or a *non-transient* failure (the inner retry layer has already absorbed transient
     * blips before this function sees the result). To opt out of inner transient retries for a
     * specific call — e.g., to count transient-failed attempts as full samples — pass
     * [innerRetryPolicy] = [RetryPolicy.None] (which the inner [runAgent] will use instead of
     * the session-default policy).
     *
     * @param innerRetryPolicy override for the transient-retry policy applied inside each
     *   re-sample attempt. `null` (default) = use the session's retry configuration policy.
     *   [RetryPolicy.None] = disable inner transient retries for this call.
     */
    public suspend fun runAgentWithRetries(
        item: TrainSetItem<Input, InputLabel>,
        maxAttempts: Int,
        until: RunUntil = RunUntil.SOLVED,
        agentToRun: GraphAIAgent<Input, Output> = trackedAgent,
        innerRetryPolicy: RetryPolicy? = null,
    ): CompletedAgentRun<Output>?

    /**
     * Runs the agent up to [maxAttempts] times, creating a fresh agent per attempt via [agentProvider].
     *
     * [innerRetryPolicy] mirrors the overload above — overrides the inner transient-retry policy
     * applied inside each [runAgent] call.
     */
    public suspend fun <RunData> runAgentWithRetries(
        item: TrainSetItem<Input, InputLabel>,
        maxAttempts: Int,
        until: RunUntil = RunUntil.SOLVED,
        innerRetryPolicy: RetryPolicy? = null,
        agentProvider: suspend () -> PreparedAgentRun<Input, Output, RunData>,
    ): MatchedAgentRun<Output, RunData>?

    // ===================================================================
    // Prompt execution (leaf)
    // ===================================================================

    /**
     * Helper function that creates a consumption-capturing executor, runs [execute] with it,
     * records the result as a [PromptExecutionRecord], and recomputes metrics.
     * Notice that this function only produces one record, so you should only make a single LLM call
     * inside the [execute] block.
     *
     * Transient-failure retries are applied automatically per the session's retry configuration —
     * each call returns with either a successful result or a *non-transient* failure (the inner
     * retry layer has already absorbed transient blips). Per-attempt detail for retried operations
     * lives in the resulting record's `previousAttempts`.
     *
     * Generally, it's better to use more clean variants like [executePrompt] or [executePromptStructured].
     *
     * @param retryPolicy override for the session-default retry policy. `null` (default) =
     *   use the session-resolved policy. [RetryPolicy.None] = disable transient retries for this call
     *   (useful in tests or where deterministic single-attempt behavior is required).
     */
    public suspend fun <T> executeWithTrackedPromptExecutor(
        name: String,
        retryPolicy: RetryPolicy? = null,
        execute: suspend CapturingPromptExecutor.() -> T,
    ): Result<T>

    // ===================================================================
    // Logging and custom metrics
    // ===================================================================

    /** Builds an action log via the [ActionLogBuilder] DSL and stores it on the current stage record. */
    public fun logAction(json: Json = defaultExperimentsJson, builder: ActionLogBuilder.() -> Unit)

    /**
     * Attaches a custom metric to the records tree of this stage. This method has two uses:
     *
     * 1. It registers this metric at the current stage, meaning it will try to recalculate it from its children.
     * 2. It records the metric value in the current stage, such that the parents can use it for recalculation.
     *
     * An example of the intended usage of this method:
     *
     * ```kotlin
     * runStage("parent") {
     *   recordCustomMetric(MyMetric(stubValue)) // registers for tracking
     *   runStage("child") {
     *     recordCustomMetric(MyMetric(childValue)) // updates the real value
     *   }
     * }
     * ```
     */
    public fun recordCustomMetric(metric: Metric)

    /** Sets some additional data in this stage's record. Overwrites if additional data was already added. */
    public fun setAdditionalDataJson(additionalData: JsonElement)

    // ===================================================================
    // Dataset iteration (special stage)
    // ===================================================================

    /**
     * Iterates over the dataset, processing each item as a substage.
     * Uses dataset-specific metrics (solved rate, item counts, agent run stats).
     *
     * @param name A custom name for the dataset iteration stage. Useful if you need multiple dataset iterations.
     * @param dataset The dataset to iterate over. Default: the entire training dataset.
     * @param customMetricsToRecord A list of custom metrics to aggregate from dataset item stages.
     * @param failureRateThreshold If the failure rate exceeds this value, the entire dataset iteration will be
     *   marked as failed (but it will NOT throw).
     * @param earlyStop A lambda that checks if the dataset iteration can be stopped prematurely.
     */
    public suspend fun iterateDataset(
        name: String = "Dataset iteration",
        dataset: TrainSet<Input, InputLabel> = this@StageScope.dataset,
        customMetricsToRecord: List<Metric>? = null,
        failureRateThreshold: Double = 0.9,
        earlyStop: (TrainSetItem<Input, InputLabel>) -> PrematureExecutionStopDecision = {
            PrematureExecutionStopDecision(false) { "Unreachable: this training never stops prematurely" }
        },
        processItem: suspend StageScope<Input, Output, InputLabel>.(TrainSetItem<Input, InputLabel>) -> Unit,
    ): StageRecord
}

// ===================================================================
// Prompt execution variants
// ===================================================================

/**
 * Executes an LLM prompt with consumption tracking.
 * Creates a PromptExecutionRecord as a leaf record, recording elapsed time and consumption.
 * On success, returns the raw LLM response messages.
 *
 * Transient-failure retries are applied automatically per the session's retry policy — see
 * [StageScope.executeWithTrackedPromptExecutor].
 *
 * @param retryPolicy override for the session-default retry policy. `null` (default) = use the
 *   session-resolved policy. [RetryPolicy.None] = disable transient retries for this call.
 */
@OptimizationExtensionApi
public suspend fun StageScope<*, *, *>.executePrompt(
    prompt: Prompt, model: LLModel,
    retryPolicy: RetryPolicy? = null,
): Result<List<Message.Response>> =
    executeWithTrackedPromptExecutor(name = prompt.id, retryPolicy = retryPolicy) {
        execute(prompt, model)
    }

/**
 * Executes a structured LLM prompt with consumption tracking.
 * Creates a PromptExecutionRecord and returns the deserialized structure.
 *
 * Transient-failure retries are applied automatically per the session's retry policy — see
 * [StageScope.executeWithTrackedPromptExecutor].
 *
 * @param retryPolicy override for the session-default retry policy. `null` (default) = use the
 *   session-resolved policy. [RetryPolicy.None] = disable transient retries for this call.
 */
@OptimizationExtensionApi
public suspend inline fun <reified T> StageScope<*, *, *>.executePromptStructured(
    prompt: Prompt, model: LLModel,
    retryPolicy: RetryPolicy? = null,
): Result<T> =
    executeWithTrackedPromptExecutor(name = prompt.id, retryPolicy = retryPolicy) {
        executeStructuredOrThrow<T>(prompt, model)
    }

// ===================================================================
// OrThrow variants
// ===================================================================

/**
 * `.getOrThrow()` shortcut over [StageScope.runStage]. See its KDoc for the [metrics] preset
 * conventions; the default and parameter semantics are identical.
 */
@OptimizationExtensionApi
public suspend fun <Input, Output, InputLabel, T> StageScope<Input, Output, InputLabel>.runStageOrThrow(
    name: String,
    substagesTotal: Int? = null,
    metrics: MetricsMap = standardStageMetrics(),
    block: suspend StageScope<Input, Output, InputLabel>.() -> T,
): T = runStage(name, substagesTotal, metrics, block).getOrThrow()

/** `.getOrThrow()` shortcut over [StageScope.runAgent]; throws on a recorded run failure. */
@OptimizationExtensionApi
public suspend fun <Input, Output, InputLabel> StageScope<Input, Output, InputLabel>.runAgentOrThrow(
    item: TrainSetItem<Input, InputLabel>,
    agentToRun: GraphAIAgent<Input, Output> = trackedAgent,
    retryPolicy: RetryPolicy? = null,
): CompletedAgentRun<Output> = runAgent(item, agentToRun, retryPolicy).getOrThrow()

/** `.getOrThrow()` shortcut over [executePrompt]; throws on a recorded execution failure. */
@OptimizationExtensionApi
public suspend fun StageScope<*, *, *>.executePromptOrThrow(
    prompt: Prompt, model: LLModel,
    retryPolicy: RetryPolicy? = null,
): List<Message.Response> = executePrompt(prompt, model, retryPolicy).getOrThrow()

/** `.getOrThrow()` shortcut over [executePromptStructured]; throws on a recorded execution failure. */
@OptimizationExtensionApi
public suspend inline fun <reified T> StageScope<*, *, *>.executePromptStructuredOrThrow(
    prompt: Prompt, model: LLModel,
    retryPolicy: RetryPolicy? = null,
): T = executePromptStructured<T>(prompt, model, retryPolicy).getOrThrow()

// ===================================================================
// Extra extensions
// ===================================================================

/**
 * Serializes [data] with [defaultExperimentsJson] and stores it as this stage's additional data
 * via [StageScope.setAdditionalDataJson]. Overwrites any previously set value.
 */
@OptimizationExtensionApi
public inline fun <reified T> StageScope<*, *, *>.setAdditionalData(data: T) {
    setAdditionalDataJson(defaultExperimentsJson.encodeToJsonElement<T>(data))
}

/**
 * Convenience wrapper that iterates a collection, enclosing it in a dedicated stage.
 * Pros: automatically sets `substagesTotal` to enable ETC calculation.
 *
 * If any item throws, the iteration is stopped, and the error is wrapped inside the returned Result.
 */
@OptimizationExtensionApi
public suspend fun <Input, Output, InputLabel, T> StageScope<Input, Output, InputLabel>.runIterableStage(
    collection: Collection<T>,
    name: String,
    itemBlock: suspend StageScope<Input, Output, InputLabel>.(T) -> Unit,
): Result<Unit> {
    val itemsTotal = collection.size
    return runStage(name = name, substagesTotal = itemsTotal) {
        collection.forEach { item -> itemBlock(item) }
    }
}

/**
 * @see runIterableStage
 */
@OptimizationExtensionApi
public suspend fun <Input, Output, InputLabel, T> StageScope<Input, Output, InputLabel>.runIterableStageOrThrow(
    collection: Collection<T>,
    name: String,
    itemBlock: suspend StageScope<Input, Output, InputLabel>.(T) -> Unit,
): Unit = runIterableStage(collection, name, itemBlock).getOrThrow()

/**
 * Convenience wrapper that maps a collection, enclosing it in a dedicated stage.
 * Pros: automatically sets `substagesTotal` to enable ETC calculation.
 *
 * If any item throws, the stage will immediately throw as well.
 */
@OptimizationExtensionApi
public suspend fun <Input, Output, InputLabel, TFrom, TInto> StageScope<Input, Output, InputLabel>.mapIterableStageOrThrow(
    collection: Collection<TFrom>,
    name: String,
    itemBlock: suspend StageScope<Input, Output, InputLabel>.(TFrom) -> TInto,
): List<TInto> {
    val itemsTotal = collection.size
    return runStageOrThrow(name = name, substagesTotal = itemsTotal) {
        collection.map { item -> itemBlock(item) }
    }
}

// ===================================================================
// Supporting types
// ===================================================================

/** Result of a successful agent run produced by [StageScope.runAgent]. */
@OptimizationExtensionApi
public data class CompletedAgentRun<Output>(
    /** The agent's output for the item. */
    val output: Output,
    /** The metric score of [output] against the item. */
    val score: Double,
    /** `true` when [score] meets the session's solved threshold. */
    val isSolved: Boolean,
    /** The agent instance that was actually used for this run (for feature access, e.g. trace collection). */
    val usedAgent: GraphAIAgent<*, *>? = null,
)

/** Stop condition for the re-sampling [StageScope.runAgentWithRetries] overloads. */
@OptimizationExtensionApi
public enum class RunUntil {
    /** Stop retrying once an agent run solves the item. */
    SOLVED,

    /** Stop retrying once an agent run completes without infrastructure failure. */
    COMPLETED,
}

/**
 * An agent paired with caller-supplied per-attempt data, returned by the `agentProvider` of the
 * generic [StageScope.runAgentWithRetries] so the data can be matched back to a successful run.
 */
@OptimizationExtensionApi
public class PreparedAgentRun<Input, Output, RunData>(
    /** The agent to run for this attempt. */
    public val agent: GraphAIAgent<Input, Output>,
    /** Caller data associated with this attempt, surfaced again in [MatchedAgentRun.runData]. */
    public val runData: RunData,
)

/**
 * A successful [CompletedAgentRun] together with the [PreparedAgentRun.runData] that produced it,
 * returned by the generic [StageScope.runAgentWithRetries].
 */
@OptimizationExtensionApi
public class MatchedAgentRun<Output, RunData>(
    /** The successful agent run. */
    public val agentRun: CompletedAgentRun<Output>,
    /** The data carried by the [PreparedAgentRun] that yielded [agentRun]. */
    public val runData: RunData,
)

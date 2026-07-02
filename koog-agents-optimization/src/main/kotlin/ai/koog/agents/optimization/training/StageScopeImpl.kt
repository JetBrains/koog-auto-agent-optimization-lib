package ai.koog.agents.optimization.training

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.annotations.OptimizationExtensionApi
import ai.koog.agents.optimization.common.abort.AbortController
import ai.koog.agents.optimization.common.abort.ExecutionAbortException
import ai.koog.agents.optimization.common.retries.AttemptMetrics
import ai.koog.agents.optimization.common.retries.RetriesOutcome
import ai.koog.agents.optimization.common.retries.RetryPolicy
import ai.koog.agents.optimization.common.retries.retryWith
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.dsl.*
import ai.koog.agents.optimization.training.metrics.MetricsMap
import ai.koog.agents.optimization.training.metrics.datasetIterationMetrics
import ai.koog.agents.optimization.training.metrics.impl.ElapsedTimeMetric
import ai.koog.agents.optimization.training.metrics.impl.Metric
import ai.koog.agents.optimization.training.metrics.impl.SubstageCountMetric
import ai.koog.agents.optimization.training.records.AgentRunRecord
import ai.koog.agents.optimization.training.records.PromptExecutionRecord
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.structures.*
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import io.github.oshai.kotlinlogging.KLogger
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.time.TimeSource

/**
 * Core implementation of [StageScope].
 *
 * Each instance owns a [StageRecord] whose [MetricsMap] is recomputed from substages on each
 * substage update, then propagated upward via [onUpdate].
 */
@OptimizationExtensionApi
public class StageScopeImpl<Input, Output, InputLabel>(
    private val resources: TrainingResources<Input, Output, InputLabel>,
    private val logger: KLogger,
    /** The record this scope owns and mutates; its metrics are recomputed on each substage update. */
    public val record: StageRecord,
    /** Enclosing scope, or null for the root scope. Walked by [ancestorsPath] for log prefixes. */
    private val parentScope: StageScopeImpl<Input, Output, InputLabel>?,
    /** Called after this stage's metrics are recomputed. Receives `isStageUpdate` flag. */
    private val onUpdate: ((isStageUpdate: Boolean) -> Unit),
) : StageScope<Input, Output, InputLabel> {

    /** The session's tracked agent, taken from [TrainingResources.trackedAgent]. */
    override val trackedAgent: GraphAIAgent<Input, Output> get() = resources.trackedAgent

    /** The session's dataset, taken from [TrainingResources.dataset]. */
    override val dataset: TrainSet<Input, InputLabel> get() = resources.dataset

    /**
     * Wall-clock start of this stage. Captured at scope construction, used to refresh
     * [StageRecord.realElapsed] on every recompute so the live value always reflects the
     * stage's true duration — not only once the block exits.
     */
    private val startTimeMark = TimeSource.Monotonic.markNow()

    /** Ancestor chain (root first, then this scope's own name) for stages nested directly under this scope. */
    private val ancestorsPath: List<String>
        get() = generateSequence(this) { it.parentScope }
            .map { it.record.name }
            .toList()
            .asReversed()

    private fun recomputeMetricsAndUpdate(isStageUpdate: Boolean) {
        refreshRealElapsed()
        record.metrics.recomputeAll(record, record.substages, isStageUpdate)
        onUpdate(isStageUpdate)
    }

    /**
     * Writes `startTimeMark.elapsedNow()` into this stage's [StageRecord.realElapsed].
     * Called before every recompute so [ElapsedTimeMetric.recompute] (which derives ETC from
     * `currentStage.realElapsed`) and consumers downstream always see an up-to-date value.
     */
    private fun refreshRealElapsed() {
        record.realElapsed = startTimeMark.elapsedNow()
    }

    // ===================================================================
    // runStage
    // ===================================================================

    /**
     * Runs [block] as a named child stage and records the result.
     *
     * On success: returns `Result.success(block's value)`.
     * On a regular exception: records the failure on the substage and returns
     * `Result.failure(StageFailedException)` so callers can choose to handle or rethrow.
     * On an [ExecutionAbortException]: records the failure on the substage and rethrows,
     * letting the execution-level abort propagate up to the session.
     */
    override suspend fun <T> runStage(
        name: String,
        substagesTotal: Int?,
        metrics: MetricsMap,
        block: suspend StageScope<Input, Output, InputLabel>.() -> T,
    ): Result<T> {
        val substageRecord = StageRecord(
            name = name,
            metrics = metrics,
            substagesTotal = substagesTotal,
        )
        record.substages.add(substageRecord)

        val substageScope =
            StageScopeImpl(resources, logger, substageRecord, parentScope = this) { substageIsStageUpdate ->
                recomputeMetricsAndUpdate(isStageUpdate = substageIsStageUpdate)
            }

        recomputeMetricsAndUpdate(isStageUpdate = true) // stage start
        logger.info { TrainingLogging.buildStageBeginConsoleLog(name, ancestorsPath, record) }

        return try {
            val result = substageScope.block()

            substageScope.refreshRealElapsed()
            substageRecord.isFinished = true
            recomputeMetricsAndUpdate(isStageUpdate = true)
            logger.info { TrainingLogging.buildStageFinishedConsoleLog(substageRecord, ancestorsPath, record) }

            Result.success(result)
        } catch (e: Exception) {
            substageScope.refreshRealElapsed()
            substageRecord.isFinished = true
            substageRecord.failure = resolveStageFailure(e, resources.failureAnalyzer)
            logger.error(e) { "Stage '$name' failed with ${e::class.simpleName}: ${e.message}" }
            // May throw an abort exception, but only once per session.
            recomputeMetricsAndUpdate(isStageUpdate = true)
            logger.info { TrainingLogging.buildStageFinishedConsoleLog(substageRecord, ancestorsPath, record) }

            if (e is ExecutionAbortException) throw e
            Result.failure(StageFailedException(substageRecord, e))
        }
    }

    // ===================================================================
    // runAgent
    // ===================================================================

    /**
     * Runs [agentToRun] on [item] and records exactly one leaf entry under the current stage —
     * `AgentRunRecord.Completed` on success, or `AgentRunRecord.Failed` on any failure (regular
     * exception, retry-exhausted transient, or [ExecutionAbortException]).
     *
     * Transient-failure retries are applied automatically per the session's retry policy
     * (overridable via [retryPolicy]); per-attempt detail lives in the resulting record's
     * `previousAttempts`.
     *
     * **Abort handling.** [ExecutionAbortException] can arise from two sites:
     * - From inside `agent.run(...)` itself (currently theoretical — the project's only abort
     *   policy fires from recompute callbacks, not from agent code). The [retryWith] `onAbort`
     *   hook records a Failed leaf with the abort's `resolvedId` AND every preceding failed-retry
     *   captured up to that point; [retryWith] then rethrows the abort after the hook returns.
     *   The post-add recompute inside the hook is safe because [AbortController] is one-shot.
     * - From inside the post-add `recomputeMetricsAndUpdate()` (spend-cap trip, etc.). The leaf
     *   is already recorded, so the abort just propagates out of this method without further
     *   bookkeeping.
     *
     * Aborts always trump retries: [retryWith]'s own abort short-circuit catches the
     * exception and rethrows immediately without entering the retry loop.
     */
    override suspend fun runAgent(
        item: TrainSetItem<Input, InputLabel>,
        agentToRun: GraphAIAgent<Input, Output>,
        retryPolicy: RetryPolicy?,
    ): Result<CompletedAgentRun<Output>> {
        val activeRetryPolicy = retryPolicy ?: resources.retriesPolicy
        var attemptTimeMark = TimeSource.Monotonic.markNow()

        val outcome = retryWith(
            policy = activeRetryPolicy,
            operationLabel = { "agent run for item '${resources.serializers.serializeItem(item)}'" },
            logger = logger,
            analyzeFailure = { exception -> resources.failureAnalyzer.analyzeAgentRunFailure(exception) },
            collectAttemptMetrics = {
                AttemptMetrics(
                    elapsedTime = attemptTimeMark.elapsedNow(),
                    consumption = LLMConsumptionOrNA.from(resources.consumptionCollector()),
                )
            },
            onAttemptStarting = {
                @Suppress("AssignedValueIsNeverRead")
                attemptTimeMark = TimeSource.Monotonic.markNow()
            },
            onAbort = { abort, previousAttempts, abortedAttemptMetrics ->
                // Abort fired inside agent.run() — record a Failed leaf with the abort's failure
                // AND every preceding retry attempt (mirrors `runStage`'s abort handling at the stage
                // level). `retryWith` rethrows after this hook returns, so we don't rethrow here.
                val agentRecord = AgentRunRecord.Failed(
                    failure = abort.toAnalyzedFailure(),
                    elapsedTime = abortedAttemptMetrics.elapsedTime,
                    consumption = abortedAttemptMetrics.consumption,
                    previousAttempts = previousAttempts,
                )
                record.substages.add(agentRecord)
                // Safe: AbortController is one-shot, so the spend-cap check inside recompute won't
                // re-fire the abort. Recompute is needed so metrics see the just-added leaf.
                recomputeMetricsAndUpdate(isStageUpdate = false)
            },
            // The attempt hook installs any critical-tool-failure propagation and returns the
            // (possibly re-prepared) agent alongside its output.
            block = { _ -> resources.runAgentAttempt(agentToRun, item.userQuery) },
        )

        return when (outcome) {
            is RetriesOutcome.Completed -> {
                val (freshAgent, output) = outcome.result
                val score = resources.metric(item, output)
                val solved = score >= resources.threshold

                val agentRecord = AgentRunRecord.Completed(
                    agentOutput = resources.serializers.serializeOutput(output),
                    score = score,
                    solved = solved,
                    elapsedTime = outcome.finalAttemptMetrics.elapsedTime,
                    consumption = outcome.finalAttemptMetrics.consumption,
                    previousAttempts = outcome.previousAttempts,
                )
                record.substages.add(agentRecord)
                // Post-add recompute may throw ExecutionAbortException (spend-cap trip). The leaf
                // is already recorded, so the abort propagates out of this method without further
                // bookkeeping — the caller never sees a duplicate Failed leaf.
                recomputeMetricsAndUpdate(isStageUpdate = false)

                Result.success(CompletedAgentRun(output, score, isSolved = solved, usedAgent = freshAgent))
            }

            is RetriesOutcome.Failed -> {
                val agentRecord = AgentRunRecord.Failed(
                    failure = outcome.finalFailure,
                    elapsedTime = outcome.finalAttemptMetrics.elapsedTime,
                    consumption = outcome.finalAttemptMetrics.consumption,
                    previousAttempts = outcome.previousAttempts,
                )
                record.substages.add(agentRecord)
                recomputeMetricsAndUpdate(isStageUpdate = false)

                logger.warn {
                    "Agent run failed: ${outcome.finalFailure.resolvedId} " +
                            "[${outcome.finalFailure.kind}] — ${outcome.finalFailure.description}"
                }
                Result.failure(AgentRunFailureException(agentRecord, outcome.finalException))
            }
        }
    }

    override suspend fun runAgentWithRetries(
        item: TrainSetItem<Input, InputLabel>,
        maxAttempts: Int,
        until: RunUntil,
        agentToRun: GraphAIAgent<Input, Output>,
        innerRetryPolicy: RetryPolicy?,
    ): CompletedAgentRun<Output>? {
        repeat(maxAttempts) {
            val execution = runAgent(item, agentToRun, retryPolicy = innerRetryPolicy)
            execution.onSuccess {
                if (shouldStop(until, it)) return it
            }
        }
        return null
    }

    override suspend fun <RunData> runAgentWithRetries(
        item: TrainSetItem<Input, InputLabel>,
        maxAttempts: Int,
        until: RunUntil,
        innerRetryPolicy: RetryPolicy?,
        agentProvider: suspend () -> PreparedAgentRun<Input, Output, RunData>,
    ): MatchedAgentRun<Output, RunData>? {
        repeat(maxAttempts) {
            val prepared = agentProvider()
            val execution = runAgent(item, prepared.agent, retryPolicy = innerRetryPolicy)
            execution.onSuccess {
                if (shouldStop(until, it)) return MatchedAgentRun(it, prepared.runData)
            }
        }
        return null
    }

    private fun shouldStop(until: RunUntil, result: CompletedAgentRun<Output>): Boolean =
        when (until) {
            RunUntil.SOLVED -> result.isSolved
            RunUntil.COMPLETED -> true
        }

    // ===================================================================
    // executePrompt
    // ===================================================================

    /**
     * Runs [execute] against a fresh consumption-capturing executor and records exactly one leaf
     * entry under the current stage — `PromptExecutionRecord.Completed` on success, or
     * `PromptExecutionRecord.Failed` on any failure. Shared by [executePrompt] and
     * [executePromptStructured].
     *
     * Transient-failure retries are applied automatically per the session's retry policy
     * (overridable via [retryPolicy]); per-attempt detail lives in the resulting record's
     * `previousAttempts`.
     *
     * **Abort handling.** Mirrors [runAgent]: [ExecutionAbortException] thrown by [execute] is
     * caught by [retryWith]'s `onAbort` hook (records a `PromptExecutionRecord.Failed` with the
     * abort's resolvedId + every preceding retry attempt), then rethrown.
     */
    override suspend fun <T> executeWithTrackedPromptExecutor(
        name: String,
        retryPolicy: RetryPolicy?,
        execute: suspend (CapturingPromptExecutor) -> T,
    ): Result<T> {
        val activeRetryPolicy = retryPolicy ?: resources.retriesPolicy
        // The capturing executor is created once per leaf call. Its `collectAndClear()` between
        // attempts isolates per-attempt consumption — the same per-attempt isolation pattern as runAgent.
        val capturingExecutor = resources.capturingExecutorFactory(resources.substepPromptExecutor)
        var attemptTimeMark = TimeSource.Monotonic.markNow()

        val outcome = retryWith(
            policy = activeRetryPolicy,
            operationLabel = { "prompt execution '$name'" },
            logger = logger,
            analyzeFailure = { exception -> resources.failureAnalyzer.analyzeTrainingFailure(exception) },
            collectAttemptMetrics = {
                AttemptMetrics(
                    elapsedTime = attemptTimeMark.elapsedNow(),
                    consumption = LLMConsumptionOrNA.from(capturingExecutor.collectAndClear()),
                )
            },
            onAttemptStarting = {
                @Suppress("AssignedValueIsNeverRead")
                attemptTimeMark = TimeSource.Monotonic.markNow()
            },
            onAbort = { abort, previousAttempts, abortedAttemptMetrics ->
                // Abort fired inside the prompt execution itself — record a `Failed` leaf with the
                // abort's failure AND every preceding retry attempt (mirrors the `runAgent` abort
                // hook above). `retryWith` rethrows after this hook returns.
                val promptRecord = PromptExecutionRecord.Failed(
                    promptName = name,
                    failure = abort.toAnalyzedFailure(),
                    elapsedTime = abortedAttemptMetrics.elapsedTime,
                    consumption = abortedAttemptMetrics.consumption,
                    previousAttempts = previousAttempts,
                )
                record.substages.add(promptRecord)
                recomputeMetricsAndUpdate(isStageUpdate = false)
            },
            block = { _ -> execute(capturingExecutor) },
        )

        return when (outcome) {
            is RetriesOutcome.Completed -> {
                val promptRecord = PromptExecutionRecord.Completed(
                    promptName = name,
                    elapsedTime = outcome.finalAttemptMetrics.elapsedTime,
                    consumption = outcome.finalAttemptMetrics.consumption,
                    previousAttempts = outcome.previousAttempts,
                )
                record.substages.add(promptRecord)
                recomputeMetricsAndUpdate(isStageUpdate = false)

                Result.success(outcome.result)
            }

            is RetriesOutcome.Failed -> {
                val promptRecord = PromptExecutionRecord.Failed(
                    promptName = name,
                    failure = outcome.finalFailure,
                    elapsedTime = outcome.finalAttemptMetrics.elapsedTime,
                    consumption = outcome.finalAttemptMetrics.consumption,
                    previousAttempts = outcome.previousAttempts,
                )
                record.substages.add(promptRecord)
                recomputeMetricsAndUpdate(isStageUpdate = false)

                Result.failure(PromptExecutionFailureException(promptRecord, outcome.finalException))
            }
        }
    }

    // ===================================================================
    // logAction and recordCustomMetric
    // ===================================================================

    override fun logAction(json: Json, builder: ActionLogBuilder.() -> Unit) {
        if (record.actionLog != null) {
            logger.warn { "Overwriting existing action log! Was: ${json.encodeToString(record.actionLog)}" }
        }
        record.actionLog = ActionLogBuilder(resources.actionLogTruncation, json).apply(builder).build()
    }

    override fun recordCustomMetric(metric: Metric) {
        record.metrics[metric.key] = metric
        onUpdate(false)
    }

    override fun setAdditionalDataJson(additionalData: JsonElement) {
        record.additionalData = additionalData
    }

    // ===================================================================
    // iterateDataset
    // ===================================================================

    /**
     * Thin delegate over [runStage] with the [datasetIterationMetrics] preset (plus any
     * [customMetricsToRecord]). Threshold breach throws inside the block so `runStage`'s catch
     * arm records it on the dataset stage — preserving the documented "marks but does not
     * throw" contract and letting per-item aborts propagate naturally.
     */
    override suspend fun iterateDataset(
        name: String,
        dataset: TrainSet<Input, InputLabel>,
        customMetricsToRecord: List<Metric>?,
        failureRateThreshold: Double,
        earlyStop: (TrainSetItem<Input, InputLabel>) -> PrematureExecutionStopDecision,
        processItem: suspend StageScope<Input, Output, InputLabel>.(TrainSetItem<Input, InputLabel>) -> Unit,
    ): StageRecord {
        val datasetMetrics = datasetIterationMetrics() +
                MetricsMap.of(*(customMetricsToRecord ?: emptyList()).toTypedArray())

        var datasetRecord: StageRecord? = null
        runStage(name = name, substagesTotal = dataset.size, metrics = datasetMetrics) {
            datasetRecord = (this as StageScopeImpl<Input, Output, InputLabel>).record

            for ((index, item) in dataset.withIndex()) {
                val itemIndex = index + 1
                val itemId = "[$name | Item #$itemIndex / ${dataset.size}]"
                val serializedItem = resources.serializers.serializeItem(item)

                val (shouldStop, stopReason) = earlyStop(item)
                if (shouldStop) {
                    logger.info { "$itemId Stopping training: ${stopReason()}" }
                    break
                }

                val itemData = buildJsonObject { put("item", serializedItem) }
                runStage(itemId) {
                    (this as StageScopeImpl).record.additionalData = itemData
                    resources.withItemContext(itemId) {
                        processItem(item)
                    }
                }
            }

            val failedRatio = record.metrics[SubstageCountMetric.KEY]!!.failedRatio
            if (failedRatio.fraction > failureRateThreshold) {
                throw DatasetMaxFailureRateExceededException(failedRatio, failureRateThreshold)
            }
        }
        return checkNotNull(datasetRecord) {
            "Dataset stage record was not captured — `runStage` block did not start"
        }
    }
}

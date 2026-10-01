package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.optimization.common.abort.DatasetFailureRateExceededAbortException
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.features.collectSubgraphTraces
import ai.koog.agents.optimization.features.installPromptOptimization
import ai.koog.agents.optimization.koogTooling.copyWith
import ai.koog.agents.optimization.koogTooling.getCollectedTraces
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.training.PrematureExecutionStopDecision
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.runStageOrThrow
import ai.koog.agents.optimization.training.metrics.impl.SubstageCountMetric
import ai.koog.agents.optimization.training.structures.AgentRunFailureException
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/** Result of running one GEPA candidate on one dataset item. */
internal sealed interface GEPARollout {
    val score: Double

    /** A completed agent rollout with score and feedback available for reflection. */
    data class Completed(val feedback: GEPAFeedbackResult) : GEPARollout {
        override val score: Double get() = feedback.score
    }

    /** A terminally failed agent rollout represented by its configured failure score. */
    data class Failed(override val score: Double) : GEPARollout
}

internal val List<GEPARollout>.scores: List<Double> get() = map { it.score }

internal val List<GEPARollout>.failedRolloutIndices: List<Int>
    get() = mapIndexedNotNull { index, rollout -> index.takeIf { rollout is GEPARollout.Failed } }

internal val List<GEPARollout>.hasFailures: Boolean get() = any { it is GEPARollout.Failed }

/**
 * Runs a prompt candidate through the training framework and converts each completed run to GEPA feedback.
 * Terminal agent-run failures retain their dataset position as [GEPARollout.Failed] and are scored as [failureScore].
 * When [abortOnFailureRateExceeded] is enabled, iteration stops once the number of failed rollouts makes a final
 * full-dataset ratio above [failureRateThreshold] unavoidable, then aborts the rollout set. When disabled, threshold
 * breaches are recorded on the dataset stage without terminating GEPA.
 */
@OptIn(InternalAgentsApi::class)
internal suspend fun <Input, Output, InputLabel> StageScope<Input, Output, InputLabel>.runGepaRollouts(
    candidate: OptimizationArtifact,
    dataset: TrainSet<Input, InputLabel>,
    stageName: String,
    gepaFeedback: GEPAFeedback<Input, Output, InputLabel>,
    failureScore: Double,
    failureRateThreshold: Double,
    abortOnFailureRateExceeded: Boolean = true,
): List<GEPARollout> = runStageOrThrow(stageName) {
    require(failureRateThreshold in 0.0..1.0) {
        "failureRateThreshold must be between 0.0 and 1.0"
    }

    val rollouts = MutableList<GEPARollout?>(dataset.size) { null }
    var stopBeforeNextItem = false
    var failedRolloutCount = 0
    var unexpectedItemFailure: Exception? = null

    // Keep positional alignment even when duplicate dataset items compare equal.
    val remainingIndicesByItem = dataset.withIndex()
        .groupBy(keySelector = { it.value }, valueTransform = { it.index })
        .mapValues { (_, indices) -> ArrayDeque(indices) }

    val agentWithArtifact = trackedAgent.copyWith(installFeatures = {
        trackedAgent.installFeatures(this)
        installPromptOptimization { artifact = candidate }
        collectSubgraphTraces { }
    })
    val traceCollector = checkNotNull(agentWithArtifact.getCollectedTraces()) {
        "SubgraphTraceCollectionFeature was not installed on the agent"
    }

    val datasetRecord = iterateDataset(
        name = "$stageName: run dataset",
        dataset = dataset,
        failureRateThreshold = failureRateThreshold,
        earlyStop = {
            PrematureExecutionStopDecision(
                conditionMet = stopBeforeNextItem,
                conditionMetReason = {
                    "Stopped GEPA rollouts early because a dataset item failed"
                },
            )
        },
    ) { item ->
        val terminalFailure: Throwable? = try {
            val index = remainingIndicesByItem.getValue(item).removeFirst()
            traceCollector.clear()

            val runResult = runAgent(item, agentWithArtifact)
            if (runResult.isFailure) {
                val exception = checkNotNull(runResult.exceptionOrNull()) {
                    "Failed GEPA agent run did not carry an exception"
                }
                val analyzedFailure = (exception as? AgentRunFailureException)?.analyzedFailure
                val failureSummary = if (analyzedFailure != null) {
                    "${analyzedFailure.resolvedId} [${analyzedFailure.kind}, " +
                            "transiency=${analyzedFailure.transiency}]"
                } else {
                    "${exception::class.simpleName}: ${exception.message}"
                }
                logger.warn {
                    "GEPA rollout ${index + 1}/${dataset.size} failed terminally during '$stageName': " +
                            "$failureSummary. Assigning failureScore=$failureScore."
                }

                rollouts[index] = GEPARollout.Failed(failureScore)
                logGepaAssignedScore(failureScore)
                failedRolloutCount++
                if (
                    abortOnFailureRateExceeded &&
                    failedRolloutCount > failureRateThreshold * dataset.size
                ) {
                    stopBeforeNextItem = true
                }

                exception
            } else {
                val run = runResult.getOrThrow()
                val collectedTraces = checkNotNull(run.usedAgent?.getCollectedTraces()) {
                    "SubgraphTraceCollectionFeature was not installed on the used agent"
                }
                val feedback = gepaFeedback(
                    input = item.userQuery,
                    output = run.output,
                    gold = item.itemLabel,
                    fullTrace = collectedTraces.getLatestFullPrompt(),
                    subgraphTraces = collectedTraces.getAllTraces(),
                )
                collectedTraces.clear()
                rollouts[index] = GEPARollout.Completed(feedback)
                logGepaAssignedScore(feedback.score)
                null
            }
        // runAgent represents terminal agent failures as Result failures; those are scored above. Exceptions reaching
        // this catch come from trace collection, feedback, or GEPA bookkeeping and must not become scoreable rollout
        // failures. iterateDataset absorbs ordinary item exceptions, so retain the original, stop the iteration, and
        // surface it again after its failure is recorded.
        } catch (exception: Exception) {
            unexpectedItemFailure = exception
            stopBeforeNextItem = true
            throw exception
        }

        // this throw outside the try makes the per-item stage record the failure and may continue the iteration
        terminalFailure?.let { throw it }
    }

    // surface an unexpected exception
    unexpectedItemFailure?.let { throw it }

    // iterateDataset applies the same threshold internally, but its regular exception is absorbed by the dataset
    // runStage and represented only in StageRecord.failure. In aborting mode, promote that recorded breach through the
    // session's shared abort controller; otherwise leave it as observability data and continue.
    val failedRatio = datasetRecord.metrics[SubstageCountMetric.KEY]!!.failedRatio
    if (abortOnFailureRateExceeded && failedRatio.fraction > failureRateThreshold) {
        abortExecution {
            DatasetFailureRateExceededAbortException(
                failedItems = failedRatio.top,
                finishedItems = failedRatio.bottom,
                failureRateThreshold = failureRateThreshold,
            )
        }
    }

    val completedRollouts = rollouts.mapIndexed { index, rollout ->
        checkNotNull(rollout) { "Missing rollout for dataset item $index" }
    }
    logGepaRolloutScores(completedRollouts.scores)

    completedRollouts
}

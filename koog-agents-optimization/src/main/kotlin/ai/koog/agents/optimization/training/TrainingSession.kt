package ai.koog.agents.optimization.training


import ai.koog.agents.optimization.common.*
import ai.koog.agents.optimization.common.abort.AbortPolicy
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.metrics.impl.ConsumptionMetric
import ai.koog.agents.optimization.training.metrics.rootStageMetrics
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingResult
import ai.koog.agents.optimization.training.structures.resolveStageFailure
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.common.toFilePathLog
import io.github.oshai.kotlinlogging.KLogger
import kotlinx.serialization.json.JsonElement

/**
 * Lifecycle wrapper for a training session.
 *
 * Creates the root [StageScopeImpl], runs the optimizer block via [use], persists the
 * resulting records, and returns a [TrainingResult] (the same shape that is written to disk).
 *
 * All provider/settings wiring (agent tracking, consumption capture, spend limits, cluster progress)
 * is supplied through [resources] and the constructor's injected collaborators; the app builds these
 * via its `TrainingSessionFactory`.
 */
public class TrainingSession<Input, Output, InputLabel>(
    private val resources: TrainingResources<Input, Output, InputLabel>,
    private val logger: KLogger,
    private val trainingName: ExperimentName,
    private val recordsFilePath: ResilientPath?,
    /** When `true`, [use] throws if [recordsFilePath] already exists. */
    private val throwIfRecordsFileExists: Boolean,
    /** Pod name recorded in the run's [ExecutionMetadata] (null when not running on a cluster). */
    private val podName: String?,
    /** Spend-limit config, pre-encoded by the app, recorded verbatim in [ExecutionMetadata]. */
    private val spendLimit: JsonElement?,
    /** Retry config, pre-encoded by the app, recorded verbatim in [ExecutionMetadata]. */
    private val retries: JsonElement?,
    /** Consulted on every recompute with the running cumulative consumption; may abort the run. */
    private val abortPolicies: List<AbortPolicy> = emptyList(),
    /** Invoked with the root record on session start, on each stage update, and on finish. */
    private val progressListener: (StageRecord) -> Unit = {},
) {
    /**
     * Runs the training session: sets up the root [StageScope], runs [block], persists final
     * records, and returns the [TrainingResult].
     *
     * If [block] throws, the exception is re-thrown after the partial records are saved
     * with `executionMetadata.completed = false` and `rootRecord.failure` populated.
     *
     * @param stagesTotal Expected number of root stages, used for progress / ETC calculation.
     *   `null` means the total is unknown.
     */
    public suspend fun use(
        stagesTotal: Int? = null,
        block: suspend StageScope<Input, Output, InputLabel>.() -> Unit,
    ): TrainingResult {
        validateRecordsFile()

        // Create root stage record
        val rootRecord = StageRecord(
            name = trainingName.runId,
            metrics = rootStageMetrics(),
            substagesTotal = stagesTotal,
        )

        // Captured at session start; updated to "completed" only on the normal exit path.
        var executionMetadata = ExecutionMetadata.createAsStartedNow(
            podName = podName,
            spendLimit = spendLimit,
            retries = retries,
        )

        logger.info { TrainingLogging.buildSessionBeginConsoleLog(trainingName, recordsFilePath) }

        writeRecordsToDisk(rootRecord, executionMetadata)

        // Root scope's onUpdate persists records, emits progress on stage updates, and evaluates
        // the abort policies (e.g. spend cap) on every recompute.
        val rootScope = StageScopeImpl(resources, logger, rootRecord, parentScope = null) { isStageUpdate ->
            writeRecordsToDisk(rootRecord, executionMetadata)
            if (isStageUpdate) {
                progressListener(rootRecord)
            }
            val cumulative = rootRecord.metrics[ConsumptionMetric.KEY]?.totalOrNull()
            abortPolicies.forEach { it.checkAndAbortIfExceeded(cumulative) }
        }

        // Emit an empty record at the beginning; equivalent to a "starting" message
        progressListener(rootRecord)

        var blockException: Throwable? = null
        try {
            rootScope.block()
        } catch (e: Exception) {
            // Mirror inner-stage behavior: record the failure on the root and re-throw.
            rootRecord.failure = resolveStageFailure(e, resources.failureAnalyzer)
            blockException = e
            logger.error(e) {
                "Training session block failed with ${e::class.simpleName}: ${e.message}\n" +
                        "WARNING: depending on the optimizer, training artifacts may have been lost!\n" +
                        "Stacktrace:\n${e.stackTraceToString()}"
            }
        } catch (e: Throwable) {
            // Errors (OOM, StackOverflow, etc.) — log and rethrow; analyzing failure on a
            // degraded JVM is risky, so the root failure is left unset for Errors.
            blockException = e
            logger.error(e) {
                "Training session block failed with ${e::class.simpleName}: ${e.message}\n" +
                        "WARNING: depending on the optimizer, training artifacts may have been lost!\n" +
                        "Stacktrace:\n${e.stackTraceToString()}"
            }
        }

        // Mark the root as finished on every exit path (success or failure), to match
        // inner-stage behavior in `runStage`.
        rootRecord.isFinished = true

        logger.info { TrainingLogging.buildSessionFinishedConsoleLog(rootRecord) }
        if (blockException == null) {
            executionMetadata = executionMetadata.copyAsCompletedNow()
        }
        progressListener(rootRecord) // report root stage as finished
        writeRecordsToDisk(rootRecord, executionMetadata)

        if (blockException != null) {
            throw blockException
        }
        return TrainingResult(rootStage = rootRecord, executionMetadata = executionMetadata)
    }

    private fun writeRecordsToDisk(record: StageRecord, executionMetadata: ExecutionMetadata) {
        val filePath = recordsFilePath ?: return
        val result = TrainingResult(rootStage = record, executionMetadata = executionMetadata)
        try {
            filePath.writeText(
                defaultExperimentsJson.encodeToStringStripped(TrainingResult.serializer(), result)
            )
        } catch (_: Exception) {
            // Best-effort progress dump
        }
    }

    private fun validateRecordsFile() {
        recordsFilePath?.createParentDirectories()
        if (recordsFilePath != null && recordsFilePath.exists() && throwIfRecordsFileExists) {
            throw IllegalArgumentException("Training records file already exists: ${recordsFilePath.toFilePathLog()}")
        }
    }
}

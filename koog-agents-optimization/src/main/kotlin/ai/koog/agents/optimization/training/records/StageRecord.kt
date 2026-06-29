package ai.koog.agents.optimization.training.records


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.training.metrics.MetricsMap
import ai.koog.agents.optimization.utils.serialization.PrettyRoundedDurationSerializer
import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlin.time.Duration

/**
 * Record of a stage -- the universal node in the training records tree.
 *
 * A stage can contain nested stages, agent runs, and prompt executions as [substages].
 * Its [metrics] are recomputed from substage records on each update.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
public data class StageRecord(
    /** Human-readable stage name; doubles as the display label in logs and the UI. */
    val name: String,

    /** Free-form stage-specific payload, serialized only when present. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    var additionalData: JsonElement? = null,

    /** Analyzed failure that aborted this stage, or null if it completed without error. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    var failure: AnalyzedFailure? = null,

    /** Free-form structured log of actions taken during this stage, serialized only when present. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    var actionLog: JsonElement? = null,

    /** Aggregated metrics for this stage, recomputed from [substages] on each update. */
    val metrics: MetricsMap = MetricsMap(),

    /** True once the stage has finished (successfully or with a [failure]). */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    var isFinished: Boolean = false,

    /**
     * Indicates the expected total number of substages for computing ETC.
     */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    var substagesTotal: Int? = null,

    /**
     * Wall-clock duration of this stage, including any work done outside of substages.
     */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    @Serializable(with = PrettyRoundedDurationSerializer::class)
    var realElapsed: Duration = Duration.ZERO,

    /** Child records (nested stages, agent runs, prompt executions) of this stage, in execution order. */
    val substages: MutableList<TrainingRecord> = mutableListOf(),
) : TrainingRecord()

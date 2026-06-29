package ai.koog.agents.optimization.training.records


import ai.koog.agents.optimization.common.ExecutionMetadata
import kotlinx.serialization.Serializable

/**
 * Top-level persisted result of a training session: the tree of [StageRecord]s plus the
 * [ExecutionMetadata] captured by the runner (pod name, start/completion timestamps).
 *
 * This is the on-disk shape of `training_records.json`. The cluster progress stream still
 * emits a bare projected [StageRecord] -- live consumers already know pod/timestamps from
 * the Kubernetes API and don't need metadata duplicated into each tick.
 */
@Serializable
public data class TrainingResult(
    /** Root of the training records tree for this session. */
    val rootStage: StageRecord,
    /** Runner-captured metadata: pod name, start and completion timestamps. */
    val executionMetadata: ExecutionMetadata,
)

package ai.koog.agents.optimization.training.records


import ai.koog.agents.optimization.common.ExecutionMetadata
import ai.koog.agents.optimization.common.ExperimentName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * Top-level persisted result of a training session, and the on-disk shape of `training_records.json`;
 * its properties are declared in the order that reads best there.
 *
 * Live cluster progress is emitted separately, as a bare projected [StageRecord].
 */
@Serializable
public data class TrainingResult(
    /** Identity of this training run (`runId`, `submissionId`, optimizer, agent). */
    val trainingName: ExperimentName,
    /** Runner-captured metadata: pod name, start and completion timestamps. */
    val executionMetadata: ExecutionMetadata,
    /** Root of the training records tree for this session. */
    val rootStage: StageRecord,
    /**
     * The run's configuration serialized as JSON, kept in the artifact for reference.
     * `null` when the caller specified none.
     */
    val resolvedConfigurationDump: JsonElement? = null,
)

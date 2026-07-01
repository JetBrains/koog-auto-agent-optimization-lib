package ai.koog.agents.optimization.common


import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Identity of an experiment run, used to name and group its persisted artifacts.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
public data class ExperimentName(
    /** Unique identifier of this run. */
    @SerialName("runId")
    val runId: String,

    /** Optional identifier of the cluster submission that launched this run. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    @SerialName("submissionId")
    val submissionId: String? = null,

    /**
     * Name of the optimizer driving this run. Plain evaluation with no optimizer uses a descriptive
     * name instead (e.g. `"Baseline (no optimizer)"`), so this is always set.
     */
    @SerialName("optimizer")
    val optimizerName: String,

    /** Name of the agent under experiment. */
    @SerialName("agent")
    val agentName: String,

    /** Optional evaluation group name (e.g. "easy", "medium", "a"). */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    @SerialName("group")
    val groupName: String? = null,
)

package ai.koog.agents.optimization.common


import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import java.time.Instant

/**
 * Metadata about the runtime environment and lifecycle of an experiment execution
 * (training session or evaluation run).
 */
@Serializable
@OptIn(ExperimentalSerializationApi::class)
public data class ExecutionMetadata(
    /** ISO 8601 timestamp of when the runner started processing. */
    val startedAt: String,

    /** ISO 8601 timestamp of when the runner finished processing; `null` while still in progress. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val completedAt: String? = null,
    /**
     * Whether the runner finished processing successfully (set to `true` alongside [completedAt]).
     * Remains `false` if the process was interrupted, crashed, or was aborted mid-run.
     */
    val completed: Boolean = false,

    /** Kubernetes pod name (set in cluster runs, `null` in local runs). */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val podName: String? = null,

    /**
     * The spend cap configured for this execution at session start, stored as the opaque JSON the
     * caller persisted (the library is provider-agnostic, so the concrete config shape lives in the
     * consuming application).
     */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val spendLimit: JsonElement? = null,

    /** The retry policy configured for this execution at session start, stored as opaque JSON like [spendLimit]. */
    @EncodeDefault(EncodeDefault.Mode.NEVER)
    val retries: JsonElement? = null,
) {
    /** Factory for [ExecutionMetadata] instances. */
    public companion object {
        /** Creates metadata for an execution starting now, with [startedAt] set to the current instant. */
        public fun createAsStartedNow(
            podName: String? = null,
            spendLimit: JsonElement? = null,
            retries: JsonElement? = null,
        ): ExecutionMetadata = ExecutionMetadata(
            startedAt = Instant.now().toString(),
            podName = podName,
            spendLimit = spendLimit,
            retries = retries,
        )
    }

    /** Returns a copy marked completed now: [completed] set to `true` and [completedAt] to the current instant. */
    public fun copyAsCompletedNow(): ExecutionMetadata = copy(
        completedAt = Instant.now().toString(),
        completed = true,
    )
}

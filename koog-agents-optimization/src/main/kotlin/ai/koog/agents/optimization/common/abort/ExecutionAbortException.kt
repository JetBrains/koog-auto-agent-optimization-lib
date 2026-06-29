package ai.koog.agents.optimization.common.abort


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.FailureKind
import ai.koog.agents.optimization.common.TransiencyLevel
import ai.koog.agents.optimization.consumption.LLMConsumption

/**
 * Marker for an execution-level abort: the runner has decided to stop the current training
 * session or evaluation run as a whole, rather than failing a single stage or leaf operation.
 *
 * Subtypes carry the specific cause (e.g. [SpendLimitExceededException]). They are thrown
 * only from [AbortController.abort], which guarantees at most one abort per execution.
 * Catch sites recognize abort exceptions via the consuming application's failure analyzers,
 * so most code paths don't need a bespoke `catch` arm for them.
 */
public sealed class ExecutionAbortException(message: String) : RuntimeException(message) {
    /** Short identifier of the abort cause (e.g. `"SpendLimitExceeded"`). */
    public abstract val abortResolvedId: String

    /** Default mapping of an abort to an [AnalyzedFailure] for record-keeping. */
    public open fun toAnalyzedFailure(): AnalyzedFailure = AnalyzedFailure(
        resolvedId = abortResolvedId,
        description = message ?: error("unreachable: Exception message is set in the constructor"),
        kind = FailureKind.EXECUTION_ABORTED,
        transiency = TransiencyLevel.NONE,
        originalStackTrace = stackTraceToString(),
    )
}

/**
 * Cumulative LLM consumption crossed the configured spend cap.
 *
 * @property configuredLimit the cap as configured at session start.
 * @property consumedAtAbort the running cumulative observed when the cap was crossed.
 */
public class SpendLimitExceededException(
    public val configuredLimit: LLMConsumption,
    public val consumedAtAbort: LLMConsumption,
) : ExecutionAbortException(
    "Spend limit exceeded: " +
            "consumed ${consumedAtAbort.toPrettyTotal()} > configured limit ${configuredLimit.toPrettyTotal()}"
) {
    override val abortResolvedId: String = "SpendLimitExceeded"
}

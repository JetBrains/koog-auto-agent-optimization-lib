package ai.koog.agents.optimization.common.abort


/**
 * Per-execution gate that lets exactly one [ExecutionAbortException] escape an experiment
 * execution. Construct one per training session / evaluation run and share it with every
 * [AbortPolicy].
 *
 * Not thread-safe — a plain `Boolean` flag is enough for the single-dispatcher runner loop.
 */
public class AbortController {
    private var aborted: Boolean = false

    /**
     * Throws an [ExecutionAbortException] produced by [produceException] on the first call;
     * silently returns on subsequent calls without invoking [produceException].
     */
    public fun abort(produceException: () -> ExecutionAbortException) {
        if (aborted) return
        aborted = true
        throw produceException()
    }
}

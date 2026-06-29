package ai.koog.agents.optimization.common.retries


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.TransiencyLevel
import ai.koog.agents.optimization.common.abort.ExecutionAbortException
import ai.koog.agents.optimization.utils.serialization.LLMConsumptionOrNA
import io.github.oshai.kotlinlogging.KLogger
import kotlinx.coroutines.delay
import kotlin.time.Duration

/**
 * Resolved-id of the heuristic recognizer for Koog's OpenTelemetry span-cleanup wrapper. Carried as
 * a constant because the retry executor reads it to emit a tailored WARN addendum on exhausted retries,
 * and failure recognizers produce it when they detect the wrapper.
 */
public const val GenAIAgentSpanCleanupResolvedId: String = "LLMCallInFlight:GenAIAgentSpanCleanup"

/**
 * Metrics the retry wrapper collects after each attempt (success or failure). The caller supplies
 * them via [retryWith]'s `collectAttemptMetrics` callback — typically by reading
 * `consumptionTracker.collectAndClear()` and a captured `TimeMark.elapsedNow()`.
 */
public data class AttemptMetrics(
    /** Wall-clock time the attempt took. */
    val elapsedTime: Duration,
    /** LLM consumption recorded for the attempt, or N/A when unavailable. */
    val consumption: LLMConsumptionOrNA,
)

/**
 * Outcome of a single [retryWith] invocation; describes the whole retry sequence, not a single retry attempt.
 * The leaf-record constructors consume this to build their `Completed` / `Failed` records.
 *
 * - On [Completed]: [previousAttempts] lists every failed attempt that preceded the successful one.
 *   [finalAttemptMetrics] holds the successful attempt's metrics.
 * - On [Failed]: [Failed.finalFailure] / [finalAttemptMetrics] describe the last (failed) attempt;
 *   [previousAttempts] lists every prior failed attempt (so `previousAttempts.size + 1` equals
 *   the total attempts made).
 */
public sealed class RetriesOutcome<out A> {
    /** Metrics of the final attempt (the successful one on [Completed], the last failed one on [Failed]). */
    public abstract val finalAttemptMetrics: AttemptMetrics

    /** Every failed attempt that preceded the final attempt, in execution order. */
    public abstract val previousAttempts: List<FailedAttempt>

    /** The operation succeeded on its final attempt. */
    public data class Completed<out A>(
        /** Value produced by the successful attempt. */
        val result: A,
        override val finalAttemptMetrics: AttemptMetrics,
        override val previousAttempts: List<FailedAttempt>,
    ) : RetriesOutcome<A>()

    /** The operation failed on its final attempt and was not retried further. */
    public data class Failed(
        /** Classified failure of the final (failed) attempt. */
        val finalFailure: AnalyzedFailure,
        /** The exception thrown by the final attempt. */
        val finalException: Exception,
        override val finalAttemptMetrics: AttemptMetrics,
        override val previousAttempts: List<FailedAttempt>,
    ) : RetriesOutcome<Nothing>()
}

/**
 * Runs [block] up to `policy.maxAttempts + 1` times (one initial attempt plus up to
 * `maxAttempts` retries), capturing per-attempt metrics via [collectAttemptMetrics] and applying
 * retry / delay decisions per [policy].
 *
 * **Abort semantics.** [ExecutionAbortException] thrown by [block] short-circuits the retry loop
 * and propagates to the caller — aborts always trump retries (a spend-limit breach or operator
 * cancel must not be silently re-attempted). Before rethrowing, [onAbort] is invoked with the
 * collected `previousAttempts` plus the metrics of the just-interrupted attempt, so the caller can
 * record a complete leaf entry (with prior failed attempts preserved) before the abort propagates.
 *
 * **Observability.** All retry-observability log events ("planned retry", "retries exhausted",
 * etc.) are emitted from inside this helper via [logger] so all three retry call sites
 * (training agent run, training prompt execution, eval per-item) produce identical log output.
 *
 * @param policy active retry policy (resolved from session settings or a per-call override).
 * @param operationLabel human-readable identifier of the operation under retry — included in log
 *   messages (e.g., `"agent run for item 7 / 50"`). Lazy: only invoked when a log event fires.
 * @param logger logger to emit retry-observability events on.
 * @param analyzeFailure converts a caught Throwable into the project's [AnalyzedFailure]. Each call
 *   site passes the analyzer matching its context (`analyzeAgentRunFailure` for runAgent,
 *   `analyzeTrainingFailure` for prompt execution, etc.).
 * @param collectAttemptMetrics returns the metrics of the just-finished attempt (success or failure).
 *   Called once after each attempt completes — including on the abort path, where the metrics of
 *   the interrupted attempt are read once and passed to [onAbort].
 * @param onAttemptStarting optional hook called *before* each attempt starts. Receives the 0-based
 *   `attemptIndex` (`0` = original attempt; `≥ 1` = retry). Used for per-attempt instrumentation
 *   (e.g., resetting consumption trackers, decorating the Langfuse trace name).
 * @param onAbort optional hook called *inside* the abort-catch arm, before the abort is rethrown.
 *   Receives the abort exception, the `previousAttempts` collected before the abort fired, and the
 *   metrics of the attempt-that-was-interrupted. The default is a no-op (suitable for call sites
 *   where the enclosing flow records the abort at a higher level — e.g. eval-side `evaluateItem`).
 *   The retry helper itself rethrows the abort after this hook returns; callers should not rethrow
 *   from inside the hook.
 * @param block the operation to retry. Receives the 0-based `attemptIndex`.
 */
public suspend fun <A> retryWith(
    policy: RetryPolicy,
    operationLabel: () -> String,
    logger: KLogger,
    analyzeFailure: (exception: Exception) -> AnalyzedFailure,
    collectAttemptMetrics: () -> AttemptMetrics,
    onAttemptStarting: (attemptIndex: Int) -> Unit = {},
    onAbort: (
        abort: ExecutionAbortException,
        previousAttempts: List<FailedAttempt>,
        abortedAttemptMetrics: AttemptMetrics,
    ) -> Unit = { _, _, _ -> },
    block: suspend (attemptIndex: Int) -> A,
): RetriesOutcome<A> {
    val previousAttempts = mutableListOf<FailedAttempt>()
    var attemptIndex = 0
    var delayBeforeAttempt: Duration = Duration.ZERO

    while (true) {
        onAttemptStarting(attemptIndex)
        try {
            // TODO: add per-attempt wall-clock cap. Observed locally during a manual Wi-Fi-cutoff
            //   test: when the network drops after an HTTP request is in-flight, the LLM HTTP
            //   client's read suspends forever (or at least for really long), block()
            //   never returns, and this loop never reaches the catch. Future work: optional
            //   `withTimeoutOrNull(perAttemptCap)` around block() driven by a new RetryPolicy
            //   field, with a typed PerAttemptTimeoutException recognised as TRANSIENT so the
            //   policy retries the next attempt instead of hanging. Deferred until the issue is
            //   observed on a real cluster run.
            val result = block(attemptIndex)
            val attemptMetrics = collectAttemptMetrics()
            return RetriesOutcome.Completed(result, attemptMetrics, previousAttempts.toList())
        } catch (abort: ExecutionAbortException) {
            // Abort short-circuits retries unconditionally. Collect this attempt's metrics so the
            // hook can record a complete leaf (including any prior failed attempts), then rethrow.
            val abortedAttemptMetrics = collectAttemptMetrics()
            onAbort(abort, previousAttempts.toList(), abortedAttemptMetrics)
            throw abort
        } catch (exception: Exception) {
            val attemptMetrics = collectAttemptMetrics()
            val failure = analyzeFailure(exception)

            if (policy.shouldRetry(failure, attemptIndex)) {
                val nextDelay = policy.nextDelay(attemptIndex)
                // Gate the Lenient WARN to the *first* retry decision
                // so the notice fires exactly once per unknown-failure sequence.
                if (attemptIndex == 0 &&
                    failure.transiency == TransiencyLevel.UNKNOWN &&
                    policy.unknownFailurePolicy == UnknownFailurePolicy.LENIENT
                ) {
                    emitUnknownFailureReportNotice(
                        logger, operationLabel, failure, mode = UnknownFailurePolicy.LENIENT,
                    )
                }
                logger.info {
                    val nextAttemptNumber = attemptIndex + 2 // human-numbered (1 = original)
                    val totalAttempts = policy.maxAttempts + 1
                    "[RETRIES] ${operationLabel()} failed with transient ${failure.resolvedId} " +
                            "(transiency=${failure.transiency}). " +
                            "Will retry: attempt $nextAttemptNumber of $totalAttempts in $nextDelay."
                }
                previousAttempts.add(
                    FailedAttempt(
                        attemptIndex = attemptIndex,
                        failure = failure,
                        elapsedTime = attemptMetrics.elapsedTime,
                        consumption = attemptMetrics.consumption,
                        delayBeforeAttempt = delayBeforeAttempt,
                    )
                )
                logger.debug { "[RETRIES] Delay before retry attempt ${attemptIndex + 2} starting ($nextDelay)." }
                delay(nextDelay)
                logger.debug { "[RETRIES] Delay before retry attempt ${attemptIndex + 2} ended." }
                delayBeforeAttempt = nextDelay
                attemptIndex++
                continue
            }

            // Not retrying — emit the appropriate observability event then return Failed outcome.
            emitNonRetryEvent(logger, operationLabel, failure, policy)
            return RetriesOutcome.Failed(
                finalFailure = failure,
                finalException = exception,
                finalAttemptMetrics = attemptMetrics,
                previousAttempts = previousAttempts.toList(),
            )
        }
    }
}

/**
 * Emits the appropriate WARN/INFO event explaining why the retry loop is *not* retrying this
 * failure: non-transient (NONE — terminal), conservative refusal (UNKNOWN + Conservative), or
 * exhausted attempts (TRANSIENT past maxAttempts, or UNKNOWN past maxAttempts under Lenient).
 * The non-transient bucket logs at INFO since it's the routine "this is not retryable" path.
 */
private fun emitNonRetryEvent(
    logger: KLogger,
    operationLabel: () -> String,
    failure: AnalyzedFailure,
    policy: RetryPolicy,
) {
    when (val transiency = failure.transiency) {
        TransiencyLevel.NONE -> {
            logger.info {
                "[RETRIES] Not retrying ${operationLabel()}: failure ${failure.resolvedId} is non-transient " +
                        "(kind=${failure.kind}, transiency=NONE)."
            }
        }
        TransiencyLevel.UNKNOWN if policy.unknownFailurePolicy == UnknownFailurePolicy.CONSERVATIVE -> {
            emitUnknownFailureReportNotice(
                logger, operationLabel, failure, mode = UnknownFailurePolicy.CONSERVATIVE,
            )
        }
        else -> {
            // Exhausted retries — TRANSIENT past maxAttempts (or UNKNOWN past maxAttempts under Lenient).
            val totalAttempts = policy.maxAttempts + 1
            val genAINote = if (failure.resolvedId == GenAIAgentSpanCleanupResolvedId) {
                "\n[RETRIES] The exhausted retries above were on a GenAIAgentSpan wrapper, whose underlying " +
                        "cause is not preserved on the exception. Some sub-cases hidden by this wrapper " +
                        "(e.g., request-timeouts) are non-retryable — they would have wasted these retry attempts. " +
                        "If exhaustion on this resolvedId is frequent, consider implementing a buffer-based " +
                        "mitigation to recover the original cause and skip retries on non-transient sub-cases."
            } else ""
            logger.warn {
                "[RETRIES] Retries exhausted on ${operationLabel()}. " +
                        "Final failure: ${failure.resolvedId} (transiency=$transiency). " +
                        "Spent $totalAttempts attempts. " +
                        "If this resolvedId frequently exhausts retries, consider extending recognition or " +
                        "raising the retry budget." + genAINote
            }
        }
    }
}

/**
 * The "please report unrecognized failure" WARN notice shared by the Conservative-refused and the
 * Lenient-retried paths. The two paths produce the same body with one differing sentence stating
 * whether the policy retried the failure or not — both surface the unrecognized failure to the
 * operator so they can decide whether to add a recognizer.
 */
private fun emitUnknownFailureReportNotice(
    logger: KLogger,
    operationLabel: () -> String,
    failure: AnalyzedFailure,
    mode: UnknownFailurePolicy,
) {
    val verdict = when (mode) {
        UnknownFailurePolicy.LENIENT -> "Lenient unknown-failure policy retried it as TRANSIENT anyway"
        UnknownFailurePolicy.CONSERVATIVE -> "Conservative unknown-failure policy did not retry it"
    }
    val toggleHint = when (mode) {
        UnknownFailurePolicy.LENIENT ->
            "To stop retrying unknowns, set `experiment.retries.unknown_failure_policy = \"conservative\"`."
        UnknownFailurePolicy.CONSERVATIVE ->
            "To retry unknowns unconditionally, set `experiment.retries.unknown_failure_policy = \"lenient\"`."
    }
    logger.warn {
        """
        [RETRIES] Encountered unrecognized failure (kind=${failure.kind}, resolvedId=${failure.resolvedId}) on ${operationLabel()}.
        $verdict. If this is a transient failure that should be retried, please file an issue or
        add a recognizer in `experimentInfrastructure/common/ErrorHandling.kt`. $toggleHint
        """.trimIndent()
    }
}

package ai.koog.agents.optimization.common.retries


import ai.koog.agents.optimization.common.AnalyzedFailure
import ai.koog.agents.optimization.common.TransiencyLevel
import ai.koog.agents.optimization.common.retries.UnknownFailurePolicy.CONSERVATIVE
import ai.koog.agents.optimization.common.retries.UnknownFailurePolicy.LENIENT
import kotlin.math.pow
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/**
 * Per-attempt wait-time growth strategy applied to transient-failure retries.
 *
 * `attemptIndex` is 0-based: index `0` is the first retry (i.e. the original attempt has already
 * failed once); index `1` is the second retry; etc.
 */
public enum class DelayStrategy {
    /** Constant delay equal to [RetryPolicy.baseDelaySeconds]. Same wait between every retry attempt. */
    FIXED,

    /**
     * Delay grows linearly with attempt index: `delay = baseDelaySeconds × (attemptIndex + 1)`,
     * then clamped to `maxDelaySeconds`. With `base=10s, max=300s`: 10s, 20s, 30s, 40s, ...
     */
    LINEAR,

    /**
     * Delay grows exponentially: `delay = baseDelaySeconds × 2^attemptIndex`, clamped to
     * `maxDelaySeconds`. With `base=10s, max=300s`:
     * attemptIndex 0 → 10s; 1 → 20s; 2 → 40s; 3 → 80s; 4 → 160s; 5 onward → 300s (capped).
     */
    EXPONENTIAL,
}

/**
 * How the retry policy treats failures of [TransiencyLevel.UNKNOWN] transiency (i.e. the recognizer
 * couldn't classify the exception).
 *
 * - [CONSERVATIVE] (default): don't retry unknown failures.
 * - [LENIENT]: retry unknown failures as if they were TRANSIENT.
 */
public enum class UnknownFailurePolicy {
    /** Don't retry unknown failures (default). */
    CONSERVATIVE,

    /** Retry unknown failures as if they were [TransiencyLevel.TRANSIENT]. */
    LENIENT,
}

/**
 * Runtime policy consumed by the retry wrapper. Translates an observed failure into
 * "should we retry now?" + "if so, wait how long?".
 *
 * `attemptIndex` is 0-based throughout: `0` means "the original attempt has just failed; should we
 * retry once?"; `1` means "the first retry has just failed; should we retry again?"; etc.
 */
public class RetryPolicy(
    /** Maximum number of retries after the original attempt (the original attempt itself is not counted). */
    public val maxAttempts: Int,
    /** Per-attempt delay growth strategy applied between retries. */
    public val strategy: DelayStrategy = DelayStrategy.EXPONENTIAL,
    /** Base delay in seconds, used as the starting point by every [DelayStrategy]. */
    public val baseDelaySeconds: Double = 10.0,
    /** Upper bound in seconds that any computed delay is clamped to. */
    public val maxDelaySeconds: Double = 300.0,
    /** How failures of [TransiencyLevel.UNKNOWN] transiency are treated. */
    public val unknownFailurePolicy: UnknownFailurePolicy = UnknownFailurePolicy.CONSERVATIVE,
) {
    /**
     * Returns `true` if the policy decides to retry after this particular failure.
     *
     * @param attemptIndex 0-based count of failures *so far* (i.e. number of retries already
     *   performed). The next retry would be `attemptIndex + 1`; we retry only while
     *   `attemptIndex < maxAttempts`.
     */
    public fun shouldRetry(failure: AnalyzedFailure, attemptIndex: Int): Boolean = when (failure.transiency) {
        TransiencyLevel.NONE -> false
        TransiencyLevel.TRANSIENT -> attemptIndex < maxAttempts
        TransiencyLevel.UNKNOWN -> when (unknownFailurePolicy) {
            UnknownFailurePolicy.CONSERVATIVE -> false
            UnknownFailurePolicy.LENIENT -> attemptIndex < maxAttempts
        }
    }

    /**
     * Computes the wall-clock delay before the next retry attempt, per the configured strategy
     * and capped at [maxDelaySeconds].
     *
     * _Note:_ Per-`resolvedId` overrides are not supported in v1
     * (kept in the design as future work), so the delay depends only on the attempt index.
     */
    public fun nextDelay(attemptIndex: Int): Duration {
        val seconds = when (strategy) {
            DelayStrategy.FIXED -> baseDelaySeconds
            DelayStrategy.LINEAR -> baseDelaySeconds * (attemptIndex + 1)
            DelayStrategy.EXPONENTIAL -> baseDelaySeconds * 2.0.pow(attemptIndex.toDouble())
        }.coerceAtMost(maxDelaySeconds)
        return seconds.seconds
    }

    /** Predefined [RetryPolicy] instances. */
    public companion object {
        /** Disabled policy — never retries. Use in tests / call-sites that need single-attempt semantics. */
        public val None: RetryPolicy = RetryPolicy(maxAttempts = 0)
    }
}

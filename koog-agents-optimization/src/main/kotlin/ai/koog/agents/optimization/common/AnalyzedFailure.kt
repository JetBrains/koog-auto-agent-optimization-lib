package ai.koog.agents.optimization.common


import kotlinx.serialization.Serializable

/**
 * A failure that has been recognized and classified by the experiment infrastructure.
 *
 * Carries a stable [resolvedId] (used to group identical failures across runs), a human-readable
 * [description], a [FailureKind] bucket, and a [TransiencyLevel] consumed by the retry policy.
 * The recognizers that produce these live in the consuming application — the library only owns the
 * vocabulary.
 */
@Serializable
public data class AnalyzedFailure(
    /** Stable identifier used to group identical failures across runs. */
    val resolvedId: String,
    /** Human-readable description of what went wrong. */
    val description: String,
    /** High-level [FailureKind] bucket this failure belongs to. */
    val kind: FailureKind,
    /**
     * Transiency assignment consumed by the retry policy. Recognizers set this honestly
     * based on what the exception alone can reveal — duration variance ("a transient outage
     * can last seconds or hours") is handled by the retry strategy, not by the enum.
     */
    val transiency: TransiencyLevel,
    /** Stack trace of the originating exception, captured as a string for record-keeping. */
    val originalStackTrace: String,
) {
    /**
     * Builds an [AnalyzedFailure] from an [exception], capturing its stack trace into
     * [originalStackTrace].
     */
    public constructor(
        resolvedId: String,
        description: String,
        kind: FailureKind,
        transiency: TransiencyLevel,
        exception: Exception,
    ) : this(
        resolvedId = resolvedId,
        description = description,
        kind = kind,
        transiency = transiency,
        originalStackTrace = exception.stackTraceToString(),
    )
}

/**
 * Resolved-ids the library itself references by value. Recognizers (which live in the consuming app)
 * produce these when they detect the corresponding failure, and the library reads them — e.g. the
 * retry executor emits a tailored WARN addendum when an exhausted retry carries
 * [GEN_AI_AGENT_SPAN_CLEANUP]. Keeping them here, next to the failure taxonomy, keeps the WARN and
 * the recognizer in sync.
 */
public object KnownResolvedIds {
    /** Koog's OpenTelemetry span-cleanup wrapper (`LLMCallInFlight:GenAIAgentSpanCleanup`). */
    public const val GEN_AI_AGENT_SPAN_CLEANUP: String = "LLMCallInFlight:GenAIAgentSpanCleanup"
}

/**
 * High-level categorization of failures occurred during experiments:
 * - AGENT_EXECUTION: likely related to agent logic / prompting / model capability (could potentially be improved).
 * - EXTERNAL_ENVIRONMENT: infrastructure issues (network / provider / API / MCP server), usually worth re-checking environment and rerunning.
 * - EXECUTION_ABORTED: the runner deliberately stopped the execution in response to a triggering condition such as a spend-limit breach.
 *   The specific cause lives in [AnalyzedFailure.resolvedId].
 * - UNKNOWN: not recognized by our heuristics yet.
 */
public enum class FailureKind {
    /** Likely related to agent logic / prompting / model capability (could potentially be improved). */
    AGENT_EXECUTION,

    /** Infrastructure issues (network / provider / API / MCP server); usually worth re-checking environment and rerunning. */
    EXTERNAL_ENVIRONMENT,

    /**
     * The runner deliberately stopped the execution in response to a triggering condition such as a
     * spend-limit breach. The specific cause lives in [AnalyzedFailure.resolvedId].
     */
    EXECUTION_ABORTED,

    /** Not recognized by our heuristics yet. */
    UNKNOWN,
}

/**
 * Retry-policy input. Recognizers classify only what they can honestly determine from the
 * exception; the retry strategy (e.g. exponential backoff with a generous cap) handles "how long
 * a transient outage will last".
 */
public enum class TransiencyLevel {
    /** Definitely not transient — don't retry (bad API key, agent stuck, context overflow, 15-min LiteLLM request timeout). */
    NONE,

    /** Transient — retry per session policy. The strategy handles "how long" via, for example, exponential backoff. */
    TRANSIENT,

    /** Recognizer doesn't know — policy decides (Conservative = don't retry, Lenient = retry as TRANSIENT). */
    UNKNOWN,
}

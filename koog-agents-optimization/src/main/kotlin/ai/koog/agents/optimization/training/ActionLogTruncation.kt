package ai.koog.agents.optimization.training


/**
 * Controls truncation of string values in substep action logs.
 *
 * Truncation is applied recursively to the entire built JSON tree — including strings
 * nested inside serialized objects or arrays.
 */
public sealed class ActionLogTruncation {

    /** No truncation applied to action log strings. */
    public data object Unlimited : ActionLogTruncation()

    /**
     * Truncate every string value in the action log to [maxChars] characters.
     *
     * @property maxChars Maximum kept length; must be positive.
     */
    public data class MaxChars(public val maxChars: Int) : ActionLogTruncation() {
        init {
            require(maxChars > 0) { "maxChars must be positive, got $maxChars" }
        }
    }

    /** Holds the library default truncation. */
    public companion object {
        /** Default truncation applied when none is configured: [MaxChars] of 1000. */
        public val DEFAULT: ActionLogTruncation = MaxChars(1000)
    }
}

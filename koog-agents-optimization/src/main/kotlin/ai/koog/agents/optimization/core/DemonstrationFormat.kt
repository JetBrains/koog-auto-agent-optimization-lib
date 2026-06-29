package ai.koog.agents.optimization.core

/**
 * Controls the level of detail included when rendering a demonstration into the prompt.
 */
public enum class DemonstrationFormat {
    /**
     * Include only the input and output of the demonstration.
     */
    COMPACT,

    /**
     * Include the full intermediate trace (tool calls, reasoning, etc.) alongside input and output.
     * Falls back to [COMPACT] if the demonstration has no intermediate messages.
     */
    FULL_TRACE,
}

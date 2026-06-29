package ai.koog.agents.optimization.core

/**
 * Controls how few-shot demonstrations are inserted into the prompt.
 */
public enum class FewShotPromptType {
    /**
     * Concatenate all demonstrations into a single string and insert as one user message.
     */
    AS_STRING,

    /**
     * Insert demonstrations as individual messages in the conversation history.
     * System messages from demonstration traces are remapped to user messages to avoid
     * system messages appearing in the middle of the conversation.
     */
    AS_MESSAGE_HISTORY,
}

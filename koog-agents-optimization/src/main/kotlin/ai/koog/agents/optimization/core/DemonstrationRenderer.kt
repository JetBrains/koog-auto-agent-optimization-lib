package ai.koog.agents.optimization.core

import ai.koog.prompt.message.Message
import ai.koog.prompt.message.RequestMetaInfo
import ai.koog.prompt.message.ResponseMetaInfo
import kotlin.time.Clock

/**
 * Renders demonstrations into prompt-ready content based on the chosen format and insertion mode.
 */
public object DemonstrationRenderer {

    /**
     * Renders a list of demonstrations as a single concatenated string.
     *
     * Used with [FewShotPromptType.AS_STRING] to produce a single user message containing
     * all demonstrations.
     *
     * @param demonstrations The demonstrations to render.
     * @param format Whether to include intermediate traces or only input/output.
     * @return A formatted string with all demonstrations, or null if the list is empty.
     */
    public fun renderAsString(
        demonstrations: List<Demonstration>,
        format: DemonstrationFormat,
    ): String? {
        if (demonstrations.isEmpty()) return null

        return demonstrations.joinToString("\n\n") { demo ->
            renderSingleDemoAsString(demo, format)
        }
    }

    /**
     * Renders a list of demonstrations as individual messages for conversation history insertion.
     *
     * Used with [FewShotPromptType.AS_MESSAGE_HISTORY] to produce user/assistant message pairs.
     * System messages from intermediate traces are remapped to user messages to prevent
     * system messages from appearing in the middle of the conversation.
     *
     * @param demonstrations The demonstrations to render.
     * @param format Whether to include intermediate traces or only input/output.
     * @return A list of messages ready for prompt insertion, empty if no demonstrations.
     */
    public fun renderAsMessages(
        demonstrations: List<Demonstration>,
        format: DemonstrationFormat,
    ): List<Message> {
        if (demonstrations.isEmpty()) return emptyList()

        return demonstrations.flatMap { demo ->
            renderSingleDemoAsMessages(demo, format)
        }
    }

    private fun renderSingleDemoAsString(demo: Demonstration, format: DemonstrationFormat): String =
        buildString {
            appendLine("Input: ${demo.input}")
            if (format == DemonstrationFormat.FULL_TRACE && demo.intermediateMessages != null) {
                appendLine("Trace:")
                for (message in demo.intermediateMessages) {
                    appendLine("  [${message.role}] ${message.content}")
                }
            }
            append("Output: ${demo.output}")
        }

    private fun renderSingleDemoAsMessages(
        demo: Demonstration,
        format: DemonstrationFormat,
    ): List<Message> = buildList {
        if (format == DemonstrationFormat.FULL_TRACE && demo.intermediateMessages != null) {
            addAll(demo.intermediateMessages.map { remapSystemToUser(it) })
        } else {
            // TODO: Does this preserve all correctness invariants on the timestamps?
            add(Message.User(demo.input, RequestMetaInfo(Clock.System.now())))
            add(Message.Assistant(demo.output, ResponseMetaInfo(Clock.System.now())))
        }
    }

    /**
     * Remaps system messages to user messages to prevent system messages
     * from appearing in the middle of the conversation history.
     */
    private fun remapSystemToUser(message: Message): Message = when (message) {
        is Message.System -> Message.User(
            content = message.content,
            metaInfo = RequestMetaInfo(timestamp = message.metaInfo.timestamp),
        )
        else -> message
    }

    /**
     * Strips a common prefix of inherited messages from a full message list.
     *
     * Used to extract only the messages that a subgraph added to the prompt,
     * excluding messages inherited from the parent context (relevant when
     * `freshHistory = false`). Messages are compared by role and content.
     *
     * @param allMessages The full prompt messages at the end of subgraph execution.
     * @param inherited The prompt messages captured before the subgraph started.
     * @return Only the messages added by the subgraph.
     */
    public fun dropInheritedPrefix(
        allMessages: List<Message>,
        inherited: List<Message>,
    ): List<Message> {
        var matchCount = 0
        for (i in inherited.indices) {
            if (i < allMessages.size
                && allMessages[i].role == inherited[i].role
                && allMessages[i].content == inherited[i].content
            ) {
                matchCount++
            } else {
                break
            }
        }
        return allMessages.drop(matchCount)
    }
}

package ai.koog.agents.optimization.core

import ai.koog.agents.optimization.core.DemonstrationRenderer.SYNTHETIC_TOOL_RESULT
import ai.koog.agents.optimization.utils.messages.renderParts
import ai.koog.agents.optimization.utils.messages.toolCalls
import ai.koog.agents.optimization.utils.messages.toolResults
import ai.koog.agents.optimization.utils.messages.withParts
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import ai.koog.prompt.message.RequestMetaInfo
import ai.koog.prompt.message.ResponseMetaInfo
import kotlin.time.Clock
import kotlin.time.Instant

/**
 * Renders demonstrations into prompt-ready content based on the chosen format and insertion mode.
 */
public object DemonstrationRenderer {

    /**
     * Output of the synthetic tool result appended for tool calls that never received one.
     *
     * A trace can legitimately end on an unanswered tool call when the strategy terminates on
     * it (e.g. a commit tool that ends the run before its result is recorded), but LLM APIs
     * reject a conversation where an assistant tool call has no matching tool result. The
     * synthetic result closes such calls so the demonstration stays a valid conversation.
     */
    internal const val SYNTHETIC_TOOL_RESULT: String = "<task ended>"

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
                    appendLine("  [${message.role}] ${message.renderParts(separator = " ")}")
                }
            }
            append("Output: ${demo.output}")
        }

    private fun renderSingleDemoAsMessages(
        demo: Demonstration,
        format: DemonstrationFormat,
    ): List<Message> = buildList {
        if (format == DemonstrationFormat.FULL_TRACE && demo.intermediateMessages != null) {
            demo.intermediateMessages
                .map { remapSystemToUser(it) }
                .closeUnansweredToolCalls()
                .let { addAll(it) }
        } else {
            // TODO: Does this preserve all correctness invariants on the timestamps?
            add(Message.User(demo.input, RequestMetaInfo(Clock.System.now())))
            add(Message.Assistant(demo.output, ResponseMetaInfo(Clock.System.now())))
        }
    }

    /**
     * Inserts a synthetic [Message.User] carrying a [MessagePart.Tool.Result]
     * (with [SYNTHETIC_TOOL_RESULT] as output) for every tool call that never received one.
     *
     * A trace can legitimately end on an unanswered tool call when the strategy terminates on it,
     * but LLM APIs reject a conversation where an assistant tool call has no matching result.
     * Unanswered calls are tracked as a list rather than a flag because a message may answer only
     * some of the parallel calls of the preceding assistant turn, and the synthetic result needs
     * the unanswered call's id and tool name.
     *
     * A message that answers only part of a batch gets the missing results appended to it rather
     * than emitted after it: Anthropic requires every `tool_result` of a batch to sit in the single
     * user turn following the `tool_use`, so a trailing second message would be rejected.
     */
    private fun List<Message>.closeUnansweredToolCalls(): List<Message> = buildList {
        val unanswered = mutableListOf<MessagePart.Tool.Call>()
        var unansweredAt: Instant? = null

        fun syntheticResults(): List<MessagePart.Tool.Result> = unanswered.map { call ->
            MessagePart.Tool.Result(id = call.id, tool = call.tool, output = SYNTHETIC_TOOL_RESULT)
        }

        fun closeUnanswered() {
            val timestamp = unansweredAt
            if (unanswered.isEmpty() || timestamp == null) return
            add(Message.User(parts = syntheticResults(), metaInfo = RequestMetaInfo(timestamp)))
            unanswered.clear()
        }

        for (message in this@closeUnansweredToolCalls) {
            val results = message.toolResults()
            if (results.isEmpty()) {
                closeUnanswered()
                add(message)
            } else {
                unanswered.removeAll { call -> results.any { it.id == call.id } }
                add(if (unanswered.isEmpty()) message else message.withParts(message.parts + syntheticResults()))
                unanswered.clear()
            }
            message.toolCalls().takeIf { it.isNotEmpty() }?.let { calls ->
                unanswered += calls
                unansweredAt = message.metaInfo.timestamp
            }
        }
        closeUnanswered()
    }

    /**
     * Remaps system messages to user messages to prevent system messages
     * from appearing in the middle of the conversation history.
     */
    private fun remapSystemToUser(message: Message): Message = when (message) {
        is Message.System -> Message.User(
            parts = message.parts,
            metaInfo = RequestMetaInfo(timestamp = message.metaInfo.timestamp),
        )
        else -> message
    }

    /**
     * Strips a common prefix of inherited messages from a full message list.
     *
     * Used to extract only the messages that a subgraph added to the prompt,
     * excluding messages inherited from the parent context (relevant when
     * `freshHistory = false`). Messages are compared by role and parts.
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
                && allMessages[i].parts == inherited[i].parts
            ) {
                matchCount++
            } else {
                break
            }
        }
        return allMessages.drop(matchCount)
    }
}

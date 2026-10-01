package ai.koog.agents.optimization.optimizers.fewShot


import ai.koog.agents.optimization.utils.messages.toolCalls
import ai.koog.agents.optimization.utils.messages.toolResults
import ai.koog.prompt.Prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.RequestMetaInfo
import ai.koog.prompt.message.ResponseMetaInfo
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Prepares demonstration messages for injection into a few-shot prompt:
 * - System messages are converted to User messages (no System messages
 *   in the middle of the conversation history)
 * - All messages have their metadata stripped (training-run timestamps,
 *   token counts, and other metadata have no place in the evaluation prompt)
 * - Orphaned Tool.Call messages (those without a matching Tool.Result) are removed,
 *   since LLM providers require every tool_call to have a corresponding tool response
 */
public fun Prompt.prepareMessagesForFewShot(): List<Message> =
    removeOrphanedToolCalls(messages).map { msg ->
        when (msg) {
            is Message.System -> Message.User(
                parts = msg.parts,
                metaInfo = RequestMetaInfo.Empty,
            )
            else -> stripMessageMetaInfo(msg)
        }
    }

/**
 * Returns a copy of this [Prompt] with all message metadata stripped
 * and orphaned Tool.Call messages removed.
 *
 * Applied when loading a serialized prompt to ensure training-run
 * metadata (timestamps, token counts) does not leak into evaluation,
 * and the message history is valid for LLM providers (no dangling tool calls).
 */
public fun Prompt.withStrippedMetaInfo(): Prompt =
    Prompt(removeOrphanedToolCalls(messages).map(::stripMessageMetaInfo), id, params)

/**
 * Removes tool call parts that have no matching tool result in the message list, dropping the
 * carrying message entirely if nothing else is left in it.
 *
 * Orphaned tool calls appear when the agent strategy does not route a tool call through
 * the standard execution loop (e.g. using it as a graph-terminating signal without
 * recording the result in the prompt). Properly implemented agents should ensure every
 * tool call produces a corresponding tool result, but older training artifacts or
 * incorrectly wired strategies may still contain orphans.
 *
 * LLM providers (OpenAI, etc.) require every tool_call_id to have a corresponding
 * tool response message, so orphaned calls must be stripped before sending.
 */
public fun removeOrphanedToolCalls(messages: List<Message>): List<Message> {
    val respondedToolCallIds = messages
        .flatMap { it.toolResults() }
        .mapTo(mutableSetOf()) { it.id }

    return messages.mapNotNull { msg ->
        // Only assistant messages can carry tool calls.
        if (msg !is Message.Assistant) return@mapNotNull msg

        val orphans = msg.toolCalls().filter { it.id !in respondedToolCallIds }
        if (orphans.isEmpty()) return@mapNotNull msg

        orphans.forEach {
            logger.info { "Removing orphaned tool call (id=${it.id}, tool=${it.tool}) — no matching tool result" }
        }
        val kept = msg.parts.filter { it !in orphans }
        if (kept.isEmpty()) null else msg.copy(parts = kept)
    }
}

/**
 * Strips metadata from a single message, clearing [ResponseMetaInfo] / [RequestMetaInfo].
 */
private fun stripMessageMetaInfo(msg: Message): Message =
    when (msg) {
        is Message.System -> {
            logger.warn { "Unexpected System message in stripMessageMetaInfo: '${msg.textContent().take(80)}...'" }
            msg.copy(metaInfo = RequestMetaInfo.Empty)
        }
        is Message.Assistant -> msg.copy(metaInfo = ResponseMetaInfo.Empty)
        is Message.User -> msg.copy(metaInfo = RequestMetaInfo.Empty)
    }

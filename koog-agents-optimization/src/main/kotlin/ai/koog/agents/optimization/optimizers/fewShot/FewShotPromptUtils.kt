package ai.koog.agents.optimization.optimizers.fewShot


import ai.koog.prompt.dsl.Prompt
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
                content = msg.content,
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
 * Removes Tool.Call messages that have no matching Tool.Result in the message list.
 *
 * Orphaned tool calls appear when the agent strategy does not route a tool call through
 * the standard execution loop (e.g. using it as a graph-terminating signal without
 * recording the result in the prompt). Properly implemented agents should ensure every
 * tool call produces a corresponding Tool.Result, but older training artifacts or
 * incorrectly wired strategies may still contain orphans.
 *
 * LLM providers (OpenAI, etc.) require every tool_call_id to have a corresponding
 * tool response message, so orphaned calls must be stripped before sending.
 */
public fun removeOrphanedToolCalls(messages: List<Message>): List<Message> {
    val respondedToolCallIds = messages
        .filterIsInstance<Message.Tool.Result>()
        .mapTo(mutableSetOf()) { it.id }

    val filtered = messages.filter { msg ->
        if (msg is Message.Tool.Call && msg.id !in respondedToolCallIds) {
            logger.info { "Removing orphaned Tool.Call (id=${msg.id}, tool=${msg.tool}) — no matching Tool.Result" }
            false
        } else {
            true
        }
    }
    return filtered
}

/**
 * Strips metadata from a single message, clearing [ResponseMetaInfo] / [RequestMetaInfo].
 */
private fun stripMessageMetaInfo(msg: Message): Message =
    when (msg) {
        is Message.System -> {
            logger.warn { "Unexpected System message in stripMessageMetaInfo: '${msg.content.take(80)}...'" }
            msg.copy(metaInfo = RequestMetaInfo.Empty)
        }
        is Message.Response -> msg.copy(updatedMetaInfo = ResponseMetaInfo.Empty)
        is Message.User -> msg.copy(metaInfo = RequestMetaInfo.Empty)
        is Message.Tool.Result -> msg.copy(metaInfo = RequestMetaInfo.Empty)
    }

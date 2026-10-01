package ai.koog.agents.optimization.utils.messages

import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Renders a message's parts as a single human-readable line.
 *
 * Unlike [Message.textContent], which only concatenates text parts, this also renders tool
 * calls and tool results, so a rendered trace does not silently lose the tool interactions
 * that make up most of an agent conversation.
 */
public fun Message.renderParts(separator: String = "\n"): String =
    parts.mapNotNull { it.render() }.joinToString(separator)

/**
 * Renders a single part, or null if it carries no textual information.
 *
 * Non-text payloads are dropped rather than rendered, and each drop is logged: this is on the
 * per-call trajectory path (see MiniSWE's `SweBenchProcessor`), so throwing would take down a run
 * over a rendering detail, but a silent drop would make a truncated trajectory look complete.
 */
public fun MessagePart.render(): String? = when (this) {
    is MessagePart.Text -> text
    is MessagePart.Tool.Call -> "$tool($args)"
    is MessagePart.Tool.Result -> {
        // `output` concatenates the text parts only, so images/files in a tool result are lost.
        val dropped = parts.count { it !is MessagePart.Text }
        if (dropped > 0) logger.warn { "Dropping $dropped non-text part(s) from the '$tool' tool result" }
        output
    }
    is MessagePart.Reasoning -> content.joinToString("\n")
    // TODO: render the payload if an optimizer ever needs to reason about image/file content.
    is MessagePart.Attachment -> {
        logger.warn { "Rendering attachment (${source::class.simpleName}) as a placeholder" }
        "<attachment>"
    }
}

/** All tool calls carried by this message. */
public fun Message.toolCalls(): List<MessagePart.Tool.Call> = parts.filterIsInstance<MessagePart.Tool.Call>()

/** All tool results carried by this message. */
public fun Message.toolResults(): List<MessagePart.Tool.Result> = parts.filterIsInstance<MessagePart.Tool.Result>()

/**
 * Returns a copy of this message with its parts replaced, keeping only the parts the
 * message role can actually carry.
 */
public fun Message.withParts(parts: List<MessagePart>): Message = when (this) {
    is Message.System -> copy(parts = parts.filterIsInstance<MessagePart.Text>())
    is Message.User -> copy(parts = parts.filterIsInstance<MessagePart.RequestPart>())
    is Message.Assistant -> copy(parts = parts.filterIsInstance<MessagePart.ResponsePart>())
}

/** True if the message carries at least one tool call part. */
public fun Message.hasToolCalls(): Boolean = parts.any { it is MessagePart.Tool.Call }

/** True if the message carries at least one tool result part. */
public fun Message.hasToolResults(): Boolean = parts.any { it is MessagePart.Tool.Result }

package ai.koog.agents.optimization.utils.agentic


import ai.koog.agents.core.dsl.builder.AIAgentSubgraphBuilderBase
import ai.koog.agents.core.dsl.builder.node
import ai.koog.agents.core.dsl.builder.AIAgentNodeDelegate
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.message.Message

/**
 * Renders the prompt's messages into a human-readable, newline-separated string,
 * each prefixed with its index and message role. Intended for logging and debugging.
 */
public fun Prompt.prettyPrint(): String =
    this.messages
        .withIndex()
        .joinToString("\n") { (i, msg) ->
            "[$i] ${msg.role}:\n${msg.content}\n"
        }

/**
 * Create a new Prompt object with the same messages, except for the system prompt.
 */
public fun Prompt.replaceSystemMessage(
    id: String = this.id,
    systemMessageEditor: (String) -> String,
): Prompt {
    return prompt(id) {
        for (m in this@replaceSystemMessage.messages) {
            if (m is Message.System) {
                this.message(Message.System(systemMessageEditor(m.content), m.metaInfo))
            } else {
                this.message(m)
            }
        }
    }
}

/**
 * Creates a new Prompt with [text] appended to the first system message.
 * If no system message exists, the prompt is returned unchanged.
 */
public fun Prompt.appendToSystemMessage(text: String): Prompt {
    val messages = this.messages
    val systemIndex = messages.indexOfFirst { it is Message.System }
    if (systemIndex < 0) return this

    val original = messages[systemIndex] as Message.System
    val appended = Message.System("${original.content}\n\n$text", original.metaInfo)
    val updatedMessages = messages.toMutableList().apply { set(systemIndex, appended) }
    return Prompt(updatedMessages, this.id, this.params)
}

/**
 * Util function for investigating the behaviour of the system on model context overflow.
 * It is very approximate and should be used mainly for preliminary experiments.
 *
 * With `contextApproxSize = 500_000` and `overflowCoef = 2.0` it generates a message of approx size:
 * - ~1.5M tokens for OpenAI GPT models
 * - ~1.5M tokens for Gemini models
 * - ~2M tokens for Claude models
 */
@Suppress("unused")
public fun generateContextOverflowMessage(contextApproxSize: Int = 500_000, overflowCoef: Double = 2.0): String {
    val phraseToRepeat =
        "This is a test phrase to check the context size and what happens if it overflows.\n" // around 15 tokens
    val repetitions = (contextApproxSize / 10.0 * overflowCoef).toInt()
    return phraseToRepeat.repeat(repetitions)
}

/** A single LLM call node with an additional user message. Intended for guiding the LLM towards with a hint. */
public inline fun <reified Input, reified Output> AIAgentSubgraphBuilderBase<Input, Output>.nodeCallLLMWithHint(hint: String): AIAgentNodeDelegate<Any?, Message.Response> = node<Any?, Message.Response> {
    llm.writeSession {
        appendPrompt {
            user(hint)
        }
        requestLLM()
    }
}

/** Hint message instructing the LLM that it must call a tool to continue, used with [nodeCallLLMWithHint]. */
public const val HINT_CALL_A_TOOL: String = "You MUST call some tool in order to continue. If you don't call a tool now, the task will be considered as failed."

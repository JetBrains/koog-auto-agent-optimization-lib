package ai.koog.agents.optimization.utils.agentic


import ai.koog.agents.core.dsl.builder.AIAgentEdgeBuilderIntermediate
import ai.koog.agents.core.dsl.builder.AIAgentNodeDelegate
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphBuilderBase
import ai.koog.agents.core.dsl.builder.node
import ai.koog.agents.core.dsl.extension.ReceivedToolResults
import ai.koog.agents.core.dsl.extension.onIsInstance
import ai.koog.agents.optimization.utils.messages.renderParts
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.PromptBuilder
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart

/**
 * Renders the prompt's messages into a human-readable, newline-separated string,
 * each prefixed with its index and message role. Intended for logging and debugging.
 */
public fun Prompt.prettyPrint(): String =
    this.messages
        .withIndex()
        .joinToString("\n") { (i, msg) ->
            "[$i] ${msg.role}:\n${msg.renderParts()}\n"
        }

/**
 * Create a new Prompt object with the same messages, except for the system prompt.
 *
 * Edits **every** system message. A prompt taken from inside a fresh optimizable subgraph has two
 * (the inherited agent-level one and the module instruction) — see `optimizableSubgraphWithTask`.
 */
public fun Prompt.replaceSystemMessage(
    id: String = this.id,
    systemMessageEditor: (String) -> String,
): Prompt {
    return prompt(id) {
        for (m in this@replaceSystemMessage.messages) {
            if (m is Message.System) {
                this.message(Message.System(systemMessageEditor(m.textContent()), m.metaInfo))
            } else {
                this.message(m)
            }
        }
    }
}

/**
 * Creates a new Prompt with [text] appended to the first system message.
 * If no system message exists, the prompt is returned unchanged.
 *
 * "First" means the agent-level system prompt for every current caller (ACE, ReasoningBank — both
 * run at strategy start, before any subgraph can append its own instruction as a second one).
 */
public fun Prompt.appendToSystemMessage(text: String): Prompt {
    val messages = this.messages
    val systemIndex = messages.indexOfFirst { it is Message.System }
    if (systemIndex < 0) return this

    val original = messages[systemIndex] as Message.System
    val appended = Message.System("${original.textContent()}\n\n$text", original.metaInfo)
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
public inline fun <reified Input, reified Output> AIAgentSubgraphBuilderBase<Input, Output>.nodeCallLLMWithHint(hint: String): AIAgentNodeDelegate<Any?, Message.Assistant> = node<Any?, Message.Assistant> {
    llm.writeSession {
        appendPrompt {
            user(hint)
        }
        requestLLM()
    }
}

/**
 * Appends every result of a tool batch as one user message, then asks the LLM to continue.
 *
 * The single message is what koog's own `nodeLLMSendToolResults` produces, and what the providers
 * expect: an assistant message with N tool calls must be answered by N results, all in the turn
 * right after it. Answering only some of them is a 400 from OpenAI, Anthropic and Vertex alike —
 * which is why the tool loops here execute the whole batch instead of just the first call.
 */
public fun PromptBuilder.toolResults(results: ReceivedToolResults) {
    user { results.toolResults.forEach { toolResult(it.toMessagePart()) } }
}

/**
 * Matches **any** assistant turn and forwards its text — the empty string when the turn carries none.
 *
 * Koog 1.1 dropped `onAssistantMessage`, and its `onTextMessage` only matches a turn that contains a
 * [MessagePart.Text]. An assistant turn made of reasoning alone — what a reasoning model returns when
 * it hits its output-token cap while thinking — then matches neither that edge nor `onToolCalls`, and
 * the run dies with `AIAgentStuckInTheNodeException`. Use this for the catch-all edge of a tool loop,
 * declared *after* the `onToolCalls` one: edges resolve in declaration order and the first match wins.
 */
public infix fun <IncomingOutput, IntermediateOutput, OutgoingInput> AIAgentEdgeBuilderIntermediate<IncomingOutput, IntermediateOutput, OutgoingInput>.onAssistantMessage(
    block: suspend (Message.Assistant) -> Boolean,
): AIAgentEdgeBuilderIntermediate<IncomingOutput, String, OutgoingInput> =
    onIsInstance(Message.Assistant::class)
        .onCondition { block(it) }
        .transformed { message -> message.parts.filterIsInstance<MessagePart.Text>().joinToString("\n") { it.text } }

/** Hint message instructing the LLM that it must call a tool to continue, used with [nodeCallLLMWithHint]. */
public const val HINT_CALL_A_TOOL: String = "You MUST call some tool in order to continue. If you don't call a tool now, the task will be considered as failed."

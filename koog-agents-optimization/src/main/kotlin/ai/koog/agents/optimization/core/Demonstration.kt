package ai.koog.agents.optimization.core

import ai.koog.prompt.message.Message
import kotlinx.serialization.Serializable

/**
 * A unified demonstration representing an input-output example for few-shot learning.
 *
 * Demonstrations can represent either compact I/O pairs (labeled examples) or full execution
 * traces with intermediate reasoning steps (bootstrapped examples). The [intermediateMessages]
 * field distinguishes these: when present, it contains the full conversation trace from a
 * successful subgraph execution (tool calls, assistant responses, etc.).
 *
 * @property input The example input, stringified at the subgraph boundary.
 * @property output The example output, stringified at the subgraph boundary.
 * @property intermediateMessages Optional intermediate conversation messages from the subgraph
 *  execution. When present, includes tool calls, tool results, and assistant responses that
 *  produced the output. Absent for labeled (non-bootstrapped) demonstrations.
 */
@Serializable
public data class Demonstration(
    val input: String,
    val output: String,
    val intermediateMessages: List<Message>? = null,
)

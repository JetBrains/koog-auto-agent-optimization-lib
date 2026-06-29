package ai.koog.agents.optimization.optimizers.mipro


import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.executePrompt
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Runs a meta-LLM prompt and extracts the assistant's text content.
 *
 * Used by MIPRO's meta-LLM components (dataset summarizer, instruction proposer)
 * for LLM calls that are separate from the agent's own execution. Returns null
 * on infrastructure failure or blank response so callers can fall back to a
 * baseline (current instruction, etc.).
 */
internal typealias MetaPromptRunner = suspend (prompt: Prompt) -> String?

/**
 * Builds a [MetaPromptRunner] bound to the given [model] that records each call
 * via the tracking-aware [executePrompt] DSL — so timing and consumption are
 * captured under the current stage rather than going through an untracked side
 * channel.
 */
internal fun StageScope<*, *, *>.metaPromptRunner(model: LLModel): MetaPromptRunner = { prompt ->
    executePrompt(prompt, model).fold(
        onSuccess = { responses ->
            responses.filterIsInstance<Message.Assistant>()
                .firstOrNull()
                ?.content
                ?.takeIf { it.isNotBlank() }
        },
        onFailure = { e ->
            logger.debug(e) { "Meta-LLM call failed for prompt '${prompt.id}'" }
            null
        }
    )
}

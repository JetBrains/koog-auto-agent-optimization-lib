package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.executePromptOrThrow
import ai.koog.agents.optimization.training.dsl.executePromptStructuredOrThrow
import ai.koog.agents.optimization.training.dsl.runStageOrThrow
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable

private val logger = KotlinLogging.logger {}

private fun makeReflectionPromptUpdatePrompt(
    moduleName: String,
    instruction: String,
    moduleFeedbackResults: List<GEPAModuleFeedbackResult>,
    useStructuredOutput: Boolean,
): String {
    check(moduleFeedbackResults.isNotEmpty()) {
        "Cannot build GEPA reflection prompt without feedback for module '$moduleName'"
    }
    val formattedFeedback = moduleFeedbackResults.joinToString("\n\n") { formatFeedback(it) }
    val outputInstruction = if (useStructuredOutput) null else "Provide the new instructions within ``` blocks."
    return """
        I provided an assistant with the following instructions to perform a task for me:
        
        ```
        $instruction
        ```
        
        The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
        
        ```
        $formattedFeedback
        ```
        
        Your task is to write a new instruction for the assistant.
        
        Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.
        
        Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.
        ${outputInstruction?.let { "\n\n$it" }.orEmpty()}
    """.trimIndent()
}

@Serializable
@LLMDescription("A GEPA reflection result with a replacement instruction.")
internal data class GEPAReflectionResponse(
    @property:LLMDescription("The full replacement instruction text. Do not wrap it in markdown fences.")
    val newInstruction: String,
)

@Serializable
@LLMDescription("A GEPA reflection result with explicit thinking and a replacement instruction.")
internal data class GEPAReflectionWithThinkingResponse(
    @property:LLMDescription("The model's reasoning about how the feedback should change the instruction.")
    val thinking: String,
    @property:LLMDescription("The full replacement instruction text. Do not wrap it in markdown fences.")
    val newInstruction: String,
)

/**
 * Runs GEPA's reflection prompt and returns a replacement instruction for one module.
 */
internal class GEPAReflectionProposer(
    private val reflectionModel: LLModel,
    private val useStructuredOutput: Boolean = false,
    private val requireThinkingFieldInOutput: Boolean = false,
) {
    init {
        require(!requireThinkingFieldInOutput || useStructuredOutput) {
            "requireThinkingFieldInOutput=true requires useStructuredOutput=true"
        }
    }

    suspend fun proposeInstruction(
        scope: StageScope<*, *, *>,
        moduleName: String,
        instruction: String,
        moduleFeedbackResults: List<GEPAModuleFeedbackResult>,
    ): String = scope.runStageOrThrow("GEPA reflection") {
        val proposedInstruction = if (useStructuredOutput) {
            reflectAndUpdatePromptStructured(
                moduleName = moduleName,
                instruction = instruction,
                moduleFeedbackResults = moduleFeedbackResults,
                executePrompt = { prompt ->
                    if (requireThinkingFieldInOutput) {
                        val response = executePromptStructuredOrThrow<GEPAReflectionWithThinkingResponse>(
                            prompt = prompt,
                            model = reflectionModel,
                        )
                        if (response.thinking.isBlank()) {
                            logger.warn { "GEPA structured reflection returned a blank thinking field." }
                        }
                        response.newInstruction
                    } else {
                        executePromptStructuredOrThrow<GEPAReflectionResponse>(
                            prompt = prompt,
                            model = reflectionModel,
                        ).newInstruction
                    }
                },
            )
        } else {
            reflectAndUpdatePrompt(
                moduleName = moduleName,
                instruction = instruction,
                moduleFeedbackResults = moduleFeedbackResults,
                executePrompt = { prompt ->
                    executePromptOrThrow(prompt, reflectionModel)
                },
            )
        }

        logGepaProposedInstruction(
            moduleName = moduleName,
            parentInstruction = instruction,
            feedbackTexts = moduleFeedbackResults.map { feedback -> feedback.feedbackText },
            proposedInstruction = proposedInstruction,
        )

        proposedInstruction
    }
}

internal fun formatFeedback(moduleFeedbackResult: GEPAModuleFeedbackResult): String {
    return """
        # Input
        ${moduleFeedbackResult.input}
        
        # Output
        ${moduleFeedbackResult.output}
        
        # Feedback
        ${moduleFeedbackResult.feedbackText}
    """.trimIndent()
}

/**
 * Extracts the updated instruction from the fenced block requested in the unstructured prompt.
 */
internal fun extractInstructionText(lmOutput: String): String {
    val firstFence = lmOutput.indexOf("```")
    val lastFence = lmOutput.lastIndexOf("```")

    if (firstFence == -1 || lastFence == -1 || firstFence == lastFence || firstFence + 3 >= lastFence) {
        val stripped = lmOutput.trim()
        val extracted = when {
            stripped.startsWith("```") -> stripped
                .replaceFirst(Regex("""^```\S*\n?"""), "")
                .trim()

            stripped.endsWith("```") -> stripped
                .dropLast(3)
                .trim()

            else -> stripped
        }
        logger.warn {
            "GEPA reflection did not return a complete fenced instruction block; " +
                    "falling back to raw output. Raw output prefix: '${stripped.take(200)}'"
        }
        return extracted
    }

    val fencedContent = lmOutput.substring(firstFence + 3, lastFence)
        .replaceFirst(Regex("""^\S*\n"""), "")

    val extracted = fencedContent.trim()
    if (extracted.isBlank()) {
        logger.warn { "GEPA reflection returned a fenced instruction block, but it was blank." }
    }
    return extracted
}


internal suspend fun reflectAndUpdatePrompt(
    moduleName: String,
    instruction: String,
    moduleFeedbackResults: List<GEPAModuleFeedbackResult>,
    executePrompt: suspend (prompt: Prompt) -> Message.Assistant,
): String {
    val prompt = prompt("gepa-reflection") {
        user(
            makeReflectionPromptUpdatePrompt(
                moduleName = moduleName,
                instruction = instruction,
                moduleFeedbackResults = moduleFeedbackResults,
                useStructuredOutput = false,
            )
        )
    }
    val result = executePrompt(prompt).textContent()
    return extractInstructionText(result)
}

internal suspend fun reflectAndUpdatePromptStructured(
    moduleName: String,
    instruction: String,
    moduleFeedbackResults: List<GEPAModuleFeedbackResult>,
    executePrompt: suspend (prompt: Prompt) -> String,
): String {
    val prompt = prompt("gepa-reflection-structured") {
        user(
            makeReflectionPromptUpdatePrompt(
                moduleName = moduleName,
                instruction = instruction,
                moduleFeedbackResults = moduleFeedbackResults,
                useStructuredOutput = true,
            )
        )
    }
    val result = executePrompt(prompt).trim()
    if (result.isBlank()) {
        logger.warn { "GEPA structured reflection produced a blank instruction." }
    }
    return result
}

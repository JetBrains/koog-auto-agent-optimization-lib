package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.prompt.Prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import kotlinx.coroutines.runBlocking
import kotlin.test.*

internal class GEPAMetaPromptsTest {
    @Test
    fun `unstructured reflection prompt contains instruction feedback examples and fenced output request`() = runBlocking {
        var capturedPrompt: Prompt? = null

        val updatedInstruction = reflectAndUpdatePrompt(
            moduleName = "synthetic_module",
            instruction = "Use the original synthetic instruction.",
            moduleFeedbackResults = listOf(
                feedback(
                    input = "input-a",
                    output = "output-a",
                    feedbackText = "feedback-a",
                ),
                feedback(
                    input = "input-b",
                    output = "output-b",
                    feedbackText = "feedback-b",
                ),
            ),
            executePrompt = { prompt ->
                capturedPrompt = prompt
                Message.Assistant(
                    """
                    The revised instruction is:

                    ```text
                    Use the improved synthetic instruction.
                    Keep the output compact.
                    ```
                    """.trimIndent(),
                    ResponseMetaInfo.Empty,
                    "stop",
                )
            },
        )

        assertEquals(
            """
            Use the improved synthetic instruction.
            Keep the output compact.
            """.trimIndent(),
            updatedInstruction,
        )

        val prompt = assertNotNull(capturedPrompt)
        assertEquals("gepa-reflection", prompt.id)
        assertEquals(1, prompt.messages.size)

        val userMessage = assertIs<Message.User>(prompt.messages.single())
        val content = userMessage.textContent()

        assertPromptContainsInOrder(
            content,
            "I provided an assistant with the following instructions to perform a task for me:",
            "Use the original synthetic instruction.",
            "The following are examples of different task inputs provided to the assistant",
            "# Input\ninput-a",
            "# Output\noutput-a",
            "# Feedback\nfeedback-a",
            "# Input\ninput-b",
            "# Output\noutput-b",
            "# Feedback\nfeedback-b",
            "Your task is to write a new instruction for the assistant.",
            "Read the inputs carefully and identify the input format",
            "Read all the assistant responses and the corresponding feedback.",
            "Provide the new instructions within ``` blocks.",
        )
    }

    @Test
    fun `structured reflection prompt omits fenced output request and returns structured instruction directly`() = runBlocking {
        var capturedPrompt: Prompt? = null

        val updatedInstruction = reflectAndUpdatePromptStructured(
            moduleName = "synthetic_module",
            instruction = "Use the original synthetic instruction.",
            moduleFeedbackResults = listOf(
                feedback(
                    input = "input-a",
                    output = "output-a",
                    feedbackText = "feedback-a",
                )
            ),
            executePrompt = { prompt ->
                capturedPrompt = prompt
                "  Use the structured synthetic instruction.  \n"
            },
        )

        assertEquals("Use the structured synthetic instruction.", updatedInstruction)

        val prompt = assertNotNull(capturedPrompt)
        assertEquals("gepa-reflection-structured", prompt.id)
        assertEquals(1, prompt.messages.size)

        val userMessage = assertIs<Message.User>(prompt.messages.single())
        val content = userMessage.textContent()

        assertPromptContainsInOrder(
            content,
            "I provided an assistant with the following instructions to perform a task for me:",
            "Use the original synthetic instruction.",
            "# Input\ninput-a",
            "# Output\noutput-a",
            "# Feedback\nfeedback-a",
            "Your task is to write a new instruction for the assistant.",
        )
        assertFalse(
            "Provide the new instructions within ``` blocks." in content,
            "Structured reflection uses typed output, so the prompt should not ask for markdown fences.",
        )
    }

    private fun feedback(
        input: String,
        output: String,
        feedbackText: String,
    ): GEPAModuleFeedbackResult =
        GEPAModuleFeedbackResult(
            input = input,
            output = output,
            feedbackText = feedbackText,
        )

    private fun assertPromptContainsInOrder(content: String, vararg fragments: String) {
        var previousIndex = -1
        for (fragment in fragments) {
            assertContains(content, fragment)
            val index = content.indexOf(fragment)
            assertTrue(
                index > previousIndex,
                "Expected fragment to appear after the previous fragment: $fragment",
            )
            previousIndex = index
        }
    }
}

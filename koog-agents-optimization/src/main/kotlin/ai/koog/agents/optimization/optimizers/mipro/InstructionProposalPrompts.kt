package ai.koog.agents.optimization.optimizers.mipro


import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt

/**
 * Prompt builders for MIPRO Step 2: instruction proposal via meta-LLM.
 *
 * These construct Koog prompts for dataset observation, summarization,
 * program/module description, and instruction generation.
 *
 * Ported from the reference MIPRO implementation with minimal changes.
 */

private const val DATASET_OBSERVATION_SYSTEM =
    """Given several examples from a dataset please write observations about trends that hold for most or all of the samples.
Some areas you may consider in your observations: topics, content, syntax, conciseness, etc.
It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative."""

private fun formatExampleBatch(examples: List<String>): String =
    examples.mapIndexed { index, example -> "Example ${index + 1}: $example" }.joinToString("\n")

/** Prompt to observe patterns in a batch of dataset examples. */
internal fun datasetDescriptorPrompt(exampleBatch: List<String>): Prompt = prompt("dataset-descriptor") {
    system(DATASET_OBSERVATION_SYSTEM)
    user(
        """EXAMPLES:
${formatExampleBatch(exampleBatch)}

Please write your observations about trends that hold for most or all of the samples."""
    )
}

/** Prompt to add observations given prior observations and a new batch. */
internal fun datasetDescriptorWithPriorObservationsPrompt(
    exampleBatch: List<String>,
    priorObservations: String,
): Prompt = prompt("dataset-descriptor-with-prior") {
    system(
        """Given several examples from a dataset please write observations about trends that hold for most or all of the samples.
I will also provide you with a few observations I have already made. Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE'.
Some areas you may consider in your observations: topics, content, syntax, conciseness, etc.
It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative."""
    )
    user(
        """PRIOR OBSERVATIONS:
$priorObservations

NEW EXAMPLES:
${formatExampleBatch(exampleBatch)}

Please add new observations or respond with 'COMPLETE' if the observations are comprehensive."""
    )
}

/** Prompt to summarize observations into 2-3 sentences. */
internal fun observationSummarizerPrompt(observations: String): Prompt = prompt("observation-summarizer") {
    system(
        """Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details."""
    )
    user(
        """OBSERVATIONS:
$observations

Please provide a two to three sentence summary of only the most significant highlights of these observations."""
    )
}

/** Prompt to describe what the overall program (strategy) does. */
internal fun describeProgramPrompt(programCode: String, programExample: String): Prompt =
    prompt("describe-program") {
        system(
            """Below is a language model program with various modules that work together to solve a task.
Please describe what task this program is designed to solve and how it accomplishes this task, based on the program structure and the example provided.
Be concise but thorough in your description."""
        )
        user(
            """PROGRAM CODE:
$programCode

EXAMPLE OF PROGRAM IN USE:
$programExample

Please describe what this program does and how it works."""
        )
    }

/** Prompt to describe a specific module's role within the program. */
internal fun describeModulePrompt(
    programCode: String,
    programDescription: String,
    programExample: String,
    moduleCode: String,
): Prompt = prompt("describe-module") {
    system(
        """Below is a language model program with various modules that work together to solve a task.
Please describe the role of the specified module within this program. Be concise but specific about what this module contributes to the overall program."""
    )
    user(
        """PROGRAM CODE:
$programCode

PROGRAM DESCRIPTION:
$programDescription

EXAMPLE OF PROGRAM IN USE:
$programExample

MODULE:
$moduleCode

Please describe this module's role in the program."""
    )
}

/** Configuration for building a GenerateModuleInstruction prompt. */
internal data class GenerateInstructionPromptConfig(
    val datasetSummary: String?,
    val programCode: String?,
    val programDescription: String?,
    val moduleCodeString: String,
    val moduleDescription: String?,
    val taskDemos: String,
    val previousInstructions: String?,
    val basicInstruction: String,
    val tip: String?,
)

/** Prompt to generate a new instruction for a module. */
internal fun generateModuleInstructionPrompt(config: GenerateInstructionPromptConfig): Prompt =
    prompt("generate-instruction") {
        system(
            """Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task."""
        )

        val sections = buildString {
            fun appendSection(header: String, content: String?) {
                if (!content.isNullOrBlank()) {
                    appendLine("$header:")
                    appendLine(content)
                    appendLine()
                }
            }

            appendSection("DATASET SUMMARY", config.datasetSummary)
            appendSection("PROGRAM CODE", config.programCode)
            appendSection("PROGRAM DESCRIPTION", config.programDescription)

            appendLine("MODULE:")
            appendLine(config.moduleCodeString)
            appendLine()

            appendSection("MODULE DESCRIPTION", config.moduleDescription)

            appendLine("TASK DEMO(S):")
            appendLine(config.taskDemos)
            appendLine()

            appendSection("PREVIOUS INSTRUCTIONS", config.previousInstructions)

            appendLine("BASIC INSTRUCTION:")
            appendLine(config.basicInstruction)

            appendSection("TIP", config.tip)

            appendLine("---")
            appendLine("Based on the above information, propose an improved instruction that will help the Language Model perform this task better. Output only the instruction text, nothing else.")
        }

        user(sections)
    }

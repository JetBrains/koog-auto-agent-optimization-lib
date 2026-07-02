package ai.koog.agents.optimization.optimizers.gepa


import ai.koog.agents.optimization.utils.agentic.prettyPrint
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt

/**
 * Prompt templates used by GEPA's reflection LM for instruction proposal and crossover.
 */
public object GEPAPrompts {

    /**
     * Builds a reflection prompt for the GEPA reflection LM.
     *
     * Given the current instruction, execution traces of failed items, and expected outputs,
     * the reflection LM produces a structured analysis of what went wrong and proposes
     * an improved instruction.
     */
    public fun reflectionAndProposalPrompt(
        moduleName: String,
        currentInstruction: String,
        failures: List<GEPAReflectionProposer.FailureFeedback>,
    ): Prompt {
        val systemText = """
You are an expert prompt engineer. Your task is to analyze execution failures from an AI agent module
and propose a better instruction for that module.

The module "$moduleName" currently uses this instruction:

<current_instruction>
$currentInstruction
</current_instruction>

Below are examples where the module produced incorrect outputs. For each failure, you will see:
- The input that was given
- The expected output (ground truth)
- The actual output produced
- The score (0.0 = complete failure, 1.0 = perfect)
- The execution trace showing the full conversation

Analyze the failure patterns and propose an improved instruction that would help the module
avoid these mistakes. Focus on:
1. What specific patterns of errors do you see?
2. What is the root cause — does the module misunderstand the task, miss key signals, or reason incorrectly?
3. What concrete guidance in the instruction would prevent these failures?

Respond with ONLY the new instruction text. Do not include any preamble, explanation, or formatting —
just the instruction itself.
        """.trimIndent()

        val failureTexts = failures.mapIndexed { i, f ->
            """
--- Failure ${i + 1} (score: ${f.score}) ---
Input: ${f.input}
Expected: ${f.expectedLabel}
Actual output: ${f.actualOutput}
Execution trace:
${f.trajectory.prettyPrint()}
            """.trimIndent()
        }.joinToString("\n\n")

        return prompt("gepa-reflection") {
            system(systemText)
            user(failureTexts)
        }
    }

    /**
     * Builds a crossover prompt: given two candidate instructions for the same module,
     * merge the best aspects of both into a single improved instruction.
     */
    public fun crossoverPrompt(
        moduleName: String,
        instructionA: String,
        scoreA: Double,
        instructionB: String,
        scoreB: Double,
    ): Prompt {
        val systemText = """
You are an expert prompt engineer. You have two different instructions for the same AI module "$moduleName".
Each performs well on different types of inputs. Your task is to merge the best aspects of both
into a single, improved instruction.

Instruction A (aggregate score: ${"%.3f".format(scoreA)}):
<instruction_a>
$instructionA
</instruction_a>

Instruction B (aggregate score: ${"%.3f".format(scoreB)}):
<instruction_b>
$instructionB
</instruction_b>

Combine the strengths of both instructions into a single, coherent instruction.
Respond with ONLY the merged instruction text. No preamble or explanation.
        """.trimIndent()

        return prompt("gepa-crossover") {
            system(systemText)
        }
    }
}

package ai.koog.agents.optimization.optimizers.gepa


import ai.koog.prompt.dsl.Prompt
import ai.koog.agents.optimization.core.OptimizableModule
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Runs a reflection LLM prompt and returns the joined message contents.
 *
 * The runner is built from a [ai.koog.agents.optimization.training.dsl.StageScope]
 * (see [GEPAOptimizer.train]), so each call is recorded as a tracked
 * [ai.koog.agents.optimization.training.records.PromptExecutionRecord]
 * with timing and consumption captured under the current stage.
 */
internal typealias GEPAReflectionRunner = suspend (Prompt) -> String

/**
 * Proposes improved instructions for GEPA modules by reflecting on execution failures.
 *
 * For each module to update, the proposer sends the current instruction, failure traces,
 * and expected outputs to a reflection LM, which analyzes the failures and proposes
 * a new instruction.
 */
public class GEPAReflectionProposer internal constructor(
    private val runReflection: GEPAReflectionRunner,
) {
    /**
     * Feedback collected from a single failed execution, fed to the reflection LM.
     *
     * @property input the agent input for the failed item.
     * @property expectedLabel the expected (ground-truth) output, as a string.
     * @property actualOutput the output the agent actually produced.
     * @property score the metric score for this item (0.0 = complete failure, 1.0 = perfect).
     * @property trajectory the full conversation trace of the failed run.
     */
    public data class FailureFeedback(
        val input: String,
        val expectedLabel: String,
        val actualOutput: String,
        val score: Double,
        val trajectory: Prompt,
    )

    /**
     * Proposes a new instruction for the given module based on failure analysis.
     *
     * @param module The module whose instruction is being improved.
     * @param currentInstruction The current instruction text.
     * @param failures Failed execution traces with expected outputs.
     * @return The proposed new instruction text.
     */
    public suspend fun proposeInstruction(
        module: OptimizableModule,
        currentInstruction: String,
        failures: List<FailureFeedback>,
    ): String {
        if (failures.isEmpty()) {
            logger.info { "  No failures for module '${module.name}', keeping current instruction." }
            return currentInstruction
        }

        logger.info { "  Proposing new instruction for '${module.name}' based on ${failures.size} failure(s)..." }

        val prompt = GEPAPrompts.reflectionAndProposalPrompt(
            moduleName = module.name,
            currentInstruction = currentInstruction,
            failures = failures,
        )

        val proposed = runReflection(prompt)
            .trim()
            .removePrefix("\"").removeSuffix("\"")
            .trim()

        if (proposed.isBlank()) {
            logger.warn { "  Reflection LM returned empty proposal for '${module.name}', keeping current." }
            return currentInstruction
        }

        logger.info { "  Proposed instruction for '${module.name}' (${proposed.length} chars)" }
        return proposed
    }

    /**
     * Proposes a merged instruction from two candidates via crossover.
     */
    public suspend fun proposeCrossover(
        module: OptimizableModule,
        instructionA: String,
        scoreA: Double,
        instructionB: String,
        scoreB: Double,
    ): String {
        val prompt = GEPAPrompts.crossoverPrompt(
            moduleName = module.name,
            instructionA = instructionA,
            scoreA = scoreA,
            instructionB = instructionB,
            scoreB = scoreB,
        )

        val merged = runReflection(prompt).trim()
        return merged.ifBlank { instructionA }
    }
}

package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.STRATEGY_MODULE_KEY
import ai.koog.prompt.Prompt
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Feedback produced for one GEPA rollout.
 *
 * @property score Continuous rollout score used for candidate acceptance and Pareto-front tracking.
 * @property moduleFeedbacks Feedback keyed by module name. Missing keys and null values both mean
 *   that no usable feedback is available for that module on this rollout.
 */
public data class GEPAFeedbackResult(
    val score: Double,
    val moduleFeedbacks: Map<String, GEPAModuleFeedbackResult?>,
)

/**
 * Textual feedback for one optimizable module on one GEPA rollout.
 *
 * @property input Human-readable module input shown to GEPA's reflection prompt.
 * @property output Human-readable module output shown to GEPA's reflection prompt.
 * @property feedbackText Natural-language critique or guidance used to propose the next instruction.
 */
public data class GEPAModuleFeedbackResult(
    val input: String,
    val output: String,
    val feedbackText: String
)

internal fun GEPAFeedbackResult.feedbackForDeclaredModule(moduleName: String): GEPAModuleFeedbackResult? {
    if (moduleName !in moduleFeedbacks) {
        logger.warn {
            "GEPA feedback did not include module '$moduleName'; treating it as unavailable for this rollout."
        }
    }
    return moduleFeedbacks[moduleName]
}

/**
 * Converts an evaluated rollout into the continuous score and textual module feedback GEPA needs.
 */
public fun interface GEPAFeedback<Input, Output, InputLabel> {
    /**
     * Returns the rollout score and module-level feedback.
     *
     * In GEPAFeedback implementations, [fullTrace] typically will be used to provide feedback for optimizing the
     * strategy instruction, while [subgraphTraces] will typically be used to provide feedback for optimizing
     * subgraph instructions.
     *
     * @param input Dataset input used for the rollout.
     * @param output Agent output produced by the rollout.
     * @param gold Expected label or reference value for [input].
     * @param fullTrace Full strategy prompt trace, when available.
     * @param subgraphTraces Demonstrations captured for each optimizable subgraph during the rollout.
     */
    public operator fun invoke(
        input: Input,
        output: Output,
        gold: InputLabel,
        fullTrace: Prompt?,
        subgraphTraces: Map<String, List<Demonstration>>
    ): GEPAFeedbackResult
}

/** Convenience helper for building a feedback function for agents that only optimize the strategy instruction. */
public fun <Input, Output, InputLabel> strategyOnlyGepaFeedback(
    formatInput: (Input) -> String = { it.toString() },
    formatOutput: (Output) -> String = { it.toString() },
    feedback: (
        input: Input,
        output: Output,
        gold: InputLabel,
        fullTrace: Prompt?,
        subgraphTraces: Map<String, List<Demonstration>>
    ) -> Pair<Double, String>,
): GEPAFeedback<Input, Output, InputLabel> =
    singleModuleGepaFeedback(
        moduleName = STRATEGY_MODULE_KEY,
        formatInput = formatInput,
        formatOutput = formatOutput,
        feedback = feedback,
    )

/** Convenience helper for building a feedback function for a single explicit module. */
public fun <Input, Output, InputLabel> singleModuleGepaFeedback(
    moduleName: String,
    formatInput: (Input) -> String = { it.toString() },
    formatOutput: (Output) -> String = { it.toString() },
    feedback: (
        input: Input,
        output: Output,
        gold: InputLabel,
        fullTrace: Prompt?,
        subgraphTraces: Map<String, List<Demonstration>>
    ) -> Pair<Double, String>,
): GEPAFeedback<Input, Output, InputLabel> =
    GEPAFeedback { input, output, gold, fullTrace, subgraphTraces ->
        val (score, feedbackText) = feedback(input, output, gold, fullTrace, subgraphTraces)
        GEPAFeedbackResult(
            score = score,
            moduleFeedbacks = mapOf(
                moduleName to GEPAModuleFeedbackResult(
                    input = formatInput(input),
                    output = formatOutput(output),
                    feedbackText = feedbackText,
                )
            ),
        )
    }

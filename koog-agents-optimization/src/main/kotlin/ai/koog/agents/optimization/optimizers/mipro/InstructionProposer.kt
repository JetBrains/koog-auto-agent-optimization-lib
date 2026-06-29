package ai.koog.agents.optimization.optimizers.mipro


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.describeForOptimization
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlin.random.Random

private val logger = KotlinLogging.logger {}

private const val NO_TASK_DEMOS = "No task demos provided."
private const val MAX_INSTRUCT_IN_HISTORY = 5

/** Tips for instruction generation diversity. */
public val TIPS: Map<String, String> = mapOf(
    "none" to "",
    "creative" to "Don't be afraid to be creative when creating the new instruction!",
    "simple" to "Keep the instruction clear and concise.",
    "description" to "Make sure your instruction is very informative and descriptive.",
    "high_stakes" to "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona" to "Include a persona that is relevant to the task in the instruction (ie. \"You are a ...\")",
)

/**
 * Configuration for [InstructionProposer], toggling which context the meta-LLM receives.
 *
 * @property useDatasetSummary include an LLM-generated summary of the training dataset in the proposal context.
 * @property programAware include the agent's program/strategy description and code in the proposal context.
 * @property useTaskDemos include few-shot task demonstrations in the proposal context.
 * @property numDemosInContext maximum number of demonstrations to include when [useTaskDemos] is on.
 * @property useTip include a stylistic tip (from [TIPS]) to diversify generated instructions.
 * @property setTipRandomly pick the tip at random rather than using the neutral default.
 * @property useInstructHistory include previously proposed instructions and their scores as context.
 * @property setHistoryRandomly decide per run, at random, whether to include instruction history.
 */
public data class InstructionProposerConfig(
    val useDatasetSummary: Boolean = true,
    val programAware: Boolean = true,
    val useTaskDemos: Boolean = true,
    val numDemosInContext: Int = 3,
    val useTip: Boolean = true,
    val setTipRandomly: Boolean = true,
    val useInstructHistory: Boolean = false,
    val setHistoryRandomly: Boolean = false,
)

/**
 * Generates instruction candidates for each optimizable module.
 *
 * MIPRO Step 2: uses a meta-LLM to propose diverse instruction variants
 * by providing context about the dataset, program structure, and few-shot demos.
 *
 * Works with both strategy-level (system prompt) and per-subgraph modules.
 */
public class InstructionProposer private constructor(
    private val runMeta: MetaPromptRunner,
    private val config: InstructionProposerConfig,
    private val random: Random,
    private val userProgramDescription: String?,
    private val datasetSummary: String?,
    private val programCode: String?,
    private val modules: List<OptimizableModule>,
) {
    /** Factory for [InstructionProposer]. */
    public companion object {
        /**
         * Creates an [InstructionProposer], generating dataset summary and program code upfront.
         *
         * @param agent The agent whose strategy to describe.
         * @param modules The optimizable modules to generate instructions for.
         * @param trainExamples Pre-rendered training examples for dataset summarization.
         * @param runMeta Tracked meta-LLM runner (see [metaPromptRunner]).
         * @param config Configuration options.
         * @param random Random instance for reproducibility.
         * @param programDescription Optional user-provided program description (skips LLM generation).
         */
        public suspend fun <Input, Output> create(
            agent: GraphAIAgent<Input, Output>,
            modules: List<OptimizableModule>,
            trainExamples: List<String>,
            runMeta: MetaPromptRunner,
            config: InstructionProposerConfig = InstructionProposerConfig(),
            random: Random = Random.Default,
            programDescription: String? = null,
        ): InstructionProposer {
            val datasetSummary = if (config.useDatasetSummary) {
                logger.info { "Generating dataset summary..." }
                createDatasetSummary(trainExamples, runMeta)
            } else null

            val programCode = if (config.programAware) {
                try {
                    agent.strategy.describeForOptimization()
                } catch (e: Exception) {
                    logger.debug(e) { "Failed to describe strategy for optimization" }
                    null
                }
            } else null

            return InstructionProposer(
                runMeta = runMeta,
                config = config,
                random = random,
                userProgramDescription = programDescription,
                datasetSummary = datasetSummary,
                programCode = programCode,
                modules = modules,
            )
        }
    }

    /**
     * Generates instruction candidates for all modules.
     *
     * @param demoCandidates Map from the module name to a list of candidate demo sets.
     * @param numCandidates Number of instruction candidates per module.
     * @param previousInstructions Previous instructions with scores for history context.
     * @param parallelism Maximum concurrent proposals per module.
     * @return Map from the module name to a list of candidate instruction strings.
     */
    public suspend fun proposeInstructionsForProgram(
        demoCandidates: Map<String, List<List<Demonstration>>>?,
        numCandidates: Int,
        previousInstructions: Map<String, List<Pair<String, Double>>> = emptyMap(),
        parallelism: Int = 1,
    ): Map<String, List<String>> {
        val effectiveUseHistory = if (config.setHistoryRandomly) random.nextBoolean() else config.useInstructHistory

        val numDemoSets = if (demoCandidates.isNullOrEmpty() || !config.useTaskDemos) {
            numCandidates
        } else {
            val firstModuleDemos = demoCandidates.values.firstOrNull()?.size ?: numCandidates
            minOf(firstModuleDemos, numCandidates)
        }

        val preSelectedTips = (0 until numDemoSets).map { selectTip() }

        return buildMap {
            for ((moduleIdx, module) in modules.withIndex()) {
                logger.info { "Proposing instructions for '${module.name}' (${moduleIdx + 1}/${modules.size})..." }

                val semaphore = Semaphore(maxOf(1, parallelism))
                val instructions = coroutineScope {
                    (0 until numDemoSets).map { demoSetIndex ->
                        async {
                            semaphore.withPermit {
                                val instruction = proposeInstructionForModule(
                                    module = module,
                                    demoCandidates = demoCandidates,
                                    demoSetIndex = demoSetIndex,
                                    tip = preSelectedTips[demoSetIndex],
                                    effectiveUseHistory = effectiveUseHistory,
                                    previousInstructions = previousInstructions,
                                )
                                logger.info { "  Candidate ${demoSetIndex + 1}/$numDemoSets for '${module.name}'" }
                                instruction
                            }
                        }
                    }.awaitAll()
                }
                put(module.name, instructions)
            }
        }
    }

    private suspend fun proposeInstructionForModule(
        module: OptimizableModule,
        demoCandidates: Map<String, List<List<Demonstration>>>?,
        demoSetIndex: Int,
        tip: String?,
        effectiveUseHistory: Boolean,
        previousInstructions: Map<String, List<Pair<String, Double>>>,
    ): String {
        val moduleCodeString = "Module \"${module.name}\"\nCurrent instruction: \"${module.currentInstruction}\""
        val taskDemos = gatherTaskDemos(module.name, demoCandidates, demoSetIndex)
        val basicInstruction = module.currentInstruction

        val currentProgramDescription = userProgramDescription
            ?: if (config.programAware && programCode != null) {
                runMeta(describeProgramPrompt(programCode, taskDemos))
            } else null

        val currentModuleDescription = module.description
            ?: if (config.programAware && programCode != null && currentProgramDescription != null) {
                runMeta(describeModulePrompt(programCode, currentProgramDescription, taskDemos, moduleCodeString))
            } else null

        val historyString = if (effectiveUseHistory) {
            formatInstructionHistory(module.name, previousInstructions)
        } else null

        val promptConfig = GenerateInstructionPromptConfig(
            datasetSummary = datasetSummary,
            programCode = programCode,
            programDescription = currentProgramDescription,
            moduleCodeString = moduleCodeString,
            moduleDescription = currentModuleDescription,
            taskDemos = taskDemos,
            previousInstructions = historyString,
            basicInstruction = basicInstruction,
            tip = if (config.useTip) tip else null,
        )

        val proposed = runMeta(generateModuleInstructionPrompt(promptConfig)) ?: return basicInstruction

        return stripInstructionPrefixes(proposed.trim()).ifBlank { basicInstruction }
    }

    private fun gatherTaskDemos(
        moduleName: String,
        demoCandidates: Map<String, List<List<Demonstration>>>?,
        demoSetIndex: Int,
    ): String {
        if (!config.useTaskDemos || demoCandidates.isNullOrEmpty()) return NO_TASK_DEMOS

        val moduleDemos = demoCandidates[moduleName]
        if (moduleDemos.isNullOrEmpty()) return NO_TASK_DEMOS
        if (demoSetIndex == 0) return NO_TASK_DEMOS

        val safeIndex = demoSetIndex.coerceAtMost(moduleDemos.size)
        val rotatedSets = moduleDemos.subList(safeIndex, moduleDemos.size) +
            moduleDemos.subList(0, safeIndex)

        val limit = config.numDemosInContext
        val allDemos = rotatedSets.flatten()
        val examples = allDemos.take(limit).map { demo ->
            buildString {
                appendLine("Input: ${demo.input}")
                appendLine("Output: ${demo.output}")
            }
        }

        return if (examples.isEmpty()) NO_TASK_DEMOS else examples.joinToString("\n")
    }

    private fun formatInstructionHistory(
        moduleName: String,
        previousInstructions: Map<String, List<Pair<String, Double>>>,
    ): String? {
        val history = previousInstructions[moduleName] ?: return null
        if (history.isEmpty()) return null

        return history
            .sortedByDescending { it.second }
            .take(MAX_INSTRUCT_IN_HISTORY)
            .reversed()
            .joinToString("\n") { (instruction, score) ->
                "\"$instruction\" | Score: ${"%.2f".format(score)}"
            }
    }

    private fun selectTip(): String? {
        if (!config.useTip) return null
        return if (config.setTipRandomly) {
            TIPS.values.toList().random(random)
        } else TIPS["none"]
    }

    private fun stripInstructionPrefixes(instruction: String): String {
        val prefixes = listOf("proposed instruction:", "instruction:", "here is", "here's")
        var result = instruction
        for (prefix in prefixes) {
            if (result.startsWith(prefix, ignoreCase = true)) {
                result = result.drop(prefix.length).trimStart()
            }
        }
        return result.trim()
    }
}

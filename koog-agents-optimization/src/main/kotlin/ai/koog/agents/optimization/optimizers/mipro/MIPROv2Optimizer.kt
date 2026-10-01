package ai.koog.agents.optimization.optimizers.mipro


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.optimization.core.*
import ai.koog.agents.optimization.features.installPromptOptimization
import ai.koog.agents.optimization.koogTooling.copyWith
import ai.koog.agents.optimization.optimizers.AgentOptimizer
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.runStageOrThrow
import ai.koog.agents.optimization.training.records.TrainingResult
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.common.toFilePathLog
import ai.koog.prompt.llm.LLModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlin.math.ceil
import kotlin.math.log2
import kotlin.random.Random

private val logger = KotlinLogging.logger {}

/**
 * Preset auto-run modes controlling candidate count, validation set size, and trial budget cap.
 *
 * The trial budget is auto-computed from the number of modules but capped by [maxTrials]
 * to prevent the formula from producing excessive trials for multi-module agents.
 *
 * @property numCandidates number of instruction/demo candidates generated per module.
 * @property valExamples number of validation examples used to score candidates.
 * @property maxTrials upper bound on optimization trials for this mode.
 */
public enum class AutoRunMode(public val numCandidates: Int, public val valExamples: Int, public val maxTrials: Int) {
    /** Smallest budget: fewest candidates, smallest validation set, lowest trial cap. */
    LIGHT(6, 25, 15),

    /** Balanced budget between [LIGHT] and [HEAVY]. */
    MEDIUM(12, 75, 40),

    /** Largest budget: most candidates, largest validation set, highest trial cap. */
    HEAVY(18, 200, 100),
}

/**
 * MIPRO v2 optimizer that tunes instructions and demonstrations via a 3-stage pipeline:
 *
 * 1. **Demo generation**: creates diverse candidate demo sets via bootstrapping.
 * 2. **Instruction proposal**: generates instruction variants via a meta-LLM.
 * 3. **Grid search**: evaluates random (instruction, demo set) combinations on a validation set.
 *
 * Produces an [OptimizationArtifact] with the best-scoring instruction and demo set
 * for each optimizable module (strategy-level and per-subgraph).
 *
 * @param metaModel LLM model for meta-LLM calls (instruction proposal, dataset summarization).
 * @param autoMode Preset mode controlling defaults (LIGHT/MEDIUM/HEAVY).
 * @param numCandidatesOverride Manual override for number of candidates per module.
 * @param valExamplesOverride Manual override for the validation-set size used to score candidates.
 * @param numTrialsOverride Manual override for grid search trials.
 * @param maxBootstrappedDemos Max demos per bootstrap run.
 * @param maxTotalDemos Total demo slots (bootstrapped and labeled).
 * @param includeLabeledExamples Whether bootstrap uses labeled fallback.
 * @param proposerConfig Configuration for an instruction proposal.
 * @param storagePath File path to save/load the optimization artifact.
 * @param randomSeed Seed for reproducibility.
 * @param parallelism Max concurrent operations (bootstrap runs, instruction proposals).
 */
public class MIPROv2Optimizer<Input, Output, InputLabel>(
    private val metaModel: LLModel,
    private val autoMode: AutoRunMode = AutoRunMode.LIGHT,
    private val numCandidatesOverride: Int? = null,
    private val valExamplesOverride: Int? = null,
    private val numTrialsOverride: Int? = null,
    private val maxBootstrappedDemos: Int = 4,
    private val maxTotalDemos: Int = 8,
    private val includeLabeledExamples: Boolean = true,
    private val proposerConfig: InstructionProposerConfig = InstructionProposerConfig(),
    public val storagePath: ResilientPath,
    private val randomSeed: Long = 42L,
    private val parallelism: Int = 1,
) : AgentOptimizer<Input, Output, InputLabel> {

    private val random = Random(randomSeed)

    init {
        require(numCandidatesOverride == null || numCandidatesOverride > 0) {
            "numCandidatesOverride must be positive"
        }
        require(valExamplesOverride == null || valExamplesOverride > 0) { "valExamplesOverride must be positive" }
        require(numTrialsOverride == null || numTrialsOverride > 0) { "numTrialsOverride must be positive" }
        // Both demo counts at zero is zero-shot mode, which `generateDemoSets` supports.
        require(maxBootstrappedDemos >= 0) { "maxBootstrappedDemos must not be negative" }
        require(maxTotalDemos >= 0) { "maxTotalDemos must not be negative" }
        require(parallelism > 0) { "parallelism must be positive" }
    }

    @OptIn(InternalAgentsApi::class)
    override fun loadOptimizedAgent(baseAgent: GraphAIAgent<Input, Output>): GraphAIAgent<Input, Output> {
        val artifact = loadArtifact(storagePath)
        return baseAgent.copyWith(installFeatures = {
            baseAgent.installFeatures(this)
            installPromptOptimization { this.artifact = artifact }
        })
    }

    @OptIn(InternalAgentsApi::class)
    override suspend fun train(
        session: TrainingSession<Input, Output, InputLabel>,
    ): TrainingResult = session.use(stagesTotal = 3) {
        require(dataset.size >= 2) { "MIPRO requires at least 2 training examples" }

        // Each number the auto mode supplies applies unless its own override is set.
        val numCandidates = numCandidatesOverride ?: autoMode.numCandidates
        val valExamples = valExamplesOverride ?: autoMode.valExamples

        // Split dataset: first 20% for bootstrap, capped validation set for grid search
        val splitIndex = maxOf(1, dataset.size / 5)
        val bootstrapSet = dataset.take(splitIndex)
        val valSet = dataset.shuffled(Random(random.nextLong())).take(valExamples)

        // Discover optimizable modules
        val modules = discoverModules(trackedAgent)

        // Compute the number of trials
        val numModules = modules.size
        val m = numModules * 2  // instruction + demos per module
        val numTrials = numTrialsOverride ?: minOf(
            autoMode.maxTrials,
            maxOf(
                ceil(2.0 * m * log2(numCandidates.toDouble())).toInt(),
                ceil(1.5 * numCandidates).toInt(),
            ),
        )

        logMiproRunSetup(
            modules = modules,
            autoMode = autoMode,
            numCandidates = numCandidates,
            numTrials = numTrials,
            overriddenBudgetValues = listOfNotNull(
                "numCandidates".takeIf { numCandidatesOverride != null },
                "valExamples".takeIf { valExamplesOverride != null },
                "numTrials".takeIf { numTrialsOverride != null },
            ),
            bootstrapSetSize = bootstrapSet.size,
            validationSetSize = valSet.size,
        )

        // Step 1: Demo Generation
        val demoCandidates = runStageOrThrow("Step 1: Demo Generation") {
            logger.info { "MIPRO Step 1: Generating $numCandidates demo candidate sets..." }
            val demos = generateDemoSets(
                dataset = bootstrapSet,
                numCandidateSets = numCandidates,
                maxBootstrappedDemos = maxBootstrappedDemos,
                maxTotalDemos = maxTotalDemos,
                includeLabeledExamples = includeLabeledExamples,
                random = Random(random.nextLong()),
                parallelism = parallelism,
                randomSeed = randomSeed,
            )
            logMiproDemoSets(demos)
            demos
        }

        // Step 2: Instruction Proposal — meta-LLM calls go through the tracked `executePrompt`
        // DSL via `metaPromptRunner`, so they show up as PromptExecutionRecords under this stage.
        val instructionCandidates = runStageOrThrow("Step 2: Instruction Proposal") {
            logger.info { "MIPRO Step 2: Proposing $numCandidates instructions per module..." }
            val trainExamples = dataset.map { "${it.userQuery}" }
            val proposer = InstructionProposer.create(
                agent = trackedAgent,
                modules = modules,
                trainExamples = trainExamples,
                runMeta = metaPromptRunner(metaModel),
                config = proposerConfig,
                random = Random(random.nextLong()),
            )
            val candidates = proposer.proposeInstructionsForProgram(
                demoCandidates = demoCandidates,
                numCandidates = numCandidates,
                parallelism = parallelism,
            )
            logMiproInstructionCandidates(candidates)
            candidates
        }

        // Step 3: Grid Search — `numTrials` random (instruction, demo set) trials plus a baseline.
        val bestArtifact = runStageOrThrow("Step 3: Grid Search", substagesTotal = numTrials + 1) {
            logger.info { "MIPRO Step 3: Running $numTrials grid search trials..." }
            gridSearch(
                modules = modules,
                instructionCandidates = instructionCandidates,
                demoCandidates = demoCandidates,
                valSet = valSet,
                numTrials = numTrials,
            )
        }

        saveArtifact(storagePath, bestArtifact)
        logger.info { "MIPRO complete. Artifact saved to ${storagePath.toFilePathLog()}" }
    }

    /**
     * Step 3: evaluate random (instruction, demo set) combinations.
     *
     * Each trial (and the baseline) runs in its own substage so per-trial timing,
     * consumption, and score are visible in the records tree and progress UI.
     */
    private suspend fun StageScope<Input, Output, InputLabel>.gridSearch(
        modules: List<OptimizableModule>,
        instructionCandidates: Map<String, List<String>>,
        demoCandidates: Map<String, List<List<Demonstration>>>?,
        valSet: TrainSet<Input, InputLabel>,
        numTrials: Int,
    ): OptimizationArtifact {
        var bestArtifact = OptimizationArtifact()
        var bestScore = -1.0
        var bestTrial: Int? = null
        var bestInstructions: Map<String, Int> = emptyMap()
        var bestDemoSets: Map<String, Int> = emptyMap()

        runStageOrThrow("Baseline (no optimization)") {
            logger.info { "MIPRO baseline (no optimization): evaluating on ${valSet.size} items..." }
            val baselineScore = evaluateArtifact(OptimizationArtifact(), valSet)
            bestScore = baselineScore
            bestArtifact = OptimizationArtifact()
            logMiproBaselineScore(baselineScore)
        }

        for (trial in 0 until numTrials) {
            runStageOrThrow("Trial ${trial + 1}/$numTrials") {
                logger.info { "MIPRO trial ${trial + 1}/$numTrials: evaluating on ${valSet.size} items..." }
                val combination = sampleTrialArtifact(modules, instructionCandidates, demoCandidates)
                logMiproTrialCombination(combination.instructionIndices, combination.demoSetIndices)
                val score = evaluateArtifact(combination.artifact, valSet)

                // Use >= to prefer optimized artifacts over baseline on tie
                val isNewBest = score >= bestScore
                if (isNewBest) {
                    bestScore = score
                    bestArtifact = combination.artifact
                    bestTrial = trial + 1
                    bestInstructions = combination.instructionIndices
                    bestDemoSets = combination.demoSetIndices
                }
                logMiproTrialOutcome(
                    trialNumber = trial + 1,
                    trialCount = numTrials,
                    averageValidationScore = score,
                    isNewBest = isNewBest,
                    bestAverageValidationScore = bestScore,
                )
            }
        }

        logMiproGridSearchResult(bestScore, bestTrial, bestInstructions, bestDemoSets)
        return bestArtifact
    }

    /**
     * Samples a random (instruction, demo set) combination into an artifact, and returns the position it
     * drew in each module's candidate list so a trial can record it.
     *
     * _Note on reproducibility._ The position comes from `nextInt(size)`, the draw `Collection.random(random)` performs.
     * Keep it that way: another way of drawing consumes the generator differently and moves MIPROv2's search.
     */
    private fun sampleTrialArtifact(
        modules: List<OptimizableModule>,
        instructionCandidates: Map<String, List<String>>,
        demoCandidates: Map<String, List<List<Demonstration>>>?,
    ): TrialCombination {
        var artifact = OptimizationArtifact()
        val instructionIndices = mutableMapOf<String, Int>()
        val demoSetIndices = mutableMapOf<String, Int>()

        for (module in modules) {
            // Sample random instruction
            val instructions = instructionCandidates[module.name]
            if (!instructions.isNullOrEmpty()) {
                val instructionIndex = random.nextInt(instructions.size)
                val instruction = instructions[instructionIndex]
                instructionIndices[module.name] = instructionIndex
                artifact = if (module.name == STRATEGY_MODULE_KEY) {
                    artifact.withStrategyInstruction(instruction)
                } else {
                    artifact.withSubgraphInstruction(module.name, instruction)
                }
            }

            // Sample random demo set
            if (demoCandidates != null) {
                val demoSets = demoCandidates[module.name]
                if (!demoSets.isNullOrEmpty()) {
                    val demoSetIndex = random.nextInt(demoSets.size)
                    val demos = demoSets[demoSetIndex]
                    demoSetIndices[module.name] = demoSetIndex
                    artifact = if (module.name == STRATEGY_MODULE_KEY) {
                        artifact.withStrategyDemonstrations(demos)
                    } else {
                        artifact.withSubgraphDemonstrations(module.name, demos)
                    }
                }
            }
        }

        return TrialCombination(artifact, instructionIndices, demoSetIndices)
    }

    /**
     * Evaluates an artifact on a validation set by running the agent with it installed.
     * Runs within a [StageScope] context from the enclosing grid search stage.
     */
    @OptIn(InternalAgentsApi::class)
    private suspend fun StageScope<Input, Output, InputLabel>.evaluateArtifact(
        artifact: OptimizationArtifact,
        valSet: TrainSet<Input, InputLabel>,
    ): Double {
        var totalScore = 0.0
        var count = 0
        val agentWithArtifact = trackedAgent.copyWith(installFeatures = {
            trackedAgent.installFeatures(this)
            installPromptOptimization { this.artifact = artifact }
        })
        iterateDataset(name = "Evaluate artifact", dataset = valSet) { item ->
            val result = runAgent(item, agentWithArtifact)
            result.onSuccess { totalScore += it.score }
            count++
        }
        return if (count > 0) totalScore / count else 0.0
    }

    /** Persistence helpers for the optimized [OptimizationArtifact]. */
    public companion object {
        private val jsonFormat = Json { ignoreUnknownKeys = true; prettyPrint = true }

        private fun saveArtifact(storagePath: ResilientPath, artifact: OptimizationArtifact) {
            storagePath.createParentDirectories()
            storagePath.writeText(jsonFormat.encodeToString(OptimizationArtifact.serializer(), artifact))
        }

        /**
         * Reads and deserializes the [OptimizationArtifact] stored at [storagePath].
         *
         * @throws IllegalArgumentException if no file exists at [storagePath].
         */
        public fun loadArtifact(storagePath: ResilientPath): OptimizationArtifact {
            if (storagePath.exists()) {
                return jsonFormat.decodeFromString(OptimizationArtifact.serializer(), storagePath.readText())
            } else {
                throw IllegalArgumentException("Cannot load artifact from $storagePath: file does not exist")
            }
        }
    }
}

/** One sampled (instruction, demo set) combination, with the candidate index it drew for each module. */
private data class TrialCombination(
    val artifact: OptimizationArtifact,
    val instructionIndices: Map<String, Int>,
    val demoSetIndices: Map<String, Int>,
)

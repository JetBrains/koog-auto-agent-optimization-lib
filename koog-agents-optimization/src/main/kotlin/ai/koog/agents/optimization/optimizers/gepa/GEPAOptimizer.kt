package ai.koog.agents.optimization.optimizers.gepa


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.STRATEGY_MODULE_KEY
import ai.koog.agents.optimization.core.discoverModules
import ai.koog.agents.optimization.features.collectSubgraphTraces
import ai.koog.agents.optimization.features.installPromptOptimization
import ai.koog.agents.optimization.koogTooling.copyWith
import ai.koog.agents.optimization.koogTooling.getCollectedTraces
import ai.koog.agents.optimization.optimizers.AgentOptimizer
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.executePromptOrThrow
import ai.koog.agents.optimization.training.dsl.runIterableStageOrThrow
import ai.koog.agents.optimization.training.dsl.runStageOrThrow
import ai.koog.agents.optimization.training.records.TrainingResult
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.common.toFilePathLog
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlin.random.Random

private val logger = KotlinLogging.logger {}

/**
 * Strategy for choosing which optimizable modules GEPA updates on each evolution iteration.
 */
public enum class ComponentSelection {
    /** Update one module per iteration, cycling through modules in order. */
    ROUND_ROBIN,

    /** Update every module on each iteration. */
    ALL_AT_ONCE,
}

/**
 * GEPA (Genetic-Pareto Optimizer for Reflective Prompt Evolution).
 *
 * An instruction-only optimizer that evolves module instructions via reflective LLM feedback
 * on failure cases, with Pareto-based candidate selection for diversity.
 *
 * **Early implementation.** This is a non-official re-implementation that deviates from the original
 * GEPA algorithm (https://arxiv.org/abs/2507.19457) and may underperform the paper. Validate on your
 * own task before relying on it; closing the gap is future work.
 *
 * Key differences from MIPROv2:
 * - Optimizes instructions only (no demonstrations) -> shorter, more generalizable prompts
 * - Uses textual feedback from a reflection LM analyzing failure traces
 * - a Pareto frontier maintains diverse candidate pool (vs. best-so-far)
 * - Optional crossover merges complementary candidates
 *
 * @param reflectionModel LLM model for reflection / instruction proposal. Calls are routed
 *   through the training DSL's tracked `executePrompt`, so timing and consumption are recorded
 *   under the appropriate iteration stage.
 * @param storagePath Path for saving the optimization artifact.
 * @param maxIterations Maximum number of evolution iterations.
 * @param minibatchSize Number of training items sampled per iteration for reflection.
 * @param componentSelection How to select which modules to update each iteration.
 * @param enableCrossover Whether to attempt merging complementary candidates during the loop.
 * @param maxMergeInvocations Maximum number of crossover attempts per optimization run.
 * @param randomSeed Seed for reproducibility.
 * @param labelExtractor Converts a dataset item's label to a string for the reflection LM.
 */
public class GEPAOptimizer<Input, Output, InputLabel>(
    private val reflectionModel: LLModel,
    public val storagePath: ResilientPath,
    private val maxIterations: Int,
    private val minibatchSize: Int,
    private val componentSelection: ComponentSelection,
    private val enableCrossover: Boolean,
    private val maxMergeInvocations: Int = 5, // TODO: wire through GEPAConfig and createGEPAOptimizer, remove default
    private val randomSeed: Long = 42L,
    private val labelExtractor: (TrainSetItem<Input, InputLabel>) -> String,
) : AgentOptimizer<Input, Output, InputLabel> {

    @OptIn(InternalAgentsApi::class)
    override fun loadOptimizedAgent(baseAgent: GraphAIAgent<Input, Output>): GraphAIAgent<Input, Output> {
        val artifact = loadArtifact(storagePath)
        return baseAgent.copyWith(installFeatures = {
            baseAgent.installFeatures(this)
            installPromptOptimization { this.artifact = artifact }
        })
    }

    /**
     * Builds a tracked reflection proposer from this stage scope. Reflection LLM calls
     * recorded under the current stage so consumption is captured correctly.
     */
    private fun StageScope<*, *, *>.buildReflectionProposer(): GEPAReflectionProposer {
        val runReflection: GEPAReflectionRunner = { prompt ->
            executePromptOrThrow(prompt, reflectionModel)
                .filterIsInstance<Message.Assistant>()
                .joinToString("\n") { it.content }
        }
        return GEPAReflectionProposer(runReflection)
    }

    @OptIn(InternalAgentsApi::class)
    override suspend fun train(
        session: TrainingSession<Input, Output, InputLabel>,
    ): TrainingResult = session.use(stagesTotal = maxIterations + 1) {
        val random = Random(randomSeed)
        val modules = discoverModules(trackedAgent)
        val frontier = ParetoFrontier(random)

        logger.info { "=== GEPA Optimization ===" }
        logger.info { "  Modules: ${modules.map { it.name }}" }
        logger.info { "  Dataset size: ${dataset.size}" }
        logger.info { "  Max iterations: $maxIterations, minibatch size: $minibatchSize" }
        logger.info { "  Component selection: $componentSelection, crossover: $enableCrossover" }

        // Step 1: Evaluate baseline
        logger.info { "Evaluating baseline (no optimization)..." }
        val baselineScores = evaluateArtifact(OptimizationArtifact(), dataset, name = "Evaluate baseline")
        val baselineCandidate = ParetoCandidate(
            artifact = OptimizationArtifact(),
            perInstanceScores = baselineScores,
        )
        frontier.addCandidate(baselineCandidate)
        logger.info { "  Baseline score: ${"%.4f".format(baselineCandidate.aggregateScore)}" }

        // Step 2: Main evolutionary loop
        var roundRobinIdx = 0
        var mergeCount = 0
        for (iteration in 1..maxIterations) runStageOrThrow("Iteration $iteration") {
            logger.info { "--- Iteration $iteration/$maxIterations ---" }

            // The proposer is built per-iteration so its tracked LLM calls record under THIS
            // iteration's stage rather than under the root scope.
            val reflectionProposer = buildReflectionProposer()

            // Select a parent candidate from a Pareto frontier
            val parent = frontier.selectCandidate()
            logger.info { "  Selected parent (score: ${"%.4f".format(parent.aggregateScore)})" }

            // Sample minibatch
            val minibatch = dataset.shuffled(random).take(minibatchSize)

            // Execute on minibatch, collect failures with traces.
            // Pre-populate keys for all modules so failures are tracked per-module.
            // `executeAndCollectFeedback` already creates a named "Collect feedback"
            // iterateDataset stage internally; no need to wrap it in another runStage.
            val failures = modules.associate { it.name to mutableListOf<GEPAReflectionProposer.FailureFeedback>() }.toMutableMap()
            executeAndCollectFeedback(parent.artifact, minibatch, failures)

            if (failures.values.all { it.isEmpty() }) {
                logger.info { "  No failures on minibatch, skipping reflection." }
                return@runStageOrThrow // effectively 'continue'
            }

            // Select which modules to update
            val modulesToUpdate = when (componentSelection) {
                ComponentSelection.ALL_AT_ONCE -> modules
                ComponentSelection.ROUND_ROBIN -> {
                    val selected = listOf(modules[roundRobinIdx % modules.size])
                    roundRobinIdx++
                    selected
                }
            }
            logger.info { "  Updating modules: ${modulesToUpdate.map { it.name }}" }

            // Propose new instructions via reflection.
            // Each module gets only the failures relevant to it (all failures for now —
            // per-subgraph trace filtering is a future improvement).
            var newArtifact = parent.artifact
            runIterableStageOrThrow(modulesToUpdate, "Propose new instructions") { module ->
                val moduleFailures = failures[module.name].orEmpty()
                if (moduleFailures.isNotEmpty()) {
                    val currentInstruction = getCurrentInstruction(module, parent.artifact)
                    val proposed = reflectionProposer.proposeInstruction(module, currentInstruction, moduleFailures)
                    newArtifact = applyInstruction(newArtifact, module.name, proposed)
                }
            }

            // Evaluate on minibatch first — only do full dataset eval if improved.
            // This saves budget: most candidates don't improve on the minibatch.
            val minibatchScores = evaluateArtifact(newArtifact, minibatch, name = "Evaluate candidate on minibatch")
            val minibatchScore = if (minibatchScores.isEmpty()) 0.0
                else minibatchScores.values.sum() / minibatchScores.size
            val parentMinibatchScore = parent.aggregateScore // approximate comparison

            if (minibatchScore <= parentMinibatchScore) {
                logger.info { "  Minibatch score ${"%.4f".format(minibatchScore)} did not improve over parent ${"%.4f".format(parentMinibatchScore)}, skipping full eval." }
                return@runStageOrThrow // essentially 'continue'
            }

            // Full dataset evaluation for candidates that improved on minibatch
            logger.info { "  Minibatch improved (${"%.4f".format(minibatchScore)} > ${"%.4f".format(parentMinibatchScore)}), running full eval..." }
            val newScores = evaluateArtifact(newArtifact, dataset, name = "Evaluate candidate")
            val newCandidate = ParetoCandidate(
                artifact = newArtifact,
                perInstanceScores = newScores,
                parentIndex = frontier.size() - 1,
            )
            frontier.addCandidate(newCandidate)
            logger.info { "  New candidate score: ${"%.4f".format(newCandidate.aggregateScore)} " +
                    "(best so far: ${"%.4f".format(frontier.getBestCandidate().aggregateScore)})" }

            // Crossover: attempt to merge complementary candidates from the frontier.
            // Runs inside the loop so merged candidates can be selected as parents in future iterations.
            // Conditions: crossover enabled, budget remaining, frontier has at least 2 above-baseline
            // candidates with different ancestry.
            if (enableCrossover && mergeCount < maxMergeInvocations && frontier.size() >= 3) {
                val candidateA = frontier.getBestCandidate()
                val mergePartner = frontier.getAllCandidates().firstOrNull { other ->
                    other !== candidateA
                            && other.aggregateScore > baselineCandidate.aggregateScore
                            && other.parentIndex != candidateA.parentIndex // disjoint lineage
                }

                if (mergePartner != null) {
                    logger.info { "  Attempting crossover (merge #${mergeCount + 1}/$maxMergeInvocations)..." }
                    var crossedArtifact = candidateA.artifact
                    runIterableStageOrThrow(modules, "Crossover instructions") { module ->
                        val instrA = getCurrentInstruction(module, candidateA.artifact)
                        val instrB = getCurrentInstruction(module, mergePartner.artifact)
                        if (instrA != instrB) {
                            val merged = reflectionProposer.proposeCrossover(
                                module, instrA, candidateA.aggregateScore, instrB, mergePartner.aggregateScore,
                            )
                            crossedArtifact = applyInstruction(crossedArtifact, module.name, merged)
                        }
                    }

                    val crossedScores = evaluateArtifact(crossedArtifact, dataset, name = "Crossover evaluation")
                    val crossedCandidate = ParetoCandidate(
                        artifact = crossedArtifact,
                        perInstanceScores = crossedScores,
                    )
                    frontier.addCandidate(crossedCandidate)
                    mergeCount++
                    logger.info { "  Crossover candidate score: ${"%.4f".format(crossedCandidate.aggregateScore)}" }
                }
            }
        }

        // Step 4: Save best artifact
        val bestCandidate = frontier.getBestCandidate()
        logger.info { "=== GEPA Complete ===" }
        logger.info { "  Best score: ${"%.4f".format(bestCandidate.aggregateScore)}" }
        logger.info { "  Candidates evaluated: ${frontier.size()}" }
        saveArtifact(storagePath, bestCandidate.artifact)
        logger.info { "Optimization completed. Artifact saved to ${storagePath.toFilePathLog()}." }
    }

    /**
     * Evaluates an artifact on a dataset, returning per-item scores.
     *
     * The [name] is forwarded to the inner [iterateDataset] stage; callers can pass a
     * distinguishing label (e.g. "Evaluate baseline" / "Evaluate candidate") so the
     * records tree shows what kind of evaluation this is, without an extra wrapping stage.
     */
    @OptIn(InternalAgentsApi::class)
    private suspend fun StageScope<Input, Output, InputLabel>.evaluateArtifact(
        artifact: OptimizationArtifact,
        dataset: TrainSet<Input, InputLabel>,
        name: String = "Evaluate artifact",
    ): Map<Int, Double> {
        val scores = mutableMapOf<Int, Double>()
        val agentWithArtifact = trackedAgent.copyWith(installFeatures = {
            trackedAgent.installFeatures(this)
            installPromptOptimization { this.artifact = artifact }
        })
        iterateDataset(name = name, dataset = dataset) { item ->
            val result = runAgent(item, agentWithArtifact)
            result.onSuccess { scores[dataset.indexOf(item)] = it.score }
        }
        return scores
    }

    /**
     * Runs the agent on a minibatch, collecting failure traces for reflection.
     */
    @OptIn(InternalAgentsApi::class)
    private suspend fun StageScope<Input, Output, InputLabel>.executeAndCollectFeedback(
        artifact: OptimizationArtifact,
        minibatch: TrainSet<Input, InputLabel>,
        failures: MutableMap<String, MutableList<GEPAReflectionProposer.FailureFeedback>>,
    ) {
        val agentWithFeatures = trackedAgent.copyWith(installFeatures = {
            trackedAgent.installFeatures(this)
            installPromptOptimization { this.artifact = artifact }
            collectSubgraphTraces { }
        })
        iterateDataset(name = "Collect feedback", dataset = minibatch) { item ->
            val result = runAgent(item, agentWithFeatures)
            val agentRun = result.getOrNull()
            if (agentRun != null && !agentRun.isSolved) {
                val trajectory = agentRun.usedAgent?.getCollectedTraces()?.getLatestFullPrompt()
                val feedback = GEPAReflectionProposer.FailureFeedback(
                    input = item.userQuery.toString(),
                    expectedLabel = labelExtractor(item),
                    actualOutput = agentRun.output.toString(),
                    score = agentRun.score,
                    trajectory = trajectory ?: error("SubgraphTraceCollectionFeature did not capture a trajectory"),
                )
                // Add failure to all modules -- the reflection proposer analyzes
                // the full trace and targets module-specific improvements.
                for (list in failures.values) {
                    list.add(feedback)
                }
            }
        }
    }

    private fun getCurrentInstruction(module: OptimizableModule, artifact: OptimizationArtifact): String {
        return if (module.name == STRATEGY_MODULE_KEY) {
            artifact.strategyInstruction ?: module.currentInstruction
        } else {
            artifact.getInstruction(module.name) ?: module.currentInstruction
        }
    }

    private fun applyInstruction(artifact: OptimizationArtifact, moduleName: String, instruction: String): OptimizationArtifact {
        return if (moduleName == STRATEGY_MODULE_KEY) {
            artifact.withStrategyInstruction(instruction)
        } else {
            artifact.withSubgraphInstruction(moduleName, instruction)
        }
    }

    /** Persistence helpers for the evolved [OptimizationArtifact]. */
    public companion object {
        private val jsonFormat = Json { ignoreUnknownKeys = true; prettyPrint = true }

        /**
         * Serializes [artifact] as pretty-printed JSON and writes it to [storagePath],
         * creating parent directories as needed.
         */
        public fun saveArtifact(storagePath: ResilientPath, artifact: OptimizationArtifact) {
            storagePath.createParentDirectories()
            storagePath.writeText(jsonFormat.encodeToString(artifact))
            logger.info { "GEPA artifact saved to $storagePath" }
        }

        /**
         * Reads and deserializes the [OptimizationArtifact] stored at [storagePath].
         *
         * @throws IllegalArgumentException if no file exists at [storagePath].
         */
        public fun loadArtifact(storagePath: ResilientPath): OptimizationArtifact {
            require(storagePath.exists()) { "Cannot load optimization artifact from $storagePath: file does not exist" }
            return jsonFormat.decodeFromString(storagePath.readText())
        }
    }
}


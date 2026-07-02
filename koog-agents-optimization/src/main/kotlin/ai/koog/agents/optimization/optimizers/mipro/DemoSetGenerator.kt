package ai.koog.agents.optimization.optimizers.mipro


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.DemonstrationRenderer
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.STRATEGY_MODULE_KEY
import ai.koog.agents.optimization.features.collectSubgraphTraces
import ai.koog.agents.optimization.koogTooling.copyWith
import ai.koog.agents.optimization.koogTooling.getCollectedTraces
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.optimizers.fewShot.BootstrapFewShotOptimizer.CollectedRun
import ai.koog.agents.optimization.training.PrematureExecutionStopDecision
import ai.koog.agents.optimization.training.dsl.PreparedAgentRun
import ai.koog.agents.optimization.training.dsl.RunUntil
import ai.koog.agents.optimization.training.dsl.StageScope
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlin.math.min
import kotlin.random.Random

private val logger = KotlinLogging.logger {}

/**
 * Generates N diverse candidate demo sets for MIPRO Step 1.
 *
 * Each candidate set is produced by running a bootstrap collection loop (similar to
 * [ai.koog.agents.optimization.optimizers.fewShot.BootstrapFewShotOptimizer])
 * with different configurations (shuffled trainsets, varying demo counts) to encourage diversity.
 *
 * Returns demo candidates keyed by module name ([STRATEGY_MODULE_KEY] for strategy-level,
 * subgraph names for per-subgraph). Each module maps to a list of candidate demo sets.
 *
 * @param dataset The training dataset for bootstrap.
 * @param scope The [StageScope] for running agent iterations within the training DSL.
 * @param numCandidateSets Total number of candidate sets to generate.
 * @param maxBootstrappedDemos Maximum demos per bootstrap run.
 * @param maxTotalDemos Total demo slots (bootstrapped and labeled fallback).
 * @param includeLabeledExamples Whether to include labeled fallback.
 * @param random Random instance for reproducibility.
 * @param parallelism Maximum concurrent bootstrap runs.
 * @param randomSeed Base seed for bootstrap runs.
 * @return Map from the module name to a list of candidate demo sets, or null if zero-shot mode.
 */
@OptIn(InternalAgentsApi::class)
internal suspend fun <Input, Output, InputLabel> StageScope<Input, Output, InputLabel>.generateDemoSets(
    dataset: TrainSet<Input, InputLabel>,
    numCandidateSets: Int,
    maxBootstrappedDemos: Int,
    maxTotalDemos: Int,
    includeLabeledExamples: Boolean,
    random: Random,
    parallelism: Int = 1,
    randomSeed: Long = 42L,
): Map<String, List<List<Demonstration>>>? {
    if (maxBootstrappedDemos == 0 && maxTotalDemos == 0) return null

    val out = mutableMapOf<String, MutableList<List<Demonstration>>>()

    // Helper: extract demos from an artifact into the output map
    fun addFromArtifact(artifact: OptimizationArtifact) {
        // Strategy-level demos
        out.getOrPut(STRATEGY_MODULE_KEY) { mutableListOf() }
            .add(artifact.strategyDemonstrations)

        // Per-subgraph demos
        for ((name, demos) in artifact.subgraphDemonstrations) {
            out.getOrPut(name) { mutableListOf() }.add(demos)
        }
    }

    var remaining = numCandidateSets

    // 1. Zero-shot set (empty demos)
    remaining--
    out.getOrPut(STRATEGY_MODULE_KEY) { mutableListOf() }.add(emptyList())

    // 2. Unshuffled bootstrap (no metric filtering -- accepts all traces)
    remaining--
    logger.info { "Demo set generation [2/$numCandidateSets]: running unshuffled bootstrap..." }
    val unshuffledArtifact = runBootstrapInScope(
        dataset = dataset,
        maxBootstrappedDemos = maxBootstrappedDemos,
        maxTotalDemos = maxTotalDemos,
        includeLabeledExamples = includeLabeledExamples,
        randomSeed = randomSeed,
        iterationName = "Bootstrap (unshuffled)",
    )
    addFromArtifact(unshuffledArtifact)

    // 3. Shuffled bootstraps to fill remaining slots
    val shuffledCount = maxOf(0, remaining)
    if (shuffledCount > 0) {
        logger.info { "Demo set generation [3-${2 + shuffledCount}/$numCandidateSets]: running $shuffledCount shuffled bootstraps..." }

        data class ShuffledParams(val seed: Long, val numDemos: Int)

        val params = (0 until shuffledCount).map {
            ShuffledParams(
                seed = random.nextLong(),
                numDemos = random.nextInt(1, maxBootstrappedDemos + 1),
            )
        }

        val semaphore = Semaphore(maxOf(1, parallelism))
        val artifacts = coroutineScope {
            params.mapIndexed { index, p ->
                async {
                    semaphore.withPermit {
                        logger.info { "  Starting shuffled bootstrap ${index + 1}/$shuffledCount (numDemos=${p.numDemos})..." }
                        val shuffledDataset = dataset.shuffled(Random(p.seed))
                        val artifact = runBootstrapInScope(
                            dataset = shuffledDataset,
                            maxBootstrappedDemos = p.numDemos,
                            maxTotalDemos = maxTotalDemos,
                            includeLabeledExamples = includeLabeledExamples,
                            randomSeed = p.seed,
                            iterationName = "Bootstrap (shuffled ${index + 1})",
                        )
                        logger.info { "  Shuffled bootstrap ${index + 1}/$shuffledCount completed" }
                        artifact
                    }
                }
            }.awaitAll()
        }

        for (artifact in artifacts) {
            addFromArtifact(artifact)
        }
    }

    return out.mapValues { (_, v) -> v.toList() }
}

/**
 * Runs a single bootstrap collection loop within the current [StageScope].
 *
 * This inlines the core logic of [ai.koog.agents.optimization.optimizers.fewShot.BootstrapFewShotOptimizer]
 * without requiring a separate [ai.koog.agents.optimization.training.TrainingSession].
 */
@OptIn(InternalAgentsApi::class)
private suspend fun <Input, Output, InputLabel> StageScope<Input, Output, InputLabel>.runBootstrapInScope(
    dataset: TrainSet<Input, InputLabel>,
    maxBootstrappedDemos: Int,
    maxTotalDemos: Int,
    includeLabeledExamples: Boolean,
    randomSeed: Long,
    iterationName: String = "Dataset iteration",
): OptimizationArtifact {
    // It's not possible to start a training of another optimizer using the same TrainingSession.
    // Therefore, we have to duplicate the code from BootstrapFewShot here.
    // TODO: deduplicate this code by adding public functions to BootstrapFewShot

    val maxRounds = maxTotalDemos
    val random = Random(randomSeed)
    val collectedRuns = mutableListOf<CollectedRun>()
    val failedDatasetItems = mutableListOf<TrainSetItem<Input, InputLabel>>()

    iterateDataset(
        name = iterationName,
        dataset = dataset,
        earlyStop = { _ ->
            PrematureExecutionStopDecision(
                conditionMet = collectedRuns.size >= maxBootstrappedDemos,
                conditionMetReason = { "Collected ${collectedRuns.size} bootstrapped demos (max: $maxBootstrappedDemos)" },
            )
        },
    ) { item ->
        val solvedRun = runAgentWithRetries(item, maxRounds, RunUntil.SOLVED) {
            val agentWithTracing = trackedAgent.copyWith(installFeatures = {
                trackedAgent.installFeatures(this)
                collectSubgraphTraces { }
            })
            PreparedAgentRun(agent = agentWithTracing, runData = Unit)
        }

        // Trace collection is pure bookkeeping (no LLM call) — flatten onto the item stage.
        if (solvedRun != null) {
            val usedAgent = solvedRun.agentRun.usedAgent
            val collected = extractTraces(usedAgent, item, solvedRun.agentRun.output)
            if (collected != null) {
                collectedRuns += collected
            }
        } else {
            failedDatasetItems += item
        }
        logAction {
            put("collected", solvedRun != null)
            put("totalCollectedRuns", collectedRuns.size)
            put("totalFailedItems", failedDatasetItems.size)
        }
    }

    return buildBootstrapArtifact(
        collectedRuns = collectedRuns,
        failedDatasetItems = failedDatasetItems,
        maxBootstrappedDemos = maxBootstrappedDemos,
        maxTotalDemos = maxTotalDemos,
        includeLabeledExamples = includeLabeledExamples,
        random = random
    )
}

// duplicated code
private suspend fun <Input, Output, InputLabel> extractTraces(
    usedAgent: GraphAIAgent<*, *>?,
    item: TrainSetItem<Input, InputLabel>,
    output: Output,
): CollectedRun? {
    if (usedAgent == null) {
        logger.warn { "usedAgent not available — cannot extract traces" }
        return null
    }

    // Access the SubgraphTraceCollectionFeature from the used agent's pipeline
    val traces = usedAgent.getCollectedTraces()

    // Strategy-level demo: full trajectory with inherited prefix stripped
    val fullPrompt = traces?.getLatestFullPrompt()
    val initialMessages = usedAgent.agentConfig.prompt.messages
    val intermediateMessages = fullPrompt?.let { prompt ->
        DemonstrationRenderer.dropInheritedPrefix(prompt.messages, initialMessages)
    }

    val strategyDemo = Demonstration(
        input = item.userQuery.toString(),
        output = output.toString(),
        intermediateMessages = intermediateMessages,
    )

    // Per-subgraph demos (automatically collected by the feature)
    val subgraphTraces = traces?.getAllTraces().orEmpty()

    return CollectedRun(strategyDemo, subgraphTraces)
}

// duplicated code
private fun <Input, InputLabel> buildBootstrapArtifact(
    maxBootstrappedDemos: Int,
    includeLabeledExamples: Boolean,
    maxTotalDemos: Int,
    random: Random,
    collectedRuns: List<CollectedRun>,
    failedDatasetItems: List<TrainSetItem<Input, InputLabel>>,
): OptimizationArtifact {
    val strategyDemos = collectedRuns.map { it.strategyDemo }.take(maxBootstrappedDemos)

    // Per-subgraph demos: merge across all runs
    val allSubgraphNames = collectedRuns.flatMap { it.subgraphTraces.keys }.toSet()
    val subgraphDemos = allSubgraphNames.associateWith { name ->
        collectedRuns.flatMap { run ->
            run.subgraphTraces[name].orEmpty()
        }.take(maxBootstrappedDemos)
    }

    // Labeled fallback for strategy-level demos
    val labeledStrategyDemos = if (includeLabeledExamples) {
        val remaining = (maxTotalDemos - strategyDemos.size).coerceAtLeast(0)
        val labeledItems = failedDatasetItems.shuffled(random).take(min(remaining, failedDatasetItems.size))
        labeledItems.map { item ->
            Demonstration(
                input = item.userQuery.toString(),
                output = item.itemLabel.toString(),
            )
        }
    } else {
        emptyList()
    }

    return OptimizationArtifact(
        strategyDemonstrations = strategyDemos + labeledStrategyDemos,
        subgraphDemonstrations = subgraphDemos,
    )
}
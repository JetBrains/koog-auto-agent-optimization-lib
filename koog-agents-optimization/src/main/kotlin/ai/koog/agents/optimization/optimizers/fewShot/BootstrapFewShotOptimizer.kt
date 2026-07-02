package ai.koog.agents.optimization.optimizers.fewShot


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.DemonstrationRenderer
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.features.collectSubgraphTraces
import ai.koog.agents.optimization.features.installPromptOptimization
import ai.koog.agents.optimization.koogTooling.copyWith
import ai.koog.agents.optimization.koogTooling.getCollectedTraces
import ai.koog.agents.optimization.optimizers.AgentOptimizer
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.PrematureExecutionStopDecision
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.dsl.PreparedAgentRun
import ai.koog.agents.optimization.training.dsl.RunUntil
import ai.koog.agents.optimization.training.records.TrainingResult
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.common.toFilePathLog
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlin.math.min
import kotlin.random.Random

/**
 * Bootstrap few-shot optimizer that collects execution traces from successful agent runs
 * and produces an [OptimizationArtifact] with demonstrations at both strategy and subgraph levels.
 *
 * For each training example, the agent is run up to [maxRounds] times. On success, the optimizer
 * collects:
 * - **Strategy-level demos**: full agent trajectory (via trace collection feature),
 *   with the inherited system prompt prefix stripped.
 * - **Per-subgraph demos**: traces from each optimizable subgraph,
 *   automatically collected when the strategy uses `optimizableSubgraphWithTask`.
 *
 * The resulting [OptimizationArtifact] is saved to [storagePath] and applied to the agent
 * via [installPromptOptimization] for inference.
 *
 * @param maxBootstrappedDemos Maximum number of successful traces to collect.
 * @param maxRounds Maximum retry attempts per training example.
 * @param maxTotalDemos Total demo slots (bootstrapped and labeled fallback).
 * @param includeLabeledExamples Whether to fill the remaining slots with labeled dataset items.
 * @param storagePath File path to save/load the optimization artifact.
 * @param randomSeed Seed for reproducible sampling.
 */
public class BootstrapFewShotOptimizer<Input, Output, InputLabel>(
    public val maxBootstrappedDemos: Int,
    public val maxRounds: Int,
    public val maxTotalDemos: Int,
    public val includeLabeledExamples: Boolean,
    public val storagePath: ResilientPath,
    randomSeed: Long,
) : AgentOptimizer<Input, Output, InputLabel> {

    private val random: Random = Random(randomSeed)
    private val logger = KotlinLogging.logger { }

    /**
     * Collected traces from a single successful agent run.
     *
     * @property strategyDemo Strategy-level demonstration (agent input/output + trajectory).
     * @property subgraphTraces Per-subgraph traces collected by the trace collection feature.
     */
    internal data class CollectedRun(
        val strategyDemo: Demonstration,
        val subgraphTraces: Map<String, List<Demonstration>>,
    )

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
    ): TrainingResult = session.use(stagesTotal = 1) {
        val collectedRuns = mutableListOf<CollectedRun>()
        val failedDatasetItems = mutableListOf<TrainSetItem<Input, InputLabel>>()

        iterateDataset(
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

            // Trace collection is pure bookkeeping (no LLM call); flatten directly onto the
            // item stage rather than spawning a substage with empty metrics.
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

        if (collectedRuns.size < maxBootstrappedDemos) {
            logger.warn { "Collected only ${collectedRuns.size}/$maxBootstrappedDemos bootstrapped demos" }
        }

        val artifact = buildArtifact(collectedRuns, failedDatasetItems)
        saveArtifact(storagePath, artifact)
        logger.info { "Optimization completed. Artifact saved to ${storagePath.toFilePathLog()}." }
    }

    /**
     * Extracts traces from the agent that ran a successful training example.
     */
    internal suspend fun extractTraces(
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

    /**
     * Builds the [OptimizationArtifact] from collected runs and optional labeled fallback.
     */
    private fun buildArtifact(
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

    /** Persistence helpers for the bootstrapped [OptimizationArtifact]. */
    public companion object {
        private val json = Json { ignoreUnknownKeys = true; prettyPrint = true }

        /**
         * Serializes [artifact] as pretty-printed JSON and writes it to [storagePath],
         * creating parent directories as needed.
         */
        public fun saveArtifact(storagePath: ResilientPath, artifact: OptimizationArtifact) {
            storagePath.createParentDirectories()
            storagePath.writeText(json.encodeToString(OptimizationArtifact.serializer(), artifact))
        }

        /**
         * Reads and deserializes the [OptimizationArtifact] stored at [storagePath].
         *
         * @throws IllegalArgumentException if no file exists at [storagePath].
         */
        public fun loadArtifact(storagePath: ResilientPath): OptimizationArtifact {
            if (storagePath.exists()) {
                val fileContent = storagePath.readText()
                return json.decodeFromString(OptimizationArtifact.serializer(), fileContent)
            } else {
                throw IllegalArgumentException("Cannot load optimization artifact from $storagePath: file does not exist")
            }
        }
    }
}


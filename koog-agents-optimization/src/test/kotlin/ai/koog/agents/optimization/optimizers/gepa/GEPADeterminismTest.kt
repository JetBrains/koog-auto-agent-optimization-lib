package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.records.AgentRunRecord
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.training.records.TrainingResult
import ai.koog.agents.optimization.training.trainingSession
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.streaming.StreamFrame
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Pins GEPA's optimization behavior on deterministic scenarios: mock executors, a fixed random seed,
 * and a rollout score sequence keyed by invocation order.
 *
 * Each expectation is a fingerprint of one run: the rollouts it spent, the reflections it ran, the
 * candidates it evaluated and accepted, and the artifact it saved. Any change to GEPA's search order,
 * its budget accounting or its acceptance rule moves at least one of these numbers, which is the point
 * of the test. A change here needs the same scrutiny as a change to the optimizer.
 */
internal class GEPADeterminismTest {
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()

    @Test
    fun `two accepted candidates over two batches`() {
        val fingerprint = runScenario(
            numRollouts = 14,
            feedbackBatchSize = 2,
            scores = listOf(
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95, 0.3, 0.2, 0.1, 0.4, 0.8, 0.85, 0.2,
                0.55, 0.65, 0.75, 0.05, 0.15, 0.25, 0.35, 0.45, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8,
            ),
        )

        assertEquals(
            Fingerprint(
                rolloutsScored = 17,
                agentRuns = 17,
                batches = 2,
                reflections = 2,
                proposalsEvaluated = 2,
                candidatesAccepted = 2,
                strategyInstruction = "Instruction v1.",
            ),
            fingerprint,
        )
    }

    @Test
    fun `a perfect parent skips every batch`() {
        val fingerprint = runScenario(
            numRollouts = 10,
            feedbackBatchSize = 1,
            scores = List(12) { 1.0 },
        )

        assertEquals(
            Fingerprint(
                rolloutsScored = 10,
                agentRuns = 10,
                batches = 7,
                reflections = 0,
                proposalsEvaluated = 0,
                candidatesAccepted = 0,
                strategyInstruction = SEED_STRATEGY_INSTRUCTION,
            ),
            fingerprint,
        )
    }

    @Test
    fun `a proposal already in the pool is skipped before it is evaluated`() {
        val fingerprint = runScenario(
            numRollouts = 12,
            feedbackBatchSize = 2,
            repeatProposal = true,
            scores = listOf(
                0.1, 0.2, 0.3, 0.4, 0.9, 0.95, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            ),
        )

        assertEquals(
            Fingerprint(
                rolloutsScored = 12,
                agentRuns = 12,
                batches = 2,
                reflections = 2,
                proposalsEvaluated = 1,
                candidatesAccepted = 1,
                strategyInstruction = "Repeated instruction.",
            ),
            fingerprint,
        )
    }

    @Test
    fun `merging enabled leaves the search itself unchanged`() {
        val fingerprint = runScenario(
            numRollouts = 40,
            feedbackBatchSize = 2,
            merge = true,
            scores = listOf(
                0.1, 0.2, 0.1, 0.2, 0.3, 0.1, 0.9, 0.95, 0.4, 0.5, 0.6, 0.2, 0.3, 0.4,
                0.85, 0.9, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.95, 0.99, 0.6, 0.7, 0.8, 0.4, 0.5, 0.6,
                0.75, 0.8, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.9, 0.95, 0.99, 0.1, 0.2,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            ),
        )

        assertEquals(
            Fingerprint(
                rolloutsScored = 41,
                agentRuns = 41,
                batches = 10,
                reflections = 6,
                proposalsEvaluated = 6,
                candidatesAccepted = 2,
                strategyInstruction = "Instruction v5.",
                mergeAttempts = 2,
            ),
            fingerprint,
        )
    }

    // ===================================================================
    // Harness
    // ===================================================================

    private data class Fingerprint(
        val rolloutsScored: Int,
        val agentRuns: Int,
        val batches: Int,
        val reflections: Int,
        val proposalsEvaluated: Int,
        val candidatesAccepted: Int,
        val strategyInstruction: String?,
        val mergeAttempts: Int = 0,
    )

    private fun runScenario(
        numRollouts: Int,
        feedbackBatchSize: Int,
        scores: List<Double>,
        merge: Boolean = false,
        repeatProposal: Boolean = false,
    ): Fingerprint {
        val rolloutExecutor = getMockExecutor { mockLLMToolCall(finishTool, "done") onCondition { true } }
        val reflectionExecutors = (1..40).map { index ->
            val instruction = if (repeatProposal) "Repeated instruction." else "Instruction v$index."
            getMockExecutor { mockLLMAnswer("```$instruction```").asDefaultResponse }
        }
        var reflections = 0
        val executor = TestPromptExecutor { prompt, model, tools ->
            if (tools.isEmpty()) {
                val chosen = reflectionExecutors[reflections.coerceAtMost(reflectionExecutors.lastIndex)]
                reflections++
                chosen.execute(prompt, model, tools)
            } else {
                rolloutExecutor.execute(prompt, model, tools)
            }
        }
        val agent = createAgent(executor)
        var rolloutsScored = 0
        val temporaryDirectory = Files.createTempDirectory("gepa-determinism-test").toFile()
        try {
            val optimizer = GEPAOptimizer<String, String, String>(
                gepaFeedback = strategyOnlyGepaFeedback { _, _, _, _, _ ->
                    val score = scores[rolloutsScored % scores.size]
                    rolloutsScored++
                    score to "Rollout $rolloutsScored scored $score."
                },
                feedbackValSplitFn = { items ->
                    GEPATrainSetSplit(feedbackSet = items.take(3), validationSet = items.drop(3))
                },
                reflectionModel = OpenAIModels.Chat.GPT4o,
                storagePath = ResilientPath(temporaryDirectory.resolve("artifact.json").toPath()),
                numRollouts = numRollouts,
                feedbackBatchSize = feedbackBatchSize,
                randomSeed = 42,
                mergeConfig = GEPAMergeConfig(enabled = merge, valSetSubsampleSize = 2),
                seedValidationFailureRateThreshold = 1.0,
            )
            val result = runBlocking { optimizer.train(session(agent, executor)) }
            return Fingerprint(
                rolloutsScored = rolloutsScored,
                agentRuns = result.countRecords { it is AgentRunRecord },
                batches = result.countStagesNamed("Feedback batch "),
                reflections = reflections,
                proposalsEvaluated = result.countStagesNamed("Evaluate proposed candidate on feedback batch "),
                candidatesAccepted = result.countStagesNamed("Evaluate accepted candidate on validation set for "),
                strategyInstruction = GEPAOptimizer.loadArtifact(optimizer.storagePath).strategyInstruction,
                mergeAttempts = result.countStagesNamed("Merge attempt "),
            )
        } finally {
            temporaryDirectory.deleteRecursively()
        }
    }

    private fun TrainingResult.countRecords(predicate: (TrainingRecord) -> Boolean): Int {
        var count = 0
        fun walk(record: TrainingRecord) {
            if (predicate(record)) count++
            if (record is StageRecord) record.substages.forEach(::walk)
        }
        walk(rootStage)
        return count
    }

    /** Counts the stages a step opens under [prefix], skipping the dataset and per-item stages beneath them. */
    private fun TrainingResult.countStagesNamed(prefix: String): Int =
        countRecords { record ->
            record is StageRecord &&
                    record.name.startsWith(prefix) &&
                    !record.name.contains(": run dataset") &&
                    !record.name.contains("| Item #")
        }

    private fun createAgent(promptExecutor: PromptExecutor): GraphAIAgent<String, String> = GraphAIAgent(
        promptExecutor = promptExecutor,
        agentConfig = AIAgentConfig(
            prompt = prompt("test") { system(SEED_STRATEGY_INSTRUCTION) },
            model = OpenAIModels.Chat.GPT4o,
            maxAgentIterations = 20,
        ),
        strategy = strategy("test-strategy") {
            val solve by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Solve carefully.",
            ) { instruction, input -> "$instruction\n$input" }
            nodeStart then solve then nodeFinish
        },
        toolRegistry = ToolRegistry { },
    )

    private fun session(agent: GraphAIAgent<String, String>, executor: PromptExecutor) = trainingSession(
        experimentName = ExperimentName(
            runId = "gepa-determinism-test",
            optimizerName = "GEPA",
            agentName = "test-agent",
        ),
        trackedAgent = agent,
        dataset = (1..6).map { TrainSetItem("item-$it", "gold-$it") },
        substepPromptExecutor = executor,
        metric = { _, _ -> 1.0 },
        serializers = DatasetExecutionSerializers(serializeItem = { it.userQuery }, serializeOutput = { it }),
    )

    private class TestPromptExecutor(
        private val executeBlock: suspend (Prompt, LLModel, List<ToolDescriptor>) -> Message.Assistant,
    ) : PromptExecutor() {
        override suspend fun execute(
            prompt: Prompt,
            model: LLModel,
            tools: List<ToolDescriptor>,
        ): Message.Assistant = executeBlock(prompt, model, tools)

        override fun executeStreaming(
            prompt: Prompt,
            model: LLModel,
            tools: List<ToolDescriptor>,
        ): Flow<StreamFrame> = error("TestPromptExecutor does not support streaming")

        override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
            error("TestPromptExecutor does not support moderation")

        override fun close() = Unit
    }

    private companion object {
        const val SEED_STRATEGY_INSTRUCTION = "Solve the task."
    }
}

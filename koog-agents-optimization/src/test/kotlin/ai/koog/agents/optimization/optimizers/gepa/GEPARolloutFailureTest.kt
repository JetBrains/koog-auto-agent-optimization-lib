package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.common.abort.DatasetFailureRateExceededAbortException
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.metrics.impl.AgentRunStatsMetric
import ai.koog.agents.optimization.training.metrics.impl.DatasetSolvedRateMetric
import ai.koog.agents.optimization.training.metrics.impl.SubstageCountMetric
import ai.koog.agents.optimization.training.records.StageRecord
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
import kotlin.test.*

internal class GEPARolloutFailureTest {
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()
    // No mock answers configured, so every call returns an empty assistant message: the subgraph never
    // gets its finish tool call and every rollout fails on `AIAgentMaxNumberOfIterationsReachedException`.
    private val executor = getMockExecutor { }

    private val agent = createAgent(executor)

    private fun createAgent(
        promptExecutor: PromptExecutor,
        maxAgentIterations: Int = 2,
    ): GraphAIAgent<String, String> = GraphAIAgent(
        promptExecutor = promptExecutor,
        agentConfig = AIAgentConfig(
            prompt = prompt("test") { system("Solve the task.") },
            model = OpenAIModels.Chat.GPT4o,
            maxAgentIterations = maxAgentIterations,
        ),
        strategy = strategy("test-strategy") {
            val solve by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Solve carefully.",
            ) { instruction, input -> "$instruction\n$input" }
            nodeStart then solve then nodeFinish
        },
        toolRegistry = ToolRegistry { },
    )

    private val dataset: TrainSet<String, String> = listOf(
        TrainSetItem("first", "gold-first"),
        TrainSetItem("second", "gold-second"),
    )

    @Test
    fun `disabled cap fills failure score and records failed dataset items`() = runBlocking {
        var captured: List<GEPARollout>? = null

        val trainingResult = session().use {
            captured = runGepaRollouts(
                candidate = OptimizationArtifact(),
                dataset = dataset,
                stageName = "failing rollouts",
                gepaFeedback = unreachableFeedback(),
                failureScore = -0.25,
                failureRateThreshold = 1.0,
            )
        }

        val rollouts = requireNotNull(captured)
        assertEquals(listOf(GEPARollout.Failed(-0.25), GEPARollout.Failed(-0.25)), rollouts)
        assertEquals(listOf(-0.25, -0.25), rollouts.scores)
        assertEquals(listOf(0, 1), rollouts.failedRolloutIndices)
        assertTrue(rollouts.hasFailures)

        val rolloutStage = trainingResult.rootStage.substages.filterIsInstance<StageRecord>().single()
        val datasetStage = rolloutStage.substages.filterIsInstance<StageRecord>().single()
        assertNull(datasetStage.failure, "A threshold of 1.0 must not fail the rollout set")
        assertEquals(2, datasetStage.metrics[SubstageCountMetric.KEY]?.failed)
        assertEquals(2, datasetStage.metrics[DatasetSolvedRateMetric.KEY]?.itemsFailed)
        assertEquals(2, datasetStage.metrics[AgentRunStatsMetric.KEY]?.failedRuns)
    }

    @Test
    fun `zero tolerance stops after first failure and surfaces threshold cause`(): Unit = runBlocking {
        val failure = assertFailsWith<DatasetFailureRateExceededAbortException> {
            session().use {
                runGepaRollouts(
                    candidate = OptimizationArtifact(),
                    dataset = dataset,
                    stageName = "failing rollouts",
                    gepaFeedback = unreachableFeedback(),
                    failureScore = -0.25,
                    failureRateThreshold = 0.0,
                )
            }
        }

        assertEquals(1.0, failure.failureRate)
        assertEquals(1, failure.failedItems)
        assertEquals(1, failure.finishedItems, "The second rollout must not run")
        assertEquals(0.0, failure.failureRateThreshold)
    }

    @Test
    fun `disabled failure-rate abort evaluates full dataset despite zero threshold`() = runBlocking {
        var captured: List<GEPARollout>? = null

        val trainingResult = session().use {
            captured = runGepaRollouts(
                candidate = OptimizationArtifact(),
                dataset = dataset,
                stageName = "non-aborting rollouts",
                gepaFeedback = unreachableFeedback(),
                failureScore = -0.25,
                failureRateThreshold = 0.0,
                abortOnFailureRateExceeded = false,
            )
        }

        assertEquals(listOf(GEPARollout.Failed(-0.25), GEPARollout.Failed(-0.25)), captured)
        val rolloutStage = trainingResult.rootStage.substages.filterIsInstance<StageRecord>().single()
        val datasetStage = rolloutStage.substages.filterIsInstance<StageRecord>().single()
        assertEquals(2, datasetStage.metrics[SubstageCountMetric.KEY]?.failed)
        assertTrue(datasetStage.failure != null, "The threshold breach must remain visible on the dataset stage")
    }

    @Test
    fun `nonzero cap stops once full dataset ratio cannot recover`(): Unit = runBlocking {
        val largerDataset = dataset + listOf(
            TrainSetItem("third", "gold-third"),
            TrainSetItem("fourth", "gold-fourth"),
        )
        val failure = assertFailsWith<DatasetFailureRateExceededAbortException> {
            session().use {
                runGepaRollouts(
                    candidate = OptimizationArtifact(),
                    dataset = largerDataset,
                    stageName = "failing validation rollouts",
                    gepaFeedback = unreachableFeedback(),
                    failureScore = -0.25,
                    failureRateThreshold = 0.5,
                )
            }
        }

        assertEquals(1.0, failure.failureRate)
        assertEquals(3, failure.failedItems)
        assertEquals(
            3,
            failure.finishedItems,
            "The fourth rollout must not run once more than half of the four-item dataset has failed",
        )
        assertEquals(0.5, failure.failureRateThreshold)
    }

    @Test
    fun `failure ratio equal to cap preserves mixed positional results`() = runBlocking {
        val successfulExecutor = getMockExecutor {
            mockLLMToolCall(finishTool, "done") onCondition { true }
        }
        var executorCalls = 0
        val mixedExecutor = TestPromptExecutor { prompt, model, tools ->
            executorCalls++
            if (executorCalls == 1) {
                successfulExecutor.execute(prompt, model, tools)
            } else {
                executor.execute(prompt, model, tools)
            }
        }
        var captured: List<GEPARollout>? = null

        val trainingResult = session(
            agent = createAgent(mixedExecutor, maxAgentIterations = 20),
            executor = mixedExecutor,
        ).use {
            captured = runGepaRollouts(
                candidate = OptimizationArtifact(),
                dataset = dataset,
                stageName = "mixed rollouts at threshold",
                gepaFeedback = singleModuleGepaFeedback(
                    moduleName = "solve",
                ) { _, _, _, _, _ -> 0.75 to "Keep this instruction." },
                failureScore = -0.25,
                failureRateThreshold = 0.5,
            )
        }

        val rollouts = requireNotNull(captured)
        assertTrue(rollouts[0] is GEPARollout.Completed)
        assertEquals(GEPARollout.Failed(-0.25), rollouts[1])
        assertEquals(listOf(0.75, -0.25), rollouts.scores)
        assertEquals(listOf(1), rollouts.failedRolloutIndices)

        val rolloutStage = trainingResult.rootStage.substages.filterIsInstance<StageRecord>().single()
        val datasetStage = rolloutStage.substages.filterIsInstance<StageRecord>().single()
        assertNull(datasetStage.failure, "A ratio equal to the cap must be tolerated")
        assertEquals(1, datasetStage.metrics[SubstageCountMetric.KEY]?.failed)
    }

    @Test
    fun `arbitrary exception thrown during agent execution becomes failed rollout`() = runBlocking {
        val throwingExecutor = TestPromptExecutor { _, _, _ ->
            throw IllegalStateException("intentional agent execution failure")
        }
        var captured: List<GEPARollout>? = null

        val trainingResult = session(
            agent = createAgent(throwingExecutor),
            executor = throwingExecutor,
        ).use {
            captured = runGepaRollouts(
                candidate = OptimizationArtifact(),
                dataset = listOf(dataset.first()),
                stageName = "throwing agent rollout",
                gepaFeedback = unreachableFeedback(),
                failureScore = -0.25,
                failureRateThreshold = 1.0,
            )
        }

        assertEquals(listOf(GEPARollout.Failed(-0.25)), captured)
        val rolloutStage = trainingResult.rootStage.substages.filterIsInstance<StageRecord>().single()
        val datasetStage = rolloutStage.substages.filterIsInstance<StageRecord>().single()
        assertEquals(1, datasetStage.metrics[SubstageCountMetric.KEY]?.failed)
        assertEquals(1, datasetStage.metrics[AgentRunStatsMetric.KEY]?.failedRuns)
    }

    @Test
    fun `lenient optimizer skips feedback batch when parent rollout fails`() = runBlocking {
        val tempDirectory = Files.createTempDirectory("gepa-rollout-failure-test").toFile()
        try {
            val storagePath = ResilientPath(tempDirectory.resolve("artifact.json").toPath())
            val optimizer = GEPAOptimizer(
                gepaFeedback = unreachableFeedback(),
                feedbackValSplitFn = { items ->
                    GEPATrainSetSplit(feedbackSet = listOf(items[0]), validationSet = listOf(items[1]))
                },
                reflectionModel = OpenAIModels.Chat.GPT4o,
                storagePath = storagePath,
                numRollouts = 2,
                feedbackBatchSize = 1,
                failureScore = -0.25,
                feedbackFailureRateThreshold = 1.0,
                seedValidationFailureRateThreshold = 1.0,
            )

            optimizer.train(session())

            assertTrue(storagePath.exists(), "The seed artifact should be saved after skipping the failed batch")
        } finally {
            tempDirectory.deleteRecursively()
        }
    }

    @Test
    fun `seed evaluation fails closed independently of later rollout policy`(): Unit = runBlocking {
        val tempDirectory = Files.createTempDirectory("gepa-seed-failure-test").toFile()
        try {
            val optimizer = GEPAOptimizer(
                gepaFeedback = unreachableFeedback(),
                feedbackValSplitFn = { items ->
                    GEPATrainSetSplit(feedbackSet = listOf(items[0]), validationSet = listOf(items[1]))
                },
                reflectionModel = OpenAIModels.Chat.GPT4o,
                storagePath = ResilientPath(tempDirectory.resolve("artifact.json").toPath()),
                numRollouts = 2,
                failureScore = -0.25,
            )

            val failure = assertFailsWith<DatasetFailureRateExceededAbortException> {
                optimizer.train(session())
            }
            assertEquals(1.0, failure.failureRate)
            assertEquals(0.0, failure.failureRateThreshold)
            assertFalse(optimizer.storagePath.exists(), "A seed-validation breach must not save an artifact")
        } finally {
            tempDirectory.deleteRecursively()
        }
    }

    @Test
    fun `disabled failure-rate abort allows failed seed evaluation to save artifact`() = runBlocking {
        val tempDirectory = Files.createTempDirectory("gepa-non-aborting-seed-failure-test").toFile()
        try {
            val optimizer = GEPAOptimizer(
                gepaFeedback = unreachableFeedback(),
                feedbackValSplitFn = { items ->
                    GEPATrainSetSplit(feedbackSet = listOf(items[0]), validationSet = listOf(items[1]))
                },
                reflectionModel = OpenAIModels.Chat.GPT4o,
                storagePath = ResilientPath(tempDirectory.resolve("artifact.json").toPath()),
                numRollouts = 2,
                failureScore = -0.25,
                seedValidationFailureRateThreshold = 0.0,
                abortOnFailureRateExceeded = false,
            )

            optimizer.train(session())

            assertTrue(optimizer.storagePath.exists(), "A record-only threshold breach must not terminate GEPA")
        } finally {
            tempDirectory.deleteRecursively()
        }
    }

    @Test
    fun `candidate validation cap breach terminates optimization without saving artifact`(): Unit = runBlocking {
        val successfulRolloutExecutor = getMockExecutor {
            mockLLMToolCall(finishTool, "done") onCondition { true }
        }
        val reflectionExecutor = getMockExecutor {
            mockLLMAnswer("```Improved instruction.```").asDefaultResponse
        }
        var rolloutCalls = 0
        val conditionalExecutor = TestPromptExecutor { prompt, model, tools ->
            if (tools.isEmpty()) {
                reflectionExecutor.execute(prompt, model, tools)
            } else {
                rolloutCalls++
                if (rolloutCalls <= 3) {
                    successfulRolloutExecutor.execute(prompt, model, tools)
                } else {
                    executor.execute(prompt, model, tools)
                }
            }
        }
        val tempDirectory = Files.createTempDirectory("gepa-validation-failure-test").toFile()
        try {
            var feedbackInvocation = 0
            val optimizer = GEPAOptimizer<String, String, String>(
                gepaFeedback = strategyOnlyGepaFeedback { _, _, _, _, _ ->
                    feedbackInvocation++
                    val score = if (feedbackInvocation == 3) 1.0 else 0.0
                    score to "Use the improved instruction."
                },
                feedbackValSplitFn = { items ->
                    GEPATrainSetSplit(feedbackSet = listOf(items[0]), validationSet = listOf(items[1]))
                },
                reflectionModel = OpenAIModels.Chat.GPT4o,
                storagePath = ResilientPath(tempDirectory.resolve("artifact.json").toPath()),
                numRollouts = 4,
                feedbackBatchSize = 1,
                validationFailureRateThreshold = 0.0,
            )

            val failure = assertFailsWith<DatasetFailureRateExceededAbortException> {
                optimizer.train(
                    session(
                        agent = createAgent(conditionalExecutor, maxAgentIterations = 20),
                        executor = conditionalExecutor,
                    )
                )
            }
            assertEquals(1.0, failure.failureRate)
            assertEquals(1, failure.failedItems)
            assertEquals(1, failure.finishedItems)
            assertEquals(0.0, failure.failureRateThreshold)
            assertEquals(3, feedbackInvocation, "Seed, parent, and candidate-feedback rollouts must complete")
            assertFalse(optimizer.storagePath.exists(), "A validation breach must prevent artifact persistence")
        } finally {
            tempDirectory.deleteRecursively()
        }
    }

    @Test
    fun `candidate feedback failure score can reject candidate without validation`() = runBlocking {
        val successfulRolloutExecutor = getMockExecutor {
            mockLLMToolCall(finishTool, "done") onCondition { true }
        }
        val reflectionExecutor = getMockExecutor {
            mockLLMAnswer("```Improved instruction.```").asDefaultResponse
        }
        var rolloutCalls = 0
        val conditionalExecutor = TestPromptExecutor { prompt, model, tools ->
            if (tools.isEmpty()) {
                reflectionExecutor.execute(prompt, model, tools)
            } else {
                rolloutCalls++
                if (rolloutCalls <= 2) {
                    successfulRolloutExecutor.execute(prompt, model, tools)
                } else {
                    executor.execute(prompt, model, tools)
                }
            }
        }
        val tempDirectory = Files.createTempDirectory("gepa-candidate-feedback-failure-test").toFile()
        try {
            var feedbackInvocations = 0
            val optimizer = GEPAOptimizer<String, String, String>(
                gepaFeedback = strategyOnlyGepaFeedback { _, _, _, _, _ ->
                    feedbackInvocations++
                    0.0 to "Use the improved instruction."
                },
                feedbackValSplitFn = { items ->
                    GEPATrainSetSplit(feedbackSet = listOf(items[0]), validationSet = listOf(items[1]))
                },
                reflectionModel = OpenAIModels.Chat.GPT4o,
                storagePath = ResilientPath(tempDirectory.resolve("artifact.json").toPath()),
                numRollouts = 3,
                feedbackBatchSize = 1,
                failureScore = -0.25,
                feedbackFailureRateThreshold = 1.0,
            )

            optimizer.train(
                session(
                    agent = createAgent(conditionalExecutor, maxAgentIterations = 20),
                    executor = conditionalExecutor,
                )
            )

            assertEquals(
                2,
                feedbackInvocations,
                "Only seed validation and parent feedback should complete; the rejected candidate must not be validated",
            )
            assertTrue(rolloutCalls > 2, "The proposed candidate feedback rollout must be attempted")
            assertEquals(
                "Solve the task.",
                GEPAOptimizer.loadArtifact(optimizer.storagePath).strategyInstruction,
                "A candidate whose feedback rollout receives failureScore must not replace its parent",
            )
        } finally {
            tempDirectory.deleteRecursively()
        }
    }

    private fun session(
        agent: GraphAIAgent<String, String> = this.agent,
        executor: PromptExecutor = this.executor,
    ) = trainingSession(
        experimentName = ExperimentName(
            runId = "gepa-rollout-failure-test",
            optimizerName = "GEPA",
            agentName = "test-agent",
        ),
        trackedAgent = agent,
        dataset = dataset,
        substepPromptExecutor = executor,
        metric = { _, _ -> 1.0 },
        serializers = DatasetExecutionSerializers(
            serializeItem = { it.userQuery },
            serializeOutput = { it },
        ),
    )

    private fun unreachableFeedback(): GEPAFeedback<String, String, String> =
        GEPAFeedback { _, _, _, _, _ -> error("Feedback must not run for a failed rollout") }

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

}

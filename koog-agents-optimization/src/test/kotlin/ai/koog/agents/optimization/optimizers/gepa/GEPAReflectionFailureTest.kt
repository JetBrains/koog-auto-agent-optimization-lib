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
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import ai.koog.agents.optimization.training.records.TrainingResult
import ai.koog.agents.optimization.training.structures.StageFailedException
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
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * How GEPA handles a reflection call that fails under each [GEPAReflectionFailurePolicy]. Runs offline: rollouts
 * finish through a mock executor, every rollout scores 0.5 (so no parent is perfect and every batch reflects), and
 * the reflection calls succeed or throw following a script.
 */
internal class GEPAReflectionFailureTest {
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()

    @Test
    fun `skipping a failed reflection skips its batch and the run completes`() {
        val run = runScenario(reflectionScript = listOf(FAIL), policy = GEPAReflectionFailurePolicy.SKIP_BATCH)

        assertTrue(run.artifactSaved, "The run should save its best artifact")
        assertEquals(1, run.outcomes.count { it == "reflection failed" })
        assertTrue(run.outcomes.count { it == "candidate rejected" } >= 1, "Later batches should reflect again")
    }

    @Test
    fun `skipping completes the run even when every reflection fails`() {
        val run = runScenario(reflectionScript = List(20) { FAIL }, policy = GEPAReflectionFailurePolicy.SKIP_BATCH)

        assertTrue(run.artifactSaved, "The run should save the seed as its best artifact")
        assertTrue(run.outcomes.count { it == "reflection failed" } >= 5, "Outcomes: ${run.outcomes}")
        assertTrue(run.outcomes.none { it.startsWith("candidate") }, "Outcomes: ${run.outcomes}")
    }

    @Test
    fun `aborting stops the run on the first failed reflection`() {
        val failure = assertFailsWith<StageFailedException> {
            runScenario(reflectionScript = listOf(FAIL), policy = GEPAReflectionFailurePolicy.ABORT)
        }

        assertTrue(failure.rootCauses().any { "reflection outage" in it.message.orEmpty() }, "Cause: $failure")
    }

    // ===================================================================
    // Harness
    // ===================================================================

    private class ScenarioRun(val artifactSaved: Boolean, val outcomes: List<String>)

    private fun newOptimizer(
        policy: GEPAReflectionFailurePolicy,
        storagePath: ResilientPath,
    ) = GEPAOptimizer<String, String, String>(
        gepaFeedback = strategyOnlyGepaFeedback { _, _, _, _, _ -> 0.5 to "Half right." },
        feedbackValSplitFn = { items -> GEPATrainSetSplit(feedbackSet = items.take(3), validationSet = items.drop(3)) },
        reflectionModel = OpenAIModels.Chat.GPT4o,
        storagePath = storagePath,
        numRollouts = 31,
        feedbackBatchSize = 2,
        randomSeed = 42,
        reflectionFailurePolicy = policy,
    )

    /**
     * Runs GEPA with reflection calls that follow [reflectionScript] in order (calls past its end succeed).
     * The feedback names only the strategy module, so round robin reflects on every other batch; the budget of
     * 31 rollouts covers the 3 seed validation runs and at least 5 reflecting batches.
     */
    private fun runScenario(reflectionScript: List<Boolean>, policy: GEPAReflectionFailurePolicy): ScenarioRun {
        val rolloutExecutor = getMockExecutor { mockLLMToolCall(finishTool, "done") onCondition { true } }
        var reflections = 0
        val executor = TestPromptExecutor { prompt, model, tools ->
            if (tools.isEmpty()) {
                val succeeds = reflectionScript.getOrElse(reflections) { SUCCEED }
                reflections++
                check(succeeds) { "reflection outage" }
                getMockExecutor { mockLLMAnswer("```Instruction v$reflections.```").asDefaultResponse }
                    .execute(prompt, model, tools)
            } else {
                rolloutExecutor.execute(prompt, model, tools)
            }
        }
        val temporaryDirectory = Files.createTempDirectory("gepa-reflection-failure-test").toFile()
        try {
            val storagePath = ResilientPath(temporaryDirectory.resolve("artifact.json").toPath())
            val optimizer = newOptimizer(policy, storagePath)
            val result = runBlocking { optimizer.train(session(createAgent(executor), executor)) }
            return ScenarioRun(artifactSaved = storagePath.exists(), outcomes = result.stepOutcomes())
        } finally {
            temporaryDirectory.deleteRecursively()
        }
    }

    /** The `outcome` every feedback batch recorded, in order. */
    private fun TrainingResult.stepOutcomes(): List<String> {
        val outcomes = mutableListOf<String>()
        fun walk(record: TrainingRecord) {
            if (record !is StageRecord) return
            if (record.name.startsWith("Feedback batch ")) {
                record.actionLog?.jsonObject?.get("outcome")?.jsonPrimitive?.content?.let(outcomes::add)
            }
            record.substages.forEach(::walk)
        }
        walk(rootStage)
        return outcomes
    }

    private fun Throwable.rootCauses(): Sequence<Throwable> = generateSequence(this) { it.cause }

    private fun createAgent(promptExecutor: PromptExecutor): GraphAIAgent<String, String> = GraphAIAgent(
        promptExecutor = promptExecutor,
        agentConfig = AIAgentConfig(
            prompt = prompt("test") { system("Solve the task.") },
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
            runId = "gepa-reflection-failure-test",
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
        const val FAIL = false
        const val SUCCEED = true
    }
}

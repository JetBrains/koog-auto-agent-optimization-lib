package ai.koog.agents.optimization.optimizers.mipro

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.optimization.optimizers.TrainSetItem
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
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The budget a run resolves: [AutoRunMode] supplies all three numbers, and each override replaces
 * its own. Every case runs the optimizer end to end against mock executors, offline, and reads the
 * numbers back from the run-setup record.
 */
internal class MIPROv2ParameterResolutionTest {

    @Test
    fun `the auto mode supplies every number when nothing is overridden`() {
        val setup = runOptimization(autoMode = AutoRunMode.LIGHT)

        assertEquals("light", setup.mode)
        assertEquals(AutoRunMode.LIGHT.numCandidates, setup.numCandidates)
        // Two modules and six candidates put the formula (21) above LIGHT's cap, so the cap applies.
        assertEquals(AutoRunMode.LIGHT.maxTrials, setup.numTrials)
        // The dataset is smaller than the mode's validation cap, so the whole of it is scored.
        assertEquals(DATASET_SIZE, setup.validationSetSize)
        assertEquals(emptyList(), setup.budgetOverrides)
    }

    @Test
    fun `a heavier mode raises every number it supplies`() {
        val setup = runOptimization(autoMode = AutoRunMode.MEDIUM, numTrials = 1)

        assertEquals("medium", setup.mode)
        assertEquals(AutoRunMode.MEDIUM.numCandidates, setup.numCandidates)
        assertTrue(
            AutoRunMode.MEDIUM.valExamples > AutoRunMode.LIGHT.valExamples,
            "the medium mode is expected to score on more validation items",
        )
    }

    @Test
    fun `overriding the candidate count recomputes the trials from it`() {
        val setup = runOptimization(autoMode = AutoRunMode.LIGHT, numCandidates = 2)

        assertEquals(2, setup.numCandidates)
        // Two modules, two candidates: max(ceil(2 * 4 * log2(2)), ceil(1.5 * 2)) = 8.
        assertEquals(8, setup.numTrials)
        assertEquals(listOf("numCandidates"), setup.budgetOverrides)
    }

    @Test
    fun `each override replaces its own number and leaves the others to the mode`() {
        val setup = runOptimization(autoMode = AutoRunMode.LIGHT, numTrials = 3)

        assertEquals(AutoRunMode.LIGHT.numCandidates, setup.numCandidates)
        assertEquals(3, setup.numTrials)
        assertEquals(listOf("numTrials"), setup.budgetOverrides)
    }

    @Test
    fun `all three overrides together decide the whole budget`() {
        val setup = runOptimization(
            autoMode = AutoRunMode.HEAVY,
            numCandidates = 2,
            valExamples = 2,
            numTrials = 2,
        )

        assertEquals("heavy", setup.mode)
        assertEquals(2, setup.numCandidates)
        assertEquals(2, setup.numTrials)
        assertEquals(2, setup.validationSetSize)
        // In the order the mode declares them, so a reader can line the names up with the numbers.
        assertEquals(listOf("numCandidates", "valExamples", "numTrials"), setup.budgetOverrides)
    }

    @Test
    fun `the validation override caps the scored set below the mode's number`() {
        val setup = runOptimization(autoMode = AutoRunMode.LIGHT, numTrials = 1, valExamples = 1)

        assertEquals(1, setup.validationSetSize)
    }

    @Test
    fun `a non-positive budget is rejected at construction`() {
        assertTrue("numCandidatesOverride" in messageOf { newOptimizer(numCandidates = 0) })
        assertTrue("valExamplesOverride" in messageOf { newOptimizer(valExamples = -1) })
        assertTrue("numTrialsOverride" in messageOf { newOptimizer(numTrials = 0) })
        assertTrue("parallelism" in messageOf { newOptimizer(parallelism = 0) })
        assertTrue("maxTotalDemos" in messageOf { newOptimizer(maxTotalDemos = -1) })
    }

    @Test
    fun `zero on both demo counts is accepted as zero-shot mode`() {
        newOptimizer(maxBootstrappedDemos = 0, maxTotalDemos = 0)
    }

    // ===================================================================
    // Harness
    // ===================================================================

    private class ResolvedSetup(
        val mode: String,
        val numCandidates: Int,
        val numTrials: Int,
        val validationSetSize: Int,
        val budgetOverrides: List<String>,
    )

    private fun messageOf(construct: () -> Unit): String =
        assertFailsWith<IllegalArgumentException> { construct() }.message.orEmpty()

    private fun newOptimizer(
        numCandidates: Int? = null,
        valExamples: Int? = null,
        numTrials: Int? = null,
        parallelism: Int = 1,
        maxBootstrappedDemos: Int = 4,
        maxTotalDemos: Int = 8,
    ): MIPROv2Optimizer<String, String, String> = MIPROv2Optimizer(
        metaModel = OpenAIModels.Chat.GPT4o,
        numCandidatesOverride = numCandidates,
        valExamplesOverride = valExamples,
        numTrialsOverride = numTrials,
        maxBootstrappedDemos = maxBootstrappedDemos,
        maxTotalDemos = maxTotalDemos,
        parallelism = parallelism,
        storagePath = ResilientPath(Files.createTempDirectory("mipro-params").resolve("artifact.json")),
    )

    private fun runOptimization(
        autoMode: AutoRunMode,
        numCandidates: Int? = null,
        valExamples: Int? = null,
        numTrials: Int? = null,
    ): ResolvedSetup {
        var proposals = 0
        val executor = TestPromptExecutor { prompt, model, tools ->
            val answer = when (prompt.id) {
                "generate-instruction" -> "Instruction ${++proposals}."
                else -> "A test answer."
            }
            getMockExecutor { mockLLMAnswer(answer).asDefaultResponse }.execute(prompt, model, tools)
        }
        val temporaryDirectory = Files.createTempDirectory("mipro-params-run").toFile()
        try {
            val optimizer = MIPROv2Optimizer<String, String, String>(
                metaModel = OpenAIModels.Chat.GPT4o,
                autoMode = autoMode,
                numCandidatesOverride = numCandidates,
                valExamplesOverride = valExamples,
                numTrialsOverride = numTrials,
                storagePath = ResilientPath(temporaryDirectory.resolve("artifact.json").toPath()),
                randomSeed = 42,
            )
            var scored = 0
            val session = trainingSession(
                experimentName = ExperimentName(
                    runId = "mipro-parameter-resolution-test",
                    optimizerName = "MIPROv2",
                    agentName = "test-agent",
                ),
                trackedAgent = createAgent(executor),
                dataset = (1..DATASET_SIZE).map { TrainSetItem("item-$it", "gold-$it") },
                substepPromptExecutor = executor,
                metric = { _, _ -> SCORES[scored++ % SCORES.size] },
                serializers = DatasetExecutionSerializers(serializeItem = { it.userQuery }, serializeOutput = { it }),
            )
            val result = runBlocking { optimizer.train(session) }
            val log = result.rootStage.actionLog!!.jsonObject
            return ResolvedSetup(
                mode = log.getValue("autoRunMode").jsonPrimitive.content,
                numCandidates = log.getValue("numCandidates").jsonPrimitive.int,
                numTrials = log.getValue("numTrials").jsonPrimitive.int,
                validationSetSize = log.getValue("validationSetSize").jsonPrimitive.int,
                budgetOverrides = log["budgetOverrides"]?.jsonArray?.map { it.jsonPrimitive.content }.orEmpty(),
            )
        } finally {
            temporaryDirectory.deleteRecursively()
        }
    }

    private fun createAgent(promptExecutor: PromptExecutor): GraphAIAgent<String, String> = GraphAIAgent(
        promptExecutor = promptExecutor,
        agentConfig = AIAgentConfig(
            prompt = prompt("mipro-parameter-resolution") { system("Solve the task.") },
            model = OpenAIModels.Chat.GPT4o,
            maxAgentIterations = 20,
        ),
        strategy = strategy("mipro-parameter-resolution") {
            val solve by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Solve carefully.",
            ) { instruction, input -> "$instruction\n$input" }
            nodeStart then solve then nodeFinish
        },
        toolRegistry = ToolRegistry { },
    )

    private companion object {
        const val DATASET_SIZE = 4
        val SCORES = listOf(0.2, 0.5, 0.9, 0.4)
    }
}

/** A [PromptExecutor] that routes every call to [execute]. */
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

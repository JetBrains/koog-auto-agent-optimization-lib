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
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.trainingSession
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.messages.renderParts
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
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * How MIPRO splits its training dataset: shuffled with the split seed, 20 % bootstraps demos and grounds the instruction
 * proposals, and validation items come only from the other 80 %, in DSPy's MIPROv2 proportions. Runs offline against
 * mock executors.
 */
internal class MIPROv2DatasetSplitTest {

    @Test
    fun `split sizes follow DSPy's proportions`() {
        // The bootstrap set takes n - floor(0.8 * n) items; the validation set takes the rest.
        val expectedByDatasetSize = mapOf(2 to (1 to 1), 5 to (1 to 4), 7 to (2 to 5), 60 to (12 to 48))
        for ((datasetSize, expected) in expectedByDatasetSize) {
            val run = runOptimization(datasetSize, valExamples = 1000)
            assertEquals(expected, run.bootstrapSetSize to run.validationSetSize, "dataset of $datasetSize items")
        }
    }

    @Test
    fun `validation items never include an item that bootstraps demos`() {
        val run = runOptimization(datasetSize = 10, valExamples = 1000)

        assertEquals(2, run.demoGenerationItems.size)
        assertEquals(emptySet(), run.demoGenerationItems intersect run.gridSearchItems)
        assertEquals(itemName(1..10), run.demoGenerationItems + run.gridSearchItems)
    }

    @Test
    fun `validation override takes items outside the bootstrap set only`() {
        val run = runOptimization(datasetSize = 10, valExamples = 3)

        assertEquals(3, run.gridSearchItems.size)
        assertEquals(emptySet(), run.demoGenerationItems intersect run.gridSearchItems)
    }

    @Test
    fun `instruction proposals see only the bootstrap items`() {
        val run = runOptimization(datasetSize = 10, valExamples = 1000)

        val leakedValidationItems = run.gridSearchItems.filter { item -> run.metaPrompts.any { item in it } }
        assertEquals(emptyList(), leakedValidationItems)
        val bootstrapItem = run.demoGenerationItems.first()
        assertTrue(run.metaPrompts.any { bootstrapItem in it }, "The proposer should see the bootstrap items")
    }

    @Test
    fun `the split depends on the split seed alone`() {
        val reference = runOptimization(datasetSize = 10, valExamples = 1000)
        val otherSearchSeed = runOptimization(datasetSize = 10, valExamples = 1000, randomSeed = 7)
        val otherSplitSeed = runOptimization(datasetSize = 10, valExamples = 1000, validationSplitSeed = 7)

        assertEquals(reference.demoGenerationItems, otherSearchSeed.demoGenerationItems)
        assertNotEquals(reference.demoGenerationItems, otherSplitSeed.demoGenerationItems)
    }

    // ===================================================================
    // Harness
    // ===================================================================

    private class SplitRun(
        val bootstrapSetSize: Int,
        val validationSetSize: Int,
        val demoGenerationItems: Set<String>,
        val gridSearchItems: Set<String>,
        val metaPrompts: List<String>,
    )

    private fun itemName(range: IntRange): Set<String> = range.map { "item-%02d".format(it) }.toSet()

    private fun runOptimization(
        datasetSize: Int,
        valExamples: Int,
        randomSeed: Long = 42,
        validationSplitSeed: Long = 42,
    ): SplitRun {
        val metaPrompts = mutableListOf<String>()
        val executor = TestPromptExecutor { prompt, model, tools ->
            val answer = if (prompt.id in META_PROMPT_IDS) {
                synchronized(metaPrompts) { metaPrompts += prompt.messages.joinToString("\n") { it.renderParts() } }
                "A meta answer."
            } else {
                "A test answer."
            }
            getMockExecutor { mockLLMAnswer(answer).asDefaultResponse }.execute(prompt, model, tools)
        }
        val temporaryDirectory = Files.createTempDirectory("mipro-split-run").toFile()
        try {
            val optimizer = MIPROv2Optimizer<String, String, String>(
                metaModel = OpenAIModels.Chat.GPT4o,
                autoMode = AutoRunMode.LIGHT,
                numCandidatesOverride = 2,
                valExamplesOverride = valExamples,
                numTrialsOverride = 1,
                storagePath = ResilientPath(temporaryDirectory.resolve("artifact.json").toPath()),
                randomSeed = randomSeed,
                validationSplitSeed = validationSplitSeed,
            )
            val session = trainingSession(
                experimentName = ExperimentName(
                    runId = "mipro-dataset-split-test",
                    optimizerName = "MIPROv2",
                    agentName = "test-agent",
                ),
                trackedAgent = createAgent(executor),
                dataset = (1..datasetSize).map { TrainSetItem("item-%02d".format(it), "gold-$it") },
                substepPromptExecutor = executor,
                metric = { _, _ -> 1.0 },
                serializers = DatasetExecutionSerializers(serializeItem = { it.userQuery }, serializeOutput = { it }),
            )
            val result = runBlocking { optimizer.train(session) }
            val setup = result.rootStage.actionLog!!.jsonObject
            return SplitRun(
                bootstrapSetSize = setup.getValue("bootstrapSetSize").jsonPrimitive.int,
                validationSetSize = setup.getValue("validationSetSize").jsonPrimitive.int,
                demoGenerationItems = result.rootStage.substage("Step 1: Demo Generation").itemsRun(),
                gridSearchItems = result.rootStage.substage("Step 3: Grid Search").itemsRun(),
                metaPrompts = metaPrompts.toList(),
            )
        } finally {
            temporaryDirectory.deleteRecursively()
        }
    }

    private fun StageRecord.substage(name: String): StageRecord =
        substages.filterIsInstance<StageRecord>().single { it.name == name }

    /** Every dataset item an item stage under this stage ran, read from the item stages' payloads. */
    private fun StageRecord.itemsRun(): Set<String> {
        val own = additionalData?.jsonObject?.get("item")?.jsonPrimitive?.content
        return setOfNotNull(own) + substages.filterIsInstance<StageRecord>().flatMap { it.itemsRun() }
    }

    private fun createAgent(promptExecutor: PromptExecutor): GraphAIAgent<String, String> = GraphAIAgent(
        promptExecutor = promptExecutor,
        agentConfig = AIAgentConfig(
            prompt = prompt("test-agent-prompt") { system("Solve the task.") },
            model = OpenAIModels.Chat.GPT4o,
            maxAgentIterations = 20,
        ),
        strategy = strategy("mipro-dataset-split") {
            val solve by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Solve carefully.",
            ) { instruction, input -> "$instruction\n$input" }
            nodeStart then solve then nodeFinish
        },
        toolRegistry = ToolRegistry { },
    )

    private companion object {
        /** Ids of the prompts MIPRO sends to its meta model (dataset summary, program description, proposals). */
        val META_PROMPT_IDS = setOf(
            "dataset-descriptor", "dataset-descriptor-with-prior", "observation-summarizer",
            "describe-program", "describe-module", "generate-instruction",
        )
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
}

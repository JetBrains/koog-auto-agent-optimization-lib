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
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.ActionLogTruncation
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
import ch.qos.logback.classic.Level
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.read.ListAppender
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import org.slf4j.LoggerFactory
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import ch.qos.logback.classic.Logger as LogbackLogger

/**
 * Asserts that a finished GEPA run explains itself: the training records carry every decision with the
 * inputs that drove it, and the run log carries the instruction texts in full.
 *
 * Every case runs the optimizer end to end against mock executors, offline.
 */
internal class GEPALoggingTest {
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()

    private val dataset: TrainSet<String, String> = listOf(
        TrainSetItem("first", "gold-first"),
        TrainSetItem("second", "gold-second"),
    )

    @Test
    fun `accepted candidate leaves the whole batch decision in the records`() {
        val result = runOptimization(numRollouts = 4, scoresInOrder = listOf(0.0, 0.0, 1.0, 1.0)).trainingResult

        val run = result.rootStage.stage("GEPA optimization").log()
        assertEquals(4, run.int("rolloutsUsed"))
        assertEquals(1, run.int("batchesRun"))
        assertEquals(1, run.int("candidatesAdded"))
        assertEquals(1, run.int("feedbackSetSize"))
        assertEquals(1, run.int("validationSetSize"))
        assertEquals("candidate-1", run.string("bestCandidateId"))
        val modules = run.strings("modules")
        assertTrue(modules.isNotEmpty(), "The discovered modules must be recorded")

        val seed = result.rootStage.stage("Initialize candidate pool").log()
        assertEquals("candidate-0", seed.string("seedCandidateId"))
        assertEquals(0.0, seed.double("averageValidationScore"))
        assertEquals(modules.toSet(), seed.stringMap("instructions").keys)

        val batch = result.rootStage.stage("Feedback batch 1").log()
        assertEquals("candidate accepted", batch.string("outcome"))
        assertEquals("candidate-0", batch.string("parentId"))
        assertEquals(mapOf("candidate-0" to 1), batch.intMap("parentSelectionWeights"))
        assertEquals(0.0, batch.double("parentBatchAverageScore"))
        assertEquals(1.0, batch.double("candidateBatchAverageScore"))
        assertEquals("candidate-1", batch.string("newCandidateId"))
        assertEquals(1.0, batch.double("averageValidationScore"))
        assertTrue(batch.string("selectedModule") in modules, "The selected module must be one of the discovered ones")

        val reflection = result.rootStage.stage("GEPA reflection").log()
        val selectedModule = batch.string("selectedModule")
        assertEquals(selectedModule, reflection.string("module"))
        assertEquals(
            seed.stringMap("instructions").getValue(selectedModule),
            reflection.string("parentInstruction"),
            "Reflection reads the parent's instruction for the module it updates",
        )
        assertEquals(1, reflection.strings("feedbackTexts").size)
        assertEquals(IMPROVED_INSTRUCTION, reflection.string("proposedInstruction"))

        val parentRollouts = result.rootStage.stage("Evaluate parent on feedback batch 1").log()
        assertEquals(listOf(0.0), parentRollouts.doubles("rolloutScores"))

        val parentItem = result.rootStage
            .stage("Evaluate parent on feedback batch 1: run dataset")
            .substages.filterIsInstance<StageRecord>().single()
        assertEquals(0.0, parentItem.log().double("assignedScore"))

        val saved = result.rootStage.stage("Save best artifact").log()
        assertEquals("candidate-1", saved.string("bestCandidateId"))
        assertEquals(1.0, saved.double("bestAverageValidationScore"))
        val candidates = saved.getValue("candidates").jsonArray.map { it.jsonObject }
        assertEquals(listOf("candidate-0", "candidate-1"), candidates.map { it.string("id") })
        assertEquals(emptyList(), candidates[0].strings("parentIds"))
        assertEquals(listOf("candidate-0"), candidates[1].strings("parentIds"))
    }

    @Test
    fun `rejected candidate records both averages that decided it`() {
        val result = runOptimization(numRollouts = 4, scoresInOrder = listOf(0.0, 0.0, 0.0, 0.0)).trainingResult

        val batch = result.rootStage.stage("Feedback batch 1").log()
        assertEquals("candidate rejected", batch.string("outcome"))
        assertEquals(0.0, batch.double("parentBatchAverageScore"))
        assertEquals(0.0, batch.double("candidateBatchAverageScore"))
        assertTrue("newCandidateId" !in batch, "A rejected candidate never enters the pool")

        val run = result.rootStage.stage("GEPA optimization").log()
        assertEquals(0, run.int("candidatesAdded"))
        assertEquals("candidate-0", run.string("bestCandidateId"))
    }

    @Test
    fun `skipped batch records the reason and the parent it skipped on`() {
        val result = runOptimization(numRollouts = 3, scoresInOrder = listOf(1.0, 1.0, 1.0)).trainingResult

        val batch = result.rootStage.stage("Feedback batch 1").log()
        assertEquals("parent already perfect", batch.string("outcome"))
        assertEquals("candidate-0", batch.string("parentId"))
        assertEquals(1.0, batch.double("parentBatchAverageScore"))

        val run = result.rootStage.stage("GEPA optimization").log()
        assertEquals(2, run.int("batchesRun"))
        assertEquals(0, run.int("candidatesAdded"))
        assertEquals("candidate-0", run.string("bestCandidateId"))
    }

    @Test
    fun `the run log carries the instruction texts in full`() {
        val longInstruction = "Improved instruction. ".repeat(120).trim()
        val optimizationRun = runOptimization(
            numRollouts = 4,
            scoresInOrder = listOf(0.0, 0.0, 1.0, 1.0),
            proposedInstruction = longInstruction,
        )
        val logLines = optimizationRun.logLines

        val seedInstructions = optimizationRun.trainingResult.rootStage
            .stage("Initialize candidate pool").log().stringMap("instructions")
        val seedLine = logLines.single { "seed candidate" in it }
        seedInstructions.forEach { (module, instruction) ->
            assertTrue(instruction in seedLine, "The seed instruction of module '$module' must reach the run log")
        }

        val proposalLine = logLines.single { "reflection proposed" in it }
        assertTrue(longInstruction in proposalLine, "The full proposed instruction must reach the run log")

        assertTrue(logLines.any { "GEPA accepted candidate-1" in it }, logLines.toString())
        assertTrue(logLines.any { "GEPA chose candidate-1" in it }, logLines.toString())
    }

    @Test
    fun `long values are clipped while the record stays within the truncation`() {
        val longInstruction = "Improved instruction. ".repeat(120).trim()
        val result = runOptimization(
            numRollouts = 4,
            scoresInOrder = listOf(0.0, 0.0, 1.0, 1.0),
            proposedInstruction = longInstruction,
        ).trainingResult

        val stored = result.rootStage.stage("GEPA reflection").log().string("proposedInstruction")
        assertEquals(maxChars + ELLIPSIS.length, stored.length)
        assertTrue(stored.endsWith(ELLIPSIS), "A clipped value keeps the truncation marker")

        val overlong = result.rootStage.allActionLogStrings().filter { it.length > maxChars + ELLIPSIS.length }
        assertEquals(emptyList(), overlong, "Every action-log string must stay within the configured truncation")
    }

    // ===================================================================
    // Harness
    // ===================================================================

    private class OptimizationRun(val trainingResult: TrainingResult, val logLines: List<String>)

    private fun runOptimization(
        numRollouts: Int,
        scoresInOrder: List<Double>,
        proposedInstruction: String = IMPROVED_INSTRUCTION,
    ): OptimizationRun {
        val rolloutExecutor = getMockExecutor {
            mockLLMToolCall(finishTool, "done") onCondition { true }
        }
        val reflectionExecutor = getMockExecutor {
            mockLLMAnswer("```$proposedInstruction```").asDefaultResponse
        }
        val executor = TestPromptExecutor { prompt, model, tools ->
            if (tools.isEmpty()) {
                reflectionExecutor.execute(prompt, model, tools)
            } else {
                rolloutExecutor.execute(prompt, model, tools)
            }
        }
        val agent = createAgent(executor)

        var scoreIndex = 0
        val temporaryDirectory = Files.createTempDirectory("gepa-action-log-test").toFile()
        try {
            val optimizer = GEPAOptimizer<String, String, String>(
                gepaFeedback = strategyOnlyGepaFeedback { _, _, _, _, _ ->
                    val score = scoresInOrder[scoreIndex.coerceAtMost(scoresInOrder.lastIndex)]
                    scoreIndex++
                    score to "Rollout $scoreIndex: scored $score."
                },
                feedbackValSplitFn = { items ->
                    GEPATrainSetSplit(feedbackSet = listOf(items[0]), validationSet = listOf(items[1]))
                },
                reflectionModel = OpenAIModels.Chat.GPT4o,
                storagePath = ResilientPath(temporaryDirectory.resolve("artifact.json").toPath()),
                numRollouts = numRollouts,
                feedbackBatchSize = 1,
                seedValidationFailureRateThreshold = 1.0,
            )

            val logEvents = mutableListOf<ILoggingEvent>()
            val trainingResult = captureGepaLogs(logEvents) {
                runBlocking { optimizer.train(session(agent, executor)) }
            }
            return OptimizationRun(trainingResult, logEvents.map { it.formattedMessage })
        } finally {
            temporaryDirectory.deleteRecursively()
        }
    }

    /** Collects what GEPA's own package logs at INFO while [body] runs. */
    private fun <T> captureGepaLogs(into: MutableList<ILoggingEvent>, body: () -> T): T {
        val logger = LoggerFactory.getLogger(GEPA_LOGGER_NAME) as LogbackLogger
        val appender = ListAppender<ILoggingEvent>().apply { start() }
        val originalLevel = logger.level
        logger.level = Level.INFO
        logger.addAppender(appender)
        try {
            return body()
        } finally {
            logger.detachAppender(appender)
            logger.level = originalLevel
            appender.stop()
            into.addAll(appender.list)
        }
    }

    private fun createAgent(promptExecutor: PromptExecutor): GraphAIAgent<String, String> = GraphAIAgent(
        promptExecutor = promptExecutor,
        agentConfig = AIAgentConfig(
            prompt = prompt("test") { system("Solve the task.") },
            model = OpenAIModels.Chat.GPT4o,
            maxAgentIterations = 20,
        ),
        strategy = strategy("test-strategy") {
            val solve by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = SEED_INSTRUCTION,
            ) { instruction, input -> "$instruction\n$input" }
            nodeStart then solve then nodeFinish
        },
        toolRegistry = ToolRegistry { },
    )

    private fun session(agent: GraphAIAgent<String, String>, executor: PromptExecutor) = trainingSession(
        experimentName = ExperimentName(
            runId = "gepa-action-log-test",
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
        const val SEED_INSTRUCTION = "Solve carefully."
        const val IMPROVED_INSTRUCTION = "Improved instruction."
        const val ELLIPSIS = "..."
        const val GEPA_LOGGER_NAME = "ai.koog.agents.optimization.optimizers.gepa"
        val maxChars = (ActionLogTruncation.DEFAULT as ActionLogTruncation.MaxChars).maxChars
    }
}

private fun StageRecord.stage(name: String): StageRecord {
    val found = if (this.name == name) this else substages.filterIsInstance<StageRecord>()
        .firstNotNullOfOrNull { substage -> runCatching { substage.stage(name) }.getOrNull() }
    return assertNotNull(found, "No stage named '$name' in the records tree")
}

private fun StageRecord.log(): JsonObject =
    assertNotNull(actionLog, "Stage '$name' carries no action log").jsonObject

private fun StageRecord.allActionLogStrings(): List<String> {
    val strings = mutableListOf<String>()
    fun collect(element: JsonElement) {
        when (element) {
            is JsonObject -> element.values.forEach(::collect)
            is JsonArray -> element.forEach(::collect)
            is JsonPrimitive -> if (element.isString) strings.add(element.content)
        }
    }

    fun walk(record: TrainingRecord) {
        if (record is StageRecord) {
            record.actionLog?.let(::collect)
            record.substages.forEach(::walk)
        }
    }
    walk(this)
    return strings
}

private fun JsonObject.string(key: String): String = getValue(key).jsonPrimitive.content
private fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int
private fun JsonObject.double(key: String): Double = getValue(key).jsonPrimitive.double
private fun JsonObject.doubles(key: String): List<Double> = getValue(key).jsonArray.map { it.jsonPrimitive.double }
private fun JsonObject.strings(key: String): List<String> = getValue(key).jsonArray.map { it.jsonPrimitive.content }
private fun JsonObject.intMap(key: String): Map<String, Int> =
    getValue(key).jsonObject.mapValues { (_, weight) -> weight.jsonPrimitive.int }

private fun JsonObject.stringMap(key: String): Map<String, String> =
    getValue(key).jsonObject.mapValues { (_, text) -> text.jsonPrimitive.content }

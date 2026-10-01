package ai.koog.agents.optimization.optimizers.mipro

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.training.records.*
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
import kotlin.test.*
import ch.qos.logback.classic.Logger as LogbackLogger

/**
 * Asserts that a finished MIPROv2 run explains itself: the records name the candidates it generated,
 * the combination each trial drew and the one it kept, and the run log carries the proposed
 * instructions in full.
 *
 * Each case also pins the search. Every expected score, instruction and count here was captured from
 * MIPROv2 as it stood before this logging landed, so a change to its sampling, its split or its budget
 * accounting moves at least one of them.
 *
 * Every case runs the optimizer end to end against mock executors, offline.
 */
internal class MIPROv2LoggingTest {
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()

    @Test
    fun `a run records its setup, its candidates and the combination it kept`() {
        val run = runOptimization(randomSeed = 42, numCandidates = 2, numTrials = 2, scores = SCORES_A)

        val setup = run.rootStage.log()
        assertContentEquals(listOf("__strategy__", "solve"), setup.strings("modules"))
        assertEquals(1, setup.int("bootstrapSetSize"))
        assertEquals(4, setup.int("validationSetSize"))

        val demoSetSizes = run.stage("Step 1").log().getValue("demoSetSizes").jsonObject
        assertEquals(listOf(0, 1), demoSetSizes.getValue("__strategy__").jsonArray.map { it.jsonPrimitive.int })
        assertEquals(listOf(1), demoSetSizes.getValue("solve").jsonArray.map { it.jsonPrimitive.int })

        val candidates = run.instructionCandidates()
        assertEquals(listOf("Instruction 1.", "Instruction 2."), candidates.getValue("__strategy__"))
        assertEquals(listOf("Instruction 3.", "Instruction 4."), candidates.getValue("solve"))

        assertEquals(0.25, run.stage("Baseline").log().double("baselineAverageValidationScore"))

        val gridSearch = run.stage("Step 3").log()
        assertEquals("trial kept", gridSearch.string("outcome"))
        assertEquals(0.65, gridSearch.double("bestAverageValidationScore"))
        assertEquals(1, gridSearch.int("bestTrialNumber"))
        assertEquals(mapOf("__strategy__" to 1, "solve" to 0), gridSearch.indices("bestInstructionIndices"))
        assertEquals(mapOf("__strategy__" to 1, "solve" to 0), gridSearch.indices("bestDemoSetIndices"))

        // The winning indices resolve, through Step 2's list, to the instructions the artifact was saved with.
        assertEquals("Instruction 2.", run.artifact.strategyInstruction)
        assertEquals(mapOf("solve" to "Instruction 3."), run.artifact.subgraphInstructions)

        assertEquals(13, run.agentRuns)
        assertEquals(14, run.promptExecutions)
    }

    @Test
    fun `each trial records the combination it drew`() {
        val run = runOptimization(randomSeed = 7, numCandidates = 3, numTrials = 3, scores = SCORES_B)

        val trials = run.stages("Trial ").map { it.log() }
        assertEquals(3, trials.size)
        assertContentEquals(
            listOf(
                mapOf("__strategy__" to 0, "solve" to 1),
                mapOf("__strategy__" to 1, "solve" to 0),
                mapOf("__strategy__" to 0, "solve" to 0),
            ),
            trials.map { it.indices("instructionIndices") },
        )
        assertContentEquals(
            listOf(
                mapOf("__strategy__" to 2, "solve" to 1),
                mapOf("__strategy__" to 1, "solve" to 1),
                mapOf("__strategy__" to 2, "solve" to 0),
            ),
            trials.map { it.indices("demoSetIndices") },
        )
        assertContentEquals(listOf(0.9, 0.1, 1.0), trials.map { it.double("trialAverageValidationScore") })
        assertContentEquals(listOf(true, false, true), trials.map { it.boolean("isNewBest") })
        assertContentEquals(listOf(0.9, 0.9, 1.0), trials.map { it.double("bestAverageValidationScore") })

        // The trial the grid search kept, resolved against the candidates Step 2 recorded.
        val gridSearch = run.stage("Step 3").log()
        assertEquals(3, gridSearch.int("bestTrialNumber"))
        val candidates = run.instructionCandidates()
        val winningIndices = gridSearch.indices("bestInstructionIndices")
        assertEquals(trials[2].indices("instructionIndices"), winningIndices)
        assertEquals(
            candidates.getValue("__strategy__")[winningIndices.getValue("__strategy__")],
            run.artifact.strategyInstruction,
        )
        assertEquals(
            candidates.getValue("solve")[winningIndices.getValue("solve")],
            run.artifact.subgraphInstructions.getValue("solve"),
        )

        assertEquals(18, run.agentRuns)
        assertEquals(20, run.promptExecutions)
    }

    @Test
    fun `a proposed instruction is clipped in the records and whole in the run log`() {
        val longInstruction = "Follow the plan. ".repeat(120)
        val runLog = mutableListOf<ILoggingEvent>()
        val run = captureMiproLogs(runLog) {
            runOptimization(
                randomSeed = 42,
                numCandidates = 2,
                numTrials = 1,
                scores = SCORES_A,
                instructionText = { index -> "$longInstruction$index" },
            )
        }

        val recorded = run.instructionCandidates().getValue("__strategy__").first()
        assertTrue(recorded.length < longInstruction.length, "the recorded candidate is not clipped")

        val proposalMessage = runLog.map { it.formattedMessage }.single { it.startsWith("MIPRO proposed") }
        assertTrue(longInstruction in proposalMessage, "the run log does not carry the instruction in full")
    }

    @Test
    fun `no trial beating the baseline leaves no winning combination`() {
        val runLog = mutableListOf<ILoggingEvent>()
        val run = captureMiproLogs(runLog) {
            runOptimization(randomSeed = 42, numCandidates = 2, numTrials = 1, scores = BASELINE_WINS_SCORES)
        }

        assertTrue(
            runLog.any { "kept the baseline" in it.formattedMessage },
            "the run log does not say the baseline was kept",
        )

        val gridSearch = run.stage("Step 3").log()
        assertEquals("baseline kept", gridSearch.string("outcome"))
        assertEquals(1.0, gridSearch.double("bestAverageValidationScore"))
        assertNull(gridSearch["bestTrialNumber"])
        assertNull(gridSearch["bestInstructionIndices"])
        assertNull(gridSearch["bestDemoSetIndices"])
        assertEquals(OptimizationArtifact(), run.artifact)
    }

    // ===================================================================
    // Harness
    // ===================================================================

    private class RunOutcome(
        val artifact: OptimizationArtifact,
        val rootStage: StageRecord,
        val stages: List<StageRecord>,
        val agentRuns: Int,
        val promptExecutions: Int,
    ) {
        fun stages(prefix: String): List<StageRecord> = stages.filter { it.name.startsWith(prefix) }

        fun stage(prefix: String): StageRecord = stages(prefix).single()

        fun instructionCandidates(): Map<String, List<String>> =
            stage("Step 2").log().getValue("instructionCandidates").jsonObject
                .mapValues { (_, texts) -> texts.jsonArray.map { it.jsonPrimitive.content } }
    }

    private fun runOptimization(
        randomSeed: Long,
        numCandidates: Int,
        numTrials: Int,
        scores: List<Double>,
        instructionText: (Int) -> String = { index -> "Instruction $index." },
    ): RunOutcome {
        val rolloutExecutor = getMockExecutor { mockLLMToolCall(finishTool, "done") onCondition { true } }
        var proposals = 0
        val executor = TestPromptExecutor { prompt, model, tools ->
            if (tools.isEmpty()) {
                val answer = when (prompt.id) {
                    "dataset-descriptor" -> "A dataset of test items."
                    "describe-program" -> "A test program."
                    "describe-module" -> "A test module."
                    "generate-instruction" -> instructionText(++proposals)
                    else -> "Unexpected meta prompt '${prompt.id}'."
                }
                getMockExecutor { mockLLMAnswer(answer).asDefaultResponse }.execute(prompt, model, tools)
            } else {
                rolloutExecutor.execute(prompt, model, tools)
            }
        }
        val agent = createAgent(executor)
        var scored = 0
        val temporaryDirectory = Files.createTempDirectory("mipro-logging-test").toFile()
        try {
            val optimizer = MIPROv2Optimizer<String, String, String>(
                metaModel = OpenAIModels.Chat.GPT4o,
                numCandidatesOverride = numCandidates,
                numTrialsOverride = numTrials,
                storagePath = ResilientPath(temporaryDirectory.resolve("artifact.json").toPath()),
                randomSeed = randomSeed,
            )
            val session = trainingSession(
                experimentName = ExperimentName(
                    runId = "mipro-logging-test",
                    optimizerName = "MIPROv2",
                    agentName = "test-agent",
                ),
                trackedAgent = agent,
                dataset = (1..4).map { TrainSetItem("item-$it", "gold-$it") },
                substepPromptExecutor = executor,
                metric = { _, _ -> scores[scored++ % scores.size] },
                serializers = DatasetExecutionSerializers(serializeItem = { it.userQuery }, serializeOutput = { it }),
            )
            val result = runBlocking { optimizer.train(session) }
            return RunOutcome(
                artifact = MIPROv2Optimizer.loadArtifact(optimizer.storagePath),
                rootStage = result.rootStage,
                stages = result.allStages(),
                agentRuns = result.countRecords { it is AgentRunRecord },
                promptExecutions = result.countRecords { it is PromptExecutionRecord },
            )
        } finally {
            temporaryDirectory.deleteRecursively()
        }
    }

    /** Collects what MIPROv2's own package logs at INFO while [body] runs. */
    private fun <T> captureMiproLogs(into: MutableList<ILoggingEvent>, body: () -> T): T {
        val logger = LoggerFactory.getLogger(MIPRO_LOGGER_NAME) as LogbackLogger
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

    private fun TrainingResult.countRecords(predicate: (TrainingRecord) -> Boolean): Int {
        var count = 0
        fun walk(record: TrainingRecord) {
            if (predicate(record)) count++
            if (record is StageRecord) record.substages.forEach(::walk)
        }
        walk(rootStage)
        return count
    }

    private fun TrainingResult.allStages(): List<StageRecord> {
        val found = mutableListOf<StageRecord>()
        fun walk(record: TrainingRecord) {
            if (record is StageRecord) {
                found += record
                record.substages.forEach(::walk)
            }
        }
        walk(rootStage)
        return found
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
                optimizableInstruction = "Solve carefully.",
            ) { instruction, input -> "$instruction\n$input" }
            nodeStart then solve then nodeFinish
        },
        toolRegistry = ToolRegistry { },
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
        const val MIPRO_LOGGER_NAME = "ai.koog.agents.optimization.optimizers.mipro"

        val SCORES_A = listOf(1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.2, 0.2, 0.2)
        val SCORES_B = listOf(
            1.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0,
        )

        /** A perfect baseline followed by a weaker trial, so the run keeps the unoptimized agent. */
        val BASELINE_WINS_SCORES = listOf(1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1)

        fun StageRecord.log(): JsonObject =
            assertNotNull(actionLog as? JsonObject, "stage '$name' recorded no action log")

        fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int

        fun JsonObject.double(key: String): Double = getValue(key).jsonPrimitive.double

        fun JsonObject.boolean(key: String): Boolean = getValue(key).jsonPrimitive.boolean

        fun JsonObject.string(key: String): String = getValue(key).jsonPrimitive.content

        fun JsonObject.strings(key: String): List<String> =
            getValue(key).jsonArray.map { it.jsonPrimitive.content }

        fun JsonObject.indices(key: String): Map<String, Int> =
            getValue(key).jsonObject.mapValues { (_, index) -> index.jsonPrimitive.int }
    }
}

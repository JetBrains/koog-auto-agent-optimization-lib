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
import ai.koog.agents.optimization.optimizers.gepa.GEPACheckpointTest.Companion.SCORES
import ai.koog.agents.optimization.training.records.AgentRunRecord
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
import java.io.File
import java.nio.file.Files
import kotlin.test.*

/**
 * GEPA's checkpoint: saved after every step, and enough for a new run to continue the search with the
 * stopped run's candidates and spent budget. Runs offline against mock executors with scripted scores.
 */
internal class GEPACheckpointTest {
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()
    private val temporaryDirectory: File = Files.createTempDirectory("gepa-checkpoint-test").toFile()

    @AfterTest
    fun deleteTemporaryDirectory() {
        temporaryDirectory.deleteRecursively()
    }

    @Test
    fun `a finished run leaves a checkpoint with its pool and spent budget`() {
        val first = runGepa(numRollouts = 14, saveCheckpoint = path("first.json"))

        val checkpoint = readGepaCheckpoint(path("first.json"))
        assertEquals(first.rolloutsSpent, checkpoint.rolloutsSpent)
        assertEquals(first.batches, checkpoint.batchesRun)
        assertEquals(listOf("candidate-0", "candidate-1", "candidate-2"), checkpoint.candidates.map { it.id })
        assertEquals(emptyList(), checkpoint.candidates.first().parentIds)
        assertTrue(checkpoint.candidates.drop(1).all { it.parentIds.size == 1 })
        assertTrue(checkpoint.candidates.all { it.validationScores.size == VALIDATION_SET_SIZE })
    }

    @Test
    fun `a resumed run keeps the pool and spends only the remaining budget`() {
        runGepa(numRollouts = 14, saveCheckpoint = path("first.json"))
        val saved = readGepaCheckpoint(path("first.json"))

        val resumed = runGepa(numRollouts = 30, saveCheckpoint = path("second.json"), resumeFrom = path("first.json"))

        assertEquals(0, resumed.seedValidations, "A resumed run must not repeat the seed validation")
        // The last step may overshoot the budget by up to one batch for the parent and the candidate plus a validation.
        val remaining = 30 - saved.rolloutsSpent
        assertTrue(resumed.agentRuns in remaining..remaining + 2 + 2 + VALIDATION_SET_SIZE, "ran ${resumed.agentRuns}")
        val continued = readGepaCheckpoint(path("second.json"))
        // Round-robin cursors move on once a restored candidate is drawn as a parent again, so they are left out.
        val withoutCursors = { candidates: List<GEPACheckpointCandidate> ->
            candidates.map { it.copy(nextModuleIndex = 0) }
        }
        assertEquals(withoutCursors(saved.candidates), withoutCursors(continued.candidates.take(saved.candidates.size)))
        assertTrue(continued.batchesRun > saved.batchesRun, "batch numbering continues")
        assertTrue(resumed.artifactSaved)
    }

    /**
     * Stops a run at each of its reflection calls in turn, so the checkpoint left behind is the one saved after the
     * step before it: the seed validation, a feedback batch of any outcome, or a merge attempt. Every one of them
     * resumes.
     */
    @Test
    fun `a run stopped at any step resumes from the checkpoint of the step before`() {
        for (merge in listOf(false, true)) {
            val reflections = runGepa(numRollouts = 30, saveCheckpoint = null, merge = merge).reflections
            assertTrue(reflections >= 3, "The full run should reflect several times, ran $reflections")
            val savedStates = mutableListOf<GEPACheckpoint>()

            for (stopAt in 1..reflections) {
                val case = "merge=$merge, stopped at reflection $stopAt"
                val stopped = path("stopped-$merge-$stopAt.json")
                assertFailsWith<StageFailedException>(case) {
                    runGepa(numRollouts = 30, saveCheckpoint = stopped, merge = merge, failingReflection = stopAt)
                }
                val saved = readGepaCheckpoint(stopped)
                savedStates += saved
                if (stopAt == 1) assertEquals(listOf("candidate-0"), saved.candidates.map { it.id }, case)

                val resumedPath = path("resumed-$merge-$stopAt.json")
                val resumed =
                    runGepa(numRollouts = 30, saveCheckpoint = resumedPath, resumeFrom = stopped, merge = merge)

                assertEquals(0, resumed.seedValidations, case)
                assertTrue(resumed.artifactSaved, case)
                val remaining = 30 - saved.rolloutsSpent
                assertTrue(resumed.agentRuns in remaining..remaining + 2 + 2 + VALIDATION_SET_SIZE, case)
                val continued = readGepaCheckpoint(resumedPath)
                assertEquals(
                    saved.candidates.map { it.copy(nextModuleIndex = 0) },
                    continued.candidates.take(saved.candidates.size).map { it.copy(nextModuleIndex = 0) },
                    case,
                )
                assertTrue(continued.batchesRun >= saved.batchesRun, case)
                assertTrue(continued.mergeAttempts >= saved.mergeAttempts, case)
            }
            assertTrue(savedStates.any { it.candidates.size > 1 }, "Some stop should follow an accepted candidate")
            if (merge) {
                assertTrue(savedStates.any { it.mergeAttempts > 0 }, "Some stop should follow a merge attempt")
            }
        }
    }

    @Test
    fun `a checkpoint with its budget already spent goes straight to the artifact`() {
        val first = runGepa(numRollouts = 14, saveCheckpoint = path("first.json"))

        val resumed = runGepa(numRollouts = first.rolloutsSpent, saveCheckpoint = null, resumeFrom = path("first.json"))

        assertEquals(0, resumed.agentRuns)
        assertTrue(resumed.artifactSaved)
    }

    @Test
    fun `resume refuses a checkpoint written for a different split`() {
        runGepa(numRollouts = 14, saveCheckpoint = path("first.json"))

        val failure = assertFailsWith<StageFailedException> {
            runGepa(
                numRollouts = 30,
                saveCheckpoint = null,
                resumeFrom = path("first.json"),
                split = { items -> GEPATrainSetSplit(feedbackSet = items.drop(3), validationSet = items.take(3)) },
            )
        }
        val messages = generateSequence<Throwable>(failure) { it.cause }.map { it.message.orEmpty() }.toList()
        assertTrue(messages.any { "different validation set" in it }, "Causes: $messages")
    }

    @Test
    fun `a checkpoint that cannot be written leaves the run going`() {
        // A regular file where the checkpoint's directory should be makes every write fail.
        temporaryDirectory.resolve("blocked").writeText("not a directory")

        val run = runGepa(numRollouts = 14, saveCheckpoint = path("blocked/checkpoint.json"))

        assertTrue(run.artifactSaved)
    }

    @Test
    fun `merge state goes into the checkpoint and back`() {
        runGepa(numRollouts = 40, saveCheckpoint = path("first.json"), merge = true)
        val saved = readGepaCheckpoint(path("first.json"))
        assertNotNull(saved.merge, "A run with merge on must checkpoint its merge state")

        val resumed = runGepa(
            numRollouts = 50,
            saveCheckpoint = path("second.json"),
            resumeFrom = path("first.json"),
            merge = true,
        )

        assertTrue(resumed.artifactSaved)
        assertTrue(readGepaCheckpoint(path("second.json")).mergeAttempts >= saved.mergeAttempts)
    }

    // ===================================================================
    // Harness
    // ===================================================================

    private class GepaRun(
        val rolloutsSpent: Int,
        val agentRuns: Int,
        val reflections: Int,
        val batches: Int,
        val seedValidations: Int,
        val artifactSaved: Boolean,
    )

    private fun path(name: String) = ResilientPath(temporaryDirectory.resolve(name).toPath())

    /**
     * Runs GEPA over six items (feedback 1-3, validation 4-6 by default). Rollouts score from [SCORES] in
     * invocation order, and every reflection proposes a new instruction; the [failingReflection]-th one (counting
     * from 1) throws instead, which stops the run.
     */
    private fun runGepa(
        numRollouts: Int,
        saveCheckpoint: ResilientPath?,
        resumeFrom: ResilientPath? = null,
        merge: Boolean = false,
        failingReflection: Int? = null,
        split: (TrainSet<String, String>) -> GEPATrainSetSplit<String, String> = { items ->
            GEPATrainSetSplit(feedbackSet = items.take(3), validationSet = items.drop(3))
        },
    ): GepaRun {
        val rolloutExecutor = getMockExecutor { mockLLMToolCall(finishTool, "done") onCondition { true } }
        var reflections = 0
        val executor = TestPromptExecutor { prompt, model, tools ->
            if (tools.isEmpty()) {
                reflections++
                check(reflections != failingReflection) { "reflection outage" }
                getMockExecutor { mockLLMAnswer("```Instruction v$reflections.```").asDefaultResponse }
                    .execute(prompt, model, tools)
            } else {
                rolloutExecutor.execute(prompt, model, tools)
            }
        }
        var rolloutsScored = 0
        val storagePath = path("artifact-${System.nanoTime()}.json")
        val optimizer = GEPAOptimizer<String, String, String>(
            gepaFeedback = strategyOnlyGepaFeedback { _, _, _, _, _ ->
                val score = SCORES[rolloutsScored % SCORES.size]
                rolloutsScored++
                score to "Rollout $rolloutsScored scored $score."
            },
            feedbackValSplitFn = split,
            reflectionModel = OpenAIModels.Chat.GPT4o,
            storagePath = storagePath,
            numRollouts = numRollouts,
            feedbackBatchSize = 2,
            randomSeed = 42,
            mergeConfig = GEPAMergeConfig(enabled = merge, valSetSubsampleSize = 2),
            seedValidationFailureRateThreshold = 1.0,
            reflectionFailurePolicy = GEPAReflectionFailurePolicy.ABORT,
            saveCheckpointPath = saveCheckpoint,
            resumeCheckpointPath = resumeFrom,
        )
        val result = runBlocking { optimizer.train(session(createAgent(executor), executor)) }
        return GepaRun(
            rolloutsSpent = result.agentRunCount() + (resumeFrom?.let { readGepaCheckpoint(it).rolloutsSpent } ?: 0),
            agentRuns = result.agentRunCount(),
            reflections = reflections,
            batches = result.stagesNamed("Feedback batch "),
            seedValidations = result.stagesNamed("Initialize candidate pool"),
            artifactSaved = storagePath.exists(),
        )
    }

    private fun TrainingResult.agentRunCount(): Int = countRecords { it is AgentRunRecord }

    private fun TrainingResult.stagesNamed(prefix: String): Int =
        countRecords { record ->
            record is StageRecord &&
                    record.name.startsWith(prefix) &&
                    "| Item #" !in record.name &&
                    ": run dataset" !in record.name
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
            runId = "gepa-checkpoint-test",
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
        const val VALIDATION_SET_SIZE = 3

        /** The determinism test's first scenario: two candidates accepted over the first two batches. */
        val SCORES = listOf(
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95, 0.3, 0.2, 0.1, 0.4, 0.8, 0.85, 0.2,
            0.55, 0.65, 0.75, 0.05, 0.15, 0.25, 0.35, 0.45, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8,
        )
    }
}

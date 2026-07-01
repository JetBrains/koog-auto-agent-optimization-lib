package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.DemonstrationFormat
import ai.koog.agents.optimization.core.FewShotPromptType
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.PromptInsertionDefaults
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.optimization.features.CollectedSubgraphTraces
import ai.koog.agents.optimization.features.SubgraphTraceCollectionFeature
import ai.koog.agents.optimization.features.collectSubgraphTraces
import ai.koog.agents.optimization.features.installPromptOptimization
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.ext.agent.subgraphWithTask
import ai.koog.agents.features.eventHandler.feature.EventHandler
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.message.Message
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Integration tests exercising the full optimization flow across various agent topologies,
 * optimizer patterns, and rendering modes.
 */
class OptimizationIntegrationTest {

    private val model = OpenAIModels.Chat.GPT4o
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()

    private fun mockExecutor(): PromptExecutor = getMockExecutor {
        mockLLMToolCall(finishTool, "done") onCondition { true }
    }

    /** Agent with NO optimizable subgraphs — just a plain subgraphWithTask. */
    private fun plainStrategy(): AIAgentGraphStrategy<String, String> =
        strategy("plain-strategy") {
            val task by subgraphWithTask<String, String>(
                freshHistory = true,
            ) { input -> "Do: $input" }

            nodeStart then task then nodeFinish
        }

    /** Agent with 1 optimizable subgraph. */
    private fun singleSubgraphStrategy(): AIAgentGraphStrategy<String, String> =
        strategy("single-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify the input.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\nInput: $input" }

            nodeStart then classify then nodeFinish
        }

    /** Agent with 2 sequential optimizable subgraphs. */
    private fun dualSubgraphStrategy(): AIAgentGraphStrategy<String, String> =
        strategy("dual-strategy") {
            val analyze by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Analyze the input.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\nInput: $input" }

            val summarize by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Summarize the analysis.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\nInput: $input" }

            nodeStart then analyze then summarize then nodeFinish
        }

    /** Runs agent with given artifact, captures prompts sent to LLM. */
    private suspend fun runAndCapture(
        strategy: AIAgentGraphStrategy<String, String>,
        artifact: OptimizationArtifact = OptimizationArtifact(),
        input: String = "test",
        systemPrompt: String = "You are helpful.",
        defaults: PromptInsertionDefaults = PromptInsertionDefaults(),
    ): List<Prompt> {
        val prompts = mutableListOf<Prompt>()
        val agent = AIAgent(
            promptExecutor = mockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("t") { system(systemPrompt) },
                model = model,
                maxAgentIterations = 40,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installPromptOptimization {
                    this.artifact = artifact
                    this.promptInsertionDefaults = defaults
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        )
        val session = agent.createSession()
        session.run(input)
        return prompts
    }

    /** Runs agent with trace collection, returns collected traces. */
    private suspend fun runAndCollectTraces(
        strategy: AIAgentGraphStrategy<String, String>,
        artifact: OptimizationArtifact = OptimizationArtifact(),
        input: String = "test",
        systemPrompt: String = "You are helpful.",
    ): CollectedSubgraphTraces {
        val agent = AIAgent(
            promptExecutor = mockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("t") { system(systemPrompt) },
                model = model,
                maxAgentIterations = 40,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installPromptOptimization { this.artifact = artifact }
                collectSubgraphTraces { }
            },
        )
        val session = agent.createSession()
        val traces = session.pipeline()?.feature(
            CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature
        ) ?: error("Trace collection not installed")
        session.run(input)
        return traces
    }

    /** Deterministically rewrites all instructions by prepending "OPTIMIZED: ". */
    private fun instructionOptimizer(
        original: OptimizationArtifact,
        strategySystemPrompt: String,
        subgraphNames: List<String>,
    ): OptimizationArtifact {
        var result = original.withStrategyInstruction("OPTIMIZED: $strategySystemPrompt")
        for (name in subgraphNames) {
            val currentInstruction = original.getInstruction(name) ?: "default"
            result = result.withSubgraphInstruction(name, "OPTIMIZED: $currentInstruction")
        }
        return result
    }

    /** Adds a labeled demo to each subgraph and the strategy. */
    private fun demoOptimizer(
        original: OptimizationArtifact,
        subgraphNames: List<String>,
        round: Int,
    ): OptimizationArtifact {
        var result = original.withStrategyDemonstrations(
            original.strategyDemonstrations + Demonstration(
                "strategy-input-r$round", "strategy-output-r$round"
            )
        )
        for (name in subgraphNames) {
            result = result.withSubgraphDemonstrations(
                name,
                original.getDemonstrations(name) + Demonstration(
                    "$name-input-r$round", "$name-output-r$round"
                )
            )
        }
        return result
    }

    /** Combines instruction and demo optimization. */
    private fun combinedOptimizer(
        original: OptimizationArtifact,
        strategySystemPrompt: String,
        subgraphNames: List<String>,
        round: Int,
    ): OptimizationArtifact {
        val withInstructions = instructionOptimizer(original, strategySystemPrompt, subgraphNames)
        return demoOptimizer(withInstructions, subgraphNames, round)
    }

    @Test
    fun testInstructionOptimizerOnSingleSubgraph() = runBlocking {
        val artifact = instructionOptimizer(
            OptimizationArtifact(),
            strategySystemPrompt = "You are helpful.",
            subgraphNames = listOf("classify"),
        )

        // With freshHistory=true, the subgraph's LLM call doesn't see the strategy prompt.
        // The subgraph instruction is what we can verify here.
        val prompts = runAndCapture(singleSubgraphStrategy(), artifact)

        // Subgraph instruction should be optimized (visible in the defineTask system message)
        assertTrue(
            prompts.first().messages.any { it.content.contains("OPTIMIZED: default") },
            "Subgraph instruction should be optimized. Got: ${prompts.first().messages.map { it.content }}"
        )
    }

    @Test
    fun testInstructionOptimizerOnDualSubgraphs() = runBlocking {
        val artifact = instructionOptimizer(
            OptimizationArtifact(),
            strategySystemPrompt = "System.",
            subgraphNames = listOf("analyze", "summarize"),
        )

        val prompts = runAndCapture(dualSubgraphStrategy(), artifact, systemPrompt = "System.")

        assertTrue(prompts.size >= 2, "Expected at least 2 LLM calls")

        // Each subgraph should use its own optimized instruction
        assertTrue(prompts[0].messages.any { it.content.contains("OPTIMIZED: default") },
            "First subgraph (analyze) should have optimized instruction")
        assertTrue(prompts[1].messages.any { it.content.contains("OPTIMIZED: default") },
            "Second subgraph (summarize) should have optimized instruction")
    }

    @Test
    fun testInstructionOptimizerMultipleRounds() = runBlocking {
        // Round 1: optimize from defaults
        val r1 = instructionOptimizer(
            OptimizationArtifact(),
            strategySystemPrompt = "Be helpful.",
            subgraphNames = listOf("classify"),
        )
        assertEquals("OPTIMIZED: Be helpful.", r1.strategyInstruction)
        assertEquals("OPTIMIZED: default", r1.getInstruction("classify"))

        // Round 2: optimize from round 1 (stacks prefixes)
        val r2 = instructionOptimizer(
            r1,
            strategySystemPrompt = r1.strategyInstruction ?: "",
            subgraphNames = listOf("classify"),
        )
        assertEquals("OPTIMIZED: OPTIMIZED: Be helpful.", r2.strategyInstruction)
        assertEquals("OPTIMIZED: OPTIMIZED: default", r2.getInstruction("classify"))

        // Verify round 2 artifact works at runtime
        val prompts = runAndCapture(singleSubgraphStrategy(), r2, systemPrompt = "Be helpful.")
        assertTrue(prompts.first().messages.any {
            it.content.contains("OPTIMIZED: OPTIMIZED: default")
        }, "Doubly-optimized instruction should appear in prompt")
    }

    @Test
    fun testPlainAgentIgnoresOptimization() = runBlocking {
        // Plain agent (no optimizable subgraphs) should run fine with an artifact installed.
        // The plain subgraphWithTask uses freshHistory=true so it doesn't see the strategy prompt,
        // but the agent should not crash. Subgraph instructions for nonexistent names are ignored.
        val artifact = OptimizationArtifact(
            strategyInstruction = "Optimized system prompt.",
            subgraphInstructions = mapOf("nonexistent" to "Should be ignored"),
        )

        val prompts = runAndCapture(plainStrategy(), artifact)
        assertTrue(prompts.isNotEmpty(), "Agent should run successfully with artifact installed")
    }

    @Test
    fun testStrategyInstructionVisibleInNonFreshSubgraph() = runBlocking {
        // Use non-freshHistory subgraph so it inherits the (optimized) strategy prompt.
        val strategy = strategy<String, String>("test") {
            val task by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Task instruction.",
                freshHistory = false,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then task then nodeFinish
        }

        val artifact = OptimizationArtifact(
            strategyInstruction = "Optimized strategy prompt.",
        )

        val prompts = runAndCapture(strategy, artifact, systemPrompt = "Original prompt.")
        val messages = prompts.first().messages

        // Strategy system prompt should be replaced (visible because freshHistory=false)
        assertTrue(
            messages.filterIsInstance<Message.System>().any {
                it.content == "Optimized strategy prompt."
            },
            "Non-fresh subgraph should see the optimized strategy prompt. Got: ${messages.map { "${it.role}: ${it.content}" }}"
        )
        assertTrue(
            messages.filterIsInstance<Message.System>().none {
                it.content == "Original prompt."
            },
            "Original prompt should be replaced"
        )
    }

    @Test
    fun testDemoOptimizerInjectsAsMessages() = runBlocking {
        val artifact = demoOptimizer(
            OptimizationArtifact(), listOf("classify"), round = 1
        )

        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact,
            defaults = PromptInsertionDefaults(
                fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            ),
        )

        val messages = prompts.first().messages
        assertTrue(messages.any { it.content == "classify-input-r1" },
            "Demo input should be injected as user message")
        assertTrue(messages.any { it.content == "classify-output-r1" },
            "Demo output should be injected as assistant message")
    }

    @Test
    fun testDemoOptimizerInjectsAsString() = runBlocking {
        val artifact = demoOptimizer(
            OptimizationArtifact(), listOf("classify"), round = 1
        )

        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact,
            defaults = PromptInsertionDefaults(
                fewShotPromptType = FewShotPromptType.AS_STRING,
            ),
        )

        val userMessages = prompts.first().messages.filterIsInstance<Message.User>()
        assertTrue(userMessages.any {
            it.content.contains("Input: classify-input-r1") &&
                it.content.contains("Output: classify-output-r1")
        }, "Demo should be rendered as a single string in a user message")
    }

    @Test
    fun testDemoOptimizerAccumulatesOverRounds() = runBlocking {
        var artifact = OptimizationArtifact()
        artifact = demoOptimizer(artifact, listOf("classify"), round = 1)
        artifact = demoOptimizer(artifact, listOf("classify"), round = 2)
        artifact = demoOptimizer(artifact, listOf("classify"), round = 3)

        assertEquals(3, artifact.getDemonstrations("classify").size,
            "Demos should accumulate over rounds")
        assertEquals(3, artifact.strategyDemonstrations.size,
            "Strategy demos should accumulate over rounds")

        // Verify all demos appear in prompt
        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact,
            defaults = PromptInsertionDefaults(fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY),
        )

        val messages = prompts.first().messages
        for (round in 1..3) {
            assertTrue(messages.any { it.content == "classify-input-r$round" },
                "Demo from round $round should be present")
        }
    }

    @Test
    fun testDemoOptimizerDualSubgraphsGetOwnDemos() = runBlocking {
        val artifact = demoOptimizer(
            OptimizationArtifact(), listOf("analyze", "summarize"), round = 1
        )

        val prompts = runAndCapture(
            dualSubgraphStrategy(), artifact,
            defaults = PromptInsertionDefaults(fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY),
        )

        assertTrue(prompts.size >= 2)

        // First subgraph (analyze) should have its own demos
        assertTrue(prompts[0].messages.any { it.content == "analyze-input-r1" })
        assertTrue(prompts[0].messages.none { it.content == "summarize-input-r1" },
            "analyze should not see summarize's demos")

        // Second subgraph (summarize) should have its own demos
        assertTrue(prompts[1].messages.any { it.content == "summarize-input-r1" })
        assertTrue(prompts[1].messages.none { it.content == "analyze-input-r1" },
            "summarize should not see analyze's demos")
    }

    @Test
    fun testStrategyLevelDemosInjected() = runBlocking {
        val artifact = demoOptimizer(
            OptimizationArtifact(), listOf("classify"), round = 1
        )

        // Strategy demos should appear in the initial prompt before any subgraph runs.
        // With freshHistory=true subgraphs, they won't see strategy demos (prompt is fresh).
        // But we can verify they're in the strategy-level prompt by checking the second
        // subgraph with freshHistory=false.
        val strategyWithNonFresh = strategy<String, String>("test") {
            val task by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Task.",
                freshHistory = false,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then task then nodeFinish
        }

        val prompts = runAndCapture(
            strategyWithNonFresh, artifact,
            defaults = PromptInsertionDefaults(fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY),
        )

        // Strategy demos should be visible in non-freshHistory subgraph
        assertTrue(prompts.first().messages.any { it.content == "strategy-input-r1" },
            "Strategy-level demo should be visible in non-freshHistory subgraph")
    }

    @Test
    fun testCombinedOptimizerModifiesBoth() = runBlocking {
        val artifact = combinedOptimizer(
            OptimizationArtifact(),
            strategySystemPrompt = "System.",
            subgraphNames = listOf("classify"),
            round = 1,
        )

        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact, systemPrompt = "System.",
            defaults = PromptInsertionDefaults(fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY),
        )

        val messages = prompts.first().messages

        // Instruction optimized
        assertTrue(messages.any { it.content.contains("OPTIMIZED:") },
            "Instruction should be optimized")

        // Demos injected
        assertTrue(messages.any { it.content == "classify-input-r1" },
            "Demo should be injected")
    }

    @Test
    fun testCombinedOptimizerMultipleRoundsOnDualSubgraphs() = runBlocking {
        var artifact = OptimizationArtifact()

        for (round in 1..3) {
            artifact = combinedOptimizer(
                artifact,
                strategySystemPrompt = artifact.strategyInstruction ?: "System.",
                subgraphNames = listOf("analyze", "summarize"),
                round = round,
            )
        }

        // After 3 rounds: instructions triple-prefixed, 3 demos each
        assertTrue(artifact.strategyInstruction?.startsWith("OPTIMIZED: OPTIMIZED: OPTIMIZED:") == true)
        assertEquals(3, artifact.getDemonstrations("analyze").size)
        assertEquals(3, artifact.getDemonstrations("summarize").size)
        assertEquals(3, artifact.strategyDemonstrations.size)

        // Verify at runtime
        val prompts = runAndCapture(
            dualSubgraphStrategy(), artifact, systemPrompt = "System.",
            defaults = PromptInsertionDefaults(fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY),
        )

        assertTrue(prompts.size >= 2)

        // analyze subgraph: 3 demos, optimized instruction
        val analyzeMessages = prompts[0].messages
        for (round in 1..3) {
            assertTrue(analyzeMessages.any { it.content == "analyze-input-r$round" },
                "analyze should have demo from round $round")
        }
        assertTrue(analyzeMessages.none { it.content.contains("summarize-input") },
            "analyze should not have summarize's demos")

        // summarize subgraph: 3 demos, optimized instruction
        val summarizeMessages = prompts[1].messages
        for (round in 1..3) {
            assertTrue(summarizeMessages.any { it.content == "summarize-input-r$round" },
                "summarize should have demo from round $round")
        }
    }

    @Test
    fun testBootstrapCollectsTracesForSubgraphs() = runBlocking {
        val traces = runAndCollectTraces(dualSubgraphStrategy(), input = "bootstrap-me")

        val analyzeTraces = traces.getTraces("analyze")
        val summarizeTraces = traces.getTraces("summarize")

        assertEquals(1, analyzeTraces.size)
        assertEquals(1, summarizeTraces.size)
        assertEquals("bootstrap-me", analyzeTraces.first().input)
    }

    @Test
    fun testBootstrapCollectsIntermediateMessages() = runBlocking {
        val traces = runAndCollectTraces(singleSubgraphStrategy(), input = "trace-me")

        val classifyTraces = traces.getTraces("classify")
        assertEquals(1, classifyTraces.size)

        val demo = classifyTraces.first()
        assertNotNull(demo.intermediateMessages,
            "Should capture intermediate messages from optimizable subgraph")
        assertTrue(demo.intermediateMessages.isNotEmpty())

        // System message (instruction) should be stripped — it's provided separately.
        assertTrue(demo.intermediateMessages.none { it is Message.System },
            "System message should be stripped from intermediate messages")

        // finalize_task_result Tool.Call should be converted to Assistant.
        assertTrue(demo.intermediateMessages.none { it is Message.Tool.Call },
            "Tool.Call should be converted to Assistant")
        assertTrue(demo.intermediateMessages.none { it is Message.Tool.Result },
            "Tool.Result should be removed")
    }

    @Test
    fun testBootstrapCollectsWholeTrajectory() = runBlocking {
        val traces = runAndCollectTraces(dualSubgraphStrategy(), input = "trajectory-test")

        val fullPrompt = traces.getLatestFullPrompt()
        assertNotNull(fullPrompt, "Whole-agent trajectory should be captured")
        assertTrue(fullPrompt.messages.isNotEmpty())
    }

    @Test
    fun testBootstrapTracesCanBeAppliedAsArtifact() = runBlocking {
        // Step 1: Collect traces
        val traces = runAndCollectTraces(singleSubgraphStrategy(), input = "training-input")
        val classifyTraces = traces.getTraces("classify")
        assertEquals(1, classifyTraces.size)

        // Step 2: Build artifact from collected traces
        val artifact = OptimizationArtifact(
            subgraphDemonstrations = mapOf("classify" to classifyTraces),
        )

        // Step 3: Apply artifact and verify demos appear in prompt
        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact,
            input = "inference-input",
            defaults = PromptInsertionDefaults(fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY),
        )

        val messages = prompts.first().messages

        // With FULL_TRACE (default), the demo's intermediate messages are injected.
        // The training input appears inside the defineTask system message, not as a standalone message.
        assertTrue(messages.any { it.content.contains("training-input") },
            "Bootstrapped demo content should appear in prompt")
    }

    @Test
    fun testBootstrapTracesAppliedAsFullTrace() = runBlocking {
        // Collect traces with intermediate messages
        val traces = runAndCollectTraces(singleSubgraphStrategy(), input = "training-input")
        val classifyTraces = traces.getTraces("classify")
        val demo = classifyTraces.first()
        assertNotNull(demo.intermediateMessages)

        // Build artifact and apply with FULL_TRACE format
        val artifact = OptimizationArtifact(
            subgraphDemonstrations = mapOf("classify" to classifyTraces),
        )
        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact,
            input = "inference-input",
            defaults = PromptInsertionDefaults(
                fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
                demonstrationFormat = DemonstrationFormat.FULL_TRACE,
            ),
        )

        val messages = prompts.first().messages

        // With FULL_TRACE, the intermediate messages should be injected
        // (system messages remapped to user messages)
        assertTrue(messages.size > 2,
            "FULL_TRACE should inject intermediate messages, producing more than just system+toolcall")
    }

    @Test
    fun testBootstrapTracesAppliedAsCompactString() = runBlocking {
        val traces = runAndCollectTraces(singleSubgraphStrategy(), input = "train")
        val classifyTraces = traces.getTraces("classify")

        val artifact = OptimizationArtifact(
            subgraphDemonstrations = mapOf("classify" to classifyTraces),
        )
        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact,
            input = "inference",
            defaults = PromptInsertionDefaults(
                fewShotPromptType = FewShotPromptType.AS_STRING,
                demonstrationFormat = DemonstrationFormat.COMPACT,
            ),
        )

        val userMessages = prompts.first().messages.filterIsInstance<Message.User>()
        assertTrue(userMessages.any {
            it.content.contains("Input: train") && it.content.contains("Output:")
        }, "AS_STRING + COMPACT should render demo as single formatted user message")
    }

    @Test
    fun testSubgraphInheritsDefaultsFromFeature() = runBlocking {
        // Subgraph doesn't specify fewShotPromptType/demonstrationFormat (null = inherit)
        // Feature defaults to AS_STRING — verify that's what the subgraph uses
        val artifact = OptimizationArtifact(
            subgraphDemonstrations = mapOf(
                "classify" to listOf(Demonstration("demo-in", "demo-out")),
            ),
        )

        val prompts = runAndCapture(
            singleSubgraphStrategy(), artifact,
            defaults = PromptInsertionDefaults(
                fewShotPromptType = FewShotPromptType.AS_STRING,
                demonstrationFormat = DemonstrationFormat.COMPACT,
            ),
        )

        // AS_STRING should produce a user message with "Input:" and "Output:" format
        val userMessages = prompts.first().messages.filterIsInstance<Message.User>()
        assertTrue(userMessages.any {
            it.content.contains("Input: demo-in") && it.content.contains("Output: demo-out")
        }, "Subgraph should inherit AS_STRING from feature defaults")
    }

    @Test
    fun testSubgraphOverridesDefaults() = runBlocking {
        // Create a strategy where the subgraph explicitly uses AS_MESSAGE_HISTORY
        val strategy = strategy<String, String>("test") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
                fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val artifact = OptimizationArtifact(
            subgraphDemonstrations = mapOf(
                "classify" to listOf(Demonstration("demo-in", "demo-out")),
            ),
        )

        // Feature defaults to AS_STRING, but subgraph overrides to AS_MESSAGE_HISTORY
        val prompts = runAndCapture(
            strategy, artifact,
            defaults = PromptInsertionDefaults(
                fewShotPromptType = FewShotPromptType.AS_STRING,
            ),
        )

        // AS_MESSAGE_HISTORY produces separate user/assistant messages
        val messages = prompts.first().messages
        assertTrue(messages.any { it is Message.User && it.content == "demo-in" },
            "Subgraph override should use AS_MESSAGE_HISTORY, not AS_STRING")
        assertTrue(messages.any { it is Message.Assistant && it.content == "demo-out" },
            "Subgraph override should produce assistant message")
    }
}

package com.jetbrains.apr.agentOptimization.optimizationCore

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.ToolCalls
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.builder.subgraph
import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.DemonstrationFormat
import ai.koog.agents.optimization.core.FewShotPromptType
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.optimization.features.CollectedSubgraphTraces
import ai.koog.agents.optimization.features.SubgraphTraceCollectionFeature
import ai.koog.agents.optimization.features.collectSubgraphTraces
import ai.koog.agents.optimization.features.installPromptOptimization
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.features.eventHandler.feature.EventHandler
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.message.Message
import ai.koog.utils.io.use
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class OptimizableSubgraphTest {

    private val model = OpenAIModels.Chat.GPT4o

    // The finish tool used internally by optimizableSubgraphWithTask (identityTool<String>)
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()

    private fun createMockExecutor(): PromptExecutor = getMockExecutor {
        mockLLMToolCall(finishTool, "done") onCondition { true }
    }

    private fun createAgent(
        freshHistory: Boolean = true,
        optimizableInstruction: String = "Default instruction.",
        config: OptimizationArtifact? = null,
        capturedPrompts: MutableList<Prompt>? = null,
        fewShotPromptType: FewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
        demonstrationFormat: DemonstrationFormat = DemonstrationFormat.COMPACT,
        executor: PromptExecutor = createMockExecutor(),
        collectTraces: Boolean = false,
    ): AIAgent<String, String> {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = optimizableInstruction,
                freshHistory = freshHistory,
                fewShotPromptType = fewShotPromptType,
                demonstrationFormat = demonstrationFormat,
                runMode = ToolCalls.SEQUENTIAL,
            ) { instruction, input -> "$instruction\nInput: $input" }

            nodeStart then classify then nodeFinish
        }

        return AIAgent(
            promptExecutor = executor,
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("test") {
                    system("Parent system prompt.")
                    user("Prior conversation.")
                    assistant("Prior response.")
                },
                model = model,
                maxAgentIterations = 20,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                if (config != null) {
                    installPromptOptimization { artifact = config }
                }
                if (capturedPrompts != null) {
                    install(EventHandler) {
                        onLLMCallStarting { capturedPrompts += it.prompt }
                    }
                }
                if (collectTraces) {
                    collectSubgraphTraces { }
                }
            },
        )
    }

    @Test
    fun testUsesDefaultInstructionWhenNoConfigInstalled() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            optimizableInstruction = "Classify the sentiment.",
            capturedPrompts = prompts,
        ).use { it.run("hello") }

        val firstPrompt = prompts.first()
        assertTrue(
            firstPrompt.messages.any { it.content.contains("Classify the sentiment.") },
            "Should use default instruction when no config installed"
        )
    }

    @Test
    fun testUsesDefaultInstructionWhenConfigHasNoEntryForThisSubgraph() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            optimizableInstruction = "Default instruction.",
            config = OptimizationArtifact(
                subgraphInstructions = mapOf("other-subgraph" to "Should not appear"),
            ),
            capturedPrompts = prompts,
        ).use { it.run("hello") }

        val firstPrompt = prompts.first()
        assertTrue(firstPrompt.messages.any { it.content.contains("Default instruction.") })
        assertTrue(firstPrompt.messages.none { it.content.contains("Should not appear") })
    }

    @Test
    fun testUsesOptimizedInstructionFromConfig() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            optimizableInstruction = "Default instruction.",
            config = OptimizationArtifact(
                subgraphInstructions = mapOf("classify" to "Optimized: classify with care."),
            ),
            capturedPrompts = prompts,
        ).use { it.run("hello") }

        val firstPrompt = prompts.first()
        assertTrue(
            firstPrompt.messages.any { it.content.contains("Optimized: classify with care.") },
            "Should use optimized instruction from config"
        )
        assertTrue(
            firstPrompt.messages.none { it.content.contains("Default instruction.") },
            "Default instruction should not appear when config overrides it"
        )
    }

    @Test
    fun testFreshHistoryStartsWithEmptyPromptAndSystemMessage() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        createAgent(freshHistory = true, capturedPrompts = prompts).use { it.run("test input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages
        val systemMessages = messages.filterIsInstance<Message.System>()
        assertEquals(1, systemMessages.size, "Expected one system message from defineTask")
        assertTrue(messages.none { it.content.contains("Parent system prompt") })
        assertTrue(messages.none { it.content.contains("Prior conversation") })
    }

    @Test
    fun testNonFreshHistoryInheritsParentPromptAndUsesUserMessage() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        createAgent(freshHistory = false, capturedPrompts = prompts).use { it.run("test input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages
        assertTrue(messages.filterIsInstance<Message.System>().any {
            it.content.contains("Parent system prompt")
        })
        assertTrue(messages.filterIsInstance<Message.User>().any {
            it.content.contains("Default instruction.")
        })
    }

    @Test
    fun testDemosInjectedAsMessageHistoryAfterTaskDescription() = runBlocking {
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(
            Demonstration("example input", "example output"),
            Demonstration("another input", "another output"),
        )

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages

        // System message (from defineTask) should come first
        assertTrue(messages.first() is Message.System)

        // Demo user/assistant pairs should be present
        assertTrue(messages.filterIsInstance<Message.User>().any { it.content == "example input" })
        assertTrue(messages.filterIsInstance<Message.Assistant>().any { it.content == "example output" })
        assertTrue(messages.filterIsInstance<Message.User>().any { it.content == "another input" })
        assertTrue(messages.filterIsInstance<Message.Assistant>().any { it.content == "another output" })
    }

    @Test
    fun testDemosInjectedAsStringProducesSingleUserMessage() = runBlocking {
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(Demonstration("example input", "example output"))

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_STRING,
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        val userMessages = firstPrompt.messages.filterIsInstance<Message.User>()
        assertTrue(
            userMessages.any {
                it.content.contains("Input: example input") && it.content.contains("Output: example output")
            },
            "AS_STRING should produce a single user message with rendered demos"
        )
    }

    @Test
    fun testNoDemosInjectedWhenConfigHasNoDemosForSubgraph() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(
                subgraphDemonstrations = mapOf("other" to listOf(Demonstration("x", "y"))),
            ),
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        assertTrue(
            firstPrompt.messages.none { it.content.contains("x") },
            "Demos for other subgraph should not leak into this subgraph"
        )
    }

    @Test
    fun testDemosInjectedInNonFreshHistoryMode() = runBlocking {
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(Demonstration("demo-in", "demo-out"))

        createAgent(
            freshHistory = false,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages

        assertTrue(messages.filterIsInstance<Message.System>().any {
            it.content.contains("Parent system prompt")
        }, "Parent system prompt should be preserved")
        assertTrue(messages.filterIsInstance<Message.User>().any {
            it.content == "demo-in"
        }, "Demo should be injected")
        assertTrue(messages.filterIsInstance<Message.Assistant>().any {
            it.content == "demo-out"
        }, "Demo should be injected")
    }

    @Test
    fun testNonFreshHistoryDemoOrderingDemosBeforeQuery() = runBlocking {
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(Demonstration("demo-input", "demo-output"))

        createAgent(
            freshHistory = false,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            capturedPrompts = prompts,
        ).use { it.run("real query") }

        val messages = prompts.first().messages

        // Find indices: demo user message and the actual query user message
        val demoIndex = messages.indexOfFirst { it is Message.User && it.content == "demo-input" }
        val queryIndex = messages.indexOfFirst {
            it is Message.User && it.content.contains("real query")
        }

        assertTrue(demoIndex >= 0, "Demo user message should be present")
        assertTrue(queryIndex >= 0, "Query user message should be present")
        assertTrue(
            demoIndex < queryIndex,
            "Demos should appear before the query in non-fresh history mode " +
                    "(demo at $demoIndex, query at $queryIndex)"
        )
    }

    @Test
    fun testFreshHistoryDemoOrderingInstructionThenDemosThenQuery() = runBlocking {
        // With freshHistory=true, the prompt should be:
        //   system(instruction) → demos → user(defineTask(instruction, input)) → LLM
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(Demonstration("demo-input", "demo-output"))

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            capturedPrompts = prompts,
        ).use { it.run("real query") }

        val messages = prompts.first().messages

        // System message should contain the instruction but the query appears
        // as a separate user message after demos
        val systemMsg = messages.first()
        assertTrue(systemMsg is Message.System, "First message should be system")
        assertTrue(
            systemMsg.content.contains("Default instruction"),
            "System message should contain the instruction"
        )

        // Demo should come after the system message
        val demoIndex = messages.indexOfFirst { it is Message.User && it.content == "demo-input" }
        assertTrue(demoIndex > 0, "Demo should appear after the system message")

        // Query user message should come after demos
        val queryIndex = messages.indexOfFirst {
            it is Message.User && it.content.contains("real query")
        }
        assertTrue(queryIndex > demoIndex,
            "Query should appear after demos (demo at $demoIndex, query at $queryIndex)")
    }

    @Test
    fun testSubgraphNameDerivedFromPropertyName() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(
                subgraphInstructions = mapOf("classify" to "Name-resolved instruction"),
            ),
            capturedPrompts = prompts,
        ).use { it.run("test") }

        assertTrue(
            prompts.first().messages.any { it.content.contains("Name-resolved instruction") },
            "Subgraph should resolve its name from the property name for config lookup"
        )
    }

    @Test
    fun testExplicitNameOverridesPropertyName() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val myProperty by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Default.",
                name = "custom-name",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then myProperty then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 20),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installPromptOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf("custom-name" to "Custom instruction"),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        assertTrue(
            prompts.first().messages.any { it.content.contains("Custom instruction") },
            "Should use explicit name 'custom-name' for config lookup, not property name 'myProperty'"
        )
    }

    @Test
    fun testMultipleSubgraphsEachGetOwnInstruction() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First default.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second default.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 40),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installPromptOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf(
                            "first" to "Optimized first",
                            "second" to "Optimized second",
                        ),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        assertTrue(prompts.size >= 2, "Expected at least two LLM calls")
        assertTrue(prompts[0].messages.any { it.content.contains("Optimized first") })
        assertTrue(prompts[0].messages.none { it.content.contains("Optimized second") })
        assertTrue(prompts[1].messages.any { it.content.contains("Optimized second") })
        assertTrue(prompts[1].messages.none { it.content.contains("Optimized first") })
    }

    @Test
    fun testDemosForOneSubgraphDoNotLeakToAnother() = runBlocking {
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 40),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installPromptOptimization {
                    artifact = OptimizationArtifact(
                        subgraphDemonstrations = mapOf(
                            "first" to listOf(Demonstration("first-demo-in", "first-demo-out")),
                        ),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        assertTrue(prompts[0].messages.any { it.content.contains("first-demo-in") },
            "First subgraph should see its demos")
        assertTrue(prompts[1].messages.none { it.content.contains("first-demo-in") },
            "Second subgraph should not see first subgraph's demos")
    }

    @Test
    fun testDuplicateNamesCauseConfigCollision() = runBlocking {
        // Documents the known limitation: two subgraphs with the same name share config entries.
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First default.",
                name = "task",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second default.",
                name = "task",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 40),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installPromptOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf("task" to "Shared instruction"),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        // Both subgraphs get the same instruction — this is the collision (known limitation).
        assertTrue(prompts[0].messages.any { it.content.contains("Shared instruction") })
        assertTrue(prompts[1].messages.any { it.content.contains("Shared instruction") },
            "Duplicate names cause both subgraphs to share the same config entry (known limitation)")
    }

    /** Helper to run an agent with trace collection and return the collected traces. */
    private suspend fun runWithTraceCollection(
        strategy: ai.koog.agents.core.agent.entity.AIAgentGraphStrategy<String, String>,
        input: String,
        maxIterations: Int = 40,
    ): CollectedSubgraphTraces {
        val agent = AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = maxIterations),
            toolRegistry = ToolRegistry { },
            installFeatures = { collectSubgraphTraces { } },
        )
        val session = agent.createSession()
        val traces = session.pipeline()?.feature(CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature)
            ?: error("Trace collection feature should be installed")
        session.run(input)
        return traces
    }

    @Test
    fun testTraceCollectionCapturesSubgraphInputOutput() = runBlocking {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "trace-input")
        val classifyTraces = traces.getTraces("classify")
        assertEquals(1, classifyTraces.size, "Should have collected one trace for 'classify'")
        assertEquals("trace-input", classifyTraces.first().input)
    }

    @Test
    fun testTraceCollectionCapturesIntermediateMessages() = runBlocking {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val demo = traces.getTraces("classify").first()
        assertNotNull(demo.intermediateMessages,
            "Intermediate messages should be captured by afterFinishToolCall hook")
        assertTrue(demo.intermediateMessages.isNotEmpty(),
            "Intermediate messages should contain the subgraph's conversation")
    }

    @Test
    fun testTraceCollectionCapturesWholeAgentTrajectory() = runBlocking {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val fullPrompt = traces.getLatestFullPrompt()
        assertNotNull(fullPrompt, "Whole-agent trajectory should be captured via interceptLLMCallCompleted")
        assertTrue(fullPrompt.messages.isNotEmpty(), "Trajectory should contain messages")
    }

    @Test
    fun testTracesPerSubgraphAreIsolated() = runBlocking {
        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val allTraces = traces.getAllTraces()

        assertTrue("first" in allTraces, "Should have traces for 'first' subgraph")
        assertTrue("second" in allTraces, "Should have traces for 'second' subgraph")
        assertEquals(1, allTraces["first"]?.size, "One trace per subgraph execution")
        assertEquals(1, allTraces["second"]?.size, "One trace per subgraph execution")

        // Each subgraph should have its own intermediate messages, not shared
        val firstDemo = allTraces.getValue("first").first()
        val secondDemo = allTraces.getValue("second").first()
        assertNotNull(firstDemo.intermediateMessages)
        assertNotNull(secondDemo.intermediateMessages)

        // The first subgraph's intermediate should contain "First." instruction
        assertTrue(
            firstDemo.intermediateMessages.any { it.content.contains("First.") },
            "First subgraph's intermediate trace should contain its own instruction"
        )
        // The second subgraph's intermediate should contain "Second." instruction
        assertTrue(
            secondDemo.intermediateMessages.any { it.content.contains("Second.") },
            "Second subgraph's intermediate trace should contain its own instruction"
        )
    }

    @Test
    fun testWholeTrajectoryAfterMultipleSubgraphs() = runBlocking {
        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val fullPrompt = traces.getLatestFullPrompt()
        assertNotNull(fullPrompt, "Whole trajectory should be captured")

        // The trajectory is the latest prompt state after the last LLM call.
        // With freshHistory, each subgraph starts fresh, so the latest prompt
        // reflects the second subgraph's conversation (last LLM call).
        assertTrue(fullPrompt.messages.isNotEmpty())
    }

    @Test
    fun testNestedOptimizableSubgraph() = runBlocking {
        // Strategy → outer (regular subgraph) → inner (optimizable)
        // Depth 3: strategy > subgraph > optimizableSubgraphWithTask
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val outer by subgraph<String, String>(name = "outer") {
                val inner by optimizableSubgraphWithTask<String, String>(
                    optimizableInstruction = "Inner default.",
                    freshHistory = true,
                ) { instruction, input -> "$instruction\n$input" }

                nodeStart then inner then nodeFinish
            }

            nodeStart then outer then nodeFinish
        }

        val agent = AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("t") { system("Strategy prompt.") },
                model = model,
                maxAgentIterations = 40,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installPromptOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf("inner" to "Optimized inner"),
                        subgraphDemonstrations = mapOf(
                            "inner" to listOf(Demonstration("demo-in", "demo-out")),
                        ),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
                collectSubgraphTraces { }
            },
        )
        val session = agent.createSession()
        val traces = session.pipeline()?.feature(
            CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature
        )
        session.run("nested-input")

        // Instruction resolved from config
        assertTrue(prompts.isNotEmpty())
        assertTrue(
            prompts.first().messages.any { it.content.contains("Optimized inner") },
            "Nested optimizable subgraph should use config instruction"
        )
        assertTrue(
            prompts.first().messages.none { it.content.contains("Inner default.") },
            "Default instruction should not appear"
        )

        // Demos injected
        assertTrue(
            prompts.first().messages.any { it.content == "demo-in" },
            "Demos should be injected into nested subgraph"
        )
        assertTrue(
            prompts.first().messages.any { it.content == "demo-out" },
            "Demos should be injected into nested subgraph"
        )

        // Traces collected for both outer and inner
        assertNotNull(traces)
        val innerTraces = traces.getTraces("inner")
        assertEquals(1, innerTraces.size, "Should collect trace for nested 'inner'")
        assertEquals("nested-input", innerTraces.first().input)
        assertNotNull(innerTraces.first().intermediateMessages,
            "Nested subgraph should export intermediate messages")

        val outerTraces = traces.getTraces("outer")
        assertEquals(1, outerTraces.size, "Should collect trace for 'outer' wrapper")
    }

    @Test
    fun testFreshHistoryIntermediateExcludesParent() = runBlocking {
        // freshHistory=true: intermediate messages should contain ONLY
        // the subgraph's own conversation, not the parent system prompt.
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify this.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val agent = AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("t") { system("Parent system prompt that should NOT appear.") },
                model = model,
                maxAgentIterations = 20,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = { collectSubgraphTraces { } },
        )
        val session = agent.createSession()
        val traces = session.pipeline()?.feature(
            CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature
        )
        session.run("hello")

        assertNotNull(traces)
        val demo = traces.getTraces("classify").first()
        assertNotNull(demo.intermediateMessages)

        // Should contain the subgraph's own system message (from defineTask)
        assertTrue(
            demo.intermediateMessages.any { it.content.contains("Classify this.") },
            "Should contain subgraph's own instruction. Got: ${demo.intermediateMessages.map { "${it.role}: ${it.content.take(50)}" }}"
        )

        // Should NOT contain the parent system prompt
        assertTrue(
            demo.intermediateMessages.none { it.content.contains("Parent system prompt that should NOT appear") },
            "Should not contain parent system prompt in freshHistory=true"
        )
    }

    @Test
    fun testNonFreshHistoryIntermediateExcludesInherited() = runBlocking {
        // freshHistory=false: intermediate messages should contain ONLY
        // what the subgraph added, not the inherited parent conversation.
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify this.",
                freshHistory = false,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val agent = AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("t") {
                    system("Inherited system prompt.")
                    user("Inherited user message.")
                    assistant("Inherited assistant reply.")
                },
                model = model,
                maxAgentIterations = 20,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = { collectSubgraphTraces { } },
        )
        val session = agent.createSession()
        val traces = session.pipeline()?.feature(
            CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature
        )
        session.run("hello")

        assertNotNull(traces)
        val demo = traces.getTraces("classify").first()
        assertNotNull(demo.intermediateMessages)

        // Should NOT contain any of the inherited messages
        assertTrue(
            demo.intermediateMessages.none { it.content.contains("Inherited system prompt") },
            "Should not contain inherited system prompt"
        )
        assertTrue(
            demo.intermediateMessages.none { it.content.contains("Inherited user message") },
            "Should not contain inherited user message"
        )
        assertTrue(
            demo.intermediateMessages.none { it.content.contains("Inherited assistant reply") },
            "Should not contain inherited assistant reply"
        )

        // Should contain the subgraph's own task description (as user message in non-fresh mode)
        assertTrue(
            demo.intermediateMessages.any { it.content.contains("Classify this.") },
            "Should contain subgraph's own task. Got: ${demo.intermediateMessages.map { "${it.role}: ${it.content.take(50)}" }}"
        )
    }

    @Test
    fun testIntermediateConvertsFinishToolCallToAssistant() = runBlocking {
        // Verify that finalize_task_result Tool.Call is converted to Assistant
        // and Tool.Result is dropped entirely.
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val demo = traces.getTraces("classify").first()
        assertNotNull(demo.intermediateMessages)

        // finalize_task_result should be converted to Assistant, not kept as Tool.Call
        assertTrue(
            demo.intermediateMessages.none { it is Message.Tool.Call },
            "finalize_task_result Tool.Call should be converted to Assistant. Got: ${demo.intermediateMessages.map { "${it::class.simpleName}(${it.content.take(30)})" }}"
        )
        assertTrue(
            demo.intermediateMessages.none { it is Message.Tool.Result },
            "finalize_task_result Tool.Result should be removed"
        )
        // The converted Assistant message should contain the tool output
        assertTrue(
            demo.intermediateMessages.any { it is Message.Assistant && it.content.contains("done") },
            "Should contain an Assistant message with the finish tool output"
        )
    }

    @Test
    fun testSequentialSubgraphsIntermediateDoNotLeak() = runBlocking {
        // Two sequential subgraphs with freshHistory=true.
        // Each should have ONLY its own messages in intermediate, not the other's.
        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First instruction.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second instruction.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")

        val firstDemo = traces.getTraces("first").first()
        val secondDemo = traces.getTraces("second").first()
        assertNotNull(firstDemo.intermediateMessages)
        assertNotNull(secondDemo.intermediateMessages)

        // First should contain "First instruction." but NOT "Second instruction."
        assertTrue(firstDemo.intermediateMessages.any { it.content.contains("First instruction.") })
        assertTrue(firstDemo.intermediateMessages.none { it.content.contains("Second instruction.") },
            "First subgraph's intermediate should not contain second's messages")

        // Second should contain "Second instruction." but NOT "First instruction."
        assertTrue(secondDemo.intermediateMessages.any { it.content.contains("Second instruction.") })
        assertTrue(secondDemo.intermediateMessages.none { it.content.contains("First instruction.") },
            "Second subgraph's intermediate should not contain first's messages")
    }
}

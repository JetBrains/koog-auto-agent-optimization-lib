package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.builder.subgraph
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.subgraphWithTask
import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.STRATEGY_MODULE_KEY
import ai.koog.agents.optimization.core.discoverModules
import ai.koog.agents.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.serialization.typeToken
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests for `discoverModules` covering two bugs that previously masked the user's
 * `optimizableInstruction` and admitted non-optimizable subgraphs into the optimizer's
 * module list.
 *
 * Bug 1: `discoverModules` collected every `AIAgentSubgraph` returned by
 *   `findAllSubgraphs()`, regardless of whether it was produced by
 *   `optimizableSubgraphWithTask`, plain `subgraph { }`, or `subgraphWithTask`.
 *   The optimizer then wasted meta-LLM budget on subgraphs whose runtime ignores
 *   the optimization artifact.
 *
 * Bug 2: For each discovered subgraph, `currentInstruction` was hard-coded to
 *   `"Subgraph: ${subgraph.name}"`. The user's actual `optimizableInstruction`
 *   was captured only inside the runtime `defineTask` closure and never reached
 *   the optimizer, so candidates were anchored on a meaningless seed.
 *
 * Both bugs are fixed by registering `(subgraph, baselineInstruction)` in a
 * private companion `WeakHashMap` on `OptimizableSubgraphDelegate` at delegation
 * time. `discoverModules` then filters via `lookupBaseline(...) ?: continue` and
 * reads the real baseline from there.
 */
class ModuleDiscoveryTest {

    private val model = OpenAIModels.Chat.GPT4o

    private fun mockExecutor(): PromptExecutor = getMockExecutor { /* unused: we never run the agent */ }

    private fun makeAgent(
        strategy: AIAgentGraphStrategy<String, String>,
        systemPrompt: String? = null,
    ): GraphAIAgent<String, String> = GraphAIAgent(
        inputType = typeToken<String>(),
        outputType = typeToken<String>(),
        promptExecutor = mockExecutor(),
        agentConfig = AIAgentConfig(
            prompt = prompt("test") {
                if (systemPrompt != null) system(systemPrompt)
            },
            model = model,
            maxAgentIterations = 20,
        ),
        strategy = strategy,
        toolRegistry = ToolRegistry { },
    )

    private fun List<OptimizableModule>.subgraphModules(): List<OptimizableModule> =
        filter { it.name != STRATEGY_MODULE_KEY }

    private fun List<OptimizableModule>.strategyModule(): OptimizableModule =
        single { it.name == STRATEGY_MODULE_KEY }

    @Test
    fun testOptimizableSubgraphSeedsModuleWithUserSuppliedInstruction() {
        val strategy = strategy<String, String>("strat") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "You are a careful sentiment classifier.",
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy))

        val subgraphModules = modules.subgraphModules()
        assertEquals(1, subgraphModules.size, "Expected one discovered subgraph module")

        val classifyModule = subgraphModules.single()
        assertEquals("classify", classifyModule.name)
        assertEquals(
            "You are a careful sentiment classifier.",
            classifyModule.currentInstruction,
            "Module's currentInstruction must be the user-supplied optimizableInstruction, " +
                    "not the old placeholder 'Subgraph: <name>'",
        )

        // Defensive: ensure the placeholder string the old code emitted does not appear anywhere.
        assertTrue(
            modules.none { it.currentInstruction == "Subgraph: classify" },
            "The old placeholder seed 'Subgraph: classify' must not appear",
        )
    }

    @Test
    fun testMultipleOptimizableSubgraphsEachKeepTheirOwnBaseline() {
        val strategy = strategy<String, String>("strat") {
            val planner by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Plan the steps carefully.",
            ) { instr, input -> "$instr\n$input" }

            val executor by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Execute one step at a time.",
            ) { instr, input -> "$instr\n$input" }

            nodeStart then planner then executor then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy)).subgraphModules()
        val byName = modules.associateBy { it.name }

        assertEquals(2, modules.size, "Expected exactly two optimizable subgraph modules")
        assertEquals("Plan the steps carefully.", byName.getValue("planner").currentInstruction)
        assertEquals("Execute one step at a time.", byName.getValue("executor").currentInstruction)
    }

    @Test
    fun testExplicitNameOverridesPropertyNameInDiscoveredModule() {
        // The lookup is identity-keyed, so the resolved name (explicit "custom-name")
        // must appear in the module — not the property name "myProperty".
        val strategy = strategy<String, String>("strat") {
            val myProperty by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Explicit-name baseline.",
                name = "custom-name",
            ) { instr, input -> "$instr\n$input" }

            nodeStart then myProperty then nodeFinish
        }

        val module = discoverModules(makeAgent(strategy)).subgraphModules().single()
        assertEquals("custom-name", module.name)
        assertEquals("Explicit-name baseline.", module.currentInstruction)
    }

    @Test
    fun testPlainSubgraphIsExcludedFromModuleList() {
        val strategy = strategy<String, String>("strat") {
            val plain by subgraph<String, String>(name = "plain-sub") {
                nodeStart then nodeFinish
            }

            nodeStart then plain then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy))

        assertEquals(
            0, modules.subgraphModules().size,
            "Plain subgraph { } produces a non-optimizable subgraph and must be filtered out",
        )
        // Strategy module should still be present (it's added unconditionally).
        assertNotNull(
            modules.find { it.name == STRATEGY_MODULE_KEY },
            "Strategy module should always be present regardless of subgraph filtering",
        )
    }

    @Test
    fun testSubgraphWithTaskIsExcludedFromModuleList() {
        val strategy = strategy<String, String>("strat") {
            val task by subgraphWithTask<String, String>(
                name = "task-sub",
            ) { input -> "do something with $input" }

            nodeStart then task then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy))

        assertEquals(
            0, modules.subgraphModules().size,
            "subgraphWithTask is not optimizable (does not consume OptimizationArtifact) " +
                    "and must be filtered out",
        )
    }

    @Test
    fun testMixedStrategyKeepsOnlyOptimizableSubgraphs() {
        // Three subgraphs side-by-side: one of each kind.
        // Only the optimizable one should appear in the module list.
        val strategy = strategy<String, String>("strat") {
            val plain by subgraph<String, String>(name = "plain-sub") {
                nodeStart then nodeFinish
            }

            val task by subgraphWithTask<String, String>(
                name = "task-sub",
            ) { input -> "task: $input" }

            val optimizable by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Optimize me.",
                name = "optimizable-sub",
            ) { instr, input -> "$instr\n$input" }

            nodeStart then plain then task then optimizable then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy)).subgraphModules()

        assertEquals(1, modules.size, "Only the optimizable subgraph should remain after filtering")
        val survivor = modules.single()
        assertEquals("optimizable-sub", survivor.name)
        assertEquals("Optimize me.", survivor.currentInstruction)

        // Defensive: ensure no module references the non-optimizable subgraphs at all.
        assertNull(modules.find { it.name == "plain-sub" })
        assertNull(modules.find { it.name == "task-sub" })
    }

    @Test
    fun testStrategyWithNoOptimizableSubgraphsReturnsOnlyStrategyModule() {
        val strategy = strategy<String, String>("strat") {
            val a by subgraph<String, String>(name = "a") { nodeStart then nodeFinish }
            val b by subgraphWithTask<String, String>(name = "b") { input -> "b: $input" }

            nodeStart then a then b then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy))

        assertEquals(
            1, modules.size,
            "When no subgraph is optimizable, only the STRATEGY_MODULE_KEY entry should remain",
        )
        assertEquals(STRATEGY_MODULE_KEY, modules.single().name)
    }

    @Test
    fun testOptimizableSubgraphNestedInsidePlainSubgraphIsStillDiscovered() {
        // findAllSubgraphs() recurses into inner subgraphs.
        // Result must contain the inner optimizable (with its real baseline) and exclude the outer.
        val strategy = strategy<String, String>("strat") {
            val outer by subgraph<String, String>(name = "outer") {
                val inner by optimizableSubgraphWithTask<String, String>(
                    optimizableInstruction = "Inner real baseline.",
                    name = "inner",
                ) { instr, input -> "$instr\n$input" }

                nodeStart then inner then nodeFinish
            }

            nodeStart then outer then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy)).subgraphModules()

        assertEquals(1, modules.size, "Outer plain subgraph filtered; inner optimizable kept")
        val innerModule = modules.single()
        assertEquals("inner", innerModule.name)
        assertEquals("Inner real baseline.", innerModule.currentInstruction)
    }

    @Test
    fun testStrategyModuleSeededWithAgentSystemPrompt() {
        val strategy = strategy<String, String>("strat") {
            nodeStart then nodeFinish
        }

        val modules = discoverModules(
            makeAgent(strategy, systemPrompt = "Agent-level system instruction."),
        )

        val strategyModule = modules.strategyModule()
        assertEquals(
            "Agent-level system instruction.",
            strategyModule.currentInstruction,
            "STRATEGY_MODULE_KEY's currentInstruction must be the agent's system prompt",
        )
    }

    @Test
    fun testStrategyModuleCurrentInstructionIsEmptyWhenNoSystemPrompt() {
        val strategy = strategy<String, String>("strat") {
            nodeStart then nodeFinish
        }

        val modules = discoverModules(makeAgent(strategy, systemPrompt = null))

        assertEquals(
            "", modules.strategyModule().currentInstruction,
            "With no system prompt the strategy module's currentInstruction should be empty " +
                    "(falls back via `?: \"\"`)",
        )
    }

    @Test
    fun testTwoStrategiesWithSameSubgraphNameKeepDistinctBaselines() {
        val strategyA = strategy<String, String>("A") {
            val shared by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Baseline from A.",
                name = "shared",
            ) { instr, input -> "$instr\n$input" }

            nodeStart then shared then nodeFinish
        }

        val strategyB = strategy<String, String>("B") {
            val shared by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Baseline from B.",
                name = "shared",
            ) { instr, input -> "$instr\n$input" }

            nodeStart then shared then nodeFinish
        }

        val modulesA = discoverModules(makeAgent(strategyA)).subgraphModules().single()
        val modulesB = discoverModules(makeAgent(strategyB)).subgraphModules().single()

        assertEquals("Baseline from A.", modulesA.currentInstruction)
        assertEquals("Baseline from B.", modulesB.currentInstruction)
        // Same name, so existing collision rules still apply at the artifact level —
        // but registry-keyed-by-identity must keep the BASELINES distinct.
        assertContains(setOf("Baseline from A.", "Baseline from B."), modulesA.currentInstruction)
    }
}

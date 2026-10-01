package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.context.AIAgentGraphContextBase
import ai.koog.agents.core.agent.entity.AIAgentSubgraph
import ai.koog.agents.core.agent.entity.ToolSelectionStrategy
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.dsl.builder.AIAgentBuilderDslMarker
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphBuilderBase
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphDelegate
import ai.koog.agents.core.dsl.builder.subgraph
import ai.koog.agents.core.tools.annotations.InternalAgentToolsApi
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.ext.agent.setupSubgraphWithTask
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.processor.ResponseProcessor
import ai.koog.serialization.typeToken
import java.util.*
import kotlin.reflect.KProperty

/**
 * Holds the resolved subgraph name for use inside node lambdas.
 *
 * The name is resolved at property delegation time (graph construction) and read
 * at node execution time. Graph construction always completes before execution starts,
 * so the name is guaranteed to be set when nodes read it.
 */
@PublishedApi
internal class SubgraphNameHolder {
    var name: String? = null
}

/**
 * Delegate wrapper for optimizable subgraphs that captures the resolved subgraph name.
 *
 * @param Input The input type for the subgraph.
 * @param Output The output type for the subgraph.
 * @property innerDelegate The underlying subgraph delegate.
 * @property nameHolder Holder populated with the resolved name at delegation time.
 */
public class OptimizableSubgraphDelegate<Input, Output> @PublishedApi internal constructor(
    private val innerDelegate: AIAgentSubgraphDelegate<Input, Output>,
    @PublishedApi internal val nameHolder: SubgraphNameHolder,
    @PublishedApi internal val baselineInstruction: String,
) {
    /**
     * Property delegation operator. Resolves the subgraph name, registers the subgraph
     * in the discovery registry, and makes the name available to node lambdas via [nameHolder].
     */
    public operator fun getValue(thisRef: Any?, property: KProperty<*>): AIAgentSubgraph<Input, Output> {
        val subgraph = innerDelegate.getValue(thisRef, property)
        nameHolder.name = subgraph.name
        registry[subgraph] = baselineInstruction
        return subgraph
    }

    /** Holds the process-wide registry mapping each optimizable subgraph to its baseline instruction. */
    public companion object {
        // Identity-keyed, weak-valued map: entries vanish when the strategy is GCed,
        // so there is no cross-test or cross-agent contamination. The map is private to
        // this companion — optimizers read via [lookupBaseline] and cannot mutate.
        private val registry: MutableMap<AIAgentSubgraph<*, *>, String> =
            Collections.synchronizedMap(WeakHashMap())

        /**
         * Returns the baseline instruction registered for [subgraph] by
         * [optimizableSubgraphWithTask], or null if the subgraph is not optimizable.
         */
        internal fun lookupBaseline(subgraph: AIAgentSubgraph<*, *>): String? = registry[subgraph]
    }
}

/**
 * Creates an optimizable subgraph that performs a single task with a tunable instruction and
 * bootstrappable few-shot demonstrations.
 *
 * This is the optimization-aware counterpart of `subgraphWithTask`. Each optimizable subgraph
 * acts as a "module" (in DSPy terms) whose instruction and demonstrations can be tuned by
 * an optimizer without modifying the graph.
 *
 * The three differences from `subgraphWithTask`:
 * 1. **Instruction resolution**: the effective instruction is read from [OptimizationArtifact] in
 *    storage (falling back to [optimizableInstruction]), then passed to [defineTask].
 * 2. **Demo injection**: demonstrations from [OptimizationArtifact] are injected into the prompt
 *    after the task description, before the LLM request.
 * 3. **Trace export**: intermediate messages are saved to storage before the prompt is discarded,
 *    enabling [SubgraphTraceCollectionFeature][ai.koog.agents.optimization.features.SubgraphTraceCollectionFeature]
 *    to capture full execution traces.
 *
 * If [PromptOptimizationFeature][ai.koog.agents.optimization.features.PromptOptimizationFeature]
 * is not installed, the subgraph uses [optimizableInstruction] and empty demonstrations.
 *
 * **Important**: subgraph names must be globally unique within a strategy for optimization to
 * work correctly. The [OptimizationArtifact] uses the subgraph name as a lookup key — duplicate
 * names will cause incorrect instruction/demo assignment. If [name] is not provided, the
 * property name is used (same convention as regular nodes and subgraphs in koog).
 *
 * TODO: enforce global uniqueness of optimizable subgraph names within a strategy at
 *  construction time. Currently, duplicate names are not detected and silently cause
 *  incorrect optimization behavior.
 *
 * @param Input The input type for the subgraph.
 * @param Output The output type for the subgraph.
 * @param optimizableInstruction Default instruction, overridable by [OptimizationArtifact].
 * @param name Optional subgraph name. If null, derived from the delegated property name.
 *   Used as the key for [OptimizationArtifact] lookup.
 * @param toolSelectionStrategy Strategy for selecting available tools.
 * @param llmModel Optional LLM model override.
 * @param llmParams Optional LLM parameters override.
 * @param parallelTools Whether the subgraph may execute tool calls in parallel.
 * @param assistantResponseRepeatMax Max retries when the model doesn't call tools.
 * @param responseProcessor Optional post-processing of LLM responses.
 * @param freshHistory When true, the subgraph drops the parent's conversation turns but keeps the
 *   parent's system messages. The module instruction is appended as one more system message after the inherited ones.
 * @param fewShotPromptType How demos are inserted. Null inherits from [PromptInsertionDefaults] in storage.
 * @param demonstrationFormat Detail level for demos. Null inherits from [PromptInsertionDefaults] in storage.
 * @param defineTask Lambda that composes the user query from the resolved instruction and input.
 *   For fresh history, the resolved instruction is also placed as its own system message separately,
 *   so demonstrations are sandwiched between the instruction and the query:
 *   `[inherited system messages] → system(instruction) → demos → user(defineTask(instruction, input)) → LLM`.
 *   The instruction is available in the lambda for convenience — if used, it will appear in both
 *   the system message and the user query (which is fine, it reinforces the instruction).
 * @return A delegate for use with Kotlin property delegation (`by`).
 */
@OptIn(InternalAgentToolsApi::class, InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified Input, reified Output> AIAgentSubgraphBuilderBase<*, *>.optimizableSubgraphWithTask(
    optimizableInstruction: String,
    name: String? = null,
    toolSelectionStrategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    llmModel: LLModel? = null,
    llmParams: LLMParams? = null,
    parallelTools: Boolean = false,
    assistantResponseRepeatMax: Int? = null,
    responseProcessor: ResponseProcessor? = null,
    freshHistory: Boolean = false,
    fewShotPromptType: FewShotPromptType? = null,
    demonstrationFormat: DemonstrationFormat? = null,
    noinline defineTask: suspend AIAgentGraphContextBase.(instruction: String, input: Input) -> String,
): OptimizableSubgraphDelegate<Input, Output> {
    val finishTool = SubgraphWithTaskUtils.finishTool<Output>()
    val nameHolder = SubgraphNameHolder()

    val innerDelegate = subgraph<Input, Output>(
        name = name,
        toolSelectionStrategy = toolSelectionStrategy,
        llmModel = llmModel,
        llmParams = llmParams,
        responseProcessor = responseProcessor,
        freshHistory = freshHistory,
    ) {
        setupSubgraphWithTask(
            finishTool = finishTool,
            inputType = typeToken<Input>(),
            outputTransformedType = typeToken<Output>(),
            parallelTools = parallelTools,
            assistantResponseRepeatMax = assistantResponseRepeatMax,

            // Resolve instruction + inject demonstrations directly into the LLM context.
            //
            // Prompt ordering:
            //   Fresh:     [parent systems] → system(instruction) → demos → user(defineTask(...)) → LLM
            //   Non-fresh: [inherited]                            → demos → user(defineTask(...)) → LLM
            defineTask = defineTask@{ input ->
                val subgraphName = nameHolder.name
                    ?: error("Optimizable subgraph name was not resolved. This is a framework bug.")

                val artifact = storage.get(OptimizationArtifact.STORAGE_KEY)
                val effectiveInstruction = artifact?.getInstruction(subgraphName)
                    ?: optimizableInstruction
                val demos = artifact?.getDemonstrations(subgraphName).orEmpty()

                val defaults = storage.get(PromptInsertionDefaults.STORAGE_KEY)
                val effectivePromptType = fewShotPromptType
                    ?: defaults?.fewShotPromptType
                    ?: FewShotPromptType.AS_MESSAGE_HISTORY
                val effectiveFormat = demonstrationFormat
                    ?: defaults?.demonstrationFormat
                    ?: DemonstrationFormat.COMPACT

                llm.writeSession {
                    appendPrompt {
                        if (freshHistory) {
                            // Since koog 1.1.1 a fresh subgraph inherits the parent's system messages and
                            // only drops the conversation turns. We leave the parent's alone — it belongs to
                            // the `__strategy__` module — and append the module instruction as a second
                            // system message. It must stay a system message: SubgraphTraceCollectionFeature
                            // drops leading ones, which is what keeps the instruction out of mined demos.
                            system(effectiveInstruction)
                        }

                        if (demos.isNotEmpty()) {
                            when (effectivePromptType) {
                                FewShotPromptType.AS_STRING -> {
                                    DemonstrationRenderer.renderAsString(demos, effectiveFormat)
                                        ?.let { user(it) }
                                }

                                FewShotPromptType.AS_MESSAGE_HISTORY -> {
                                    val demoMessages =
                                        DemonstrationRenderer.renderAsMessages(demos, effectiveFormat)
                                    if (demoMessages.isNotEmpty()) {
                                        messages(demoMessages)
                                    }
                                }
                            }
                        }
                    }
                }

                // Returned string is appended as a user message by setupSubgraphWithTask.
                defineTask(effectiveInstruction, input)
            },
        )
    }

    return OptimizableSubgraphDelegate(innerDelegate, nameHolder, optimizableInstruction)
}

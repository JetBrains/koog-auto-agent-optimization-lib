package ai.koog.agents.optimization.features

import ai.koog.agents.core.agent.GraphAIAgent.FeatureContext
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.feature.AIAgentGraphFeature
import ai.koog.agents.core.feature.config.FeatureConfig
import ai.koog.agents.core.feature.pipeline.AIAgentGraphPipeline
import ai.koog.agents.core.tools.annotations.InternalAgentToolsApi
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.optimization.core.Demonstration
import ai.koog.agents.optimization.core.DemonstrationRenderer
import ai.koog.agents.optimization.utils.messages.withParts
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Configuration for [SubgraphTraceCollectionFeature].
 *
 * @property maxTracesPerSubgraph Maximum traces to store per subgraph. Zero or negative means unlimited.
 * @property stringifyInput Converts subgraph input to string for demonstration storage.
 * @property stringifyOutput Converts subgraph output to string for demonstration storage.
 */
public class SubgraphTraceCollectionConfig : FeatureConfig() {
    /** Maximum traces per subgraph. Zero or negative means unlimited. */
    public var maxTracesPerSubgraph: Int = 100

    /** Converts subgraph input to string for demonstration storage. */
    public var stringifyInput: (Any?) -> String = { it.toString() }

    /** Converts subgraph output to string for demonstration storage. */
    public var stringifyOutput: (Any?) -> String = { it.toString() }
}

/**
 * Thread-safe container for collected subgraph execution traces and whole-agent trajectory.
 *
 * Stores per-subgraph [Demonstration] objects keyed by subgraph name, and the latest
 * whole-agent conversation trajectory. Safe for concurrent access via internal mutex.
 *
 * Intermediate-message capture is driven purely by pipeline events:
 * - On subgraph start, the prompt's current messages are snapshotted as the "inherited" prefix.
 * - On every LLM call completion, the prompt+responses are recorded against the innermost
 *   active subgraph for that run.
 * - On subgraph completion, the latest recorded snapshot is diffed against the inherited prefix
 *   and cleaned up into [Demonstration.intermediateMessages].
 */
public class CollectedSubgraphTraces(
    private val maxPerSubgraph: Int,
) {
    private val traces = mutableMapOf<String, MutableList<Demonstration>>()
    private var latestFullPrompt: Prompt? = null
    private val mutex = Mutex()

    /** (runId, subgraphName) -> prompt messages captured at subgraph start. */
    private val inheritedByRunSubgraph = mutableMapOf<Pair<String, String>, List<Message>>()

    /** (runId, subgraphName) -> latest (prompt.messages + responses) seen during the subgraph. */
    private val latestSnapshotByRunSubgraph = mutableMapOf<Pair<String, String>, List<Message>>()

    /** runId -> stack of currently active subgraph names (top = innermost). */
    private val activeStacks = mutableMapOf<String, ArrayDeque<String>>()

    internal suspend fun onSubgraphStarting(
        runId: String,
        subgraphName: String,
        inherited: List<Message>,
    ) {
        mutex.withLock {
            inheritedByRunSubgraph[runId to subgraphName] = inherited
            activeStacks.getOrPut(runId) { ArrayDeque() }.addLast(subgraphName)
        }
    }

    internal suspend fun onLLMCallCompleted(
        runId: String,
        currentPrompt: Prompt,
        response: Message.Assistant,
    ) {
        mutex.withLock {
            val active = activeStacks[runId]?.lastOrNull()
            if (active != null) {
                latestSnapshotByRunSubgraph[runId to active] = currentPrompt.messages + response
            }
            latestFullPrompt = prompt(currentPrompt) { messages(listOf(response)) }
        }
    }

    @OptIn(InternalAgentToolsApi::class)
    internal suspend fun onSubgraphCompleted(
        runId: String,
        subgraphName: String,
        demo: (intermediate: List<Message>?) -> Demonstration,
    ) {
        mutex.withLock {
            val inherited = inheritedByRunSubgraph.remove(runId to subgraphName)
            val snapshot = latestSnapshotByRunSubgraph.remove(runId to subgraphName)
            popActive(runId, subgraphName)

            val intermediate = if (snapshot != null && inherited != null) {
                cleanCapturedMessages(snapshot, inherited)
            } else {
                null
            }

            val subgraphTraces = traces.getOrPut(subgraphName) { mutableListOf() }
            if (maxPerSubgraph > 0 && subgraphTraces.size >= maxPerSubgraph) {
                subgraphTraces.removeAt(0)
            }
            subgraphTraces.add(demo(intermediate))
        }
    }

    internal suspend fun onSubgraphFailed(runId: String, subgraphName: String) {
        mutex.withLock {
            inheritedByRunSubgraph.remove(runId to subgraphName)
            latestSnapshotByRunSubgraph.remove(runId to subgraphName)
            popActive(runId, subgraphName)
        }
    }

    private fun popActive(runId: String, subgraphName: String) {
        val stack = activeStacks[runId] ?: return
        stack.remove(subgraphName)
        if (stack.isEmpty()) activeStacks.remove(runId)
    }

    /** Returns all collected traces for a specific subgraph. */
    public suspend fun getTraces(subgraphName: String): List<Demonstration> =
        mutex.withLock { traces[subgraphName]?.toList().orEmpty() }

    /** Returns all collected traces grouped by subgraph name. */
    public suspend fun getAllTraces(): Map<String, List<Demonstration>> =
        mutex.withLock { traces.mapValues { (_, v) -> v.toList() } }

    /**
     * Returns the latest whole-agent conversation trajectory, or null if no LLM call was made.
     *
     * This is the full prompt (including all LLM responses) captured after the last LLM call
     * in the agent run. Useful for trajectory-based optimizers (ACE, ReasoningBank).
     */
    public suspend fun getLatestFullPrompt(): Prompt? =
        mutex.withLock { latestFullPrompt }

    /** Clears all collected traces and the trajectory. */
    public suspend fun clear() {
        mutex.withLock {
            traces.clear()
            latestFullPrompt = null
            inheritedByRunSubgraph.clear()
            latestSnapshotByRunSubgraph.clear()
            activeStacks.clear()
        }
    }
}

/**
 * Strips the inherited prefix, drops the leading system messages, converts the finalize
 * tool call into plain text, and drops the finalize tool result. The result is the
 * subgraph's own conversation in a shape suitable for use as a few-shot demonstration.
 */
@OptIn(InternalAgentToolsApi::class)
private fun cleanCapturedMessages(
    allMessages: List<Message>,
    inherited: List<Message>,
): List<Message> {
    val subgraphOnly = DemonstrationRenderer.dropInheritedPrefix(allMessages, inherited)
    return subgraphOnly
        // Two kinds of system messages sit at the head of a fresh subgraph's prompt: the parent's
        // (inherited since koog 1.1.1, already gone with the prefix) and the module instruction that
        // `optimizableSubgraphWithTask` appends. Both must stay out of demonstrations: under FULL_TRACE
        // these messages are replayed verbatim, so a captured instruction would later be shown next to
        // a newer candidate instruction.
        .dropWhile { it is Message.System }
        .mapNotNull { msg ->
            val kept = msg.parts.mapNotNull { part ->
                when {
                    part !is MessagePart.Tool -> part
                    part.tool != SubgraphWithTaskUtils.FINALIZE_SUBGRAPH_TOOL_NAME -> part
                    part is MessagePart.Tool.Call -> MessagePart.Text(part.args)
                    else -> null
                }
            }
            when {
                kept == msg.parts -> msg
                kept.isEmpty() -> null
                else -> msg.withParts(kept)
            }
        }
}

/**
 * `Optimizer-internal` feature that captures both per-subgraph traces and whole-agent trajectory.
 *
 * Installed by optimizers on temporary agents during training to collect demonstrations
 * from successful runs. Not intended for direct end-user use.
 *
 * Driven entirely by pipeline events — no special hooks inside subgraphs are required:
 * - **Per-subgraph traces**: at subgraph start, the prompt messages are snapshotted; every LLM
 *   call updates the running snapshot for the innermost active subgraph; on subgraph completion
 *   the snapshot is diffed against the start state and cleaned into [Demonstration].
 * - **Whole-agent trajectory**: the prompt+responses from the last LLM call in the run.
 *   Useful for trajectory-based optimizers (ACE, ReasoningBank).
 */
public object SubgraphTraceCollectionFeature :
    AIAgentGraphFeature<SubgraphTraceCollectionConfig, CollectedSubgraphTraces> {

    /** Storage key under which the [CollectedSubgraphTraces] container is published. */
    override val key: AIAgentStorageKey<CollectedSubgraphTraces> =
        createStorageKey("optimization-subgraph-trace-collection")

    /** Creates the default [SubgraphTraceCollectionConfig]. */
    override fun createInitialConfig(agentConfig: AIAgentConfig): SubgraphTraceCollectionConfig =
        SubgraphTraceCollectionConfig()

    /** Registers the pipeline interceptors that capture per-subgraph traces and the whole-agent trajectory. */
    override fun install(
        config: SubgraphTraceCollectionConfig,
        pipeline: AIAgentGraphPipeline,
    ): CollectedSubgraphTraces {
        val collected = CollectedSubgraphTraces(config.maxTracesPerSubgraph)

        pipeline.interceptSubgraphExecutionStarting(this) { eventContext ->
            val inherited = eventContext.context.llm.readSession { prompt.messages }
            collected.onSubgraphStarting(
                runId = eventContext.context.runId,
                subgraphName = eventContext.subgraph.name,
                inherited = inherited,
            )
        }

        pipeline.interceptLLMCallCompleted(this) { eventContext ->
            // A moderation-only call completes without an assistant response; nothing to record.
            val response = eventContext.response ?: return@interceptLLMCallCompleted
            collected.onLLMCallCompleted(
                runId = eventContext.runId,
                currentPrompt = eventContext.prompt,
                response = response,
            )
        }

        pipeline.interceptSubgraphExecutionCompleted(this) { eventContext ->
            collected.onSubgraphCompleted(
                runId = eventContext.context.runId,
                subgraphName = eventContext.subgraph.name,
            ) { intermediate ->
                Demonstration(
                    input = config.stringifyInput(eventContext.input),
                    output = config.stringifyOutput(eventContext.output),
                    intermediateMessages = intermediate,
                )
            }
        }

        pipeline.interceptSubgraphExecutionFailed(this) { eventContext ->
            collected.onSubgraphFailed(
                runId = eventContext.context.runId,
                subgraphName = eventContext.subgraph.name,
            )
        }

        return collected
    }
}

/**
 * Installs the [SubgraphTraceCollectionFeature] to collect subgraph execution traces
 * and whole-agent trajectory.
 *
 * @param configure Lambda to customize trace collection behavior.
 */
public fun FeatureContext.collectSubgraphTraces(
    configure: SubgraphTraceCollectionConfig.() -> Unit = {},
) {
    install(SubgraphTraceCollectionFeature) {
        configure()
    }
}

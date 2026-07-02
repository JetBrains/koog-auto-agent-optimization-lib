package ai.koog.agents.optimization.training


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.retries.RetryPolicy
import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.prompt.executor.model.PromptExecutor

/**
 * Shared execution resources created once per training session.
 *
 * Holds the tracked agent, the injected execution collaborators (consumption capture, failure
 * analysis, per-item context, agent invocation), and configuration shared across all scope instances.
 *
 * These collaborators ([failureAnalyzer], [consumptionCollector], [capturingExecutorFactory],
 * [runAgentAttempt], [withItemContext]) are supplied by the app; the defaults make the resources
 * usable on their own (no consumption, no provider-specific recognition, plain agent runs).
 */
public class TrainingResources<Input, Output, InputLabel>(
    /** The agent being trained/evaluated; the default target of `runAgent`. */
    public val trackedAgent: GraphAIAgent<Input, Output>,
    /** The training dataset iterated by the DSL. */
    public val dataset: TrainSet<Input, InputLabel>,
    /** Executor used for the session's leaf prompt calls (`executePrompt` and variants). */
    public val substepPromptExecutor: PromptExecutor,
    /** Scores an agent's output against its dataset item; higher is better. */
    public val metric: (TrainSetItem<Input, InputLabel>, Output) -> Double,
    /** Score at or above which a run counts as solved. */
    public val threshold: Double,
    /** Serializers for dataset items and agent outputs used when recording runs. */
    public val serializers: DatasetExecutionSerializers<Input, Output, InputLabel>,
    /** Truncation applied to action-log strings recorded during the session. */
    public val actionLogTruncation: ActionLogTruncation,
    /**
     * Session-level retry policy applied inside leaf primitives (`runAgent` /
     * `executeWithTrackedPromptExecutor`).
     */
    public val retriesPolicy: RetryPolicy,
    /** Recognizes caught exceptions into the project's failure taxonomy. */
    public val failureAnalyzer: FailureAnalyzer,
    /** Wraps a delegate executor into a consumption-capturing one, created once per leaf prompt call. */
    public val capturingExecutorFactory: (PromptExecutor) -> CapturingPromptExecutor,
    /** Returns the consumption accumulated by the tracked agent since the last call (cleared each call). */
    public val consumptionCollector: () -> LLMConsumption? = { null },
    /** Runs one agent attempt; returns the (possibly re-prepared) agent and its output. */
    public val runAgentAttempt: suspend (agent: GraphAIAgent<Input, Output>, input: Input) -> Pair<GraphAIAgent<Input, Output>, Output> =
        { agent, input -> agent to agent.run(input) },
    /** Establishes any per-item context (e.g. tracing) around a dataset item's processing. */
    public val withItemContext: suspend (itemId: String, block: suspend () -> Unit) -> Unit =
        { _, block -> block() },
)

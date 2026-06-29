package ai.koog.agents.optimization.training

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.common.DatasetExecutionSerializers
import ai.koog.agents.optimization.common.ExperimentName
import ai.koog.agents.optimization.common.retries.RetryPolicy
import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.prompt.executor.model.PromptExecutor
import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging

/**
 * Builds a [TrainingSession] for the common case in a single call, filling in library defaults:
 * no smart failure recognition ([NoOpFailureAnalyzer]), LiteLLM token capture
 * ([LiteLLMConsumptionCapturingPromptExecutor], which reads koog's standard token counts),
 * single-attempt runs ([RetryPolicy.None]), [ActionLogTruncation.DEFAULT], no records file, and no
 * cluster/spend-limit metadata.
 *
 * Reach for the [TrainingResources]/[TrainingSession] constructors directly when you need to
 * customize any of those seams (provider-specific failure analysis, retries, abort policies,
 * progress listeners, cluster metadata, …).
 *
 * To capture real token consumption per agent run, build [trackedAgent] on a
 * [LiteLLMConsumptionCapturingPromptExecutor] and pass its `collectAndClear` as [consumptionCollector]:
 * ```
 * val capturing = LiteLLMConsumptionCapturingPromptExecutor(executor)
 * val agent = AIAgent.invokeGraphAgent(promptExecutor = capturing, /* … */)
 * val session = trainingSession(
 *     experimentName = ExperimentName(runId = "demo", optimizerName = "GEPA", agentName = "agent"),
 *     trackedAgent = agent,
 *     dataset = dataset,
 *     substepPromptExecutor = executor,
 *     metric = metric,
 *     serializers = serializers,
 *     consumptionCollector = { capturing.collectAndClear() },
 * )
 * ```
 */
public fun <Input, Output, InputLabel> trainingSession(
    experimentName: ExperimentName,
    trackedAgent: GraphAIAgent<Input, Output>,
    dataset: TrainSet<Input, InputLabel>,
    substepPromptExecutor: PromptExecutor,
    metric: (TrainSetItem<Input, InputLabel>, Output) -> Double,
    serializers: DatasetExecutionSerializers<Input, Output, InputLabel>,
    threshold: Double = 0.9,
    consumptionCollector: () -> LLMConsumption? = { null },
    logger: KLogger = KotlinLogging.logger {},
): TrainingSession<Input, Output, InputLabel> =
    TrainingSession(
        resources = TrainingResources(
            trackedAgent = trackedAgent,
            dataset = dataset,
            substepPromptExecutor = substepPromptExecutor,
            metric = metric,
            threshold = threshold,
            serializers = serializers,
            actionLogTruncation = ActionLogTruncation.DEFAULT,
            retriesPolicy = RetryPolicy.None,
            failureAnalyzer = NoOpFailureAnalyzer,
            capturingExecutorFactory = ::LiteLLMConsumptionCapturingPromptExecutor,
            consumptionCollector = consumptionCollector,
        ),
        logger = logger,
        trainingName = experimentName,
        recordsFilePath = null,
        throwIfRecordsFileExists = false,
        podName = null,
        spendLimit = null,
        retries = null,
    )

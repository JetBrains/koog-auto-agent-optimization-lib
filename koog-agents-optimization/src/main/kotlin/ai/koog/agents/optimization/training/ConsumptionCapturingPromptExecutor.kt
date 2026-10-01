package ai.koog.agents.optimization.training


import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.agents.optimization.consumption.LiteLLMTokenConsumption
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.executor.model.PromptExecutorOperation
import ai.koog.prompt.executor.model.ResolvedModel
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.streaming.StreamFrame
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.Flow
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * A [CapturingPromptExecutor] that wraps a [delegate] executor, runs each call through it, and
 * accumulates the [LLMConsumption] extracted from the returned [Message.Assistant]'s metadata.
 *
 * The training infrastructure creates one per leaf prompt call (via the session's
 * `capturingExecutorFactory`) and reads accumulated consumption between retry attempts via
 * [collectAndClear]. Subclasses supply the provider-specific [extractConsumption].
 */
public abstract class ConsumptionCapturingPromptExecutor(
    protected val delegate: PromptExecutor,
) : CapturingPromptExecutor() {

    protected val logger: io.github.oshai.kotlinlogging.KLogger = KotlinLogging.logger {}

    private val lock = ReentrantLock()
    private var accumulated: LLMConsumption? = null

    /**
     * Runs the call through [delegate] and accumulates any [LLMConsumption] reported by
     * [extractConsumption] for the returned response. Accumulation is thread-safe.
     */
    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Message.Assistant = delegate.execute(prompt, model, tools).also(::accumulate)

    /** Pre-resolved-model overload of [execute]; captures consumption the same way. */
    override suspend fun execute(
        prompt: Prompt,
        model: ResolvedModel,
        tools: List<ToolDescriptor>,
    ): Message.Assistant = delegate.execute(prompt, model, tools).also(::accumulate)

    /** Delegates streaming to [delegate]; no consumption is captured from streamed frames. */
    override fun executeStreaming(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Flow<StreamFrame> = delegate.executeStreaming(prompt, model, tools)

    /** Pre-resolved-model overload of [executeStreaming]; also captures nothing. */
    override fun executeStreaming(
        prompt: Prompt,
        resolvedModel: ResolvedModel,
        tools: List<ToolDescriptor>,
    ): Flow<StreamFrame> = delegate.executeStreaming(prompt, resolvedModel, tools)

    /** Delegates moderation to [delegate]; no consumption is captured. */
    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        delegate.moderate(prompt, model)

    /** Pre-resolved-model overload of [moderate]; also captures nothing. */
    override suspend fun moderate(prompt: Prompt, model: ResolvedModel): ModerationResult =
        delegate.moderate(prompt, model)

    /** Delegates model resolution, so the wrapped executor keeps its fallback/routing behaviour. */
    override suspend fun resolveModel(model: LLModel, promptExecutorOperation: PromptExecutorOperation): ResolvedModel =
        delegate.resolveModel(model, promptExecutorOperation)

    /** Closes the underlying [delegate] executor. */
    override fun close(): Unit = delegate.close()

    private fun accumulate(response: Message.Assistant) {
        val consumption = extractConsumption(response) ?: return
        lock.withLock {
            val current = accumulated
            accumulated = if (current == null) consumption else current + consumption
        }
    }

    /**
     * Extracts [LLMConsumption] from the [Message.Assistant] returned by a single [execute] call.
     * Returns `null` if no consumption data is present.
     */
    protected abstract fun extractConsumption(response: Message.Assistant): LLMConsumption?

    override fun collectAndClear(): LLMConsumption? = lock.withLock {
        val result = accumulated
        accumulated = null
        result
    }
}

/**
 * LiteLLM variant: reads `inputTokensCount` / `outputTokensCount` / `totalTokensCount` from the
 * response metadata.
 */
public class LiteLLMConsumptionCapturingPromptExecutor(
    delegate: PromptExecutor,
) : ConsumptionCapturingPromptExecutor(delegate) {

    override fun extractConsumption(response: Message.Assistant): LiteLLMTokenConsumption? {
        val metaInfo = response.metaInfo.takeIf { it.inputTokensCount != null } ?: return null

        // Passed through as reported, so with a thinking model, input + output can fall below the total.
        // TODO: track reasoning/thinking tokens as their own count -- see `LiteLLMTokenConsumption`.
        val inputTokens = metaInfo.inputTokensCount?.toLong() ?: return null
        val outputTokens = metaInfo.outputTokensCount?.toLong() ?: return null
        val totalTokens = metaInfo.totalTokensCount?.toLong() ?: return null

        val consumption = LiteLLMTokenConsumption(
            inputTokens = inputTokens,
            outputTokens = outputTokens,
            totalTokens = totalTokens,
        )
        logger.debug { "Captured tokens: ${consumption.toPrettyString()}" }
        return consumption
    }
}

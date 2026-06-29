package ai.koog.agents.optimization.training


import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.optimization.consumption.LLMConsumption
import ai.koog.agents.optimization.consumption.LiteLLMTokenConsumption
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.streaming.StreamFrame
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.Flow
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * A [CapturingPromptExecutor] that wraps a [delegate] executor, runs each call through it, and
 * accumulates the [LLMConsumption] extracted from the returned [Message.Response] metadata.
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
     * [extractConsumption] for the returned responses. Accumulation is thread-safe.
     */
    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): List<Message.Response> {
        val responses = delegate.execute(prompt, model, tools)
        extractConsumption(responses)?.let { consumption ->
            lock.withLock {
                val current = accumulated
                accumulated = if (current == null) consumption else current + consumption
            }
        }
        return responses
    }

    /** Delegates streaming to [delegate]; no consumption is captured from streamed frames. */
    override fun executeStreaming(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Flow<StreamFrame> = delegate.executeStreaming(prompt, model, tools)

    /** Delegates moderation to [delegate]; no consumption is captured. */
    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        delegate.moderate(prompt, model)

    /** Closes the underlying [delegate] executor. */
    override fun close(): Unit = delegate.close()

    /**
     * Extracts [LLMConsumption] from one batch of [Message.Response] objects returned by a single
     * [execute] call. Returns `null` if no consumption data is present.
     */
    protected abstract fun extractConsumption(responses: List<Message.Response>): LLMConsumption?

    override fun collectAndClear(): LLMConsumption? = lock.withLock {
        val result = accumulated
        accumulated = null
        result
    }
}

/**
 * LiteLLM variant: reads `inputTokensCount` / `outputTokensCount` / `totalTokensCount` from the first
 * response that carries token metadata.
 *
 * LiteLLM populates these fields on the first (and typically only) response in the list; subsequent
 * responses are tool-call objects with no token data.
 */
public class LiteLLMConsumptionCapturingPromptExecutor(
    delegate: PromptExecutor,
) : ConsumptionCapturingPromptExecutor(delegate) {

    override fun extractConsumption(responses: List<Message.Response>): LiteLLMTokenConsumption? {
        val metaInfo = responses.firstOrNull { it.metaInfo.inputTokensCount != null }?.metaInfo
            ?: return null

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

package ai.koog.agents.optimization.consumption

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.round

/** Token-based consumption unit (LiteLLM). */
public object LiteLLMTokensUnit : LLMConsumptionUnit {
    override val shortLabel: String = "tokens"
    override val displayLabel: String = "LiteLLM tokens"
}

/**
 * Token-based consumption for the LiteLLM provider.
 *
 * Tracks tokens with detailed breakdown:
 * - [inputTokens]: Tokens in the prompt/input
 * - [outputTokens]: Tokens in the completion/output
 * - [totalTokens]: Total tokens consumed
 */
@Serializable
public data class LiteLLMTokenConsumption(
    /** Tokens in the prompt/input. */
    public val inputTokens: Long = 0,
    /** Tokens in the completion/output. */
    public val outputTokens: Long = 0,
    /** Total tokens consumed. */
    public val totalTokens: Long = 0,
) : LLMConsumption() {
    override val unit: LLMConsumptionUnit get() = LiteLLMTokensUnit

    override val total: Double get() = totalTokens.toDouble()

    override fun plus(consumption: LLMConsumption): LLMConsumption {
        if (consumption is LiteLLMTokenConsumption) {
            return LiteLLMTokenConsumption(
                inputTokens = inputTokens + consumption.inputTokens,
                outputTokens = outputTokens + consumption.outputTokens,
                totalTokens = totalTokens + consumption.totalTokens,
            )
        } else {
            error("Can plus `LiteLLMTokenConsumption` only to `LiteLLMTokenConsumption`, but got: $consumption")
        }
    }

    override fun toPrettyJson(): JsonObject = buildJsonObject {
        put("inputTokens", JsonPrimitive(formatAmount(inputTokens.toDouble(), unit)))
        put("outputTokens", JsonPrimitive(formatAmount(outputTokens.toDouble(), unit)))
        put("totalTokens", JsonPrimitive(formatAmount(totalTokens.toDouble(), unit)))
    }

    override fun toPrettyString(): String {
        val totalStr = formatAmount(totalTokens.toDouble(), unit)
        val inputStr = formatAmount(inputTokens.toDouble(), unit)
        val outputStr = formatAmount(outputTokens.toDouble(), unit)

        return "$totalStr (input = $inputStr, output = $outputStr)"
    }

    /** Factory for reconstructing [LiteLLMTokenConsumption] from its pretty-JSON form. */
    public companion object {
        /** Parses the pretty fields of [obj] (`inputTokens`/`outputTokens`/`totalTokens`) back into token counts. */
        public fun fromPrettyJson(obj: JsonObject): LiteLLMTokenConsumption {
            fun readLong(key: String): Long {
                val (value, unit) = parseAmount(obj.getValue(key).jsonPrimitive.content)
                require(unit == LiteLLMTokensUnit) { "Expected tokens for '$key', got ${unit.displayLabel}" }
                return round(value).toLong()
            }
            return LiteLLMTokenConsumption(
                inputTokens = readLong("inputTokens"),
                outputTokens = readLong("outputTokens"),
                totalTokens = readLong("totalTokens"),
            )
        }
    }
}

/** Registers LiteLLM token consumption with [LLMConsumptionRegistry] via ServiceLoader. */
public class LiteLLMConsumptionType : LLMConsumptionType {
    override val unit: LLMConsumptionUnit = LiteLLMTokensUnit
    override fun fromPrettyJson(obj: JsonObject): LLMConsumption = LiteLLMTokenConsumption.fromPrettyJson(obj)
}

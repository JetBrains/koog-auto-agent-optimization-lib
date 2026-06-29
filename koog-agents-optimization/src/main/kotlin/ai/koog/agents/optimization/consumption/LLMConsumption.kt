package ai.koog.agents.optimization.consumption

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * Base type for an amount of LLM consumption recorded during an optimization run.
 *
 * This is an `abstract class` (not `sealed`) so that providers outside the library can define
 * their own consumption flavors. Each flavor pairs with an [LLMConsumptionUnit] and registers a
 * decoder via [LLMConsumptionType] (discovered through [LLMConsumptionRegistry]); serialization
 * is handled uniformly by [LLMConsumptionSerializer].
 */
@Serializable(with = LLMConsumptionSerializer::class)
public abstract class LLMConsumption {
    /** The unit this consumption is measured in. */
    public abstract val unit: LLMConsumptionUnit

    /** The single scalar total used for comparisons (e.g. against a spend cap). */
    public abstract val total: Double

    /** Adds two consumptions of the same flavor. Implementations reject mismatched flavors. */
    public abstract operator fun plus(consumption: LLMConsumption): LLMConsumption

    /** The flavor-specific fields as pretty JSON (without the top-level `"unit"` field). */
    public abstract fun toPrettyJson(): JsonObject

    /** Human-readable form, typically with input/output breakdown where applicable. */
    public abstract fun toPrettyString(): String

    /**
     * Total-only short form, e.g. `"330,329.00 tokens"` / `"150.00 credits"`.
     * Used where the input/output breakdown would be misleading or off-topic — most
     * notably the spend-limit abort message, where there is no such thing as a separate
     * input/output cap (cap is a single total).
     */
    public fun toPrettyTotal(): String = formatAmount(total, unit)
}

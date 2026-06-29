package ai.koog.agents.optimization.consumption

import kotlinx.serialization.json.JsonObject

/**
 * SPI for contributing an [LLMConsumption] flavor to [LLMConsumptionRegistry].
 *
 * Implementations are discovered at runtime via [java.util.ServiceLoader], so each provider
 * must list its implementation in
 * `META-INF/services/ai.koog.agents.optimization.consumption.LLMConsumptionType`. This lets the
 * library decode any consumption flavor present on the classpath without compile-time knowledge
 * of provider-specific types (e.g. the library never references Grazie consumption).
 */
public interface LLMConsumptionType {
    /** The unit this type decodes; its [LLMConsumptionUnit.displayLabel] is the registry key. */
    public val unit: LLMConsumptionUnit

    /** Decodes the flavor-specific JSON (the object with the `"unit"` field already stripped). */
    public fun fromPrettyJson(obj: JsonObject): LLMConsumption
}

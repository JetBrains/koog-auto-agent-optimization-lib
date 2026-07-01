package ai.koog.agents.optimization.consumption

import kotlinx.serialization.json.JsonObject
import java.util.ServiceLoader

/**
 * Resolves consumption flavors discovered on the classpath via [LLMConsumptionType] services.
 *
 * Discovery is lazy (on first use) and uses the thread context classloader, so there is no
 * startup-ordering requirement: whichever provider jars are present contribute their flavors.
 * The library ships LiteLLM; the consuming app contributes its own provider-specific flavors via its
 * own `META-INF/services` entry.
 */
public object LLMConsumptionRegistry {
    private val types: List<LLMConsumptionType> by lazy {
        ServiceLoader.load(LLMConsumptionType::class.java).toList()
    }
    private val byDisplayLabel: Map<String, LLMConsumptionType> by lazy {
        types.associateByUnique("displayLabel") { it.unit.displayLabel }
    }
    private val byShortLabel: Map<String, LLMConsumptionType> by lazy {
        types.associateByUnique("shortLabel") { it.unit.shortLabel.lowercase() }
    }

    /** Like [associateBy], but fails loudly on a key collision. */
    private fun List<LLMConsumptionType>.associateByUnique(
        keyName: String,
        key: (LLMConsumptionType) -> String,
    ): Map<String, LLMConsumptionType> =
        groupBy(key)
            .onEach { (k, dup) ->
                require(dup.size == 1) {
                    "Duplicate consumption $keyName '$k' from providers: " +
                        dup.joinToString { it::class.qualifiedName ?: "?" }
                }
            }
            .mapValues { it.value.single() }

    /** Decodes consumption identified by its top-level `"unit"` [displayLabel]. */
    public fun decode(displayLabel: String, obj: JsonObject): LLMConsumption =
        typeForDisplayLabel(displayLabel).fromPrettyJson(obj)

    /** Resolves the unit from a top-level `"unit"` display label (e.g. `"LiteLLM tokens"`). */
    public fun unitForDisplayLabel(displayLabel: String): LLMConsumptionUnit =
        typeForDisplayLabel(displayLabel).unit

    /** Resolves the unit from a formatted-amount short suffix (e.g. `"tokens"`). */
    public fun unitForShortLabel(shortLabel: String): LLMConsumptionUnit =
        (byShortLabel[shortLabel.trim().lowercase()]
            ?: error("Unknown consumption unit suffix: '$shortLabel'")).unit

    private fun typeForDisplayLabel(displayLabel: String): LLMConsumptionType =
        byDisplayLabel[displayLabel.trim()]
            ?: error("Unknown consumption unit label: '$displayLabel'")
}

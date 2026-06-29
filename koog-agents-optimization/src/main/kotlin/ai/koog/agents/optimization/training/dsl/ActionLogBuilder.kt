package ai.koog.agents.optimization.training.dsl

import ai.koog.agents.optimization.annotations.OptimizationExtensionApi

import ai.koog.agents.optimization.training.ActionLogTruncation
import kotlinx.serialization.json.*

/**
 * DSL for building a stage action log. Obtained via [StageScope.logAction].
 *
 * - Primitive overloads ([put] for [Int], [Long], [Double], [Boolean]) emit JSON primitives directly.
 * - [put] for [String] emits a JSON string; any truncation is applied globally in [build].
 * - Generic [put] with a `@Serializable` type encodes the value as a proper JSON element
 *   (array, object, …) using the [json] instance provided to [StageScope.logAction] —
 *   **never wrapped as a string**.
 *
 * After all `put` calls, [build] applies [ActionLogTruncation] recursively to the entire JSON
 * tree, so strings nested inside serialized objects or arrays are truncated by the same rule.
 */
@TrainingDsl
@OptimizationExtensionApi
public class ActionLogBuilder(
    /** Truncation applied to all string values when [build] is called. */
    public val truncation: ActionLogTruncation,
    /** JSON instance used to serialize values passed to the generic [put]. */
    public val json: Json,
) {
    /** The accumulated log entries, keyed by name; mutated by the `put` calls. */
    public val properties: MutableMap<String, JsonElement> = mutableMapOf()

    /** Records [value] under [key] as a JSON number. */
    public fun put(key: String, value: Int) {
        properties[key] = JsonPrimitive(value)
    }

    /** Records [value] under [key] as a JSON number. */
    public fun put(key: String, value: Long) {
        properties[key] = JsonPrimitive(value)
    }

    /** Records [value] under [key] as a JSON number. */
    public fun put(key: String, value: Double) {
        properties[key] = JsonPrimitive(value)
    }

    /** Records [value] under [key] as a JSON boolean. */
    public fun put(key: String, value: Boolean) {
        properties[key] = JsonPrimitive(value)
    }

    /** Records [value] under [key] as a JSON string (subject to [truncation] in [build]). */
    public fun put(key: String, value: String) {
        properties[key] = JsonPrimitive(value)
    }

    /** Serializes [value] as a proper JSON element (object, array, …) using this builder's [json]. */
    public inline fun <reified T> put(key: String, value: T) {
        properties[key] = json.encodeToJsonElement(value)
    }

    /** Adds [list] under [key] only when it is non-empty. */
    public inline fun <reified T> putIfNonEmpty(key: String, list: List<T>) {
        if (list.isNotEmpty()) put(key, list)
    }

    /**
     * Assembles the accumulated [properties] into a [JsonObject] and applies [truncation]
     * recursively to every string in the tree.
     */
    public fun build(): JsonObject = JsonObject(properties).truncateStrings(truncation) as JsonObject
}

/**
 * Recursive string truncation over the entire JSON element tree
 */

@OptimizationExtensionApi
public fun JsonElement.truncateStrings(truncation: ActionLogTruncation): JsonElement {
    val maxChars = when (truncation) {
        is ActionLogTruncation.Unlimited -> return this
        is ActionLogTruncation.MaxChars -> truncation.maxChars
    }
    return truncateStrings(maxChars)
}

private fun JsonElement.truncateStrings(maxChars: Int): JsonElement = when (this) {
    is JsonObject -> JsonObject(entries.associate { (k, v) -> k to v.truncateStrings(maxChars) })
    is JsonArray -> JsonArray(map { it.truncateStrings(maxChars) })
    is JsonPrimitive -> if (isString && content.length > maxChars)
        JsonPrimitive("${content.take(maxChars)}...") else this
}

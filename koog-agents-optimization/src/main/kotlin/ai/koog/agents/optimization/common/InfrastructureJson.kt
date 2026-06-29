package ai.koog.agents.optimization.common


import ai.koog.agents.optimization.training.metrics.impl.Metric
import kotlinx.serialization.KSerializer
import kotlinx.serialization.json.*

/**
 * Canonical [Json] instance for serializing training and evaluation results to disk.
 *
 * Used by training sessions, evaluation, and as the default serializer
 * in stage action logs (`StageScope.logAction`).
 */
public val defaultExperimentsJson: Json = Json {
    prettyPrint = true
    classDiscriminator = "kind"
    encodeDefaults = true
    explicitNulls = true
    serializersModule = Metric.serializersModule
}

/**
 * Recursively strips the class discriminator field from all [JsonObject] nodes.
 *
 * Used to remove the `"kind"` field that kotlinx.serialization adds for polymorphic types,
 * keeping the JSON output clean for disk records and cluster progress logs.
 */
public fun JsonElement.stripClassDiscriminator(discriminator: String = "kind"): JsonElement = when (this) {
    is JsonObject -> JsonObject(
        filterKeys { it != discriminator }.mapValues { (_, v) -> v.stripClassDiscriminator(discriminator) }
    )
    is JsonArray -> JsonArray(map { it.stripClassDiscriminator(discriminator) })
    is JsonPrimitive -> this
}

/**
 * Encodes [value] to a JSON string, stripping the class discriminator from all objects.
 */
public fun <T> Json.encodeToStringStripped(serializer: KSerializer<T>, value: T): String {
    val element = encodeToJsonElement(serializer, value)
    val stripped = element.stripClassDiscriminator()
    return encodeToString(JsonElement.serializer(), stripped)
}

package ai.koog.agents.optimization.consumption

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.descriptors.element
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * JSON serializer for any [LLMConsumption] flavor. The on-disk shape is a single object with a
 * top-level `"unit"` field (the flavor's [LLMConsumptionUnit.displayLabel]) plus the flavor's own
 * pretty fields. On read, the `"unit"` label is resolved through [LLMConsumptionRegistry] to the
 * right decoder — so no provider-specific dispatch lives here.
 */
public object LLMConsumptionSerializer : KSerializer<LLMConsumption> {
    /** Descriptor with a single `"unit"` field; subtype fields are dynamic and not described here. */
    override val descriptor: SerialDescriptor =
        buildClassSerialDescriptor("LLMConsumptionSerializer") {
            element<String>("unit")
            // subtype fields are dynamic; descriptor here is mostly informational anyway
        }

    /** Encodes [value] as a JSON object of its pretty fields plus a `"unit"` label. JSON only. */
    override fun serialize(encoder: Encoder, value: LLMConsumption) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: throw SerializationException("LLMConsumptionSerializer supports JSON only")

        val obj = buildJsonObject {
            put("unit", JsonPrimitive(value.unit.displayLabel))

            // subtype-specific pretty fields (no unit inside values except suffix "tokens/credits")
            value.toPrettyJson().forEach { (key, value) -> put(key, value) }
        }

        jsonEncoder.encodeJsonElement(obj)
    }

    /** Decodes a JSON object, resolving its `"unit"` label through [LLMConsumptionRegistry]. JSON only. */
    override fun deserialize(decoder: Decoder): LLMConsumption {
        val jsonDecoder = decoder as? JsonDecoder
            ?: throw SerializationException("LLMConsumptionSerializer supports JSON only")

        val obj = jsonDecoder.decodeJsonElement().jsonObject
        val unitLabel = obj.getValue("unit").jsonPrimitive.content

        // remove "unit" before passing down (children don't expect it)
        val withoutUnit = buildJsonObject {
            obj.forEach { (key, value) -> if (key != "unit") put(key, value) }
        }

        return LLMConsumptionRegistry.decode(unitLabel, withoutUnit)
    }
}

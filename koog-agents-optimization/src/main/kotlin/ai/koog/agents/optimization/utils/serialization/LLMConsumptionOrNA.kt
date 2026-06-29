package ai.koog.agents.optimization.utils.serialization

import ai.koog.agents.optimization.consumption.LLMConsumptionSerializer

import ai.koog.agents.optimization.consumption.LLMConsumption
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.*

/**
 * A wrapper class to serialize nullable consumption properly.
 */
@Serializable(with = LLMConsumptionOrNASerializer::class)
public sealed class LLMConsumptionOrNA {
    /** Absence of a consumption value, serialized as `"n/a"`. */
    @Serializable
    public data object NA : LLMConsumptionOrNA()

    /**
     * A present consumption value.
     *
     * @property consumption The wrapped [LLMConsumption].
     */
    @Serializable
    public data class Value(val consumption: LLMConsumption) : LLMConsumptionOrNA()

    /** Factory for wrapping a possibly-null [LLMConsumption]. */
    public companion object {
        /** Returns [Value] when [consumption] is non-null, otherwise [NA]. */
        public fun from(consumption: LLMConsumption?): LLMConsumptionOrNA =
            if (consumption == null) NA else Value(consumption)
    }
}

/** Serializes [LLMConsumptionOrNA] as `"n/a"` for [LLMConsumptionOrNA.NA], or the wrapped consumption otherwise. JSON only. */
public object LLMConsumptionOrNASerializer : KSerializer<LLMConsumptionOrNA> {
    /** String descriptor backing the serialized value. */
    override val descriptor: SerialDescriptor =
        PrimitiveSerialDescriptor("LLMConsumptionOrNA", PrimitiveKind.STRING)

    /** Encodes [value] as `"n/a"` or the delegated consumption JSON. JSON only. */
    override fun serialize(encoder: Encoder, value: LLMConsumptionOrNA) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: throw SerializationException("LLMConsumptionOrNASerializer supports JSON only")

        when (value) {
            LLMConsumptionOrNA.NA -> jsonEncoder.encodeJsonElement(JsonPrimitive("n/a"))
            is LLMConsumptionOrNA.Value -> jsonEncoder.encodeJsonElement(
                jsonEncoder.json.encodeToJsonElement(LLMConsumptionSerializer, value.consumption)
            )
        }
    }

    /** Parses `"n/a"` into [LLMConsumptionOrNA.NA], otherwise into a [LLMConsumptionOrNA.Value]. JSON only. */
    override fun deserialize(decoder: Decoder): LLMConsumptionOrNA {
        val jsonDecoder = decoder as? JsonDecoder
            ?: throw SerializationException("LLMConsumptionOrNASerializer supports JSON only")

        val el = jsonDecoder.decodeJsonElement()
        if (el is JsonPrimitive && el.isString && el.content.equals("n/a", ignoreCase = true)) {
            return LLMConsumptionOrNA.NA
        }

        val c = jsonDecoder.json.decodeFromJsonElement(LLMConsumptionSerializer, el)
        return LLMConsumptionOrNA.Value(c)
    }
}

/**
 * Sums the present consumption values in [consumptions], skipping [LLMConsumptionOrNA.NA] entries.
 * Returns [LLMConsumptionOrNA.NA] when no value is present.
 */
public fun sumConsumptions(consumptions: Iterable<LLMConsumptionOrNA>): LLMConsumptionOrNA {
    val values = consumptions.filterIsInstance<LLMConsumptionOrNA.Value>().map { it.consumption }
    if (values.isEmpty()) return LLMConsumptionOrNA.NA
    return LLMConsumptionOrNA.Value(values.reduce { a, b -> a + b })
}

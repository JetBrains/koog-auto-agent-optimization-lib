package ai.koog.agents.optimization.utils.serialization


import ai.koog.agents.optimization.common.results.ConsumptionWithRetries
import ai.koog.agents.optimization.consumption.parseAmount
import ai.koog.agents.optimization.consumption.toPrettyAmountOrNA
import ai.koog.agents.optimization.training.metrics.impl.ConsumptionMetric
import ai.koog.agents.optimization.training.metrics.impl.SolvedAwareConsumptionMetric
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.descriptors.element
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.*

/**
 * Serializes [ConsumptionMetric] with min / max / avg-per-substage emitted as formatted strings
 * (`"N,NNN.NN tokens"` / `"N,NNN.NN credits"`) carrying the unit from `totalConsumption`. The
 * kotlinx default would emit the aggregates as bare `Double`s without a unit suffix and without
 * thousand separators.
 */
public object ConsumptionMetricSerializer : KSerializer<ConsumptionMetric> {
    /** Class descriptor with the total, per-substage aggregates, and optional `withRetries` fields. */
    override val descriptor: SerialDescriptor =
        buildClassSerialDescriptor("ConsumptionMetric") {
            element("totalConsumption", buildClassSerialDescriptor("totalConsumption"))
            element<String>("minPerSubstage")
            element<String>("maxPerSubstage")
            element<String>("avgPerSubstage")
            element("withRetries", buildClassSerialDescriptor("withRetries"))
        }

    /** Encodes [value] as a JSON object, emitting aggregates as unit-suffixed formatted strings. JSON only. */
    override fun serialize(encoder: Encoder, value: ConsumptionMetric) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: throw SerializationException("ConsumptionMetricSerializer supports JSON only")

        val unit = value.totalOrNull()?.unit  // null when totalConsumption is NA
        val obj = buildJsonObject {
            put(
                "totalConsumption",
                jsonEncoder.json.encodeToJsonElement(LLMConsumptionOrNASerializer, value.totalConsumption)
            )
            put("minPerSubstage", JsonPrimitive(value.minPerSubstage.toPrettyAmountOrNA(unit)))
            put("maxPerSubstage", JsonPrimitive(value.maxPerSubstage.toPrettyAmountOrNA(unit)))
            put("avgPerSubstage", JsonPrimitive(value.avgPerSubstage.toPrettyAmountOrNA(unit)))
            value.withRetries?.let { wr ->
                put("withRetries", buildJsonObject {
                    put(
                        "totalConsumptionWithRetries",
                        jsonEncoder.json.encodeToJsonElement(LLMConsumptionOrNASerializer, wr.totalConsumptionWithRetries)
                    )
                    put(
                        "totalConsumptionInRetries",
                        jsonEncoder.json.encodeToJsonElement(LLMConsumptionOrNASerializer, wr.totalConsumptionInRetries)
                    )
                })
            }
        }
        jsonEncoder.encodeJsonElement(obj)
    }

    /** Parses the JSON object back into a [ConsumptionMetric], reading `n/a` aggregates as `null`. JSON only. */
    override fun deserialize(decoder: Decoder): ConsumptionMetric {
        val jsonDecoder = decoder as? JsonDecoder
            ?: throw SerializationException("ConsumptionMetricSerializer supports JSON only")
        val obj = jsonDecoder.decodeJsonElement().jsonObject

        val totalConsumption = obj["totalConsumption"]?.let {
            jsonDecoder.json.decodeFromJsonElement(LLMConsumptionOrNASerializer, it)
        } ?: LLMConsumptionOrNA.NA

        fun readAggregate(key: String): Double? {
            val s = (obj[key] as? JsonPrimitive)?.content ?: return null
            if (s.equals("n/a", ignoreCase = true)) return null
            return parseAmount(s).first
        }

        val withRetries = (obj["withRetries"] as? JsonObject)?.let { wrObj ->
            ConsumptionWithRetries(
                totalConsumptionWithRetries = wrObj["totalConsumptionWithRetries"]
                    ?.let { jsonDecoder.json.decodeFromJsonElement(LLMConsumptionOrNASerializer, it) }
                    ?: LLMConsumptionOrNA.NA,
                totalConsumptionInRetries = wrObj["totalConsumptionInRetries"]
                    ?.let { jsonDecoder.json.decodeFromJsonElement(LLMConsumptionOrNASerializer, it) }
                    ?: LLMConsumptionOrNA.NA,
            )
        }

        return ConsumptionMetric(
            totalConsumption = totalConsumption,
            minPerSubstage = readAggregate("minPerSubstage"),
            maxPerSubstage = readAggregate("maxPerSubstage"),
            avgPerSubstage = readAggregate("avgPerSubstage"),
            withRetries = withRetries,
        )
    }
}

/**
 * Serializes [SolvedAwareConsumptionMetric] with all per-solved aggregates emitted as
 * formatted strings carrying the unit from `totalSolvedConsumption`. Mirrors
 * [ConsumptionMetricSerializer] for the solved-aware variant attached to
 * `iterateDataset` stages.
 */
public object SolvedAwareConsumptionMetricSerializer : KSerializer<SolvedAwareConsumptionMetric> {
    /** Class descriptor with the total solved consumption, per-solved aggregates, and optional `withRetries` fields. */
    override val descriptor: SerialDescriptor =
        buildClassSerialDescriptor("SolvedAwareConsumptionMetric") {
            element("totalSolvedConsumption", buildClassSerialDescriptor("totalSolvedConsumption"))
            element<String>("avgPerSolved")
            element<String>("minPerSolved")
            element<String>("maxPerSolved")
            element<String>("averageSolvedCost")
            element("withRetries", buildClassSerialDescriptor("withRetries"))
        }

    /** Encodes [value] as a JSON object, emitting aggregates as unit-suffixed formatted strings. JSON only. */
    override fun serialize(encoder: Encoder, value: SolvedAwareConsumptionMetric) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: throw SerializationException("SolvedAwareConsumptionMetricSerializer supports JSON only")

        val unit = (value.totalSolvedConsumption as? LLMConsumptionOrNA.Value)?.consumption?.unit  // null when NA
        val obj = buildJsonObject {
            put(
                "totalSolvedConsumption",
                jsonEncoder.json.encodeToJsonElement(LLMConsumptionOrNASerializer, value.totalSolvedConsumption)
            )
            put("avgPerSolved", JsonPrimitive(value.avgPerSolved.toPrettyAmountOrNA(unit)))
            put("minPerSolved", JsonPrimitive(value.minPerSolved.toPrettyAmountOrNA(unit)))
            put("maxPerSolved", JsonPrimitive(value.maxPerSolved.toPrettyAmountOrNA(unit)))
            put("averageSolvedCost", JsonPrimitive(value.averageSolvedCost.toPrettyAmountOrNA(unit)))
            value.withRetries?.let { wr ->
                put("withRetries", buildJsonObject {
                    put(
                        "totalConsumptionWithRetries",
                        jsonEncoder.json.encodeToJsonElement(LLMConsumptionOrNASerializer, wr.totalConsumptionWithRetries)
                    )
                    put(
                        "totalConsumptionInRetries",
                        jsonEncoder.json.encodeToJsonElement(LLMConsumptionOrNASerializer, wr.totalConsumptionInRetries)
                    )
                })
            }
        }
        jsonEncoder.encodeJsonElement(obj)
    }

    /** Parses the JSON object back into a [SolvedAwareConsumptionMetric], reading `n/a` aggregates as `null`. JSON only. */
    override fun deserialize(decoder: Decoder): SolvedAwareConsumptionMetric {
        val jsonDecoder = decoder as? JsonDecoder
            ?: throw SerializationException("SolvedAwareConsumptionMetricSerializer supports JSON only")
        val obj = jsonDecoder.decodeJsonElement().jsonObject

        val totalSolvedConsumption = obj["totalSolvedConsumption"]?.let {
            jsonDecoder.json.decodeFromJsonElement(LLMConsumptionOrNASerializer, it)
        } ?: LLMConsumptionOrNA.NA

        fun readAggregate(key: String): Double? {
            val s = (obj[key] as? JsonPrimitive)?.content ?: return null
            if (s.equals("n/a", ignoreCase = true)) return null
            return parseAmount(s).first
        }

        val withRetries = (obj["withRetries"] as? JsonObject)?.let { wrObj ->
            ConsumptionWithRetries(
                totalConsumptionWithRetries = wrObj["totalConsumptionWithRetries"]
                    ?.let { jsonDecoder.json.decodeFromJsonElement(LLMConsumptionOrNASerializer, it) }
                    ?: LLMConsumptionOrNA.NA,
                totalConsumptionInRetries = wrObj["totalConsumptionInRetries"]
                    ?.let { jsonDecoder.json.decodeFromJsonElement(LLMConsumptionOrNASerializer, it) }
                    ?: LLMConsumptionOrNA.NA,
            )
        }

        return SolvedAwareConsumptionMetric(
            totalSolvedConsumption = totalSolvedConsumption,
            avgPerSolved = readAggregate("avgPerSolved"),
            minPerSolved = readAggregate("minPerSolved"),
            maxPerSolved = readAggregate("maxPerSolved"),
            averageSolvedCost = readAggregate("averageSolvedCost"),
            withRetries = withRetries,
        )
    }
}

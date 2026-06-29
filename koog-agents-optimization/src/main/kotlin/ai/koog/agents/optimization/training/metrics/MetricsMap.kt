package ai.koog.agents.optimization.training.metrics


import ai.koog.agents.optimization.training.metrics.impl.Metric
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import kotlinx.serialization.KSerializer
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * A mutable, typed wrapper around `Map<MetricKey<*>, Metric>`.
 *
 * **Mutability invariant**: MetricsMap is always updated in-place via [recomputeAll] and [set].
 * It is never replaced wholesale on a [StageRecord] -- the same instance is mutated throughout
 * the stage's lifetime. The [plus] operator creates a new instance (used only during preset
 * composition at stage creation time).
 *
 * Serialized like a plain Map, e.g. `{ "metric_key_id": { ... metric fields ... }, ... }`.
 */
@Serializable(with = MetricsMapSerializer::class)
public class MetricsMap(
    private val map: MutableMap<MetricKey<Metric>, Metric> = mutableMapOf(),
) {
    /** Returns the metric stored under [key], or `null` if absent. */
    @Suppress("UNCHECKED_CAST")
    public operator fun <T : Metric> get(key: MetricKey<T>): T? = map[key as MetricKey<Metric>] as? T

    /** Stores [value] under [key], replacing any existing metric with that key. */
    @Suppress("UNCHECKED_CAST")
    public operator fun <T : Metric> set(key: MetricKey<T>, value: T) {
        map[key as MetricKey<Metric>] = value
    }

    /** The key/metric pairs currently held by this map. */
    public val entries: Set<Map.Entry<MetricKey<*>, Metric>> get() = map.entries

    /** Returns an immutable snapshot of the underlying key-to-metric map. */
    public fun toMap(): Map<MetricKey<Metric>, Metric> = map.toMap()

    /** Returns a new [MetricsMap] containing this map's entries plus [other]'s (the latter wins on key clashes). */
    public operator fun plus(other: MetricsMap): MetricsMap = MetricsMap((map + other.map).toMutableMap())

    /** Recompute all metrics in-place from the given [substages]. */
    public fun recomputeAll(currentStage: StageRecord, substages: List<TrainingRecord>, isStageUpdate: Boolean) {
        for ((key, metric) in map.toMap()) {
            map[key] = metric.recompute(currentStage, substages, isStageUpdate)
        }
    }

    /** Renders the map as `{key_id: metric, ...}`. */
    override fun toString(): String =
        map.entries.joinToString(prefix = "{", postfix = "}") { (k, v) -> "${k.id}: $v" }

    /** Factory helpers for building a [MetricsMap]. */
    public companion object {
        /** Create a [MetricsMap] from a list of metrics, using each metric's own [Metric.key]. */
        public fun of(vararg metrics: Metric): MetricsMap {
            val map = metrics.associateBy { it.key }
            @Suppress("UNCHECKED_CAST")
            return MetricsMap(map.toMutableMap())
        }
    }
}

/** [KSerializer] that encodes a [MetricsMap] as a plain map of metric-key id to polymorphic [Metric]. */
public object MetricsMapSerializer : KSerializer<MetricsMap> {
    private val metricSerializer = PolymorphicSerializer(Metric::class)
    private val delegateSerializer = MapSerializer(MetricKey.serializer(metricSerializer), metricSerializer)

    /** Serial descriptor, delegated to the underlying map serializer. */
    override val descriptor: SerialDescriptor = delegateSerializer.descriptor

    /** Encodes [value] as its underlying key-to-metric map. */
    override fun serialize(encoder: Encoder, value: MetricsMap) {
        encoder.encodeSerializableValue(delegateSerializer, value.toMap())
    }

    /** Decodes a key-to-metric map and wraps it in a [MetricsMap]. */
    override fun deserialize(decoder: Decoder): MetricsMap {
        val delegateMap = decoder.decodeSerializableValue(delegateSerializer)
        return MetricsMap(delegateMap.toMutableMap())
    }
}

package ai.koog.agents.optimization.training.metrics


import kotlinx.serialization.Serializable

/**
 * Typed key for identifying a metric within a [MetricsMap].
 *
 * The type parameter [T] ensures compile-time safety when retrieving metrics:
 * `metricsMap[ai.koog.agents.optimization.training.metrics.impl.ElapsedTimeMetric.KEY]` returns `ElapsedTimeMetric?`.
 */
@JvmInline
@Serializable
public value class MetricKey<out Metric>(
    /** Stable string identifier of the metric, used as the serialized map key. */
    public val id: String,
) {
    /** Returns the metric [id]. */
    override fun toString(): String = id
}

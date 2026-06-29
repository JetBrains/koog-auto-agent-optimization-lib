package ai.koog.agents.optimization.training.metrics.impl


import ai.koog.agents.optimization.training.metrics.MetricKey
import ai.koog.agents.optimization.training.records.StageRecord
import ai.koog.agents.optimization.training.records.TrainingRecord
import kotlinx.serialization.InternalSerializationApi
import kotlinx.serialization.SerializationStrategy
import kotlinx.serialization.Transient
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.serializer

/**
 * A metric that can be computed from substage records.
 *
 * Each concrete metric is a `@Serializable` data class. The core invariant is that
 * metrics calculate **only** from substage records — no incremental updates are passed.
 *
 * Every metric carries its own [key] so that [ai.koog.agents.optimization.training.metrics.MetricsMap]
 * can iterate and recompute all metrics without external key sets.
 */
public interface Metric {
    /** The key that identifies this metric in a [ai.koog.agents.optimization.training.metrics.MetricsMap]. */
    @Transient
    public val key: MetricKey<Metric>

    /**
     * Recompute this metric from the full list of [substages] records.
     *
     * @param currentStage The stage record that owns this metric.
     * @param substages All substages of the stage that owns this metric.
     * @param isStageUpdate `true` when a stage-level event occurred (substage started or finished).
     *   `false` when a leaf event occurred (agent run or prompt execution completed).
     *   Metrics that should only update on significant progress (e.g. [MinSolvedRateMetric])
     *   can skip recalculation when this is `false`.
     * @return A new metric instance with updated values.
     */
    public fun recompute(currentStage: StageRecord, substages: List<TrainingRecord>, isStageUpdate: Boolean): Metric

    /** Serialization support for the polymorphic [Metric] hierarchy. */
    public companion object {
        /**
         * [SerializersModule] that registers a default serializer for every concrete [Metric]
         * subclass, avoiding an explicit listing. Note: deserialization is not yet supported.
         */
        @OptIn(InternalSerializationApi::class)
        public val serializersModule: SerializersModule = SerializersModule {
            // This workaround allows to avoid listing all Metric subclasses explicitly
            polymorphicDefaultSerializer(Metric::class) { instance ->
                @Suppress("UNCHECKED_CAST")
                instance::class.serializer() as SerializationStrategy<Metric>
            }
            // TODO: deserialization will not work at all, but it's not needed yet.
            //   If we ever need it, the correct workaround is to manually list all metric classes here.
            //   Custom metrics will unfortunately need to be listed too.
            //   See https://github.com/Kotlin/kotlinx.serialization/blob/master/docs/polymorphism.md#open-polymorphism
        }
    }
}


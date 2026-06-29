package ai.koog.agents.optimization.training.metrics


import ai.koog.agents.optimization.training.metrics.impl.*

/** Standard metrics for any stage: elapsed time, substage counts, consumption, failures, retry stats. */
public fun standardStageMetrics(): MetricsMap = MetricsMap.of(
    ElapsedTimeMetric(),
    SubstageCountMetric(),
    ConsumptionMetric(),
    FailureBreakdownMetric(),
    RetryStatsMetric(),
)

/** Root stage metrics: standard + cross-stage summary metrics. */
public fun rootStageMetrics(): MetricsMap = standardStageMetrics() + MetricsMap.of(
    MinSolvedRateMetric(),
    MaxFailureRateMetric(),
)

/** Dataset iteration metrics: standard + dataset-specific solved rate + agent run stats + solved-aware consumption. */
public fun datasetIterationMetrics(): MetricsMap = standardStageMetrics() + MetricsMap.of(
    DatasetSolvedRateMetric(),
    AgentRunStatsMetric(),
    SolvedAwareConsumptionMetric(),
)

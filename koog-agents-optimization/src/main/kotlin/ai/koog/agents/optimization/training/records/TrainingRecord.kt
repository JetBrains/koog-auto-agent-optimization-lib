package ai.koog.agents.optimization.training.records


import kotlinx.serialization.Serializable

/**
 * Base class for all records in the training records tree.
 *
 * The tree is formed by [StageRecord] nodes (which have [StageRecord.substages])
 * and leaf records: [AgentRunRecord] and [PromptExecutionRecord].
 */
@Serializable
public sealed class TrainingRecord

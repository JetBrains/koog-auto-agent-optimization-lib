package ai.koog.agents.optimization.optimizers


import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.records.TrainingResult

/**
 * Common contract for agent optimizers: algorithms that improve a [GraphAIAgent] by learning
 * from execution trajectories collected over a training dataset.
 *
 * An implementation runs in two phases. [train] executes the base agent over a dataset, scores the
 * trajectories with a caller-supplied metric, and persists optimizer-specific artifacts to disk.
 * [loadOptimizedAgent] then reads those artifacts and applies them to a base agent.
 *
 * @param Input the agent input type.
 * @param Output the agent output type.
 * @param InputLabel the golden-label type associated with each dataset item (see [TrainSetItem]).
 */
public interface AgentOptimizer<Input, Output, InputLabel> {
    /**
     * Trains the agent using the provided [session], which gives access to a
     * [TrainingSession.use] block where the training scope (agent, dataset,
     * tracked execution methods) is available.
     *
     * **Implementation note:** when writing to the filesystem (artifact storage paths), use
     * `ResilientPath` (and its methods `writeText`, `createParentDirectories`, `deleteIfExists`)
     * instead of bare `java.nio.file.Path` stdlib equivalents. On GCS-fuse mounts in the cluster,
     * bare I/O operations can fail with transient errors that `ResilientPath` handles automatically
     * via retry with exponential backoff.
     *
     * Training produces optimizer-specific artifacts (saved to disk). Use [loadOptimizedAgent]
     * to apply those artifacts to a base agent.
     *
     * @return The [TrainingResult] of the training session -- a wrapper around the root stage
     * record and the execution metadata (pod name / timestamps).
     */
    public suspend fun train(
        session: TrainingSession<Input, Output, InputLabel>,
    ): TrainingResult

    /**
     * Loads training artifacts produced by [train] and applies them to the given [baseAgent].
     *
     * Note: there is an implicit contract to first call [train] and only then call this method.
     * Otherwise, behavior is not defined, but expect some exception, likely FileNotFound.
     *
     * @return An optimized version of [baseAgent].
     */
    public fun loadOptimizedAgent(baseAgent: GraphAIAgent<Input, Output>): GraphAIAgent<Input, Output>
}

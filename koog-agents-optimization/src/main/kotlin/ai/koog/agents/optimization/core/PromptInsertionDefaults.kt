package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.agent.entity.createStorageKey
import kotlinx.serialization.Serializable

/**
 * Default settings for how demonstrations are inserted into prompts.
 *
 * These defaults are set at the feature level via [ai.koog.agents.optimization.features.PromptOptimizationFeature]
 * and propagated to all optimizable subgraphs via agent storage. Individual subgraphs can override
 * these defaults by specifying explicit values in `optimizableSubgraphWithTask`.
 *
 * @property fewShotPromptType How demonstrations are inserted (as a single string or as messages).
 * @property demonstrationFormat Level of detail for each demonstration (compact I/O or full trace).
 */
@Serializable
public data class PromptInsertionDefaults(
    val fewShotPromptType: FewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
    val demonstrationFormat: DemonstrationFormat = DemonstrationFormat.FULL_TRACE,
) {
    /** Holds the agent-storage key under which these defaults are propagated to subgraphs. */
    public companion object {
        /** Storage key for propagating defaults to optimizable subgraphs. */
        public val STORAGE_KEY: AIAgentStorageKey<PromptInsertionDefaults> =
            createStorageKey("optimization-prompt-insertion-defaults")
    }
}

package ai.koog.agents.optimization.optimizers.ace


/**
 * What an [ACEOptimizer] should do when a playbook already exists at the target storage path.
 *
 * - [OVERWRITE] — delete the existing playbook and start fresh.
 * - [OPTIMIZE_FURTHER] — load the existing playbook and keep refining it.
 * - [THROW] — refuse to start, to avoid clobbering prior results.
 */
public enum class OnExistingPlaybookAction {
    /** Delete the existing playbook and start fresh. */
    OVERWRITE,

    /** Load the existing playbook and keep refining it. */
    OPTIMIZE_FURTHER,

    /** Refuse to start, to avoid clobbering prior results. */
    THROW,
}

package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.training.dsl.StageScope
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Proposes child artifacts by reflecting on module-specific feedback from parent rollouts.
 */
internal class GEPACandidateProposer(
    private val reflectionProposer: GEPAReflectionProposer,
    private val modules: List<OptimizableModule>,
) {
    /** Returns the proposed child artifact, or null when no module had feedback on this batch. */
    suspend fun proposeCandidate(
        scope: StageScope<*, *, *>,
        parent: OptimizationArtifact,
        parentRollouts: List<GEPARollout>,
        selectedModule: OptimizableModule? = null,
    ): OptimizationArtifact? {
        val candidate: OptimizationArtifact = if (selectedModule == null) {
            var out = parent
            var updatedModules = 0
            for (module in modules) {
                val moduleInstruction = parent.getModulePrompt(module.name) ?: module.currentInstruction
                val moduleFeedbackResults = feedbackForModule(module.name, parentRollouts)
                if (moduleFeedbackResults.isEmpty()) {
                    logger.warn { "No GEPA feedback for module '${module.name}' on this batch; skipping update." }
                    continue
                }
                val candidateInstruction =
                    reflectionProposer.proposeInstruction(scope, module.name, moduleInstruction, moduleFeedbackResults)
                out = out.withModulePrompt(module.name, candidateInstruction)
                updatedModules++
            }
            if (updatedModules == 0) return null
            out
        } else {
            val moduleInstruction = parent.getModulePrompt(selectedModule.name) ?: selectedModule.currentInstruction
            val moduleFeedbackResults = feedbackForModule(selectedModule.name, parentRollouts)
            if (moduleFeedbackResults.isEmpty()) {
                logger.warn { "No GEPA feedback for module '${selectedModule.name}' on this batch; skipping update." }
                return null
            }
            val candidateInstruction =
                reflectionProposer.proposeInstruction(scope,
                    selectedModule.name, moduleInstruction, moduleFeedbackResults)
            parent.withModulePrompt(selectedModule.name, candidateInstruction)
        }
        return candidate
    }

    private fun feedbackForModule(
        moduleName: String,
        parentRollouts: List<GEPARollout>,
    ): List<GEPAModuleFeedbackResult> =
        parentRollouts.mapNotNull { rollout ->
            (rollout as? GEPARollout.Completed)?.feedback?.feedbackForDeclaredModule(moduleName)
        }
}

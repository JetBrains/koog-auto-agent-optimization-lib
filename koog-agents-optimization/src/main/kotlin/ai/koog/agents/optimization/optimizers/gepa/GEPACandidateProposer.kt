package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.structures.StageFailedException
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlin.coroutines.cancellation.CancellationException

private val logger = KotlinLogging.logger {}

/** How one proposal attempt ended. */
internal sealed interface GEPAProposal {
    /** Reflection produced [candidate] from the parent's rollouts. */
    data class Proposed(val candidate: OptimizationArtifact) : GEPAProposal

    /** No module had feedback on the batch, so there was nothing to reflect on. */
    data object NoModuleFeedback : GEPAProposal

    /** A reflection call still failed after its retries; [failure] is the failed reflection stage. */
    data class ReflectionFailed(val failure: StageFailedException) : GEPAProposal
}

/**
 * Proposes child artifacts by reflecting on module-specific feedback from parent rollouts.
 */
internal class GEPACandidateProposer(
    private val reflectionProposer: GEPAReflectionProposer,
    private val modules: List<OptimizableModule>,
) {
    /**
     * Reflects on [parentRollouts] for [selectedModule], or for every module when it is null. A failed reflection
     * call ends the attempt as [GEPAProposal.ReflectionFailed], even when other modules were already updated, as the
     * reference GEPA implementation drops the whole proposal.
     */
    suspend fun proposeCandidate(
        scope: StageScope<*, *, *>,
        parent: OptimizationArtifact,
        parentRollouts: List<GEPARollout>,
        selectedModule: OptimizableModule? = null,
    ): GEPAProposal {
        var candidate = parent
        var updatedModules = 0
        for (module in selectedModule?.let(::listOf) ?: modules) {
            val moduleInstruction = parent.getModulePrompt(module.name) ?: module.currentInstruction
            val moduleFeedbackResults = feedbackForModule(module.name, parentRollouts)
            if (moduleFeedbackResults.isEmpty()) {
                logger.warn { "No GEPA feedback for module '${module.name}' on this batch; skipping update." }
                continue
            }
            val candidateInstruction = try {
                reflectionProposer.proposeInstruction(scope, module.name, moduleInstruction, moduleFeedbackResults)
            } catch (exception: StageFailedException) {
                if (exception.cause is CancellationException) throw exception
                return GEPAProposal.ReflectionFailed(exception)
            }
            candidate = candidate.withModulePrompt(module.name, candidateInstruction)
            updatedModules++
        }
        return if (updatedModules == 0) GEPAProposal.NoModuleFeedback else GEPAProposal.Proposed(candidate)
    }

    private fun feedbackForModule(
        moduleName: String,
        parentRollouts: List<GEPARollout>,
    ): List<GEPAModuleFeedbackResult> =
        parentRollouts.mapNotNull { rollout ->
            (rollout as? GEPARollout.Completed)?.feedback?.feedbackForDeclaredModule(moduleName)
        }
}

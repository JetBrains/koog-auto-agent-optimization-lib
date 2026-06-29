package ai.koog.agents.optimization.optimizers.ace


import ai.koog.agents.optimization.training.records.TrainingResult

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.optimization.features.collectSubgraphTraces
import ai.koog.prompt.llm.LLModel
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.dsl.executePromptStructuredOrThrow
import ai.koog.agents.optimization.training.dsl.runAgentOrThrow
import ai.koog.agents.optimization.koogTooling.copyWith
import ai.koog.agents.optimization.koogTooling.getCollectedTraces
import ai.koog.agents.optimization.optimizers.AgentOptimizer
import ai.koog.agents.optimization.optimizers.TrainSetItem
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.agents.optimization.utils.common.koogJson
import ai.koog.agents.optimization.utils.common.toFilePathLog
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Agentic Context Engineering optimizer: improves an agent by curating an [ACEPlaybook] of bullet
 * insights instead of editing its prompt directly.
 *
 * For each training item the agent is run and its full trajectory captured; a reflector LLM diagnoses
 * the trajectory into [TrajectoryInsight]s (also tagging which existing bullets helped or hurt), and a
 * curator LLM turns those insights into [DeltaUpdate]s that grow and refine the playbook. The resulting
 * playbook is injected into the agent's system prompt at run time via [ACEPlaybookFeature].
 *
 * @param playbookStoragePath Where the optimized [ACEPlaybook] is read from and written to.
 * @param onExistingPlaybook How to handle a playbook already present at [playbookStoragePath]; see [OnExistingPlaybookAction].
 * @param reflectorModel Model used to diagnose trajectories into insights.
 * @param curatorModel Model used to convert insights into playbook deltas.
 * @param labelExtractor Extracts the ground-truth label string for a dataset item, fed to the reflector.
 */
public class ACEOptimizer<Input, Output, InputLabel>(
    private val playbookStoragePath: ResilientPath,
    private val onExistingPlaybook: OnExistingPlaybookAction,
    private val reflectorModel: LLModel,
    private val curatorModel: LLModel,
    private val labelExtractor: (datasetItem: TrainSetItem<Input, InputLabel>) -> String,
) : AgentOptimizer<Input, Output, InputLabel> {

    private val logger = KotlinLogging.logger { }

    @OptIn(InternalAgentsApi::class)
    override fun loadOptimizedAgent(baseAgent: GraphAIAgent<Input, Output>): GraphAIAgent<Input, Output> {
        val playbook = ACEPlaybook(playbookStoragePath).load()
        return baseAgent.copyWith(installFeatures = {
            baseAgent.installFeatures(this)
            installACEPlaybook { this.playbook = playbook }
        })
    }

    @OptIn(InternalAgentsApi::class)
    override suspend fun train(
        session: TrainingSession<Input, Output, InputLabel>,
    ): TrainingResult = session.use(stagesTotal = 1) {
        // init
        val playbook = ACEPlaybook(playbookStoragePath)
        if (playbookStoragePath.exists()) {
            when (onExistingPlaybook) {
                OnExistingPlaybookAction.OVERWRITE -> {
                    logger.warn { "Overwriting existing ACE playbook at $playbookStoragePath." }
                    playbookStoragePath.deleteIfExists()
                }

                OnExistingPlaybookAction.OPTIMIZE_FURTHER -> {
                    logger.warn { "Using existing playbook at $playbookStoragePath: its content will be modified during further optimization." }
                    playbook.load()
                }

                OnExistingPlaybookAction.THROW -> {
                    throw IllegalArgumentException("Refusing to start optimization: playbook already exists at $playbookStoragePath.")
                }
            }
        } else {
            logger.info { "Creating new ACE playbook at $playbookStoragePath." }
        }

        // train — install ACEPlaybookFeature for playbook injection and
        // SubgraphTraceCollectionFeature for trajectory capture
        val agentWithFeatures = trackedAgent.copyWith(installFeatures = {
            trackedAgent.installFeatures(this)
            installACEPlaybook { this.playbook = playbook }
            collectSubgraphTraces { }
        })

        iterateDataset { item ->
            val agentResult = runAgentOrThrow(item, agentWithFeatures)

            // Get whole-agent trajectory via SubgraphTraceCollectionFeature
            val trajectory = agentResult.usedAgent?.getCollectedTraces()?.getLatestFullPrompt()
                ?: error("SubgraphTraceCollectionFeature did not capture a trajectory")

            // Reflector — single LLM call, captured directly as a leaf PromptExecutionRecord.
            val insights = executePromptStructuredOrThrow<ReflectorResponse>(
                prompt = ACEPrompts.reflectorPrompt(
                    groundTruthLabel = labelExtractor(item),
                    usedPlaybook = playbook,
                    isSolved = agentResult.isSolved,
                    trajectory = trajectory,
                ),
                model = reflectorModel,
            ).insights

            val usedBulletTags = insights.flatMap { it.bulletTags }
            val updatedBullets = playbook.updateCounters(usedBulletTags)

            // Curator — single LLM call.
            val rawDeltaUpdates = executePromptStructuredOrThrow<CuratorResponse>(
                prompt = ACEPrompts.curatorPrompt(insights, playbook),
                model = curatorModel,
            ).operations

            // Update the playbook
            val failedDeltas = rawDeltaUpdates
                .map { it.toDeltaUpdate() }
                .filterNot { playbook.applyDelta(it) }
            playbook.save()

            // Single combined action log on the item stage. koogJson is used so
            // [RawDeltaUpdate]'s sealed-class discriminator survives serialization
            // (defaultExperimentsJson would strip "kind", losing ADD/UPDATE/DELETE).
            logAction(koogJson) {
                put("insightsExtracted", insights.size)
                put("insights", insights)
                with(updatedBullets) {
                    put("bulletsToUpdate", usedBulletTags.size)
                    put("totalUpdated", updatedBullets.totalUpdated)
                    putIfNonEmpty("helpfulBullets", helpfulBullets)
                    putIfNonEmpty("harmfulBullets", harmfulBullets)
                    putIfNonEmpty("neutralBullets", neutralBullets)
                    putIfNonEmpty("notFoundBullets", notFoundBullets)
                    putIfNonEmpty("unknownBulletTags", unknownBulletTags)
                }
                put("deltaUpdatesGenerated", rawDeltaUpdates.size)
                put("deltaUpdates", rawDeltaUpdates)
                put("totalDeltasFailed", failedDeltas.size)
                put("failedDeltas", failedDeltas)
            }
        }

        // Explicit final save guarantees the playbook file exists even if the training
        // loop processed zero items or individual save steps failed.
        playbook.save()

        if (playbookStoragePath.exists()) {
            logger.info { "Optimization completed. Resulting playbook is saved to ${playbookStoragePath.toFilePathLog()}." }
        } else {
            logger.error { "Optimization completed but playbook file was NOT created at $playbookStoragePath." }
        }
    }

    /**
     * Structured reflector output: the [TrajectoryInsight]s diagnosed from a single agent trajectory.
     *
     * @property insights One or more insights extracted from the trajectory.
     */
    @Serializable
    public data class ReflectorResponse(
        val insights: List<TrajectoryInsight>,
    )

    /**
     * A single diagnosis of an agent trajectory produced by the reflector.
     *
     * @property reasoning Chain-of-thought analysis of the trajectory.
     * @property errorIdentification What specifically went wrong in the reasoning.
     * @property rootCauseAnalysis Why the error occurred / which concept was misunderstood.
     * @property correctApproach What the model should have done instead.
     * @property keyInsight Strategy, formula, or principle to remember to avoid the error.
     * @property bulletTags Helpful/harmful/neutral tags for each playbook bullet the trajectory used.
     */
    @Serializable
    @LLMDescription("An insight extracted from an agent trajectory.")
    public data class TrajectoryInsight(
        @property:LLMDescription("Your chain of thought / reasoning / thinking process, detailed analysis and calculations")
        val reasoning: String,
        // TODO: separate prompts for solved trajectories
        @property:LLMDescription("What specifically went wrong in the reasoning?")
        val errorIdentification: String,
        @property:LLMDescription("Why did this error occur? What concept was misunderstood?")
        val rootCauseAnalysis: String,
        @property:LLMDescription("What should the model have done instead?")
        val correctApproach: String,
        @property:LLMDescription("What strategy, formula, or principle should be remembered to avoid this error?")
        val keyInsight: String,
        @property:LLMDescription("A list of json objects with bullet ID and tag for each bulletpoint used by the trajectory")
        val bulletTags: List<BulletTag>,
    )

    /**
     * A reflector's assessment of one playbook bullet's contribution to a trajectory.
     *
     * @property id The [PlaybookBulletItem.id] being assessed.
     * @property tag One of `helpful`, `harmful`, or `neutral`.
     */
    @Serializable
    public data class BulletTag(
        val id: String,
        @property:LLMDescription("One of: helpful | harmful | neutral")
        val tag: String,
    )

    /**
     * Structured curator output: the playbook mutations derived from a batch of reflector insights.
     *
     * @property reasoning Chain-of-thought behind the proposed operations.
     * @property operations Playbook mutations to apply, each a [RawDeltaUpdate].
     */
    @Serializable
    @LLMDescription("The updates to the playbook that should be applied to the agent to improve its performance on the current task.")
    public data class CuratorResponse(
        @property:LLMDescription("Your chain of thought / reasoning / thinking process, detailed analysis and calculations")
        val reasoning: String,
        @property:LLMDescription("A list of updates to be applied to the playbook for future runs")
        val operations: List<RawDeltaUpdate>,
    )

    /**
     * A single playbook mutation as returned by the curator LLM, convertible to a [DeltaUpdate] via [toDeltaUpdate].
     *
     * Distinct from [DeltaUpdate] because it carries the `"kind"` polymorphic discriminator the LLM
     * must emit; see the serialization note below.
     */
    // SERIALIZATION NOTE — "kind" discriminator:
    //   Koog's JsonStructuredData.defaultJson uses classDiscriminator = "kind", so polymorphic
    //   subclasses of this sealed class are identified in JSON by a "kind" field whose value
    //   equals the @SerialName of each subclass ("ADD", "UPDATE", "DELETE").
    //   The field is NOT a Kotlin property — it is added/consumed automatically by kotlinx.serialization.
    //
    //   Consequences:
    //   - Always use JsonStructuredData.defaultJson (or an equivalent Json with classDiscriminator="kind")
    //     for both encoding and decoding. Do NOT use the default kotlinx Json companion, which uses
    //     classDiscriminator = "type" and would produce/expect a different field name.
    //   - The prompt examples in ACEPrompts.curatorPrompt must also reference "kind", not "type",
    //     so the LLM sees consistent instructions.
    //
    // NOTE: without examples this will definitely break! be sure to include them in prompts
    @Serializable
    @SerialName("DeltaUpdate")
    public sealed class RawDeltaUpdate {
        /** Converts this raw update into the corresponding [DeltaUpdate]. */
        public abstract fun toDeltaUpdate(): DeltaUpdate

        /**
         * Raw curator request to add a new bullet.
         *
         * @property content The content of the bullet item to add.
         * @property sectionName The name of the section the bullet should be added to.
         * @property sectionShortName The short name of that section.
         */
        @Serializable
        @SerialName("ADD")
        public data class Add(
            @property:LLMDescription("The content of the bullet item to be added")
            val content: String,
            @property:LLMDescription("The name of the section to which the bullet item should be added")
            val sectionName: String,
            @property:LLMDescription("The short name of the section")
            val sectionShortName: String,
        ) : RawDeltaUpdate() {
            /** Converts this to a [DeltaUpdate.Add]. */
            override fun toDeltaUpdate(): DeltaUpdate = DeltaUpdate.Add(content, sectionName, sectionShortName)
        }

        /**
         * Raw curator request to replace a bullet's content.
         *
         * @property id The ID of the bullet item to update.
         * @property newContent The new content for that bullet.
         */
        @Serializable
        @SerialName("UPDATE")
        public data class Update(
            @property:LLMDescription("The ID of the bullet item to be updated")
            val id: String,
            @property:LLMDescription("The new content of the bullet item")
            val newContent: String,
        ) : RawDeltaUpdate() {
            /** Converts this to a [DeltaUpdate.Update]. */
            override fun toDeltaUpdate(): DeltaUpdate = DeltaUpdate.Update(id, newContent)
        }

        /**
         * Raw curator request to remove a bullet.
         *
         * @property id The ID of the bullet item to delete.
         */
        @Serializable
        @SerialName("DELETE")
        public data class Delete(
            @property:LLMDescription("The ID of the bullet item to be deleted")
            val id: String,
        ) : RawDeltaUpdate() {
            /** Converts this to a [DeltaUpdate.Delete]. */
            override fun toDeltaUpdate(): DeltaUpdate = DeltaUpdate.Delete(id)
        }
    }
}


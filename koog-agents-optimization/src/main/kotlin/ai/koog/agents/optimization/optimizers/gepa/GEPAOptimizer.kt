package ai.koog.agents.optimization.optimizers.gepa

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.optimization.core.OptimizableModule
import ai.koog.agents.optimization.core.OptimizationArtifact
import ai.koog.agents.optimization.core.discoverModules
import ai.koog.agents.optimization.core.initialArtifactOf
import ai.koog.agents.optimization.features.installPromptOptimization
import ai.koog.agents.optimization.koogTooling.copyWith
import ai.koog.agents.optimization.optimizers.AgentOptimizer
import ai.koog.agents.optimization.optimizers.TrainSet
import ai.koog.agents.optimization.training.TrainingSession
import ai.koog.agents.optimization.training.dsl.StageScope
import ai.koog.agents.optimization.training.dsl.runStageOrThrow
import ai.koog.agents.optimization.training.records.TrainingResult
import ai.koog.agents.optimization.utils.common.ResilientPath
import ai.koog.prompt.llm.LLModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlin.random.Random

private val logger = KotlinLogging.logger {}

private fun <T> repeatedShuffled(items: List<T>, random: Random): Sequence<T> = sequence {
    check(items.isNotEmpty()) { "Cannot sample batches from an empty dataset" }

    while (true) {
        yieldAll(items.shuffled(random))
    }
}

/**
 * Strategy for choosing which optimizable modules GEPA updates on each feedback batch.
 */
public enum class GEPAModuleSelectionStrategy {
    /** Update one module per accepted proposal opportunity, cycling through modules per parent candidate. */
    ROUND_ROBIN,

    /** Update every discovered module from the same feedback batch. */
    ALL,
}

/**
 * The two sets GEPA trains on, split off a training set.
 *
 * @property feedbackSet Items GEPA collects module-level textual feedback on (`D_feedback` in the paper).
 * @property validationSet Items an accepted candidate is scored on (`D_pareto` in the paper).
 */
public data class GEPATrainSetSplit<Input, InputLabel>(
    val feedbackSet: TrainSet<Input, InputLabel>,
    val validationSet: TrainSet<Input, InputLabel>,
)

/**
 * GEPA optimizer, based on "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
 * (https://arxiv.org/abs/2507.19457).
 *
 * Training splits the dataset into feedback and validation sets, proposes instruction updates from
 * module-level textual feedback, and keeps candidates that improve over their parent on feedback batches.
 * Accepted candidates are evaluated on the validation set and tracked in a Pareto-style candidate pool.
 *
 * @property storagePath Path where the best optimization artifact is written and later loaded from.
 * @property numRollouts Total rollout budget, including the initial validation-set evaluation. A practical starting point is 6–20
 * times the input training dataset size, matching the hyperparameter choice for runs reported in the GEPA paper.
 * @property failureScore Score assigned to a rollout whose agent execution still fails after retries are exhausted.
 * @property feedbackFailureRateThreshold Maximum failed-item ratio tolerated in parent and candidate feedback batches.
 * @property validationFailureRateThreshold Maximum failed-item ratio tolerated in candidate validation runs.
 * @property seedValidationFailureRateThreshold Maximum failed-item ratio tolerated in the initial seed validation.
 * @property abortOnFailureRateExceeded Whether exceeding a rollout failure-rate threshold aborts the optimization.
 */
public class GEPAOptimizer<Input, Output, InputLabel>(
    private val gepaFeedback: GEPAFeedback<Input, Output, InputLabel>,
    private val feedbackValSplitFn: (TrainSet<Input, InputLabel>) -> GEPATrainSetSplit<Input, InputLabel>,
    private val reflectionModel: LLModel,
    public val storagePath: ResilientPath,
    private val numRollouts: Int,
    private val feedbackBatchSize: Int = 3,
    private val skipPerfectFeedbackBatches: Boolean = true,
    private val perfectScoreThreshold: Double = 1.0,
    private val randomSeed: Long = 42,
    private val moduleSelectionStrategy: GEPAModuleSelectionStrategy = GEPAModuleSelectionStrategy.ROUND_ROBIN,
    private val useStructuredOutput: Boolean = false,
    private val requireThinkingFieldInOutput: Boolean = false,
    private val mergeConfig: GEPAMergeConfig = GEPAMergeConfig(),
    private val failureScore: Double = 0.0,
    private val feedbackFailureRateThreshold: Double = 1.0,
    private val validationFailureRateThreshold: Double = 0.9,
    private val seedValidationFailureRateThreshold: Double = 0.0,
    private val abortOnFailureRateExceeded: Boolean = true,
) : AgentOptimizer<Input, Output, InputLabel> {
    private val reflectionProposer = GEPAReflectionProposer(
        reflectionModel = reflectionModel,
        useStructuredOutput = useStructuredOutput,
        requireThinkingFieldInOutput = requireThinkingFieldInOutput,
    )

    init {
        require(numRollouts > 0) { "numRollouts must be positive" }
        require(feedbackBatchSize > 0) { "feedbackBatchSize must be positive" }
        require(perfectScoreThreshold.isFinite()) { "perfectScore must be finite" }
        require(failureScore.isFinite()) { "failureScore must be finite" }
        require(feedbackFailureRateThreshold in 0.0..1.0) {
            "feedbackFailureRateThreshold must be between 0.0 and 1.0"
        }
        require(validationFailureRateThreshold in 0.0..1.0) {
            "validationFailureRateThreshold must be between 0.0 and 1.0"
        }
        require(seedValidationFailureRateThreshold in 0.0..1.0) {
            "seedValidationFailureRateThreshold must be between 0.0 and 1.0"
        }
        require(!requireThinkingFieldInOutput || useStructuredOutput) {
            "requireThinkingFieldInOutput=true requires useStructuredOutput=true"
        }
    }

    override suspend fun train(session: TrainingSession<Input, Output, InputLabel>): TrainingResult =
        session.use(stagesTotal = 1) {
            runStageOrThrow("GEPA optimization") {
                val random = Random(randomSeed)
                val modules = discoverModules(trackedAgent)
                require(modules.isNotEmpty()) { "No modules found" }

                // Split into feedback and validation sets (D_feedback and D_pareto in the paper).
                val (feedbackSet, valSet) = feedbackValSplitFn(dataset)
                require(feedbackSet.isNotEmpty()) { "feedbackSet must not be empty" }
                require(valSet.isNotEmpty()) { "valSet must not be empty" }
                require(numRollouts > valSet.size) {
                    "numRollouts ($numRollouts) must exceed the initial validation evaluation cost (${valSet.size})"
                }
                logGepaRunSetup(modules, feedbackSetSize = feedbackSet.size, validationSetSize = valSet.size)

                // GEPA runs until the rollout budget is exhausted.
                // Budget is consumed by evaluating candidates on any item in the feedback or validation set.
                // LLM calls for reflection and prompt updates do not count against the budget.
                val budget = RolloutBudget(numRollouts)

                // Initialize candidate pool with the unoptimized prompts.
                val candidatePool = initializeCandidatePool(valSet, random, budget)

                // Proposes new prompt candidates from parent rollouts.
                val candidateProposer = GEPACandidateProposer(reflectionProposer, modules)

                val gepaMerge = if (mergeConfig.enabled) {
                    GEPAMerge(candidatePool, mergeConfig, random)
                } else {
                    null
                }

                // Main optimization loop
                // 1. Sample batch from feedback set
                // 2. Select parent from candidate pool, select module to optimize
                // 3. Evaluate parent on feedback batch
                // 4. Skip the batch if the parent already solves every item
                // 5. Propose new instruction via GEPA's meta reflection and update prompt
                // 6. Evaluate proposed candidate on feedback batch
                // 7. If proposed candidate is better on feedback set, evaluate on validation set and add it to the pool
                var batchIndex = 0
                var mergeAttemptIndex = 0
                val feedbackBatches = repeatedShuffled(feedbackSet, random).chunked(feedbackBatchSize).iterator()
                while (budget.hasRemaining) {
                    // If merge is due, attempt it.
                    if (gepaMerge != null && gepaMerge.shouldAttemptMerge()) {
                        mergeAttemptIndex++
                        runMergeAttempt(mergeAttemptIndex, gepaMerge, candidatePool, valSet, budget)
                        gepaMerge.notifyMergeAttemptFinished()
                        continue
                    }

                    val batch = feedbackBatches.next()
                    batchIndex++
                    runFeedbackBatch(
                        batchIndex = batchIndex,
                        batch = batch,
                        candidatePool = candidatePool,
                        candidateProposer = candidateProposer,
                        modules = modules,
                        valSet = valSet,
                        gepaMerge = gepaMerge,
                        budget = budget,
                    )
                }

                val bestCandidateId = saveBestArtifact(candidatePool)
                logGepaRunTotals(
                    rolloutsUsed = budget.spent,
                    batchesRun = batchIndex,
                    mergeAttempts = mergeAttemptIndex.takeIf { gepaMerge != null },
                    candidatesAdded = candidatePool.size - 1,
                    bestCandidateId = bestCandidateId,
                )
            }
        }

    /** Evaluates the agent's current instructions on [valSet] and seeds the pool with them. */
    private suspend fun StageScope<Input, Output, InputLabel>.initializeCandidatePool(
        valSet: TrainSet<Input, InputLabel>,
        random: Random,
        budget: RolloutBudget,
    ): GEPACandidatePool = runStageOrThrow("Initialize candidate pool") {
        val candidatePool = GEPACandidatePool(valSet.size, random)
        val initialArtifact = initialArtifactOf(trackedAgent)
        val initialCandidateEvaluation = runGepaRollouts(
            candidate = initialArtifact,
            dataset = valSet,
            stageName = "Evaluate initial candidate on validation set",
            gepaFeedback = gepaFeedback,
            failureScore = failureScore,
            failureRateThreshold = seedValidationFailureRateThreshold,
            abortOnFailureRateExceeded = abortOnFailureRateExceeded,
        )
        budget.spend(valSet.size)
        candidatePool.add(initialArtifact, emptyList(), initialCandidateEvaluation.scores)
        logGepaSeedCandidate(
            seedCandidateId = candidatePool.idOf(initialArtifact),
            seedArtifact = initialArtifact,
            averageValidationScore = initialCandidateEvaluation.scores.average(),
        )
        candidatePool
    }

    /**
     * Runs one feedback batch: draws a parent from the pool, proposes a candidate from the parent's
     * rollouts, and adds the candidate to the pool when it improves on the parent.
     */
    private suspend fun StageScope<Input, Output, InputLabel>.runFeedbackBatch(
        batchIndex: Int,
        batch: TrainSet<Input, InputLabel>,
        candidatePool: GEPACandidatePool,
        candidateProposer: GEPACandidateProposer,
        modules: List<OptimizableModule>,
        valSet: TrainSet<Input, InputLabel>,
        gepaMerge: GEPAMerge?,
        budget: RolloutBudget,
    ) {
        runStageOrThrow("Feedback batch $batchIndex") {
            // Sample among Pareto-front representatives, weighted by validation items where they are representative.
            val (parent, frontierWeights) = candidatePool.selectParent()
            logGepaParentSelection(candidatePool.idOf(parent), frontierWeights)

            val parentRollouts = runGepaRollouts(
                candidate = parent,
                dataset = batch,
                stageName = "Evaluate parent on feedback batch $batchIndex",
                gepaFeedback = gepaFeedback,
                failureScore = failureScore,
                failureRateThreshold = feedbackFailureRateThreshold,
                abortOnFailureRateExceeded = abortOnFailureRateExceeded,
            )
            budget.spend(batch.size)
            // Parent failures make it easier for the candidate to improve on the parent
            // This risks spending budget on a bad candidate, which is why we always skip
            if (parentRollouts.hasFailures) {
                logger.warn {
                    "Skipping feedback batch $batchIndex because the parent failed on rollout indices " +
                            "${parentRollouts.failedRolloutIndices.sorted()}."
                }
                logGepaStepOutcome(GEPAStepOutcome.PARENT_ROLLOUTS_FAILED)
                return@runStageOrThrow
            }
            val avgParentScore = parentRollouts.scores.average()
            logGepaParentBatchScore(avgParentScore)

            if (skipPerfectFeedbackBatches && avgParentScore >= perfectScoreThreshold) {
                logger.info {
                    "Skipping feedback batch $batchIndex because the selected parent reached the perfect score."
                }
                logGepaStepOutcome(GEPAStepOutcome.PARENT_ALREADY_PERFECT)
                return@runStageOrThrow
            }

            val selectedModule = when (moduleSelectionStrategy) {
                GEPAModuleSelectionStrategy.ROUND_ROBIN ->
                    modules[candidatePool.advanceNextModuleIndex(parent, modules.size)]

                GEPAModuleSelectionStrategy.ALL -> null
            }
            logGepaSelectedModule(selectedModule)
            val candidate = candidateProposer.proposeCandidate(this, parent, parentRollouts, selectedModule)
            if (candidate == null) {
                logGepaStepOutcome(GEPAStepOutcome.NO_MODULE_FEEDBACK)
                return@runStageOrThrow
            }
            if (candidate in candidatePool) {
                logger.info { "Skipping proposed candidate because it already exists in the candidate pool." }
                logGepaStepOutcome(GEPAStepOutcome.CANDIDATE_ALREADY_IN_POOL)
                return@runStageOrThrow
            }

            val candidateBatchRollouts = runGepaRollouts(
                candidate = candidate,
                dataset = batch,
                stageName = "Evaluate proposed candidate on feedback batch $batchIndex",
                gepaFeedback = gepaFeedback,
                failureScore = failureScore,
                failureRateThreshold = feedbackFailureRateThreshold,
                abortOnFailureRateExceeded = abortOnFailureRateExceeded,
            )
            budget.spend(batch.size)
            val avgCandidateScore = candidateBatchRollouts.scores.average()
            logGepaCandidateBatchScore(avgCandidateScore)

            // Only candidates that improve on the feedback batch are evaluated on the validation set.
            // Failed rollouts get assigned the fallback failureScore
            if (avgCandidateScore > avgParentScore) {
                gepaMerge?.notifySuccessfulReflection()
                val candidateValSetRollouts = runGepaRollouts(
                    candidate = candidate,
                    dataset = valSet,
                    stageName = "Evaluate accepted candidate on validation set for batch $batchIndex",
                    gepaFeedback = gepaFeedback,
                    failureScore = failureScore,
                    failureRateThreshold = validationFailureRateThreshold,
                    abortOnFailureRateExceeded = abortOnFailureRateExceeded,
                )
                candidatePool.add(candidate, listOf(parent), candidateValSetRollouts.scores)
                budget.spend(valSet.size)
                logGepaAcceptedCandidate(
                    candidateId = candidatePool.idOf(candidate),
                    averageValidationScore = candidateValSetRollouts.scores.average(),
                )
                logGepaStepOutcome(GEPAStepOutcome.CANDIDATE_ACCEPTED)
            } else {
                logGepaStepOutcome(GEPAStepOutcome.CANDIDATE_REJECTED)
            }
        }
    }

    /**
     * Attempts one merge: recombines two frontier candidates over their shared ancestor and adds the
     * result to the pool when it holds up against both parents on a validation subsample.
     */
    private suspend fun StageScope<Input, Output, InputLabel>.runMergeAttempt(
        mergeAttemptIndex: Int,
        gepaMerge: GEPAMerge,
        candidatePool: GEPACandidatePool,
        valSet: TrainSet<Input, InputLabel>,
        budget: RolloutBudget,
    ) {
        runStageOrThrow("Merge attempt $mergeAttemptIndex") {
            val proposal = gepaMerge.propose()
            if (proposal == null) {
                logGepaStepOutcome(GEPAStepOutcome.NO_MERGE_PROPOSAL)
                return@runStageOrThrow
            }

            val (firstParent, secondParent) = proposal.parents
            logGepaMergeProposal(
                parentIds = proposal.parents.map { parent -> candidatePool.idOf(parent) },
                ancestorId = candidatePool.idOf(proposal.ancestor),
            )
            val validationSubsampleIndices = candidatePool.selectMergeValidationSubsample(
                firstParent,
                secondParent,
                mergeConfig.valSetSubsampleSize,
            )
            val validationSubsample = validationSubsampleIndices.sorted().map { valSet[it] }

            val subsampleRollouts = runGepaRollouts(
                candidate = proposal.candidate,
                dataset = validationSubsample,
                stageName = "Evaluate merge proposal on validation set subsample " +
                        "for merge attempt $mergeAttemptIndex",
                gepaFeedback = gepaFeedback,
                failureScore = failureScore,
                failureRateThreshold = validationFailureRateThreshold,
                abortOnFailureRateExceeded = abortOnFailureRateExceeded,
            )
            budget.spend(validationSubsample.size)

            val firstParentSubsampleScore =
                candidatePool.sumValidationScores(firstParent, validationSubsampleIndices)
            val secondParentSubsampleScore =
                candidatePool.sumValidationScores(secondParent, validationSubsampleIndices)
            val bestParentSubsampleScore =
                maxOf(firstParentSubsampleScore, secondParentSubsampleScore)
            val subsampleScore = subsampleRollouts.scores.sum()
            logGepaMergeSubsample(validationSubsampleIndices, subsampleScore, bestParentSubsampleScore)

            if (subsampleScore >= bestParentSubsampleScore) {
                val proposalValSetRollouts = runGepaRollouts(
                    candidate = proposal.candidate,
                    dataset = valSet,
                    stageName = "Evaluate merge proposal on validation set for merge attempt $mergeAttemptIndex",
                    gepaFeedback = gepaFeedback,
                    failureScore = failureScore,
                    failureRateThreshold = validationFailureRateThreshold,
                    abortOnFailureRateExceeded = abortOnFailureRateExceeded,
                )
                candidatePool.add(proposal.candidate, proposal.parents, proposalValSetRollouts.scores)
                budget.spend(valSet.size)
                gepaMerge.notifyAcceptedMerge()
                logGepaAcceptedCandidate(
                    candidateId = candidatePool.idOf(proposal.candidate),
                    averageValidationScore = proposalValSetRollouts.scores.average(),
                )
                logGepaStepOutcome(GEPAStepOutcome.CANDIDATE_ACCEPTED)
            } else {
                logGepaStepOutcome(GEPAStepOutcome.CANDIDATE_REJECTED)
            }
        }
    }

    // TODO: support restarting an interrupted optimization. A restart needs GEPA's own search state,
    //  most likely written as a separate output file or directory: every candidate's instructions,
    //  validation scores and parents, plus the rollout budget already spent.
    /** Writes the pool's best candidate to [storagePath] and returns that candidate's id. */
    private suspend fun StageScope<Input, Output, InputLabel>.saveBestArtifact(
        candidatePool: GEPACandidatePool,
    ): String = runStageOrThrow("Save best artifact") {
        val (bestCandidate, bestAverageValidationScore) = candidatePool.getBestCandidate()
        saveArtifact(storagePath, bestCandidate)
        val bestCandidateId = candidatePool.idOf(bestCandidate)
        logGepaBestCandidate(
            bestCandidateId = bestCandidateId,
            bestAverageValidationScore = bestAverageValidationScore,
            candidates = candidatePool.summarizeCandidates(),
            storagePath = storagePath,
        )
        bestCandidateId
    }

    @OptIn(InternalAgentsApi::class)
    override fun loadOptimizedAgent(baseAgent: GraphAIAgent<Input, Output>): GraphAIAgent<Input, Output> {
        val artifact = loadArtifact(storagePath)
        return baseAgent.copyWith(installFeatures = {
            baseAgent.installFeatures(this)
            installPromptOptimization { this.artifact = artifact }
        })
    }


    /** Utilities for reading GEPA artifacts from persistent storage. */
    public companion object {
        private val jsonFormat = Json { ignoreUnknownKeys = true; prettyPrint = true }

        private fun saveArtifact(storagePath: ResilientPath, artifact: OptimizationArtifact) {
            storagePath.createParentDirectories()
            storagePath.writeText(jsonFormat.encodeToString(OptimizationArtifact.serializer(), artifact))
        }

        /**
         * Loads a previously saved [OptimizationArtifact] from [storagePath].
         *
         * @throws IllegalArgumentException when [storagePath] does not exist.
         */
        public fun loadArtifact(storagePath: ResilientPath): OptimizationArtifact {
            if (storagePath.exists()) {
                return jsonFormat.decodeFromString(OptimizationArtifact.serializer(), storagePath.readText())
            } else {
                throw IllegalArgumentException("Cannot load artifact from $storagePath: file does not exist")
            }
        }
    }
}

/** GEPA's rollout budget: how many candidate evaluations a run may spend, and how many it has spent. */
private class RolloutBudget(private val total: Int) {
    var spent: Int = 0
        private set

    val hasRemaining: Boolean get() = spent < total

    fun spend(rollouts: Int) {
        spent += rollouts
    }
}

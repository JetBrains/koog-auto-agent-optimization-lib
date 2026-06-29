package ai.koog.agents.optimization.optimizers.mipro


import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

private const val DEFAULT_BATCH_SIZE = 10
private const val MAX_ITERATIONS = 10
private const val COMPLETE_SKIP_THRESHOLD = 5

/**
 * Creates a summary of the dataset by iteratively observing patterns in batches of examples.
 *
 * The process:
 * 1. Show the first batch to the LLM, ask for observations
 * 2. Show subsequent batches with prior observations, ask for additions or "COMPLETE"
 * 3. Repeat until "COMPLETE" is returned multiple times or max iterations reached
 * 4. Summarize all observations into 2-3 sentences
 *
 * @param renderedExamples Pre-rendered string representations of training examples.
 * @param runMeta Tracked meta-LLM runner (built from a [ai.koog.agents.optimization.training.dsl.StageScope]).
 * @param batchSize Number of examples to show per batch.
 * @return A 2-3 sentence summary of the dataset, or null if summarization fails.
 */
internal suspend fun createDatasetSummary(
    renderedExamples: List<String>,
    runMeta: MetaPromptRunner,
    batchSize: Int = DEFAULT_BATCH_SIZE,
): String? {
    if (renderedExamples.isEmpty()) return null

    logger.info { "Dataset summarizer: observing ${renderedExamples.size} examples in batches of $batchSize..." }
    val firstBatch = renderedExamples.take(batchSize)
    val initialObservations = runMeta(datasetDescriptorPrompt(firstBatch))
    if (initialObservations == null) {
        logger.warn { "Failed to get initial observations from LLM" }
        return null
    }

    var observations = initialObservations
    var skips = 0
    var iterationCount = 0

    for (batchStart in batchSize until renderedExamples.size step batchSize) {
        iterationCount++
        if (iterationCount >= MAX_ITERATIONS) break

        val batchEnd = minOf(batchStart + batchSize, renderedExamples.size)
        val batch = renderedExamples.subList(batchStart, batchEnd)

        val newObservations = runMeta(
            datasetDescriptorWithPriorObservationsPrompt(batch, observations)
        )
        if (newObservations == null) {
            logger.debug { "LLM call failed during refinement at batch $iterationCount, stopping" }
            break
        }

        if (newObservations.trimStart().uppercase().startsWith("COMPLETE")) {
            skips++
            if (skips >= COMPLETE_SKIP_THRESHOLD) break
            continue
        }

        observations += "\n" + newObservations
    }

    logger.info { "Dataset summarizer: summarizing observations..." }
    val summary = runMeta(observationSummarizerPrompt(observations))
    if (summary == null) {
        logger.debug { "Failed to summarize observations, falling back to raw" }
        return observations.take(500).ifBlank { null }
    }

    return summary
}

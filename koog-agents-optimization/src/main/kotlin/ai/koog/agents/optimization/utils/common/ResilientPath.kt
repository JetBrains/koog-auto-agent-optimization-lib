package ai.koog.agents.optimization.utils.common


import io.github.oshai.kotlinlogging.KotlinLogging
import java.io.IOException
import java.nio.file.Path
import kotlin.io.path.*

/**
 * A wrapper around [Path] whose write operations retry on transient I/O failures.
 *
 * Intended for paths that may live on a GCS-fuse mount, where `close()`-time uploads can fail with
 * transient errors (e.g. "Stale file handle"). Read operations delegate directly to [Path]; mutating
 * operations ([writeText], [createDirectories], [deleteRecursively], etc.) wrap the call in an
 * exponential-backoff retry loop. See [writeText] for the full rationale.
 */
public class ResilientPath(
    private val path: Path,
) {
    /** Constructs a [ResilientPath] from a string path, parsed via [Path.of]. */
    public constructor(path: String) : this(Path.of(path))

    /** The normalized, absolute string form of this path. */
    public val absolutePathString: String = path.normalize().absolutePathString()

    /** Whether the underlying path exists on the filesystem. */
    public fun exists(): Boolean = path.exists()

    /** Reads the entire file content as text. */
    public fun readText(): String = path.readText()

    /** Whether the path's file name has a `.json` extension (case-insensitive). */
    public fun isJsonFile(): Boolean = path.name.lowercase().endsWith(".json")

    /** Returns [absolutePathString]. */
    override fun toString(): String = absolutePathString

    /** Resolves [other] against this path, returning a new [ResilientPath]. */
    public fun resolve(other: String): ResilientPath = ResilientPath(path.resolve(other))

    /** Resolves [other] against this path, returning a new [ResilientPath]. */
    public fun resolve(other: ResilientPath): ResilientPath = ResilientPath(path.resolve(other.path))

    /** Path-join operator, equivalent to [resolve]. */
    public operator fun div(other: String): ResilientPath = this.resolve(other)

    /** Path-join operator, equivalent to [resolve]. */
    public operator fun div(other: ResilientPath): ResilientPath = this.resolve(other)

    /** Returns the wrapped [Path], bypassing the retry behavior; avoid unless strictly necessary. */
    @Deprecated("Do not use unless absolutely necessary!", ReplaceWith("this"))
    public fun getPath(): Path = path

    /**
     * Writes [text] to this path with automatic retry on transient I/O failures.
     *
     * On GCS-fuse mounts, `Path.writeText()` can fail with "Stale file handle"
     * (`ESTALE`) during the `close()` call, which triggers the actual GCS upload.
     * This happens when:
     * - Multiple pods concurrently write to different subdirectories of the same
     *   GCS-fuse mount, causing metadata/generation conflicts on shared parent
     *   directory objects in GCS.
     * - The GCS-fuse sidecar experiences transient memory pressure.
     * - Stat cache serves stale generation numbers.
     *
     * Since the error is transient, retrying after a short delay reliably resolves it.
     *
     * @param maxRetries Maximum number of retry attempts (default: [WriteWithRetryConstants.DEFAULT_MAX_RETRIES]).
     * @param initialDelayMs Initial delay before the first retry in milliseconds
     *   (default: [WriteWithRetryConstants.DEFAULT_INITIAL_DELAY_MS]).
     *   Each subsequent retry doubles the delay (exponential backoff).
     * @throws IOException if all retry attempts are exhausted.
     */
    public fun writeText(
        text: String,
        maxRetries: Int = DEFAULT_MAX_RETRIES,
        initialDelayMs: Long = DEFAULT_INITIAL_DELAY_MS,
    ) {
        writeWithRetry(
            writeAction = { writeText(text) },
            contextDescription = "writeText to $this",
            maxRetries = maxRetries,
            initialDelayMs = initialDelayMs,
        )
    }

    /** Retry-wrapped [createParentDirectories]. See [writeText] for rationale. */
    public fun createParentDirectories(
        maxRetries: Int = DEFAULT_MAX_RETRIES,
        initialDelayMs: Long = DEFAULT_INITIAL_DELAY_MS,
    ) {
        writeWithRetry(
            writeAction = { createParentDirectories() },
            contextDescription = "createParentDirectories for $this",
            maxRetries = maxRetries,
            initialDelayMs = initialDelayMs,
        )
    }

    /** Retry-wrapped directory creation (including this path). See [writeText] for rationale. */
    public fun createDirectories(
        maxRetries: Int = DEFAULT_MAX_RETRIES,
        initialDelayMs: Long = DEFAULT_INITIAL_DELAY_MS,
    ) {
        writeWithRetry(
            writeAction = { createDirectories() },
            contextDescription = "createDirectories for $this",
            maxRetries = maxRetries,
            initialDelayMs = initialDelayMs,
        )
    }

    /** Retry-wrapped recursive delete. See [writeText] for rationale. */
    @OptIn(ExperimentalPathApi::class)
    public fun deleteRecursively(
        maxRetries: Int = DEFAULT_MAX_RETRIES,
        initialDelayMs: Long = DEFAULT_INITIAL_DELAY_MS,
    ) {
        writeWithRetry(
            writeAction = { deleteRecursively() },
            contextDescription = "deleteRecursively for $this",
            maxRetries = maxRetries,
            initialDelayMs = initialDelayMs,
        )
    }

    /** Retry-wrapped [deleteIfExists]. See [writeText] for rationale. */
    public fun deleteIfExists(
        maxRetries: Int = DEFAULT_MAX_RETRIES,
        initialDelayMs: Long = DEFAULT_INITIAL_DELAY_MS,
    ) {
        writeWithRetry(
            writeAction = { deleteIfExists() },
            contextDescription = "deleteIfExists $this",
            maxRetries = maxRetries,
            initialDelayMs = initialDelayMs,
        )
    }

    /** Factory helpers for constructing [ResilientPath] instances. */
    public companion object {

        /** Builds a [ResilientPath] from path segments, joined via [Path.of]. */
        public fun of(first: String, vararg more: String): ResilientPath {
            return ResilientPath(Path.of(first, *more))
        }

        private const val DEFAULT_MAX_RETRIES = 3
        private const val DEFAULT_INITIAL_DELAY_MS = 1000L
    }

    private val logger = KotlinLogging.logger {}

    /**
     * Core retry loop for I/O write operations.
     *
     * Retries on **any** [IOException]. On a GCS-fuse mount, virtually all IOExceptions
     * are either transient (metadata race, sidecar restart — retrying helps) or fatal
     * (mount gone due to node eviction — retrying is harmless, just delays the inevitable
     * by a few seconds). The cost of a missed retry (failed experiment, hours of LLM credits
     * wasted) far exceeds the cost of a false retry (a few seconds of delay).
     *
     * Uses [Thread.sleep] rather than coroutine `delay()` because several callers
     * (`ACEPlaybook.save()`, `ReasoningBank.save()`, `saveAgentState()`,
     * `saveOptimizedPrompt()`) are non-suspend functions. Making them all suspend
     * would propagate the modifier through ~6 function signatures for zero practical
     * benefit — the retry fires only on rare transient errors, blocking a thread for
     * 1–4 seconds during a multi-hour experiment is negligible.
     *
     * Extracted from [writeText] so the retry logic can be tested
     * independently of the real filesystem (by injecting a [writeAction] that
     * simulates failures).
     *
     * @param writeAction The write operation to attempt.
     * @param contextDescription Human-readable description for log messages (e.g., the file path).
     * @param maxRetries Maximum number of retry attempts.
     * @param initialDelayMs Initial backoff delay; doubles on each subsequent retry.
     */
    private fun writeWithRetry(
        writeAction: Path.() -> Unit,
        contextDescription: String,
        maxRetries: Int = DEFAULT_MAX_RETRIES,
        initialDelayMs: Long = DEFAULT_INITIAL_DELAY_MS,
    ) {
        require(maxRetries >= 0) { "'maxRetries' must be non-negative, got: $maxRetries" }
        require(initialDelayMs > 0) { "'initialDelayMs' must be positive, got: $initialDelayMs" }

        val totalAttempts = 1 + maxRetries

        for (attempt in 1..totalAttempts) {
            try {
                path.writeAction()
                return
            } catch (e: IOException) {
                if (attempt < totalAttempts) {
                    val delayMs = initialDelayMs * (1L shl (attempt - 1)) // exponential backoff
                    logger.warn {
                        "I/O error writing to $contextDescription " +
                                "(attempt $attempt/$totalAttempts): ${e.message}. " +
                                "Retrying in ${delayMs}ms..."
                    }
                    Thread.sleep(delayMs)
                } else {
                    throw e
                }
            }
        }
        error("Unreachable: the loop always returns on success or throws on the last failed attempt")
    }
}

/** Wraps this [Path] in a [ResilientPath], adding retry-on-failure semantics to its write operations. */
public fun Path.asResilientPath(): ResilientPath = ResilientPath(this)

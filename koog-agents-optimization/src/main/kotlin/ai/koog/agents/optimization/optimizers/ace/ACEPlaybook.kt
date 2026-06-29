package ai.koog.agents.optimization.optimizers.ace


import ai.koog.agents.optimization.utils.common.ResilientPath
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * A single curated insight in an [ACEPlaybook], living inside a [PlaybookSection].
 *
 * @property content The advice text shown to the agent in its system prompt.
 * @property id Stable identifier of the form `"<sectionShortName>-<number>"`, used by the
 * reflector and curator to reference this bullet.
 * @property helpfulCount Number of trajectories in which this bullet was tagged `"helpful"`.
 * @property harmfulCount Number of trajectories in which this bullet was tagged `"harmful"`.
 */
@Serializable
public data class PlaybookBulletItem(
    var content: String,
    val id: String,
    var helpfulCount: Int,
    var harmfulCount: Int,
) {
    /** Factory helpers for [PlaybookBulletItem]. */
    public companion object {
        // IDs follow the format "text-5"
        /**
         * Generates the next unused bullet [id] for [section], one greater than the highest
         * existing bullet number (or `1` if the section is empty).
         */
        public fun generateNewId(section: PlaybookSection): String = with(section) {
            val bulletNumber = bullets.maxOfOrNull { b ->
                b.id.split("-").last().toInt() + 1
            } ?: 1
            "$shortName-$bulletNumber"
        }
    }

    /**
     * Creates a bullet whose [id] is freshly allocated within [section] via [generateNewId].
     */
    public constructor(
        content: String,
        section: PlaybookSection,
        helpfulCount: Int,
        harmfulCount: Int,
    ) : this(content, generateNewId(section), helpfulCount, harmfulCount)
}

/**
 * A named group of [PlaybookBulletItem]s within an [ACEPlaybook].
 *
 * @property name Human-readable section title shown to the agent.
 * @property shortName Compact identifier used as the prefix of contained bullet [PlaybookBulletItem.id]s.
 * @property bullets The bullets in this section, mutated in place as deltas are applied.
 */
@Serializable
public data class PlaybookSection(
    val name: String,
    // TODO: maybe not ask LLM for this?
    val shortName: String,
    val bullets: MutableList<PlaybookBulletItem>,
)

/**
 * A single mutation to apply to an [ACEPlaybook], as produced by the curator and consumed by
 * [ACEPlaybook.applyDelta].
 */
@Serializable
public sealed class DeltaUpdate {
    /**
     * Adds a bullet with [content] to the section identified by [sectionShortName], creating it (named [sectionName]) if absent.
     *
     * @property content Text of the new bullet.
     * @property sectionName Human-readable name of the target section.
     * @property sectionShortName Short identifier of the target section.
     */
    public data class Add(val content: String, val sectionName: String, val sectionShortName: String) : DeltaUpdate()

    /**
     * Replaces the content of the bullet with [id] by [newContent].
     *
     * @property id Identifier of the bullet to update.
     * @property newContent Replacement text.
     */
    public data class Update(val id: String, val newContent: String) : DeltaUpdate()

    /**
     * Removes the bullet with [id] from the playbook.
     *
     * @property id Identifier of the bullet to delete.
     */
    public data class Delete(val id: String) : DeltaUpdate()
}

/**
 * Mutable collection of curated [PlaybookSection]s persisted to [storagePath].
 *
 * The playbook is the artifact optimized by [ACEOptimizer]: insights are accumulated as bullets,
 * refined via [DeltaUpdate]s, and injected into the agent's system prompt at run time. Backed by a
 * JSON file that is loaded with [load] and persisted with [save].
 *
 * @property storagePath File location the playbook is read from and written to.
 */
public class ACEPlaybook(
    private val storagePath: ResilientPath,
) {
    // TODO: add deduplication using embeddings (lazy vs not lazy)
    // TODO: also deduplicate sections

    private val logger = KotlinLogging.logger { }

    private val sections = mutableListOf<PlaybookSection>()
    private val json = Json { ignoreUnknownKeys = true; prettyPrint = true }

    /**
     * Reads the playbook contents from [storagePath] into this instance and returns it.
     *
     * @throws IllegalArgumentException if no file exists at [storagePath].
     */
    public fun load(): ACEPlaybook {
        if (storagePath.exists()) {
            val fileContent = storagePath.readText()
            sections.addAll(json.decodeFromString(fileContent))
        } else {
            throw IllegalArgumentException("Cannot load playbook from $storagePath: file does not exist")
        }
        return this
    }

    /** Persists the current sections to [storagePath] as pretty-printed JSON, creating parent directories as needed. */
    public fun save() {
        storagePath.createParentDirectories()
        storagePath.writeText(json.encodeToString(sections))
    }

    /**
     * Returns true if [delta] was applied; false otherwise.
     */
    public fun applyDelta(delta: DeltaUpdate): Boolean {
        when (delta) {
            is DeltaUpdate.Add -> {
                val targetSection = if (sections.any { it.shortName == delta.sectionShortName }) {
                    sections.first { it.shortName == delta.sectionShortName }
                } else {
                    val newSection = PlaybookSection(delta.sectionName, delta.sectionShortName, mutableListOf())
                    assert(sections.none { it.shortName == delta.sectionShortName })
                    assert(sections.none { it.name == delta.sectionName })
                    sections.add(newSection)
                    newSection
                }
                targetSection.bullets.add(PlaybookBulletItem(delta.content, targetSection, 0, 0))
            }

            is DeltaUpdate.Update -> {
                val bulletItem = sections.flatMap { it.bullets }.find { it.id == delta.id }
                if (bulletItem == null) {
                    logger.warn { "Cannot apply update: bullet with id ${delta.id} not found" }
                    return false
                }
                bulletItem.content = delta.newContent
            }

            is DeltaUpdate.Delete -> {
                val bulletItem = sections.flatMap { it.bullets }.find { it.id == delta.id }
                var deleteCount = 0
                for (section in sections) {
                    val deleted = section.bullets.remove(bulletItem)
                    if (deleted) deleteCount++
                }
                if (deleteCount != 1) logger.warn { "Delta-delete: removed $deleteCount bullets instead of 1" }
            }
        }

        return true
    }

    /**
     * Outcome of an [updateCounters] call, partitioning the processed bullet tags by how they were handled.
     *
     * @property helpfulBullets IDs of bullets whose [PlaybookBulletItem.helpfulCount] was incremented.
     * @property harmfulBullets IDs of bullets whose [PlaybookBulletItem.harmfulCount] was incremented.
     * @property neutralBullets IDs of bullets tagged neutral (no counter changed).
     * @property notFoundBullets Tagged IDs that matched no existing bullet.
     * @property unknownBulletTags Tags whose value was none of `helpful`/`harmful`/`neutral`.
     */
    public data class UpdatedBullets(
        val helpfulBullets: List<String>,
        val harmfulBullets: List<String>,
        val neutralBullets: List<String>,
        val notFoundBullets: List<String>,
        val unknownBulletTags: List<ACEOptimizer.BulletTag>,
    ) {
        /** Total number of bullets actually accounted for: helpful, harmful, and neutral combined. */
        val totalUpdated: Int get() = helpfulBullets.size + harmfulBullets.size + neutralBullets.size
    }

    /**
     * Applies [bulletTags] to the playbook, incrementing each referenced bullet's helpful/harmful counter,
     * and returns a breakdown of what happened as [UpdatedBullets].
     */
    public fun updateCounters(bulletTags: List<ACEOptimizer.BulletTag>): UpdatedBullets {
        val helpfulBullets = mutableListOf<String>()
        val harmfulBullets = mutableListOf<String>()
        val neutralBullets = mutableListOf<String>()
        val notFoundBullets = mutableListOf<String>()
        val unknownBullets = mutableListOf<ACEOptimizer.BulletTag>()

        for (bulletTag in bulletTags) {
            val (bulletId, tag) = bulletTag
            val bulletItem = sections.flatMap { it.bullets }.find { it.id == bulletId }
            if (bulletItem == null) {
                notFoundBullets.add(bulletId)
                logger.warn { "Cannot update counter: bullet with id $bulletId not found" }
                continue
            }
            when (tag) {
                "helpful" -> {
                    helpfulBullets.add(bulletId)
                    bulletItem.helpfulCount++
                }
                "harmful" -> {
                    harmfulBullets.add(bulletId)
                    bulletItem.harmfulCount++
                }
                "neutral" -> {
                    neutralBullets.add(bulletId)
                }
                else -> {
                    unknownBullets.add(bulletTag)
                    logger.warn { "Unknown tag $tag for bullet $bulletId" }
                }
            }
        }
        return UpdatedBullets(
            helpfulBullets = helpfulBullets,
            harmfulBullets = harmfulBullets,
            neutralBullets = neutralBullets,
            notFoundBullets = notFoundBullets,
            unknownBulletTags = unknownBullets,
        )
    }

    /** Renders the playbook as a human-readable, sectioned text block for embedding in an LLM prompt. */
    public fun toPromptRepresentation(): String = StringBuilder().apply {
        appendLine(" ======= PLAYBOOK BEGIN =======\n")
        for (section in sections) {
            appendLine(" === SECTION ${section.name} === \n")
            for (bullet in section.bullets) {
                appendLine("[${bullet.id}]")
                appendLine(bullet.content)
                appendLine()
            }
        }
        appendLine("======= PLAYBOOK END =======")
    }.toString()

    /** Serializes the playbook sections to a JSON string using the provided [json] instance. */
    public fun toJson(json: Json): String = json.encodeToString(sections)
}

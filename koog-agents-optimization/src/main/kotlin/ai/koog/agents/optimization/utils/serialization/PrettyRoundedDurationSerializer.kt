package ai.koog.agents.optimization.utils.serialization


import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlin.time.Duration
import kotlin.time.Duration.Companion.ZERO
import kotlin.time.DurationUnit
import kotlin.time.toDuration

/**
 * Serializes a [Duration] as a second-rounded, human-readable string (e.g. `"1h 2m 3s"`) via [pretty],
 * and parses it back with [parsePrettyDuration]. Sub-second precision is lost on round-trip.
 */
public object PrettyRoundedDurationSerializer : KSerializer<Duration> {
    /** String descriptor backing the serialized [Duration]. */
    override val descriptor: SerialDescriptor =
        PrimitiveSerialDescriptor("PrettyRoundedDuration", PrimitiveKind.STRING)

    /** Encodes [value] using its [pretty] string form. */
    override fun serialize(encoder: Encoder, value: Duration) {
        encoder.encodeString(value.pretty())
    }

    /** Parses a pretty duration string back into a [Duration]. */
    override fun deserialize(decoder: Decoder): Duration {
        val s = decoder.decodeString().trim()
        return parsePrettyDuration(s)
    }
}

/**
 * Formats this duration as a second-rounded string with `h`/`m`/`s` components (e.g. `"1h 2m 3s"`,
 * `"2m 3s"`, `"3s"`), omitting leading zero components. Negative durations are prefixed with `-`,
 * and [Duration.ZERO] renders as `"0s"`.
 */
public fun Duration.pretty(): String {
    if (this == ZERO) return "0s"

    // Round to nearest second
    val rounded = ((this.inWholeMilliseconds + if (this.isNegative()) -500 else 500) / 1000)
        .toDuration(DurationUnit.SECONDS)

    val sign = if (rounded.isNegative()) "-" else ""
    val positive = rounded.absoluteValue

    return positive.toComponents { hours, minutes, seconds, _ ->
        buildString {
            append(sign)
            when {
                hours > 0 -> append("${hours}h ${minutes}m ${seconds}s")
                minutes > 0 -> append("${minutes}m ${seconds}s")
                else -> append("${seconds}s")
            }
        }
    }
}

private val durationStringRegex = Regex("""(-?\d+)\s*([hms])""")

/**
 * Parses a pretty duration string produced by [pretty] (e.g. `"1h 2m 3s"`) back into a [Duration].
 *
 * @throws IllegalStateException if [text] contains no recognizable `h`/`m`/`s` components.
 */
public fun parsePrettyDuration(text: String): Duration {
    if (text == "0s") return ZERO

    var total = ZERO
    val matches = durationStringRegex.findAll(text)

    var found = false
    for (m in matches) {
        found = true
        val amount = m.groupValues[1].toLong()
        val unit = when (m.groupValues[2]) {
            "h" -> DurationUnit.HOURS
            "m" -> DurationUnit.MINUTES
            "s" -> DurationUnit.SECONDS
            else -> error("Unsupported unit")
        }
        total += amount.toDuration(unit)
    }

    if (!found) error("Cannot parse duration: '$text'")
    return total
}

package ai.koog.agents.optimization.utils.common


import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * Represents a ratio of [top] out of [bottom], serialized as a human-readable string
 * like `"3 / 10 (30.00%)"`.
 *
 * @property top Numerator of the ratio.
 * @property bottom Denominator of the ratio; a value of `0` yields a `0%` ratio rather than a division error.
 */
@Serializable(with = RatioSerializer::class)
public data class Ratio(val top: Int, val bottom: Int) {
    /** Renders the ratio as `"top / bottom (percent%)"`, e.g. `"3 / 10 (30.00%)"`. */
    override fun toString(): String {
        val percent = if (bottom == 0) 0.0 else top.toDouble() / bottom * 100
        return "%d / %d (%.2f%%)".format(top, bottom, percent)
    }

    /** The ratio as a fraction in `[0, 1]`, or `0.0` when [bottom] is `0`. */
    val fraction: Double get() = if (bottom == 0) 0.0 else top.toDouble() / bottom

    /** Predefined [Ratio] constants. */
    public companion object {
        /** The empty ratio `0 / 0`, treated as `0%`. */
        public val ZERO: Ratio = Ratio(0, 0)
    }
}

/** Serializes [Ratio] as its [toString][Ratio.toString] string form (e.g. `"3 / 10 (30.00%)"`). */
public object RatioSerializer : KSerializer<Ratio> {
    /** String descriptor backing the serialized [Ratio]. */
    override val descriptor: SerialDescriptor =
        PrimitiveSerialDescriptor("Ratio", PrimitiveKind.STRING)

    /** Encodes [value] using its string form. */
    override fun serialize(encoder: Encoder, value: Ratio) {
        encoder.encodeString(value.toString())
    }

    /** Parses a `"top / bottom (percent%)"` string back into a [Ratio], falling back to [Ratio.ZERO] when malformed. */
    override fun deserialize(decoder: Decoder): Ratio {
        val str = decoder.decodeString()
        // Parse "3 / 10 (30.00%)"
        val parts = str.split("/", "(", "%", ")").map { it.trim() }.filter { it.isNotEmpty() }
        return if (parts.size >= 2) {
            Ratio(parts[0].toInt(), parts[1].toInt())
        } else {
            Ratio.ZERO
        }
    }
}

package ai.koog.agents.optimization.consumption

import java.util.Locale

/**
 * Format an amount as `"N,NNN.NN <shortLabel>"` (e.g. `"12,345.67 tokens"`), or `"n/a"` for null.
 *
 * `Locale.ROOT` pins the thousand-separator to `,` and the decimal mark to `.` regardless of the
 * JVM's default locale — the produced string is consumed by the cluster Python parser
 * (`parseAmount` plus the cluster's `format_consumption`), and a `12.345,67` from a German JVM
 * would break round-trips.
 */
public fun formatAmount(value: Double?, unit: LLMConsumptionUnit): String =
    if (value != null) "%,.2f %s".format(Locale.ROOT, value, unit.shortLabel) else "n/a"

/** Extension form of [formatAmount]. */
public fun Double?.toPrettyAmount(unit: LLMConsumptionUnit): String = formatAmount(this, unit)

/**
 * Format a finite, non-null amount as `"N,NNN.NN <unit>"`, or `"n/a"` for null,
 * non-finite, or null-unit inputs. Null unit reads as "n/a" because there's no
 * sensible label to attach.
 */
public fun Double?.toPrettyAmountOrNA(unit: LLMConsumptionUnit?): String =
    if (this != null && unit != null && this.isFinite()) formatAmount(this, unit) else "n/a"

// Accepts either separator-less "12345.67 tokens" (legacy) or grouped "12,345.67 tokens".
// The unit suffix is matched generically and resolved through the registry, so the library
// carries no hardcoded provider-specific unit names.
private val amountRegex = Regex(
    """^\s*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$""",
    RegexOption.IGNORE_CASE,
)

/**
 * Parse a formatted amount like `"12,345.67 tokens"` into its value and resolved unit.
 * The unit suffix is resolved through [LLMConsumptionRegistry], so any registered flavor parses.
 */
public fun parseAmount(text: String): Pair<Double, LLMConsumptionUnit> {
    val m = amountRegex.matchEntire(text)
        ?: throw IllegalArgumentException(
            "Cannot parse amount '$text'. Expected like '12,345.67 tokens' or '0.50 credits'.",
        )
    val value = m.groupValues[1].replace(",", "").toDouble()
    val unit = LLMConsumptionRegistry.unitForShortLabel(m.groupValues[2])
    return value to unit
}

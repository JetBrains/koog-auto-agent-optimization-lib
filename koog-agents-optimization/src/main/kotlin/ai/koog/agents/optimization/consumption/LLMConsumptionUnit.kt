package ai.koog.agents.optimization.consumption

/**
 * The unit a particular [LLMConsumption] flavor is measured in (e.g. tokens, credits).
 *
 * This is an open interface rather than a closed enum so that providers living outside the
 * library can contribute their own units. A unit is identified by its [displayLabel], which is
 * also the value written to the `"unit"` field of serialized consumption JSON.
 *
 * Implementations are expected to be singletons (objects) so that identity comparison
 * (`unit == SomeUnit`) is meaningful.
 *
 * Both [shortLabel] and [displayLabel] must be globally unique across all providers on the
 * classpath: they are registry keys, and [LLMConsumptionRegistry] fails loudly on a collision.
 */
public interface LLMConsumptionUnit {
    /** Short suffix appended to formatted amounts, e.g. `"tokens"` / `"credits"`. Must be globally unique. */
    public val shortLabel: String

    /** Long label written to the top-level `"unit"` field, e.g. `"LiteLLM tokens"`. Must be globally unique. */
    public val displayLabel: String
}

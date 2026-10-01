package ai.koog.agents.optimization.common


import ai.koog.agents.optimization.optimizers.TrainSetItem

/**
 * Caller-supplied serializers that render dataset items and agent outputs to strings for
 * persistence and logging. Lets the infrastructure record execution data without knowing the
 * concrete `Input`/`Output`/`InputLabel` types.
 */
public data class DatasetExecutionSerializers<Input, Output, InputLabel>(
    /** Renders a [TrainSetItem] in full, for the persisted records. */
    public val serializeItem: (item: TrainSetItem<Input, InputLabel>) -> String,

    /** Renders an agent output to a string. */
    public val serializeOutput: (output: Output) -> String,

    /**
     * Renders a [TrainSetItem] as a short label, used wherever an item is named in a log line or a
     * stage name. An identifier is the usual choice.
     *
     * Keep the label on one line and short: a label is rendered once per item, and a dataset of a few
     * hundred items then fills the log with them. Also, avoid backticks: a log line that embeds a label
     * wraps it in backticks itself.
     *
     * Defaults to [serializeItem]; override it when [serializeItem] renders more than an identifier.
     */
    public val describeItem: (item: TrainSetItem<Input, InputLabel>) -> String = serializeItem,
)

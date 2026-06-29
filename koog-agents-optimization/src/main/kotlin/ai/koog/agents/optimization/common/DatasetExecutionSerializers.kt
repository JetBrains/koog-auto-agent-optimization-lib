package ai.koog.agents.optimization.common


import ai.koog.agents.optimization.optimizers.TrainSetItem

/**
 * Caller-supplied serializers that render dataset items and agent outputs to strings for
 * persistence and logging. Lets the infrastructure record execution data without knowing the
 * concrete `Input`/`Output`/`InputLabel` types.
 */
public data class DatasetExecutionSerializers<Input, Output, InputLabel>(
    /** Renders a [TrainSetItem] (input plus its label) to a string. */
    public val serializeItem: (item: TrainSetItem<Input, InputLabel>) -> String,
    /** Renders an agent output to a string. */
    public val serializeOutput: (output: Output) -> String,
)
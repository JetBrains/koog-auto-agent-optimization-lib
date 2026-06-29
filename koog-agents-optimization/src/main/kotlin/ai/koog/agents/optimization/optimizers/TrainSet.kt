package ai.koog.agents.optimization.optimizers

/**
 * A training dataset: an ordered list of [TrainSetItem]s an optimizer runs the agent on.
 */
public typealias TrainSet<AgentInput, ItemLabel> = List<TrainSetItem<AgentInput, ItemLabel>>

/**
 * An optimizer requires dataset items to run an agent on them and collect
 * execution trajectories. DatasetItems consist of a userQuery, which is being
 * passed to the Agent, and a label, which is the golden answer. The one using
 * an optimizer is in charge of providing an optimizer with a relevant metric
 * that can evaluate the produced result over the golden label.
 *
 * @property userQuery the input passed to the agent for this dataset item.
 * @property itemLabel the golden answer this item is evaluated against.
 */
public data class TrainSetItem<AgentInput, ItemLabel>(
    public val userQuery: AgentInput,
    public val itemLabel: ItemLabel,
)

package ai.koog.agents.optimization.annotations

/**
 * Marks the **stage-authoring DSL** of this library — the `StageScope` builder, its `run*`/`execute*`
 * extensions and helper records, `StageScopeImpl`, `TrainingDsl`, and `ActionLogBuilder`. This is the
 * surface you touch only when composing an optimizer's own training stages.
 *
 * Everything else an optimizer author uses is plain-public and needs no opt-in: `AgentOptimizer`,
 * `TrainingSession`, the records, metrics, and the abort framework, as well as the `core`, `features`,
 * and `consumption` packages and `optimizers.TrainSet`. Code behind this marker is public but
 * **subject to change** — opt in deliberately with `@OptIn(OptimizationExtensionApi::class)` (or a
 * build-wide opt-in) when writing stage logic against `StageScope`.
 */
@RequiresOptIn(
    message = "This is part of the stage-authoring DSL and may change between releases. " +
        "Opt in with @OptIn(OptimizationExtensionApi::class).",
    level = RequiresOptIn.Level.WARNING,
)
@Retention(AnnotationRetention.BINARY)
@Target(
    AnnotationTarget.CLASS,
    AnnotationTarget.FUNCTION,
    AnnotationTarget.PROPERTY,
)
public annotation class OptimizationExtensionApi

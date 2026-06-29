package ai.koog.agents.optimization.annotations

/**
 * Marks the **optimizer-authoring SPI** of this library: [ai.koog.agents.optimization.optimizers.AgentOptimizer],
 * the training session and DSL, records, metrics, and the abort framework.
 *
 * Code that merely *makes an agent optimizable* or *reads results* (the `core`, `features`, and
 * `consumption` packages plus `optimizers.TrainSet`) is plain-public and needs no opt-in. Everything
 * behind this marker is public but **subject to change** — opt in deliberately with
 * `@OptIn(OptimizationExtensionApi::class)` (or a build-wide opt-in) when authoring an optimizer.
 */
@RequiresOptIn(
    message = "This is part of the optimizer-authoring SPI and may change between releases. " +
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

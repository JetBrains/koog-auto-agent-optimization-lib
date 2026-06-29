package ai.koog.agents.optimization.utils.common


/**
 * Formats this path as a clickable `file://` URI for log output, or `file://<file not specified>`
 * when the receiver is `null`.
 */
public fun ResilientPath?.toFilePathLog(): String =
    "file://${this?.absolutePathString ?: "<file not specified>"}"

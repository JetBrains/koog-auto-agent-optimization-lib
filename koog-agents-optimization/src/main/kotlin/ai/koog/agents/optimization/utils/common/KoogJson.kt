package ai.koog.agents.optimization.utils.common


import kotlinx.serialization.json.Json

/**
 * Lenient [Json] instance for parsing LLM-produced JSON (e.g. ACE playbook deltas).
 *
 * Tolerates unknown keys and relaxed syntax, since model output is not guaranteed
 * to match the target schema exactly.
 */
public val koogJson: Json = Json { ignoreUnknownKeys = true; isLenient = true }

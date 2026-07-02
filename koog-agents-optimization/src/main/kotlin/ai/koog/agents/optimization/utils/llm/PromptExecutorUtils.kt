package ai.koog.agents.optimization.utils.llm


import ai.koog.agents.optimization.utils.common.mapError
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.executor.model.executeStructured
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.structure.StructuredRequest
import ai.koog.prompt.structure.StructuredRequestConfig
import ai.koog.prompt.structure.json.JsonStructure
import ai.koog.prompt.structure.json.generator.StandardJsonSchemaGenerator

/**
 * Executes [prompt] against [model], requesting a structured response of type [OutputStructT], and
 * returns the parsed data.
 *
 * The JSON schema for [OutputStructT] is generated automatically via [StandardJsonSchemaGenerator].
 *
 * @throws RuntimeException wrapping the underlying cause if the structured call or parsing fails.
 */
// TODO: find a way to support model temperature here
public suspend inline fun <reified OutputStructT> PromptExecutor.executeStructuredOrThrow(
    prompt: Prompt,
    model: LLModel,
): OutputStructT {
    val outputStructure = JsonStructure.create<OutputStructT>(
        schemaGenerator = StandardJsonSchemaGenerator,
    )
    val structuredSchema = StructuredRequestConfig(
        default = StructuredRequest.Manual(outputStructure),
    )
    return executeStructured(prompt, model, structuredSchema)
        .mapError { RuntimeException("Structured LLM call failed", it) }
        .getOrThrow().data
}

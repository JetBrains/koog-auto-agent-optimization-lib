package ai.koog.agents.optimization.utils.common


/**
 * Transforms the exception wrapped in [this] Result, if it is an exception.
 * Otherwise, preserves the same Result.
 */
public inline fun <T> Result<T>.mapError(transform: (e: Throwable) -> Throwable): Result<T> {
    val e = exceptionOrNull()
    return if (e == null) this else Result.failure(transform(e))
}

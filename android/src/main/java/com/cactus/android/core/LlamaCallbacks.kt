package com.cactus.android

/**
 * Callback interface for receiving model loading progress updates.
 */
fun interface LoadProgressCallback {
    /**
     * Called periodically during model loading.
     * @param progress Percentage progress (0-100).
     * @return true to continue loading, false to interrupt.
     */
    fun onProgress(progress: Int): Boolean
}

/**
 * Callback interface for receiving partial completion results during inference.
 */
fun interface PartialCompletionCallback {
    /**
     * Called when a new chunk of text is generated.
     * @param partialResult A map containing the generated token chunk ("token": String)
     *                      and optionally probability info ("completion_probabilities": List<Map<String, Any?>>).
     *                      It's recommended to use data classes for parsing this map.
     */
    fun onPartialCompletion(partialResult: Map<String, Any?>)
}

/**
 * Callback interface for receiving native log messages.
 */
fun interface LogCallback {
    /**
     * Called when a log message is emitted from the native code.
     * @param level Log level ("ERROR", "WARN", "INFO", "DEBUG").
     * @param message The log message content.
     */
    fun onNativeLog(level: String, message: String)
} 
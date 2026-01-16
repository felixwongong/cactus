package com.cactus

expect class Cactus : AutoCloseable {
    companion object {
        fun create(modelPath: String, corpusDir: String? = null): Cactus
        fun setTelemetryToken(token: String)
        fun setProKey(key: String)
    }

    fun complete(prompt: String, options: CompletionOptions = CompletionOptions()): CompletionResult
    fun complete(
        messages: List<Message>,
        options: CompletionOptions = CompletionOptions(),
        tools: List<Map<String, Any>>? = null,
        callback: TokenCallback? = null
    ): CompletionResult

    fun transcribe(
        audioPath: String,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult

    fun transcribe(
        pcmData: ByteArray,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult

    fun embed(text: String, normalize: Boolean = true): FloatArray
    fun ragQuery(query: String, topK: Int = 5): String
    fun reset()
    fun stop()
    override fun close()
}

data class Message(val role: String, val content: String) {
    companion object {
        fun system(content: String) = Message("system", content)
        fun user(content: String) = Message("user", content)
        fun assistant(content: String) = Message("assistant", content)
    }
}

data class CompletionOptions(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val maxTokens: Int = 512,
    val stopSequences: List<String> = emptyList(),
    val confidenceThreshold: Float = 0f
)

data class CompletionResult(
    val text: String,
    val functionCalls: List<Map<String, Any>>?,
    val promptTokens: Int,
    val completionTokens: Int,
    val timeToFirstToken: Double,
    val totalTime: Double,
    val prefillTokensPerSecond: Double,
    val decodeTokensPerSecond: Double,
    val confidence: Double,
    val needsCloudHandoff: Boolean
)

data class TranscriptionResult(
    val text: String,
    val segments: List<Map<String, Any>>?,
    val totalTime: Double
)

fun interface TokenCallback {
    fun onToken(token: String, tokenId: Int)
}

class CactusException(message: String) : Exception(message)

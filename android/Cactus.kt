package com.cactus

import org.json.JSONArray
import org.json.JSONObject
import java.io.Closeable

class Cactus private constructor(private var handle: Long) : Closeable {

    companion object {
        init {
            System.loadLibrary("cactus")
        }

        @JvmStatic
        fun create(modelPath: String, corpusDir: String? = null): Cactus {
            val handle = nativeInit(modelPath, corpusDir)
            if (handle == 0L) {
                throw CactusException(nativeGetLastError().ifEmpty { "Failed to initialize model" })
            }
            return Cactus(handle)
        }

        @JvmStatic
        fun setTelemetryToken(token: String) = nativeSetTelemetryToken(token)

        @JvmStatic
        fun setProKey(key: String) = nativeSetProKey(key)

        @JvmStatic
        private external fun nativeInit(modelPath: String, corpusDir: String?): Long
        @JvmStatic
        private external fun nativeGetLastError(): String
        @JvmStatic
        private external fun nativeSetTelemetryToken(token: String)
        @JvmStatic
        private external fun nativeSetProKey(key: String)
    }

    data class Message(val role: String, val content: String) {
        companion object {
            fun system(content: String) = Message("system", content)
            fun user(content: String) = Message("user", content)
            fun assistant(content: String) = Message("assistant", content)
        }

        fun toJson(): JSONObject = JSONObject().apply {
            put("role", role)
            put("content", content)
        }
    }

    data class CompletionOptions(
        val temperature: Float = 0.7f,
        val topP: Float = 0.9f,
        val topK: Int = 40,
        val maxTokens: Int = 512,
        val stopSequences: List<String> = emptyList(),
        val confidenceThreshold: Float = 0f
    ) {
        fun toJson(): String = JSONObject().apply {
            put("temperature", temperature)
            put("top_p", topP)
            put("top_k", topK)
            put("max_tokens", maxTokens)
            put("stop", JSONArray(stopSequences))
            put("confidence_threshold", confidenceThreshold)
        }.toString()
    }

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
    ) {
        companion object {
            fun fromJson(json: JSONObject): CompletionResult {
                val functionCalls = json.optJSONArray("function_calls")?.let { arr ->
                    (0 until arr.length()).map { arr.getJSONObject(it).toMap() }
                }
                return CompletionResult(
                    text = json.optString("text", ""),
                    functionCalls = functionCalls,
                    promptTokens = json.optInt("prompt_tokens", 0),
                    completionTokens = json.optInt("completion_tokens", 0),
                    timeToFirstToken = json.optDouble("time_to_first_token_ms", 0.0),
                    totalTime = json.optDouble("total_time_ms", 0.0),
                    prefillTokensPerSecond = json.optDouble("prefill_tokens_per_second", 0.0),
                    decodeTokensPerSecond = json.optDouble("decode_tokens_per_second", 0.0),
                    confidence = json.optDouble("confidence", 1.0),
                    needsCloudHandoff = json.optBoolean("cloud_handoff", false)
                )
            }
        }
    }

    data class TranscriptionResult(
        val text: String,
        val segments: List<Map<String, Any>>?,
        val totalTime: Double
    ) {
        companion object {
            fun fromJson(json: JSONObject): TranscriptionResult {
                val segments = json.optJSONArray("segments")?.let { arr ->
                    (0 until arr.length()).map { arr.getJSONObject(it).toMap() }
                }
                return TranscriptionResult(
                    text = json.optString("text", ""),
                    segments = segments,
                    totalTime = json.optDouble("total_time_ms", 0.0)
                )
            }
        }
    }

    fun interface TokenCallback {
        fun onToken(token: String, tokenId: Int)
    }

    fun complete(prompt: String, options: CompletionOptions = CompletionOptions()): CompletionResult {
        return complete(listOf(Message.user(prompt)), options)
    }

    fun complete(
        messages: List<Message>,
        options: CompletionOptions = CompletionOptions(),
        tools: List<Map<String, Any>>? = null,
        callback: TokenCallback? = null
    ): CompletionResult {
        checkHandle()
        val messagesJson = JSONArray(messages.map { it.toJson() }).toString()
        val toolsJson = tools?.let { JSONArray(it.map { t -> JSONObject(t) }).toString() }

        val responseJson = nativeComplete(handle, messagesJson, options.toJson(), toolsJson, callback)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return CompletionResult.fromJson(json)
    }

    fun transcribe(
        audioPath: String,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult {
        checkHandle()
        val optionsJson = JSONObject().apply {
            language?.let { put("language", it) }
            put("translate", translate)
        }.toString()

        val responseJson = nativeTranscribe(handle, audioPath, prompt, optionsJson, null)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return TranscriptionResult.fromJson(json)
    }

    fun transcribe(
        pcmData: ByteArray,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult {
        checkHandle()
        val optionsJson = JSONObject().apply {
            language?.let { put("language", it) }
            put("translate", translate)
        }.toString()

        val responseJson = nativeTranscribe(handle, null, prompt, optionsJson, pcmData)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return TranscriptionResult.fromJson(json)
    }

    fun embed(text: String, normalize: Boolean = true): FloatArray {
        checkHandle()
        return nativeEmbed(handle, text, normalize)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate embedding" })
    }

    fun ragQuery(query: String, topK: Int = 5): String {
        checkHandle()
        val responseJson = nativeRagQuery(handle, query, topK)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return responseJson
    }

    fun reset() {
        checkHandle()
        nativeReset(handle)
    }

    fun stop() {
        checkHandle()
        nativeStop(handle)
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkHandle() {
        if (handle == 0L) {
            throw CactusException("Model has been closed")
        }
    }

    private external fun nativeDestroy(handle: Long)
    private external fun nativeReset(handle: Long)
    private external fun nativeStop(handle: Long)
    private external fun nativeComplete(
        handle: Long,
        messagesJson: String,
        optionsJson: String?,
        toolsJson: String?,
        callback: TokenCallback?
    ): String
    private external fun nativeTranscribe(
        handle: Long,
        audioPath: String?,
        prompt: String?,
        optionsJson: String?,
        pcmData: ByteArray?
    ): String
    private external fun nativeEmbed(handle: Long, text: String, normalize: Boolean): FloatArray?
    private external fun nativeRagQuery(handle: Long, query: String, topK: Int): String
}

class CactusException(message: String) : Exception(message)

private fun JSONObject.toMap(): Map<String, Any> {
    val map = mutableMapOf<String, Any>()
    keys().forEach { key ->
        map[key] = when (val value = get(key)) {
            is JSONObject -> value.toMap()
            is JSONArray -> value.toList()
            JSONObject.NULL -> Unit
            else -> value
        }
    }
    return map
}

private fun JSONArray.toList(): List<Any> {
    return (0 until length()).map { i ->
        when (val value = get(i)) {
            is JSONObject -> value.toMap()
            is JSONArray -> value.toList()
            JSONObject.NULL -> Unit
            else -> value
        }
    }
}

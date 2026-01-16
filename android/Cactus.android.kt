package com.cactus

import org.json.JSONArray
import org.json.JSONObject

actual class Cactus private constructor(private var handle: Long) : AutoCloseable {

    actual companion object {
        init {
            System.loadLibrary("cactus")
        }

        actual fun create(modelPath: String, corpusDir: String?): Cactus {
            val handle = nativeInit(modelPath, corpusDir)
            if (handle == 0L) {
                throw CactusException(nativeGetLastError().ifEmpty { "Failed to initialize model" })
            }
            return Cactus(handle)
        }

        actual fun setTelemetryToken(token: String) = nativeSetTelemetryToken(token)
        actual fun setProKey(key: String) = nativeSetProKey(key)

        @JvmStatic
        private external fun nativeInit(modelPath: String, corpusDir: String?): Long
        @JvmStatic
        private external fun nativeGetLastError(): String
        @JvmStatic
        private external fun nativeSetTelemetryToken(token: String)
        @JvmStatic
        private external fun nativeSetProKey(key: String)
    }

    actual fun complete(prompt: String, options: CompletionOptions): CompletionResult {
        return complete(listOf(Message.user(prompt)), options, null, null)
    }

    actual fun complete(
        messages: List<Message>,
        options: CompletionOptions,
        tools: List<Map<String, Any>>?,
        callback: TokenCallback?
    ): CompletionResult {
        checkHandle()
        val messagesJson = JSONArray(messages.map { it.toJson() }).toString()
        val optionsJson = options.toJson()
        val toolsJson = tools?.let { JSONArray(it.map { t -> JSONObject(t) }).toString() }

        val responseJson = nativeComplete(handle, messagesJson, optionsJson, toolsJson, callback)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toCompletionResult()
    }

    actual fun transcribe(
        audioPath: String,
        prompt: String?,
        language: String?,
        translate: Boolean
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

        return json.toTranscriptionResult()
    }

    actual fun transcribe(
        pcmData: ByteArray,
        prompt: String?,
        language: String?,
        translate: Boolean
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

        return json.toTranscriptionResult()
    }

    actual fun embed(text: String, normalize: Boolean): FloatArray {
        checkHandle()
        return nativeEmbed(handle, text, normalize)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate embedding" })
    }

    actual fun ragQuery(query: String, topK: Int): String {
        checkHandle()
        val responseJson = nativeRagQuery(handle, query, topK)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return responseJson
    }

    actual fun reset() {
        checkHandle()
        nativeReset(handle)
    }

    actual fun stop() {
        checkHandle()
        nativeStop(handle)
    }

    actual override fun close() {
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
    private external fun nativeComplete(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: TokenCallback?): String
    private external fun nativeTranscribe(handle: Long, audioPath: String?, prompt: String?, optionsJson: String?, pcmData: ByteArray?): String
    private external fun nativeEmbed(handle: Long, text: String, normalize: Boolean): FloatArray?
    private external fun nativeRagQuery(handle: Long, query: String, topK: Int): String
}

private fun Message.toJson(): JSONObject = JSONObject().apply {
    put("role", role)
    put("content", content)
}

private fun CompletionOptions.toJson(): String = JSONObject().apply {
    put("temperature", temperature)
    put("top_p", topP)
    put("top_k", topK)
    put("max_tokens", maxTokens)
    put("stop", JSONArray(stopSequences))
    put("confidence_threshold", confidenceThreshold)
}.toString()

private fun JSONObject.toCompletionResult(): CompletionResult {
    val functionCalls = optJSONArray("function_calls")?.let { arr ->
        (0 until arr.length()).map { arr.getJSONObject(it).toMap() }
    }
    return CompletionResult(
        text = optString("text", ""),
        functionCalls = functionCalls,
        promptTokens = optInt("prompt_tokens", 0),
        completionTokens = optInt("completion_tokens", 0),
        timeToFirstToken = optDouble("time_to_first_token_ms", 0.0),
        totalTime = optDouble("total_time_ms", 0.0),
        prefillTokensPerSecond = optDouble("prefill_tokens_per_second", 0.0),
        decodeTokensPerSecond = optDouble("decode_tokens_per_second", 0.0),
        confidence = optDouble("confidence", 1.0),
        needsCloudHandoff = optBoolean("cloud_handoff", false)
    )
}

private fun JSONObject.toTranscriptionResult(): TranscriptionResult {
    val segments = optJSONArray("segments")?.let { arr ->
        (0 until arr.length()).map { arr.getJSONObject(it).toMap() }
    }
    return TranscriptionResult(
        text = optString("text", ""),
        segments = segments,
        totalTime = optDouble("total_time_ms", 0.0)
    )
}

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

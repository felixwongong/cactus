package com.cactus

import cactus.*
import kotlinx.cinterop.*
import kotlinx.serialization.json.*
import platform.Foundation.NSLog

@OptIn(ExperimentalForeignApi::class)
actual class Cactus private constructor(private var handle: COpaquePointer?) : AutoCloseable {

    actual companion object {
        actual fun create(modelPath: String, corpusDir: String?): Cactus {
            val handle = cactus_init(modelPath, corpusDir)
            if (handle == null) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }
            return Cactus(handle)
        }

        actual fun setTelemetryToken(token: String) {
            cactus_set_telemetry_token(token)
        }

        actual fun setProKey(key: String) {
            cactus_set_pro_key(key)
        }
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
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val messagesJson = serializeMessages(messages)
            val optionsJson = serializeOptions(options)
            val toolsJson = tools?.let { serializeTools(it) }

            val result = cactus_complete(
                handle,
                messagesJson,
                buffer,
                65536u,
                optionsJson,
                toolsJson,
                null,
                null
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            val response = buffer.toKString()
            return parseCompletionResult(response)
        }
    }

    actual fun transcribe(
        audioPath: String,
        prompt: String?,
        language: String?,
        translate: Boolean
    ): TranscriptionResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val optionsJson = buildJsonObject {
                language?.let { put("language", it) }
                put("translate", translate)
            }.toString()

            val result = cactus_transcribe(
                handle,
                audioPath,
                prompt,
                buffer,
                65536u,
                optionsJson,
                null,
                null,
                null,
                0u
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseTranscriptionResult(buffer.toKString())
        }
    }

    actual fun transcribe(
        pcmData: ByteArray,
        prompt: String?,
        language: String?,
        translate: Boolean
    ): TranscriptionResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val optionsJson = buildJsonObject {
                language?.let { put("language", it) }
                put("translate", translate)
            }.toString()

            val pcmPtr = pcmData.refTo(0).getPointer(this)

            val result = cactus_transcribe(
                handle,
                null,
                prompt,
                buffer,
                65536u,
                optionsJson,
                null,
                null,
                pcmPtr.reinterpret(),
                pcmData.size.toULong()
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseTranscriptionResult(buffer.toKString())
        }
    }

    actual fun embed(text: String, normalize: Boolean): FloatArray {
        checkHandle()
        memScoped {
            val buffer = allocArray<FloatVar>(4096)
            val dimPtr = alloc<ULongVar>()

            val result = cactus_embed(
                handle,
                text,
                buffer,
                4096u,
                dimPtr.ptr,
                normalize
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            val dim = dimPtr.value.toInt()
            return FloatArray(dim) { buffer[it] }
        }
    }

    actual fun ragQuery(query: String, topK: Int): String {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)

            val result = cactus_rag_query(
                handle,
                query,
                buffer,
                65536u,
                topK.toULong()
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return buffer.toKString()
        }
    }

    actual fun reset() {
        checkHandle()
        cactus_reset(handle)
    }

    actual fun stop() {
        checkHandle()
        cactus_stop(handle)
    }

    actual override fun close() {
        handle?.let { cactus_destroy(it) }
        handle = null
    }

    private fun checkHandle() {
        if (handle == null) throw CactusException("Model has been closed")
    }

    private fun serializeMessages(messages: List<Message>): String {
        return buildJsonArray {
            messages.forEach { msg ->
                addJsonObject {
                    put("role", msg.role)
                    put("content", msg.content)
                }
            }
        }.toString()
    }

    private fun serializeOptions(options: CompletionOptions): String {
        return buildJsonObject {
            put("temperature", options.temperature)
            put("top_p", options.topP)
            put("top_k", options.topK)
            put("max_tokens", options.maxTokens)
            putJsonArray("stop") { options.stopSequences.forEach { add(it) } }
            put("confidence_threshold", options.confidenceThreshold)
        }.toString()
    }

    private fun serializeTools(tools: List<Map<String, Any>>): String {
        return Json.encodeToString(tools)
    }

    private fun parseCompletionResult(json: String): CompletionResult {
        val obj = Json.parseToJsonElement(json).jsonObject
        return CompletionResult(
            text = obj["text"]?.jsonPrimitive?.contentOrNull ?: "",
            functionCalls = obj["function_calls"]?.jsonArray?.map { it.jsonObject.toMap() },
            promptTokens = obj["prompt_tokens"]?.jsonPrimitive?.intOrNull ?: 0,
            completionTokens = obj["completion_tokens"]?.jsonPrimitive?.intOrNull ?: 0,
            timeToFirstToken = obj["time_to_first_token_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            totalTime = obj["total_time_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            prefillTokensPerSecond = obj["prefill_tokens_per_second"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            decodeTokensPerSecond = obj["decode_tokens_per_second"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            confidence = obj["confidence"]?.jsonPrimitive?.doubleOrNull ?: 1.0,
            needsCloudHandoff = obj["cloud_handoff"]?.jsonPrimitive?.booleanOrNull ?: false
        )
    }

    private fun parseTranscriptionResult(json: String): TranscriptionResult {
        val obj = Json.parseToJsonElement(json).jsonObject
        return TranscriptionResult(
            text = obj["text"]?.jsonPrimitive?.contentOrNull ?: "",
            segments = obj["segments"]?.jsonArray?.map { it.jsonObject.toMap() },
            totalTime = obj["total_time_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0
        )
    }

    private fun JsonObject.toMap(): Map<String, Any> {
        return entries.associate { (k, v) ->
            k to when (v) {
                is JsonPrimitive -> v.contentOrNull ?: v.toString()
                is JsonObject -> v.toMap()
                is JsonArray -> v.map { it.toString() }
                else -> v.toString()
            }
        }
    }
}

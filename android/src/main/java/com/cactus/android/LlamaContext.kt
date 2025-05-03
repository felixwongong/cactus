package com.cactus.android

import android.util.Log
import java.io.Closeable

/**
 * Provides access to the native Cactus/Llama inference engine.
 *
 * Use the companion object methods to initialize contexts and perform operations.
 * Remember to call [close] on the returned [LlamaContext] instance when finished,
 * preferably using a `use` block:
 * ```kotlin
 * LlamaContext.create(initParams).use { context ->
 *     val result = context.complete("Hello", completionParams)
 *     // ...
 * }
 * ```
 */
class LlamaContext private constructor(
    /** Internal pointer to the native context. Do not use directly. */
    public val contextPtr: Long
) : Closeable {

    // Keep track of whether the native resource is already freed
    private var isClosed = false

    // --- Instance Methods (Operating on an existing context) ---

    private fun checkClosed() {
        if (isClosed) {
            throw IllegalStateException("LlamaContext has been closed.")
        }
        if (contextPtr <= 0) {
            throw IllegalStateException("Invalid native context pointer ($contextPtr).")
        }
    }

    /**
     * Checks if this context is currently performing inference.
     */
    fun isPredicting(): Boolean {
        checkClosed()
        return isPredictingNative(contextPtr)
    }

    /**
     * Interrupts any ongoing completion task for this context.
     */
    fun stopCompletion() {
        checkClosed()
        stopCompletionNative(contextPtr)
    }

    /**
     * Retrieves detailed information about the loaded model.
     * @return Parsed model details.
     * @throws IllegalStateException if the context pointer is invalid or native call fails.
     */
    fun getModelDetails(): LlamaModelInfo {
        checkClosed()
        val map = loadModelDetailsNative(contextPtr)
            ?: throw IllegalStateException("Failed to load model details, context invalid or native error.")

        // Use safeGet for basic properties
        val description = map.safeGet("desc", "")
        val sizeBytes = map.safeGet("size", 0.0) // JNI likely returns Double
        val embeddingDim = map.safeGet("nEmbd", 0.0)
        val paramCount = map.safeGet("nParams", 0.0)

        // Handle nested metadata map carefully
        val rawMetadata = map.safeGet<Map<*,*>>("metadata", emptyMap<Any, Any>())
        val metadata = rawMetadata.mapNotNull { (key, value) ->
            if (key is String && value is String) {
                key to value
            } else {
                Log.w(TAG, "Skipping invalid metadata entry: key type=${key?.javaClass?.simpleName}, value type=${value?.javaClass?.simpleName}")
                null
            }
        }.toMap()

        // TODO: Parse chatTemplates map fully - Requires structure definition or more info
        val chatTemplates = ChatTemplateInfo(false, MinjaTemplateInfo(false, false, null, null)) // Placeholder remains

        return LlamaModelInfo(
            description = description,
            sizeBytes = sizeBytes,
            embeddingDim = embeddingDim,
            paramCount = paramCount,
            metadata = metadata,
            chatTemplates = chatTemplates
        )
    }

    /**
     * Formats a chat history using the model's Jinja template capabilities.
     *
     * @param messagesJson JSON string representing the list of messages (e.g., `[{"role": "user", "content": "..."}]`).
     * @param chatTemplate Optional custom Jinja template override.
     * @param jsonSchema Optional JSON schema for function/tool calling grammar generation.
     * @param toolsJson Optional JSON string describing available tools (OpenAI format).
     * @param parallelToolCalls Allow model to request multiple tool calls simultaneously.
     * @param toolChoice Force the model to use a specific tool: "none", "auto", or JSON like `{"type": "function", "function": {"name": "..."}}`.
     * @return Parsed result containing the formatted prompt, grammar, and other parameters.
     * @throws IllegalStateException if the context pointer is invalid.
     * @throws RuntimeException if formatting fails in native code.
     */
    fun getFormattedChatWithJinja(
        messagesJson: String,
        chatTemplate: String? = null,
        jsonSchema: String? = null,
        toolsJson: String? = null,
        parallelToolCalls: Boolean = false,
        toolChoice: String? = null
    ): FormattedChatResult {
        checkClosed()
        val map = getFormattedChatWithJinjaNative(
            contextPtr,
            messagesJson,
            chatTemplate ?: "", // Pass empty string if null
            jsonSchema ?: "",
            toolsJson ?: "",
            parallelToolCalls,
            toolChoice ?: ""
        ) ?: throw IllegalStateException("Failed to format chat with Jinja, context invalid or native error occurred.")

        // Use safeGet and handle nested lists
        val rawTriggersList = map.safeGet<List<*>>("grammar_triggers", emptyList<Any>())
        val triggersList = rawTriggersList.mapNotNull { triggerMap ->
            if (triggerMap !is Map<*, *>) {
                Log.w(TAG, "Skipping invalid grammar_trigger entry: not a map")
                return@mapNotNull null
            }
            GrammarTrigger(
                type = triggerMap.safeGet("type", -1),
                value = triggerMap.safeGet("value", ""),
                token = triggerMap.safeGet<Number?>("token", null)?.toInt() // Handle potential null and number type
            )
        }

        val rawPreservedList = map.safeGet<List<*>>("preserved_tokens", emptyList<Any>())
        val preservedList = rawPreservedList.mapNotNull { it as? String ?: it?.toString()?.also { Log.w(TAG, "Non-string in preserved_tokens, using toString()") } }

        val rawStopsList = map.safeGet<List<*>>("additional_stops", emptyList<Any>())
        val stopsList = rawStopsList.mapNotNull { it as? String ?: it?.toString()?.also { Log.w(TAG, "Non-string in additional_stops, using toString()") } }

        return FormattedChatResult(
            prompt = map.safeGet("prompt", ""),
            chatFormat = map.safeGet("chat_format", -1),
            grammar = map.safeGet<String?>("grammar", null), // Allow null grammar
            grammarLazy = map.safeGet("grammar_lazy", false),
            grammarTriggers = triggersList,
            preservedTokens = preservedList,
            additionalStops = stopsList
        )
    }

     /**
     * Formats a chat history using legacy (non-Jinja) chat templates.
     *
     * @param messagesJson JSON string representing the list of messages.
     * @param chatTemplate Optional custom template name/identifier.
     * @return The formatted prompt string.
     * @throws IllegalStateException if the context pointer is invalid.
     * @throws RuntimeException if formatting fails in native code.
     */
     fun getFormattedChat(
         messagesJson: String,
         chatTemplate: String? = null
     ): String {
         checkClosed()
         return getFormattedChatNative(
             contextPtr,
             messagesJson,
             chatTemplate ?: ""
         ) ?: throw IllegalStateException("Failed to format chat, context invalid or native error occurred.")
     }

    /**
     * Loads a previously saved session state into the context.
     * @param path The path to the session file.
     * @return Parsed result containing the number of tokens loaded and the reconstructed prompt.
     * @throws IllegalStateException if the context pointer is invalid.
     * @throws java.io.IOException if the file cannot be read.
     * @throws RuntimeException on other native errors.
     */
    fun loadSession(path: String): SessionLoadResult {
        checkClosed()
        val map = loadSessionNative(contextPtr, path)
            ?: throw IllegalStateException("Failed to load session, context invalid or native error occurred.")
        // Use safeGet
        return SessionLoadResult(
            tokensLoaded = map.safeGet("tokens_loaded", 0L), // Expect Long
            prompt = map.safeGet("prompt", "")
        )
    }

    /**
     * Saves the current context state (KV cache) to a file.
     * @param path The path to save the session file.
     * @param size The maximum number of tokens to save (<= 0 means save all).
     * @return The number of tokens actually saved, or -1 on error.
     * @throws IllegalStateException if the context pointer is invalid.
     * @throws java.io.IOException if the file cannot be written.
     * @throws RuntimeException on other native errors.
     */
    fun saveSession(path: String, size: Int = 0): Int {
        checkClosed()
        val result = saveSessionNative(contextPtr, path, size)
        if (result < 0) {
             // Check if an exception was thrown by JNI, otherwise create a generic one
             // Note: JNI exception check isn't possible here directly.
             // Rely on the native code throwing appropriate exceptions for I/O errors.
             throw RuntimeException("Failed to save session (returned $result)")
        }
        return result
    }


    /**
     * Performs text completion based on the provided parameters.
     *
     * @param prompt The input prompt text.
     * @param params Parameters controlling the generation process.
     * @param partialCompletionCallback Optional callback for receiving generated tokens incrementally.
     * @return Parsed result containing the full completion, stats, and timings.
     * @throws IllegalStateException if the context pointer is invalid or completion is already running.
     * @throws RuntimeException on native errors during generation.
     */
    fun complete(
        prompt: String,
        params: LlamaCompletionParams,
        partialCompletionCallback: PartialCompletionCallback? = null
    ): LlamaCompletionResult {
        checkClosed()
        
        // Convert complex parameters to formats expected by JNI
        // Note: These conversions might need more specific JNI helpers depending on complexity.
        val grammarTriggersList = params.grammarTriggers?.map { mapOf("type" to it.type, "value" to it.value, "token" to it.token) } ?: emptyList()
        val preservedTokensList = params.preservedTokens ?: emptyList()
        val stopArray = params.stop?.toTypedArray() ?: emptyArray()
        val logitBiasMap = params.logitBias ?: emptyMap()
        val drySequenceBreakersArray = params.drySequenceBreakers?.toTypedArray() ?: emptyArray()

        val resultMap = doCompletionNative(
            contextPtr,
            prompt,
            params.chatFormat,
            params.grammar ?: "",
            params.grammarLazy,
            grammarTriggersList, // Passing List<Map> directly
            preservedTokensList, // Passing List<String> directly
            params.temperature,
            params.nThreads,
            params.nPredict,
            params.nProbs,
            params.penaltyLastN,
            params.penaltyRepeat,
            params.penaltyFreq,
            params.penaltyPresent,
            params.mirostat,
            params.mirostatTau,
            params.mirostatEta,
            params.topK,
            params.topP,
            params.minP,
            params.xtcThreshold,
            params.xtcProbability,
            params.typicalP,
            params.seed,
            stopArray,
            params.ignoreEos,
            logitBiasMap, // Passing Map<Int, Float> directly
            params.dryMultiplier,
            params.dryBase,
            params.dryAllowedLength,
            params.dryPenaltyLastN,
            params.topNSigma,
            drySequenceBreakersArray,
            partialCompletionCallback
        ) ?: throw IllegalStateException("Completion failed, context invalid or native error occurred.")

        // Use safeGet and handle timings map
        val timingsMap = resultMap.safeGet<Map<*, *>>("timings", emptyMap<Any, Any>())
        val timings = Timings(
            promptN = timingsMap.safeGet("prompt_n", 0),
            promptMs = timingsMap.safeGet("prompt_ms", 0L),
            promptPerTokenMs = timingsMap.safeGet("prompt_per_token_ms", 0.0),
            promptPerSecond = timingsMap.safeGet("prompt_per_second", 0.0),
            predictedN = timingsMap.safeGet("predicted_n", 0),
            predictedMs = timingsMap.safeGet("predicted_ms", 0L),
            predictedPerTokenMs = timingsMap.safeGet("predicted_per_token_ms", 0.0),
            predictedPerSecond = timingsMap.safeGet("predicted_per_second", 0.0)
        )
        
        // TODO: Parse tool calls, probabilities etc. robustly if their structure is known

        return LlamaCompletionResult(
            text = resultMap.safeGet("text", ""),
            content = resultMap.safeGet<String?>("content", null), // Allow null
            reasoningContent = resultMap.safeGet<String?>("reasoning_content", null), // Allow null
            toolCalls = null, // Placeholder
            completionProbabilities = null, // Placeholder
            tokensPredicted = resultMap.safeGet("tokens_predicted", 0),
            tokensEvaluated = resultMap.safeGet("tokens_evaluated", 0),
            truncated = resultMap.safeGet("truncated", false),
            stoppedEos = resultMap.safeGet("stopped_eos", false),
            stoppedWord = resultMap.safeGet("stopped_word", false),
            stoppedLimit = resultMap.safeGet("stopped_limit", false),
            stoppingWord = resultMap.safeGet("stopping_word", ""),
            tokensCached = resultMap.safeGet("tokens_cached", 0),
            timings = timings
        )
    }

    /**
     * Tokenizes the given text using the context's vocabulary.
     * @param text The text to tokenize.
     * @param addBos Whether to prepend the Beginning-Of-Sequence token.
     * @param parseSpecial Whether to parse special tokens (e.g., <|user|>).
     * @return A list of integer token IDs.
     * @throws IllegalStateException if the context pointer is invalid or tokenization fails.
     */
    fun tokenize(text: String, addBos: Boolean = false, parseSpecial: Boolean = false): List<Int> {
        checkClosed()
        return tokenizeNative(contextPtr, text, addBos, parseSpecial)
            ?: throw IllegalStateException("Tokenization failed, context invalid or native error occurred.")
    }

    /**
     * Converts a list of token IDs back into a string.
     * @param tokens The list or array of integer token IDs.
     * @return The reconstructed string.
     * @throws IllegalStateException if the context pointer is invalid or detokenization fails.
     */
    fun detokenize(tokens: IntArray): String {
        checkClosed()
        return detokenizeNative(contextPtr, tokens)
            ?: throw IllegalStateException("Detokenization failed, context invalid or native error occurred.")
    }
    fun detokenize(tokens: List<Int>): String = detokenize(tokens.toIntArray())


    /**
     * Checks if embedding generation was enabled when this context was created.
     * Returns false if the context is invalid.
     */
    fun isEmbeddingEnabled(): Boolean {
        if (isClosed || contextPtr <= 0) return false
        return isEmbeddingEnabledNative(contextPtr)
    }

    /**
     * Generates embeddings for the given text.
     * @param text The text to embed.
     * @param normalize Override the normalization setting (-1 uses context default, 0=off, 1=on).
     * @return Parsed embedding result.
     * @throws IllegalStateException if the context is invalid, embedding was not enabled, or native error occurs.
     */
    fun embedding(text: String, normalize: Int = -1): LlamaEmbeddingResult {
        checkClosed()
        if (!isEmbeddingEnabled()) {
            throw IllegalStateException("Embedding mode not enabled for this context.")
        }
        val map = embeddingNative(contextPtr, text, normalize)
            ?: throw IllegalStateException("Embedding generation failed, context invalid or native error occurred.")
        
        // Use safeGet and handle lists
        val rawEmbeddingList = map.safeGet<List<*>>("embedding", emptyList<Any>())
        val embeddingList = rawEmbeddingList.mapNotNull { 
            (it as? Number)?.toFloat() ?: run {
                Log.w(TAG, "Non-number found in embedding list: $it")
                null
            }
        }
        
        val rawTokensList = map.safeGet<List<*>>("prompt_tokens", emptyList<Any>())
        val tokensList = rawTokensList.mapNotNull { 
            it as? String ?: run {
                Log.w(TAG, "Non-string found in prompt_tokens list: $it")
                it?.toString() // Fallback to toString()
            }
        }

        return LlamaEmbeddingResult(
            embedding = embeddingList,
            promptTokens = tokensList
        )
    }

    /**
     * Runs a benchmark on the loaded model.
     * @param pp Prompt processing tokens count.
     * @param tg Text generation tokens count.
     * @param pl Parallel processing level.
     * @param nr Number of repetitions.
     * @return A JSON string containing benchmark results.
     * @throws IllegalStateException if the context pointer is invalid or benchmark fails.
     */
    fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String {
        checkClosed()
        return benchNative(contextPtr, pp, tg, pl, nr)
            ?: throw IllegalStateException("Benchmarking failed, context invalid or native error occurred.")
    }

    /**
     * Applies one or more LoRA adapters to the model.
     * @param loraAdapters A list of LoRA adapter configurations.
     * @return 0 on success, non-zero on failure.
     * @throws IllegalStateException if the context pointer is invalid.
     * @throws RuntimeException on native error during application.
     */
    fun applyLoraAdapters(loraAdapters: List<LoraAdapterInfo>): Int {
        checkClosed()
        // Convert LoraAdapterInfo data class list to List<Map<String, Any?>> for JNI
        val loraListForJni = loraAdapters.map { mapOf("path" to it.path, "scaled" to it.scale) }
        val result = applyLoraAdaptersNative(contextPtr, loraListForJni)
        if (result != 0) {
            throw RuntimeException("Failed to apply LoRA adapters (native code returned $result)")
        }
        return result
    }

    /**
     * Removes all applied LoRA adapters.
     * @throws IllegalStateException if the context pointer is invalid.
     * @throws RuntimeException on native error during removal.
     */
    fun removeLoraAdapters() {
        checkClosed()
        // Consider if native method can throw, otherwise wrap
        removeLoraAdaptersNative(contextPtr)
    }

    /**
     * Gets a list of currently applied LoRA adapters.
     * @return A list LoRA adapter configurations.
     * @throws IllegalStateException if the context pointer is invalid or native call fails.
     */
    fun getLoadedLoraAdapters(): List<LoraAdapterInfo> {
       checkClosed()
       val listMap = getLoadedLoraAdaptersNative(contextPtr)
           ?: throw IllegalStateException("Failed to get loaded LoRA adapters, context invalid or native error.")
        
        // Parse List<Map<String, Any?>> back to List<LoraAdapterInfo>
        return listMap.mapNotNull { map ->
            val path = map["path"] as? String
            val scale = (map["scaled"] as? Number)?.toFloat() // JNI might return Double
            if (path != null && scale != null) {
                LoraAdapterInfo(path, scale)
            } else {
                Log.w(TAG, "Skipping invalid LoRA info map from native: $map")
                null
            }
        }
    }

    /**
     * Releases the native context resources. Call this when done with the context.
     * Safe to call multiple times.
     */
    override fun close() {
        if (!isClosed && contextPtr > 0) {
            freeContextNative(contextPtr)
            isClosed = true
            // Optionally remove from a global tracking map if needed
        }
    }

    protected fun finalize() {
        // Warn if context wasn't closed explicitly
        if (!isClosed && contextPtr > 0) {
            Log.w(TAG, "LlamaContext was not closed explicitly. Call close() to release native resources.")
            // close() // Optionally force close on finalize, but risks issues if called from wrong thread
        }
    }

    // --- Companion Object (Static Methods and Native Declarations) --- 
    companion object {
        private const val TAG = "LlamaContext"

        // Static initialization block to load the native library
        init {
            try {
                // Try loading optimized versions first, fallback to generic
                // The exact names depend on your CMakeLists.txt `build_library` calls
                val libs = arrayOf(
                    // ARMv8.2 variants (ordered most specific to least)
                    "cactus_v8_2_dotprod_i8mm",
                    "cactus_v8_2_i8mm",
                    "cactus_v8_2_dotprod",
                    "cactus_v8_2",
                    // ARMv8 generic
                    "cactus_v8",
                    // x86_64 variants
                    "cactus_x86_64",
                    // Generic fallback
                    "cactus"
                )
                var loaded = false
                val errors = mutableListOf<String>()
                for (lib in libs) {
                    try {
                        System.loadLibrary(lib)
                        Log.i(TAG, "Successfully loaded native library: lib$lib.so")
                        loaded = true
                        break // Stop trying once one loads
                    } catch (e: UnsatisfiedLinkError) {
                        // This is expected if a specific CPU feature isn't available
                        errors.add("Failed to load library '$lib': ${e.message}")
                    }
                }
                if (!loaded) {
                    Log.e(TAG, "-------------------------------------------------")
                    Log.e(TAG, "FATAL: Could not load any native cactus library variant.")
                    errors.forEach { Log.e(TAG, "- $it") }
                    Log.e(TAG, "Please ensure native libraries for the device architecture are bundled.")
                    Log.e(TAG, "-------------------------------------------------")
                    // Throw an exception to prevent usage if no library loads
                     throw UnsatisfiedLinkError("Failed to load any native cactus library variant. Check Logcat for details.")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Exception loading native library", t)
                // Re-throw if needed, or handle appropriately
                if (t is UnsatisfiedLinkError) throw t
            }
        }

        /**
         * Initializes a new Llama context.
         *
         * @param params Parameters for initializing the context.
         * @param loadProgressCallback Optional callback for load progress updates.
         * @return A [LlamaContext] instance managing the native resources.
         * @throws IllegalArgumentException if required parameters are missing.
         * @throws RuntimeException if context initialization fails in native code.
         */
        @JvmStatic
        fun create(
            params: LlamaInitParams,
            loadProgressCallback: LoadProgressCallback? = null
        ): LlamaContext {
            
            // Convert LoraAdapterInfo list to List<Map<String, Any?>> for JNI
            val loraListForJni = params.loraAdapters?.map { mapOf("path" to it.path, "scaled" to it.scale) } ?: emptyList()

            val contextPtr = initContextNative(
                modelPath = params.modelPath,
                chatTemplate = params.chatTemplate ?: "",
                reasoningFormat = params.reasoningFormat ?: "",
                embedding = params.embedding,
                embdNormalize = params.embdNormalize,
                nCtx = params.nCtx,
                nBatch = params.nBatch,
                nUbatch = params.nUbatch,
                nThreads = params.nThreads,
                nGpuLayers = params.nGpuLayers,
                flashAttn = params.flashAttn,
                cacheTypeK = params.cacheTypeK,
                cacheTypeV = params.cacheTypeV,
                useMlock = params.useMlock,
                useMmap = params.useMmap,
                vocabOnly = params.vocabOnly,
                loraList = loraListForJni,
                ropeFreqBase = params.ropeFreqBase,
                ropeFreqScale = params.ropeFreqScale,
                poolingType = params.poolingType,
                loadProgressCallback = loadProgressCallback
            )

            if (contextPtr == 0L) { // Check only for 0, as valid pointers might cast to negative jlong
                // Check if JNI threw an exception, otherwise create a generic one
                // Note: Cannot directly check for pending JNI exceptions here.
                throw RuntimeException("Failed to initialize native Llama context (returned pointer 0)")
            }
            Log.i(TAG, "Successfully initialized Llama context (ptr=$contextPtr)")
            return LlamaContext(contextPtr)
        }

        /**
         * Retrieves information about a model file without loading the full context.
         * @param modelPath Path to the GGUF model file.
         * @param skipKeys Optional array of metadata keys to skip.
         * @return Parsed model file information, or null if the file is invalid or cannot be read.
         */
        @JvmStatic
        fun getModelFileInfo(modelPath: String, skipKeys: Array<String> = emptyArray()): Map<String, Any?>? {
            // Native function might throw IOException which needs to be caught by caller
            // Or return null on failure
            return try {
                 modelInfoNative(modelPath, skipKeys)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get model file info for $modelPath", e)
                null
            }
        }

        /**
         * Sets up a global callback for receiving native log messages.
         * NOTE: This is a global setting affecting all contexts.
         * @param logCallback The object implementing the LogCallback interface, or null to disable native logging via callback.
         */
        @JvmStatic
        external fun setupLog(logCallback: LogCallback?)

        /**
         * Disables the global native log callback set by [setupLog].
         */
        @JvmStatic
        external fun unsetLog()

        // --- Native external function declarations --- 
        // Intentionally making doCompletionNative public for testing access from app module.
        // Others are internal as they should ideally only be called via the public wrapper methods.

        @JvmStatic
        internal external fun initContextNative(
            modelPath: String,
            chatTemplate: String,
            reasoningFormat: String,
            embedding: Boolean,
            embdNormalize: Int,
            nCtx: Int,
            nBatch: Int,
            nUbatch: Int,
            nThreads: Int,
            nGpuLayers: Int,
            flashAttn: Boolean,
            cacheTypeK: String,
            cacheTypeV: String,
            useMlock: Boolean,
            useMmap: Boolean,
            vocabOnly: Boolean,
            loraList: List<Map<String, Any?>>, // JNI expects jobject (List)
            ropeFreqBase: Float,
            ropeFreqScale: Float,
            poolingType: Int,
            loadProgressCallback: LoadProgressCallback? // JNI expects jobject
        ): Long // Returns context pointer or <= 0 on error

        // TODO: Define interruptLoadNative if needed and possible
        // @JvmStatic
        internal external fun interruptLoadNative(contextPtr: Long)

        @JvmStatic
        internal external fun freeContextNative(contextPtr: Long)

        @JvmStatic
        internal external fun modelInfoNative(modelPath: String, skipKeys: Array<String>): Map<String, Any?>?

        @JvmStatic
        internal external fun loadModelDetailsNative(contextPtr: Long): Map<String, Any?>?

        @JvmStatic
        internal external fun getFormattedChatWithJinjaNative(
            contextPtr: Long,
            messagesJson: String,
            chatTemplate: String,
            jsonSchema: String,
            toolsJson: String,
            parallelToolCalls: Boolean,
            toolChoice: String
        ): Map<String, Any?>? // JNI returns jobject (Map)

        @JvmStatic
        internal external fun getFormattedChatNative(
            contextPtr: Long,
            messagesJson: String,
            chatTemplate: String
        ): String?

        @JvmStatic
        internal external fun loadSessionNative(contextPtr: Long, path: String): Map<String, Any?>? // JNI returns jobject (Map)

        @JvmStatic
        internal external fun saveSessionNative(contextPtr: Long, path: String, size: Int): Int

        @JvmStatic
        public external fun doCompletionNative( // Made public temporarily for testing
            contextPtr: Long,
            prompt: String,
            chatFormat: Int,
            grammar: String,
            grammarLazy: Boolean,
            grammarTriggers: List<Map<String, Any?>>, // JNI expects jobject (List)
            preservedTokens: List<String>,          // JNI expects jobject (List)
            temperature: Float,
            nThreads: Int,
            nPredict: Int,
            nProbs: Int,
            penaltyLastN: Int,
            penaltyRepeat: Float,
            penaltyFreq: Float,
            penaltyPresent: Float,
            mirostat: Float,
            mirostatTau: Float,
            mirostatEta: Float,
            topK: Int,
            topP: Float,
            minP: Float,
            xtcThreshold: Float,
            xtcProbability: Float,
            typicalP: Float,
            seed: Int,
            stop: Array<String>,
            ignoreEos: Boolean,
            logitBias: Map<Int, Float>, // JNI expects jobject (Map)
            dryMultiplier: Float,
            dryBase: Float,
            dryAllowedLength: Int,
            dryPenaltyLastN: Int,
            topNSigma: Float,
            drySequenceBreakers: Array<String>,
            partialCompletionCallback: PartialCompletionCallback? // JNI expects jobject
        ): Map<String, Any?>? // JNI returns jobject (Map)

        @JvmStatic
        internal external fun stopCompletionNative(contextPtr: Long)

        @JvmStatic
        internal external fun isPredictingNative(contextPtr: Long): Boolean

        @JvmStatic
        internal external fun tokenizeNative(contextPtr: Long, text: String, addBos: Boolean, parseSpecial: Boolean): List<Int>? // JNI returns jobject (List)

        @JvmStatic
        internal external fun detokenizeNative(contextPtr: Long, tokens: IntArray): String?

        @JvmStatic
        internal external fun isEmbeddingEnabledNative(contextPtr: Long): Boolean

        @JvmStatic
        internal external fun embeddingNative(contextPtr: Long, text: String, normalize: Int): Map<String, Any?>? // JNI returns jobject (Map)

        @JvmStatic
        internal external fun benchNative(contextPtr: Long, pp: Int, tg: Int, pl: Int, nr: Int): String?

        @JvmStatic
        internal external fun applyLoraAdaptersNative(contextPtr: Long, loraAdapters: List<Map<String, Any?>>): Int // JNI expects jobject (List)

        @JvmStatic
        internal external fun removeLoraAdaptersNative(contextPtr: Long)

        @JvmStatic
        internal external fun getLoadedLoraAdaptersNative(contextPtr: Long): List<Map<String, Any?>>? // JNI returns jobject (List)
    }
}

// Helper extension function for safer map access from JNI results
private fun <T> Map<*, *>?.safeGet(key: String, defaultValue: T): T {
    if (this == null) {
        Log.w("LlamaContext.safeGet", "Attempting to get '$key' from a null map. Returning default.")
        return defaultValue
    }
    if (!this.containsKey(key)) {
        Log.w("LlamaContext.safeGet", "Key '$key' not found in map. Returning default.")
        return defaultValue
    }
    val value = this[key]
    if (value == null) {
        // Allow null default value to be returned if the key exists but value is null
        if (defaultValue == null) {
            @Suppress("UNCHECKED_CAST")
            return null as T
        }
        Log.w("LlamaContext.safeGet", "Value for key '$key' is null. Returning default.")
        return defaultValue
    }
    // Check if the value's type matches the default value's type (simple check)
    // Note: This won't work perfectly for generic types, but covers basic cases like String, Double, Int, Boolean, Map, List
    if (defaultValue != null && value::class != defaultValue::class && !(value is Number && defaultValue is Number)) {
         // Allow casting between number types (e.g., JNI Double to Kotlin Int/Long/Float)
        if (value is Number && defaultValue is Number) {
             // Attempt numeric conversion
            return when (defaultValue) {
                is Int -> value.toInt() as? T ?: defaultValue.also { Log.w("LlamaContext.safeGet", "Failed Number cast for '$key' from ${value::class.simpleName} to Int. Returning default.") }
                is Long -> value.toLong() as? T ?: defaultValue.also { Log.w("LlamaContext.safeGet", "Failed Number cast for '$key' from ${value::class.simpleName} to Long. Returning default.") }
                is Float -> value.toFloat() as? T ?: defaultValue.also { Log.w("LlamaContext.safeGet", "Failed Number cast for '$key' from ${value::class.simpleName} to Float. Returning default.") }
                is Double -> value.toDouble() as? T ?: defaultValue.also { Log.w("LlamaContext.safeGet", "Failed Number cast for '$key' from ${value::class.simpleName} to Double. Returning default.") }
                else -> defaultValue.also { Log.w("LlamaContext.safeGet", "Unsupported Number cast for '$key' to ${defaultValue::class.simpleName}. Returning default.") }
            }
        }
        Log.w("LlamaContext.safeGet", "Type mismatch for key '$key'. Expected ~${defaultValue::class.simpleName}, got ${value::class.simpleName}. Returning default.")
        return defaultValue
    }
    try {
        @Suppress("UNCHECKED_CAST")
        return value as T
    } catch (e: ClassCastException) {
        Log.w("LlamaContext.safeGet", "ClassCastException for key '$key'. Expected ${defaultValue!!::class.simpleName}, got ${value::class.simpleName}. Returning default.", e)
        return defaultValue
    }
} 
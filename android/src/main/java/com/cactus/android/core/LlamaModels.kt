package com.cactus.android

// --- Initialization Parameters --- 

data class LlamaInitParams(
    val modelPath: String,
    val chatTemplate: String? = null,
    val reasoningFormat: String? = null, // e.g., "deepseek"
    val embedding: Boolean = false,
    val embdNormalize: Int = -1, // -1: use model default, 0: off, 1: on
    val nCtx: Int = 2048,
    val nBatch: Int = 512,
    val nUbatch: Int = 512,
    val nThreads: Int = 0, // 0: auto
    val nGpuLayers: Int = 0,
    val flashAttn: Boolean = false,
    val cacheTypeK: String = "f16", // e.g., "f16", "q8_0"
    val cacheTypeV: String = "f16",
    val useMlock: Boolean = false,
    val useMmap: Boolean = true,
    val vocabOnly: Boolean = false,
    val loraAdapters: List<LoraAdapterInfo>? = null, // List of adapters to apply at init
    val ropeFreqBase: Float = 0.0f, // 0: use model default
    val ropeFreqScale: Float = 0.0f, // 0: use model default
    val poolingType: Int = -1 // -1: use model default (e.g., LLAMA_POOLING_TYPE_NONE, LLAMA_POOLING_TYPE_MEAN, LLAMA_POOLING_TYPE_CLS)
)

data class LoraAdapterInfo(
    val path: String,
    val scale: Float = 1.0f
)

// --- Completion Parameters --- 

data class LlamaCompletionParams(
    val chatFormat: Int = 0, // Corresponds to common_chat_format enum
    val grammar: String? = null,
    val grammarLazy: Boolean = false,
    val grammarTriggers: List<GrammarTrigger>? = null,
    val preservedTokens: List<String>? = null, // Strings that should map to single tokens
    val temperature: Float = 0.8f,
    val nThreads: Int = 0, // Override init threads (0 = use init setting)
    val nPredict: Int = -1, // Max tokens to generate (-1 = infinite / until stop)
    val nProbs: Int = 0, // Number of top probabilities to return
    val penaltyLastN: Int = 64,
    val penaltyRepeat: Float = 1.1f,
    val penaltyFreq: Float = 0.0f,
    val penaltyPresent: Float = 0.0f,
    val mirostat: Float = 0.0f, // 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0
    val mirostatTau: Float = 5.0f,
    val mirostatEta: Float = 0.1f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val xtcThreshold: Float = 0.0f, // Experimental: X-Factor Temperature Control
    val xtcProbability: Float = 0.0f,
    val typicalP: Float = 1.0f,
    val seed: Int = -1, // -1 = random seed
    val stop: List<String>? = null,
    val ignoreEos: Boolean = false,
    val logitBias: Map<Int, Float>? = null, // Map<TokenID, Bias>
    val dryMultiplier: Float = 0.0f, // Experimental: Dynamic Repetition Yield
    val dryBase: Float = 1.5f,
    val dryAllowedLength: Int = 256,
    val dryPenaltyLastN: Int = 64,
    val topNSigma: Float = 0.0f, // Experimental
    val drySequenceBreakers: List<String>? = null
)

data class GrammarTrigger(
    val type: Int, // Corresponds to common_grammar_trigger_type enum (WORD, TOKEN, etc.)
    val value: String,
    val token: Int? = null // Only relevant if type is TOKEN
)

// --- Completion Result --- 

data class LlamaCompletionResult(
    val text: String,
    val content: String? = null, // Parsed content if using chat format
    val reasoningContent: String? = null, // Parsed reasoning if available
    val toolCalls: List<ToolCall>? = null, // Parsed tool calls
    val completionProbabilities: List<TokenProbability>? = null, // Overall probabilities
    val tokensPredicted: Int,
    val tokensEvaluated: Int,
    val truncated: Boolean,
    val stoppedEos: Boolean,
    val stoppedWord: Boolean,
    val stoppedLimit: Boolean,
    val stoppingWord: String,
    val tokensCached: Int,
    val timings: Timings
)

data class ToolCall(
    val type: String, // e.g., "function"
    val id: String?, // Optional ID for the tool call
    val function: ToolFunction
)

data class ToolFunction(
    val name: String,
    val arguments: String // JSON string
)

data class TokenProbability(
    val content: String, // String representation of the predicted token
    val probs: List<TokenProbDetail> // List of probabilities for alternative tokens
)

data class TokenProbDetail(
    val tokStr: String, // String representation of the alternative token
    val prob: Double
)

data class Timings(
    val promptN: Int,
    val promptMs: Long,
    val promptPerTokenMs: Double,
    val promptPerSecond: Double,
    val predictedN: Int,
    val predictedMs: Long,
    val predictedPerTokenMs: Double,
    val predictedPerSecond: Double
)

// --- Embedding Result --- 

data class LlamaEmbeddingResult(
    val embedding: List<Float>,
    val promptTokens: List<String> // Tokens used to generate the embedding
)

// --- Model Info --- 

data class LlamaModelInfo(
    val description: String,
    val sizeBytes: Double, // Consider Long if size exceeds Double precision needs
    val embeddingDim: Double,
    val paramCount: Double, // Consider Long
    val metadata: Map<String, String>,
    val chatTemplates: ChatTemplateInfo
)

data class ChatTemplateInfo(
    val isLegacySupported: Boolean,
    val minja: MinjaTemplateInfo
)

data class MinjaTemplateInfo(
    val hasDefault: Boolean,
    val hasToolUse: Boolean,
    val defaultCaps: TemplateCapabilities?,
    val toolUseCaps: TemplateCapabilities?
)

data class TemplateCapabilities(
    val tools: Boolean,
    val toolCalls: Boolean,
    val parallelToolCalls: Boolean,
    val toolResponses: Boolean,
    val systemRole: Boolean,
    val toolCallId: Boolean
)

// --- Session Load Result ---

data class SessionLoadResult(
    val tokensLoaded: Long,
    val prompt: String
)

// --- Formatted Chat Result (Jinja) ---

data class FormattedChatResult(
    val prompt: String,
    val chatFormat: Int, // Corresponds to common_chat_format
    val grammar: String?,
    val grammarLazy: Boolean,
    val grammarTriggers: List<GrammarTrigger>?,
    val preservedTokens: List<String>?,
    val additionalStops: List<String>?
)
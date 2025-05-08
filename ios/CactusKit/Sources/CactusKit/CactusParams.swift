import Foundation

// MARK: - Enums

/// Specifies the reasoning format hint for certain models (e.g., Deepseek Coder).
public enum ReasoningFormat {
    case none
    case deepseek
}

/// Specifies the pooling strategy for embedding models.
/// Matches `llama_pooling_type` values.
public enum PoolingType: Int32 {
    case unspecified = -1 // LLAMA_POOLING_TYPE_UNSPECIFIED
    case none = 0        // LLAMA_POOLING_TYPE_NONE
    case mean = 1        // LLAMA_POOLING_TYPE_MEAN
    case cls = 2         // LLAMA_POOLING_TYPE_CLS
}

/// Specifies the data type for the key-value cache.
/// Maps to values used by `cactus::kv_cache_type_from_str` or similar C++ logic.
public enum CacheType: String, CaseIterable {
    // Exact string values depend on C++ implementation
    case float32 = "f32"
    case float16 = "f16"
    case int8 = "i8"    // Placeholder, check actual C++ strings
    case int4 = "i4"    // Placeholder
    case uint8 = "u8"   // Placeholder
    case uint4 = "u4"   // Placeholder
    // Add others as needed
}

// MARK: - Parameter Structs

/// Parameters for loading a language model.
public struct ModelLoadParams: Equatable {
    /// Path to the model file (gguf format).
    public var modelPath: String
    /// If `true`, treat `modelPath` as an asset name within the main bundle.
    public var isModelAsset: Bool = false
    /// Optional chat template string to override the one in the model file.
    public var chatTemplate: String? = nil
    /// Hint for models requiring specific reasoning formatting (e.g., Deepseek Coder).
    public var reasoningFormat: ReasoningFormat = .none
    /// Context size (max sequence length). Default based on model.
    public var contextSize: Int? = nil // n_ctx
    /// Number of GPU layers to offload (-1 for max possible). Requires Metal support.
    public var gpuLayers: Int? = -1 // n_gpu_layers, -1 means "all"
    /// Lock the model in memory (prevent paging). Defaults to false.
    public var useMlock: Bool = false
    /// Use memory mapping if possible. Defaults to true.
    public var useMmap: Bool = true
    /// Load the model in embedding mode. Defaults to false.
    public var embeddingMode: Bool = false
    /// Specifies the pooling type for embedding models. Only used if `embeddingMode` is true.
    public var poolingType: PoolingType = .unspecified
    /// Normalize embeddings. Only used if `embeddingMode` is true.
    public var normalizeEmbeddings: Bool = true // embd_normalize
    /// RoPE base frequency. Default based on model.
    public var ropeFreqBase: Float? = nil
    /// RoPE frequency scaling factor. Default based on model.
    public var ropeFreqScale: Float? = nil
    /// Use Flash Attention if available. Defaults to false.
    public var flashAttention: Bool = false
    /// KV cache data type for Keys. Default based on model/build.
    public var cacheTypeK: CacheType? = nil
    /// KV cache data type for Values. Default based on model/build.
    public var cacheTypeV: CacheType? = nil
    /// Number of threads for CPU computation (0 for default). Default: system-dependent.
    public var threadCount: Int = 0 // n_threads
    /// Batch size for prompt processing. Default: 512.
    public var batchSize: Int = 512 // n_batch
    /// Batch size for parallel decoding. Default: batchSize.
    public var ubatchSize: Int? = nil // n_ubatch, defaults to n_batch if nil
    /// Optional list of LoRA adapters to apply during initial load.
    public var loraAdapters: [LoraAdapterParams]? = nil
    /// Set to true to receive progress updates via Combine publisher or callback during model load.
    public var reportLoadProgress: Bool = true

    public init(modelPath: String) {
        self.modelPath = modelPath
    }
}

/// Parameters for loading and applying a LoRA adapter.
public struct LoraAdapterParams: Equatable, Hashable {
    /// Filesystem path to the LoRA adapter file.
    public let path: String
    /// Scaling factor for the adapter's influence (usually 0.0 to 1.0).
    public let scale: Float

    public init(path: String, scale: Float = 1.0) {
        self.path = path
        self.scale = scale
    }
}

/// Parameters controlling the text generation (completion) process.
public struct CompletionParams: Equatable {
    /// The text prompt to start generation from.
    public var prompt: String
    /// Temperature for sampling (higher = more random). Default: 0.8.
    public var temperature: Double = 0.8
    /// Max number of tokens to predict. -1 for infinite (until EOS or stop). Default: -1.
    public var maxPredictedTokens: Int = -1 // n_predict
    /// Set of strings that will cause generation to stop.
    public var stopWords: [String] = [] // anti_prompt / stop
    /// Keep generating even if EOS token is encountered.
    public var ignoreEOS: Bool = false
    /// Enable streaming of generated tokens via the async sequence.
    public var streamTokens: Bool = true // If false, only CompletionResult is returned.
    /// Random seed (-1 for unpredictable). Default: -1.
    public var seed: Int = -1
    /// Number of alternative token probabilities to return for each step (0 = disabled). Default: 0.
    public var nProbs: Int = 0

    // --- Standard Sampling Params ---
    /// Top-K sampling threshold. Default: 40.
    public var topK: Int = 40
    /// Top-P (nucleus) sampling threshold. Default: 0.95.
    public var topP: Double = 0.95
    /// Minimum probability threshold for sampling. Default: 0.05.
    public var minP: Double = 0.05
    /// Typicality sampling threshold (typ_p). Default: 1.0 (disabled).
    public var typicalP: Double = 1.0

    // --- Repetition Penalties ---
    /// Penalty for repeating tokens. Default: 1.1.
    public var repeatPenalty: Double = 1.1 // penalty_repeat
    /// Number of recent tokens to consider for penalty. Default: 64.
    public var repeatLastN: Int = 64 // penalty_last_n
    /// Penalty based on frequency in prompt & generation. Default: 0.0.
    public var frequencyPenalty: Double = 0.0 // penalty_freq
    /// Penalty based on presence in prompt & generation. Default: 0.0.
    public var presencePenalty: Double = 0.0 // penalty_present

    // --- Mirostat Sampling (Alternative to Top-K/P) ---
    /// Mirostat mode (0 = disabled, 1 = v1, 2 = v2). Default: 0.
    public var mirostat: Int = 0
    /// Mirostat target entropy. Default: 5.0.
    public var mirostatTau: Double = 5.0
    /// Mirostat learning rate. Default: 0.1.
    public var mirostatEta: Double = 0.1

    // --- Grammar / Formatting ---
    /// GBNF grammar string to constrain generation.
    public var grammar: String? = nil
    /// JSON schema string (converted to grammar internally if `grammar` is nil).
    public var jsonSchema: String? = nil

    // --- Cactus-Specific? (Check if these exist in your C++ layer) ---
    // public var xtcThreshold: Double? = nil
    // public var xtcProbability: Double? = nil
    // public var dryMultiplier: Double? = nil
    // ... etc for DRY params, top_n_sigma ...

    public init(prompt: String) {
        self.prompt = prompt
    }
}

/// Parameters for generating embeddings.
public struct EmbeddingParams: Equatable {
    /// Number of threads for CPU computation. Defaults to session's thread count.
    public var threadCount: Int? = nil
    /// Batch size for processing. Defaults to session's batch size.
    public var batchSize: Int? = nil

    public init() {}
}

/// Parameters for formatting chat messages.
public struct ChatFormatParams: Equatable {
    /// JSON string representing the list of chat messages (e.g., `[{"role": "user", "content": "..."}]`).
    public var messagesJson: String
    /// Optional: Override the model's default chat template string.
    public var chatTemplate: String? = nil
    /// If true, use Jinja2 template processing (requires minja). If false, use basic Llama2-style formatting.
    public var useJinja: Bool = true

    // --- Jinja-Specific Options (only if `useJinja` is true) ---
    /// Optional JSON schema string for structured output constraints.
    public var jsonSchema: String? = nil
    /// Optional JSON string defining available tools for function calling.
    public var toolsJson: String? = nil
    /// Allow the model to request multiple tool calls in parallel.
    public var parallelToolCalls: Bool = false
    /// Constrain the model to use a specific tool (if defined in `toolsJson`).
    public var toolChoice: String? = nil

    public init(messagesJson: String) {
        self.messagesJson = messagesJson
    }
}

/// Parameters for saving the model's KV cache state.
public struct SessionSaveParams: Equatable {
    /// Path to save the session file.
    public var path: String
    /// Optional: List of tokens to exclude from the saved state.
    public var tokensToExclude: [CactusToken]? = nil // Assuming llama_context_save_session_file takes this
    /// Target size for the saved session file (implementation defined).
    public var targetSize: Int = 0 // Check C++ API

    public init(path: String) {
        self.path = path
    }
}

/// Parameters for loading a previously saved KV cache state.
public struct SessionLoadParams: Equatable {
    /// Path to the session file to load.
    public var path: String
    /// Target size for loading (implementation defined).
    public var targetSize: Int = 0 // Check C++ API

    public init(path: String) {
        self.path = path
    }
}

/// Parameters for running a benchmark.
public struct BenchmarkParams: Equatable {
    /// Prompt processing batch size.
    public var ppBatchSize: Int // pp
    /// Token generation batch size.
    public var tgBatchSize: Int // tg
    /// Prompt length.
    public var promptLength: Int // pl
    /// Number of tokens to generate.
    public var generationLength: Int // nr
    /// Number of threads.
    public var threadCount: Int

    public init(ppBatchSize: Int, tgBatchSize: Int, promptLength: Int, generationLength: Int, threadCount: Int) {
        self.ppBatchSize = ppBatchSize
        self.tgBatchSize = tgBatchSize
        self.promptLength = promptLength
        self.generationLength = generationLength
        self.threadCount = threadCount
    }
} 
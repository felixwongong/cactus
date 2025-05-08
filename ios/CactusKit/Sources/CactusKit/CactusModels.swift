import Foundation

// MARK: - Completion Models

/// Represents a single token generated during completion, potentially with probabilities.
public struct TokenResult: Equatable, Hashable {
    /// The string content of the generated token.
    public let content: String
    /// Indicates if this token generation caused a stop condition (e.g., EOS, stop word).
    /// This might need to be derived after the token is yielded, based on C++ context state.
    public let stop: Bool
    /// Optional: The probabilities of likely alternative tokens for the step that generated this token,
    /// if requested via `CompletionParams.nProbs`.
    public let probabilities: [TokenProbability]?
}

/// Represents the probability of a specific token at a generation step.
public struct TokenProbability: Equatable, Hashable {
    /// The string representation of the alternative token.
    public let tokenString: String
    /// The probability assigned to this token.
    public let probability: Double
}

/// Represents the final result of a completion task, including timings.
public struct CompletionResult: Equatable {
    /// The complete generated text string.
    public let text: String
    /// Performance timings for the generation process.
    public let timings: Timings
    /// Reason why the generation stopped (e.g., "eos", "max_tokens", "stop_word").
    public let stopReason: String // Consider an enum for common reasons
    // TODO: Add fields like tokens_evaluated, tokens_predicted etc. if available from C++.
}

/// Represents performance timings captured during model operations.
/// Mirrors the structure of `llama_timings`.
public struct Timings: Equatable {
    public let promptN: Int
    public let promptMs: Double
    public let promptPerTokenMs: Double
    public let promptPerSecond: Double

    public let predictN: Int
    public let predictMs: Double
    public let predictPerTokenMs: Double
    public let predictPerSecond: Double

    // Add other timing fields if cactus.h/llama.h exposes them via C functions
    // e.g., t_load_ms, t_sample_ms etc.
}

// MARK: - Embedding Models

/// Represents the result of an embedding generation task.
public struct EmbeddingResult: Equatable {
    /// The vector of embedding values.
    public let values: [Float]
}

// MARK: - Model Info Models

/// Represents information about the loaded model.
public struct ModelInfo: Equatable {
    /// Model description string.
    public let description: String
    /// Size of the model file in bytes.
    public let size: Int
    /// Dimension of the model's embeddings.
    public let embeddingDim: Int
    /// Total number of parameters in the model.
    public let paramCount: Int
    /// Model metadata key-value pairs.
    public let metadata: [String: String]
    /// Vocabulary type.
    public let vocabType: String // Example: Assuming this is available
    // TODO: Add chat template info, architecture details, etc.
}

// MARK: - Chat Formatting Models

/// Represents the result of formatting chat messages.
public struct FormattedChatResult: Equatable {
    /// The fully formatted prompt string ready for the model.
    public let prompt: String
    /// Internal format identifier (consider an enum).
    public let chatFormat: Int
    /// Optional grammar string generated based on schema/tools.
    public let grammar: String?
    /// Whether the grammar should be applied lazily.
    public let grammarLazy: Bool?
    /// Specific tokens that should be preserved during processing.
    public let preservedTokens: [String]?
    /// Additional stop sequences derived from the chat format.
    public let additionalStops: [String]?
    // TODO: Add grammar_triggers if needed
}

// MARK: - LoRA Models

/// Represents information about a loaded LoRA adapter.
public struct LoraAdapterInfo: Equatable, Identifiable, Hashable {
    /// Unique identifier, using the path.
    public var id: String { path }
    /// Filesystem path to the LoRA adapter file.
    public let path: String
    /// Scaling factor applied to the LoRA adapter.
    public let scale: Float
}

// MARK: - Tokenization Models

/// Type alias for llama tokens (integers).
public typealias CactusToken = Int32 
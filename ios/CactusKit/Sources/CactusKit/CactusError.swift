import Foundation // Using Foundation for NSError bridging potential, can remove if not needed

/// Errors that can be thrown by the CactusKit package.
public enum CactusError: Error, Equatable {
    /// Failed to load the model. Contains an optional reason string.
    case modelLoadFailed(String?)
    /// The provided model path was invalid or inaccessible.
    case invalidModelPath(String)
    /// An error occurred during text prediction/generation. Contains an optional reason string.
    case predictionFailed(String?)
    /// Failed to tokenize the input text.
    case tokenizationFailed
    /// Failed to detokenize the input tokens.
    case detokenizationFailed
    /// Failed to generate embeddings.
    case embeddingFailed
    /// Failed to format the chat input.
    case chatFormattingFailed
    /// Failed to save the session state.
    case sessionSaveFailed
    /// Failed to load the session state.
    case sessionLoadFailed
    /// Failed to apply LoRA adapters.
    case loraApplyFailed
    /// Metal acceleration is not supported or available on this device. Contains an optional reason string.
    case metalNotSupported(String?)
    /// A generic underlying error from the C++ layer or bridging.
    case underlyingError(String)
    /// Operation was cancelled.
    case cancelled
    /// Context is not available or invalid.
    case invalidContext
} 
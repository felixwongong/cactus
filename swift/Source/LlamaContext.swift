import Foundation

/// Represents a Llama model context that can perform text completion and other operations
public class LlamaContext {
    // MARK: - Properties
    
    /// Unique identifier for this context
    public let id: Int
    
    /// Whether GPU acceleration is available
    public let gpu: Bool
    
    /// Reason why GPU acceleration is not available (if applicable)
    public let reasonNoGPU: String
    
    /// Information about the loaded model
    public let model: ModelInfo
    
    /// Reference to the native context
    private let contextRef: CactusContextRef
    
    // MARK: - Initialization
    
    /// Internal initializer used by factory methods
    internal init(id: Int, gpu: Bool, reasonNoGPU: String, model: ModelInfo, contextRef: CactusContextRef) {
        self.id = id
        self.gpu = gpu
        self.reasonNoGPU = reasonNoGPU
        self.model = model
        self.contextRef = contextRef
    }
    
    deinit {
        // Attempt to release resources
        try? release()
    }
    
    // MARK: - Public Methods
    
    /// Load cached prompt & completion state from a file
    /// - Parameter filepath: Path to the session file
    /// - Returns: Result of loading the session
    public func loadSession(filepath: String) async throws -> SessionLoadResult {
        var path = filepath
        if path.hasPrefix("file://") {
            path = String(path.dropFirst(7))
        }
        
        // Call bridge function
        guard let resultPtr = path.withCString({ cPath in
            cactus_context_load_session(contextRef, cPath)
        }) else {
            throw CactusError.sessionOperationFailed
        }
        
        defer {
            cactus_free_string(resultPtr)
        }
        
        // Parse the JSON result
        let resultString = String(cString: resultPtr)
        guard let resultData = resultString.data(using: .utf8),
              let resultJson = try? JSONSerialization.jsonObject(with: resultData) as? [String: Any],
              let tokens = resultJson["tokens"] as? Int,
              let success = resultJson["success"] as? Bool else {
            throw CactusError.invalidResponse
        }
        
        return SessionLoadResult(tokens: tokens, success: success)
    }
    
    /// Save current cached prompt & completion state to a file
    /// - Parameters:
    ///   - filepath: Path where to save the session
    ///   - tokenSize: Max token size to save, -1 for all
    /// - Returns: Number of tokens saved
    public func saveSession(filepath: String, tokenSize: Int = -1) async throws -> Int {
        let result = filepath.withCString { cPath in
            cactus_context_save_session(contextRef, cPath, Int32(tokenSize))
        }
        
        if result < 0 {
            throw CactusError.sessionOperationFailed
        }
        
        return Int(result)
    }
    
    /// Perform text completion with the given parameters
    /// - Parameters:
    ///   - params: Parameters for the completion
    ///   - callback: Optional callback for receiving tokens as they're generated
    /// - Returns: Result of the completion
    public func completion(params: CompletionParams, callback: ((TokenData) -> Void)? = nil) async throws -> CompletionResult {
        var callbackContext: CallbackContext? = nil
        
        // Set up callback if provided
        if let callback = callback {
            callbackContext = CallbackContext(callback: callback)
        }
        
        // Set up completion parameters
        var messagesJson: String? = nil
        if let messages = params.messages {
            // Convert messages to JSON
            let encoder = JSONEncoder()
            if let messagesData = try? encoder.encode(messages),
               let messagesJsonStr = String(data: messagesData, encoding: .utf8) {
                messagesJson = messagesJsonStr
            }
        }
        
        // Response format JSON
        var responseFormatJson: String? = nil
        if let responseFormat = params.responseFormat {
            var formatDict: [String: Any] = ["type": responseFormat.type.rawValue]
            if let jsonSchema = responseFormat.jsonSchema {
                formatDict["json_schema"] = jsonSchema
            }
            
            if let formatData = try? JSONSerialization.data(withJSONObject: formatDict),
               let formatJsonStr = String(data: formatData, encoding: .utf8) {
                responseFormatJson = formatJsonStr
            }
        }
        
        // Tools JSON
        var toolsJson: String? = nil
        if let tools = params.tools {
            if let toolsData = try? JSONSerialization.data(withJSONObject: tools),
               let toolsJsonStr = String(data: toolsData, encoding: .utf8) {
                toolsJson = toolsJsonStr
            }
        }
        
        // Create C parameters
        var cParams = CactusCompletionParams(
            prompt: params.prompt?.cString(using: .utf8),
            messages_json: messagesJson?.cString(using: .utf8),
            chat_template: params.chatTemplate?.cString(using: .utf8),
            jinja: params.jinja,
            max_tokens: Int32(params.maxTokens),
            temperature: params.temperature,
            top_p: params.topP,
            top_k: Int32(params.topK),
            frequency_penalty: params.frequencyPenalty,
            presence_penalty: params.presencePenalty,
            logprobs: params.logprobs ?? false,
            top_logprobs: Int32(params.topLogprobs ?? 0),
            response_format: responseFormatJson?.cString(using: .utf8),
            tools_json: toolsJson?.cString(using: .utf8)
        )
        
        // Set up C callback
        let cCallback: CactusTokenCallback? = callback != nil ? { tokenData, userData in
            guard let userData = userData else { return }
            let contextPtr = Unmanaged<CallbackContext>.fromOpaque(userData).takeUnretainedValue()
            
            // Convert token data
            let token = String(cString: tokenData.token)
            var probabilities: [TokenProbability]? = nil
            
            if let probsJson = tokenData.probs_json {
                let probsJsonStr = String(cString: probsJson)
                if let probsData = probsJsonStr.data(using: .utf8),
                   let probsList = try? JSONSerialization.jsonObject(with: probsData) as? [[String: Any]] {
                    probabilities = probsList.compactMap { dict in
                        guard let token = dict["token"] as? String,
                              let id = dict["id"] as? Int,
                              let prob = dict["prob"] as? Double else {
                            return nil
                        }
                        return TokenProbability(token: token, id: id, prob: prob)
                    }
                }
            }
            
            let tokenData = TokenData(token: token, completionProbabilities: probabilities)
            contextPtr.callback(tokenData)
        } : nil
        
        // Call C function
        let resultPtr = cactus_context_completion(
            contextRef,
            cParams,
            cCallback,
            callbackContext.map { Unmanaged.passUnretained($0).toOpaque() }
        )
        
        // Check and handle result
        guard let resultPtr = resultPtr else {
            throw CactusError.completionFailed
        }
        
        defer {
            cactus_free_string(resultPtr)
        }
        
        // Parse the JSON result
        let resultString = String(cString: resultPtr)
        guard let resultData = resultString.data(using: .utf8),
              let resultJson = try? JSONSerialization.jsonObject(with: resultData) as? [String: Any] else {
            throw CactusError.invalidResponse
        }
        
        // Extract fields from JSON
        guard let text = resultJson["text"] as? String,
              let usageDict = resultJson["usage"] as? [String: Any],
              let promptTokens = usageDict["prompt_tokens"] as? Int,
              let completionTokens = usageDict["completion_tokens"] as? Int,
              let totalTokens = usageDict["total_tokens"] as? Int,
              let finishReasonStr = resultJson["finish_reason"] as? String,
              let timingsDict = resultJson["timings"] as? [String: Any],
              let totalDuration = timingsDict["total_duration"] as? Double else {
            throw CactusError.invalidResponse
        }
        
        // Convert finish reason
        let finishReason: CompletionFinishReason
        switch finishReasonStr {
        case "length":
            finishReason = .length
        case "stop":
            finishReason = .stop
        case "error":
            finishReason = .error
        default:
            finishReason = .unknown
        }
        
        // Create result
        let usage = CompletionUsage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: totalTokens
        )
        
        let timings = CompletionTimings(totalDuration: totalDuration)
        
        return CompletionResult(
            text: text,
            usage: usage,
            finishReason: finishReason,
            timings: timings
        )
    }
    
    /// Stop an ongoing completion
    public func stopCompletion() async throws {
        cactus_context_stop_completion(contextRef)
    }
    
    /// Tokenize a text string
    /// - Parameter text: Text to tokenize
    /// - Returns: Tokenization result
    public func tokenize(text: String) async throws -> TokenizeResult {
        guard let resultPtr = text.withCString({ cText in
            cactus_context_tokenize(contextRef, cText)
        }) else {
            throw CactusError.invalidOperation
        }
        
        defer {
            cactus_free_string(resultPtr)
        }
        
        // Parse the JSON result
        let resultString = String(cString: resultPtr)
        guard let resultData = resultString.data(using: .utf8),
              let resultJson = try? JSONSerialization.jsonObject(with: resultData) as? [String: Any],
              let tokens = resultJson["tokens"] as? [Int] else {
            throw CactusError.invalidResponse
        }
        
        return TokenizeResult(tokens: tokens)
    }
    
    /// Convert tokens back to text
    /// - Parameter tokens: Tokens to convert
    /// - Returns: Detokenized text
    public func detokenize(tokens: [Int]) async throws -> String {
        // Convert tokens to JSON string
        let encoder = JSONEncoder()
        guard let tokensData = try? encoder.encode(tokens),
              let tokensJsonStr = String(data: tokensData, encoding: .utf8) else {
            throw CactusError.invalidOperation
        }
        
        guard let resultPtr = tokensJsonStr.withCString({ cTokens in
            cactus_context_detokenize(contextRef, cTokens)
        }) else {
            throw CactusError.invalidOperation
        }
        
        defer {
            cactus_free_string(resultPtr)
        }
        
        return String(cString: resultPtr)
    }
    
    /// Generate embeddings for the provided text
    /// - Parameters:
    ///   - text: Text to embed
    ///   - params: Optional embedding parameters
    /// - Returns: Embedding result
    public func embedding(text: String, params: EmbeddingParams? = nil) async throws -> EmbeddingResult {
        let normalize = params?.normalize ?? true
        
        guard let resultPtr = text.withCString({ cText in
            cactus_context_embedding(contextRef, cText, normalize)
        }) else {
            throw CactusError.invalidOperation
        }
        
        defer {
            cactus_free_string(resultPtr)
        }
        
        // Parse the JSON result
        let resultString = String(cString: resultPtr)
        guard let resultData = resultString.data(using: .utf8),
              let resultJson = try? JSONSerialization.jsonObject(with: resultData) as? [String: Any],
              let embedding = resultJson["embedding"] as? [Float] else {
            throw CactusError.invalidResponse
        }
        
        return EmbeddingResult(embedding: embedding)
    }
    
    /// Release resources associated with this context
    public func release() async throws {
        cactus_context_destroy(contextRef)
    }
}

// MARK: - Supporting Types

/// Information about a loaded model
public struct ModelInfo {
    /// Model type
    public let type: String
    
    /// Number of parameters in the model
    public let nParams: Int
    
    /// Number of layers in the model
    public let nLayers: Int
    
    /// Model's context size
    public let contextSize: Int
    
    /// Embedding size
    public let embeddingSize: Int
    
    /// Available chat templates
    public let chatTemplates: ChatTemplates
}

/// Available chat template formats
public struct ChatTemplates {
    /// Whether Llama chat is supported
    public let llamaChat: Bool
    
    /// Whether Minja (Jinja) templates are supported
    public let minja: Bool
}

/// Result of loading a session
public struct SessionLoadResult {
    /// Number of tokens loaded
    public let tokens: Int
    
    /// Whether loading was successful
    public let success: Bool
}

/// Data for a generated token
public struct TokenData {
    /// The token text
    public let token: String
    
    /// Probabilities for completion tokens (if requested)
    public let completionProbabilities: [TokenProbability]?
}

/// Probability information for a token
public struct TokenProbability {
    /// The token text
    public let token: String
    
    /// The token ID
    public let id: Int
    
    /// Probability score
    public let prob: Double
}

/// Parameters for completion
public struct CompletionParams {
    /// The prompt text (for non-chat completions)
    public var prompt: String?
    
    /// Messages for chat completions
    public var messages: [ChatMessage]?
    
    /// Custom chat template to use
    public var chatTemplate: String?
    
    /// Whether to use Jinja templating
    public var jinja: Bool
    
    /// Maximum number of tokens to generate
    public var maxTokens: Int
    
    /// Temperature for sampling
    public var temperature: Float
    
    /// Top-p for nucleus sampling
    public var topP: Float
    
    /// Top-k for sampling
    public var topK: Int
    
    /// Frequency penalty
    public var frequencyPenalty: Float
    
    /// Presence penalty
    public var presencePenalty: Float
    
    /// Whether to return token probabilities
    public var logprobs: Bool?
    
    /// Number of tokens to return probabilities for
    public var topLogprobs: Int?
    
    /// Response format
    public var responseFormat: CompletionResponseFormat?
    
    /// Tools to use during completion
    public var tools: [Any]?
    
    public init(
        prompt: String? = nil,
        messages: [ChatMessage]? = nil,
        chatTemplate: String? = nil,
        jinja: Bool = false,
        maxTokens: Int = 256,
        temperature: Float = 0.8,
        topP: Float = 0.95,
        topK: Int = 40,
        frequencyPenalty: Float = 0,
        presencePenalty: Float = 0,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        responseFormat: CompletionResponseFormat? = nil,
        tools: [Any]? = nil
    ) {
        self.prompt = prompt
        self.messages = messages
        self.chatTemplate = chatTemplate
        self.jinja = jinja
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.responseFormat = responseFormat
        self.tools = tools
    }
}

/// Result of a completion operation
public struct CompletionResult {
    /// Generated text
    public let text: String
    
    /// Token usage information
    public let usage: CompletionUsage
    
    /// Reason why the completion finished
    public let finishReason: CompletionFinishReason
    
    /// Timing information
    public let timings: CompletionTimings
}

/// Token usage information for a completion
public struct CompletionUsage {
    /// Number of tokens in the prompt
    public let promptTokens: Int
    
    /// Number of tokens in the completion
    public let completionTokens: Int
    
    /// Total number of tokens
    public let totalTokens: Int
}

/// Timing information for a completion
public struct CompletionTimings {
    /// Time spent in text generation (in seconds)
    public let totalDuration: Double
    
    // Additional timing details can be added as needed
    
    init(totalDuration: Double = 0) {
        self.totalDuration = totalDuration
    }
}

/// Reasons for completion finishing
public enum CompletionFinishReason {
    /// Model reached its designed token limit
    case length
    
    /// Model completed successfully (EOS token)
    case stop
    
    /// An error occurred
    case error
    
    /// Unknown reason
    case unknown
}

/// Format specification for completion responses
public struct CompletionResponseFormat {
    /// Response format type
    public let type: CompletionResponseType
    
    /// JSON schema (for JSON outputs)
    public let jsonSchema: [String: Any]?
    
    public init(type: CompletionResponseType, jsonSchema: [String: Any]? = nil) {
        self.type = type
        self.jsonSchema = jsonSchema
    }
}

/// Type of response format
public enum CompletionResponseType: String, Codable {
    /// Plain text output
    case text
    
    /// JSON object output
    case jsonObject = "json_object"
    
    /// JSON conforming to a schema
    case jsonSchema = "json_schema"
}

/// Message for chat completions
public struct ChatMessage: Codable {
    /// Role of the message sender
    public let role: ChatRole
    
    /// Content of the message
    public let content: String
    
    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

/// Role in a chat conversation
public enum ChatRole: String, Codable {
    case system
    case user
    case assistant
    case tool
}

/// Result of tokenization
public struct TokenizeResult {
    /// The token IDs
    public let tokens: [Int]
}

/// Parameters for embedding generation
public struct EmbeddingParams {
    /// Whether to normalize the output vectors
    public let normalize: Bool
    
    public init(normalize: Bool = true) {
        self.normalize = normalize
    }
}

/// Result of embedding generation
public struct EmbeddingResult {
    /// The embedding vector
    public let embedding: [Float]
}

// MARK: - Internal Helpers

// Callback context for token generation
class CallbackContext {
    let callback: (TokenData) -> Void
    
    init(callback: @escaping (TokenData) -> Void) {
        self.callback = callback
    }
}

// Error types
enum CactusError: Error {
    case modelLoadFailed
    case invalidOperation
    case completionFailed
    case sessionOperationFailed
    case invalidResponse
} 
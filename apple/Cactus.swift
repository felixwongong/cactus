import Foundation
import cactus

public final class Cactus: @unchecked Sendable {


    public struct CompletionResult {
        public let text: String
        public let functionCalls: [[String: Any]]?
        public let promptTokens: Int
        public let completionTokens: Int
        public let timeToFirstToken: Double
        public let totalTime: Double
        public let prefillTokensPerSecond: Double
        public let decodeTokensPerSecond: Double
        public let confidence: Double
        public let needsCloudHandoff: Bool

        init(json: [String: Any]) {
            self.text = json["text"] as? String ?? ""
            self.functionCalls = json["function_calls"] as? [[String: Any]]
            self.promptTokens = json["prompt_tokens"] as? Int ?? 0
            self.completionTokens = json["completion_tokens"] as? Int ?? 0
            self.timeToFirstToken = json["time_to_first_token_ms"] as? Double ?? 0
            self.totalTime = json["total_time_ms"] as? Double ?? 0
            self.prefillTokensPerSecond = json["prefill_tokens_per_second"] as? Double ?? 0
            self.decodeTokensPerSecond = json["decode_tokens_per_second"] as? Double ?? 0
            self.confidence = json["confidence"] as? Double ?? 1.0
            self.needsCloudHandoff = json["cloud_handoff"] as? Bool ?? false
        }
    }

    public struct TranscriptionResult {
        public let text: String
        public let segments: [[String: Any]]?
        public let totalTime: Double

        init(json: [String: Any]) {
            self.text = json["text"] as? String ?? ""
            self.segments = json["segments"] as? [[String: Any]]
            self.totalTime = json["total_time_ms"] as? Double ?? 0
        }
    }

    public struct Message {
        public let role: String
        public let content: String

        public init(role: String, content: String) {
            self.role = role
            self.content = content
        }

        public static func system(_ content: String) -> Message {
            Message(role: "system", content: content)
        }

        public static func user(_ content: String) -> Message {
            Message(role: "user", content: content)
        }

        public static func assistant(_ content: String) -> Message {
            Message(role: "assistant", content: content)
        }

        func toDict() -> [String: String] {
            ["role": role, "content": content]
        }
    }

    public struct CompletionOptions {
        public var temperature: Float
        public var topP: Float
        public var topK: Int
        public var maxTokens: Int
        public var stopSequences: [String]
        public var confidenceThreshold: Float

        public init(
            temperature: Float = 0.7,
            topP: Float = 0.9,
            topK: Int = 40,
            maxTokens: Int = 512,
            stopSequences: [String] = [],
            confidenceThreshold: Float = 0.0
        ) {
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.maxTokens = maxTokens
            self.stopSequences = stopSequences
            self.confidenceThreshold = confidenceThreshold
        }

        public static let `default` = CompletionOptions()

        func toJSON() -> String? {
            let dict: [String: Any] = [
                "temperature": temperature,
                "top_p": topP,
                "top_k": topK,
                "max_tokens": maxTokens,
                "stop": stopSequences,
                "confidence_threshold": confidenceThreshold
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: dict),
                  let json = String(data: data, encoding: .utf8) else {
                return nil
            }
            return json
        }
    }

    public struct TranscriptionOptions {
        public var language: String?
        public var translateToEnglish: Bool

        public init(language: String? = nil, translateToEnglish: Bool = false) {
            self.language = language
            self.translateToEnglish = translateToEnglish
        }

        public static let `default` = TranscriptionOptions()

        func toJSON() -> String? {
            var dict: [String: Any] = [
                "translate": translateToEnglish
            ]
            if let lang = language {
                dict["language"] = lang
            }
            guard let data = try? JSONSerialization.data(withJSONObject: dict),
                  let json = String(data: data, encoding: .utf8) else {
                return nil
            }
            return json
        }
    }

    public enum CactusError: Error, LocalizedError {
        case initializationFailed(String)
        case completionFailed(String)
        case transcriptionFailed(String)
        case embeddingFailed(String)
        case invalidResponse

        public var errorDescription: String? {
            switch self {
            case .initializationFailed(let msg): return "Initialization failed: \(msg)"
            case .completionFailed(let msg): return "Completion failed: \(msg)"
            case .transcriptionFailed(let msg): return "Transcription failed: \(msg)"
            case .embeddingFailed(let msg): return "Embedding failed: \(msg)"
            case .invalidResponse: return "Invalid response from model"
            }
        }
    }


    private let handle: OpaquePointer
    private static let defaultBufferSize = 65536

    public init(modelPath: String, corpusDir: String? = nil) throws {
        guard let h = cactus_init(modelPath, corpusDir) else {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.initializationFailed(error.isEmpty ? "Unknown error" : error)
        }
        self.handle = h
    }

    deinit {
        cactus_destroy(handle)
    }

    public func complete(
        messages: [Message],
        options: CompletionOptions = .default,
        tools: [[String: Any]]? = nil,
        onToken: ((String, UInt32) -> Void)? = nil
    ) throws -> CompletionResult {
        let messagesJSON = try serializeMessages(messages)
        let optionsJSON = options.toJSON()
        let toolsJSON = try serializeTools(tools)

        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)

        let callbackContext = onToken.map { TokenCallbackContext(callback: $0) }
        let contextPtr = callbackContext.map { Unmanaged.passUnretained($0).toOpaque() }

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_complete(
                handle,
                messagesJSON,
                bufferPtr.baseAddress,
                bufferPtr.count,
                optionsJSON,
                toolsJSON,
                onToken != nil ? tokenCallbackBridge : nil,
                contextPtr
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.completionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        if let errorMsg = json["error"] as? String {
            throw CactusError.completionFailed(errorMsg)
        }

        return CompletionResult(json: json)
    }

    public func complete(
        _ prompt: String,
        options: CompletionOptions = .default
    ) throws -> CompletionResult {
        try complete(messages: [.user(prompt)], options: options)
    }

    public func transcribe(
        audioPath: String,
        prompt: String? = nil,
        options: TranscriptionOptions = .default
    ) throws -> TranscriptionResult {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)
        let optionsJSON = options.toJSON()

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_transcribe(
                handle,
                audioPath,
                prompt,
                bufferPtr.baseAddress,
                bufferPtr.count,
                optionsJSON,
                nil,
                nil,
                nil,
                0
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.transcriptionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        return TranscriptionResult(json: json)
    }

    public func transcribe(
        pcmData: Data,
        prompt: String? = nil,
        options: TranscriptionOptions = .default
    ) throws -> TranscriptionResult {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)
        let optionsJSON = options.toJSON()

        let result = pcmData.withUnsafeBytes { pcmPtr in
            buffer.withUnsafeMutableBufferPointer { bufferPtr in
                cactus_transcribe(
                    handle,
                    nil,
                    prompt,
                    bufferPtr.baseAddress,
                    bufferPtr.count,
                    optionsJSON,
                    nil,
                    nil,
                    pcmPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    pcmData.count
                )
            }
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.transcriptionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        return TranscriptionResult(json: json)
    }

    public func embed(text: String, normalize: Bool = true) throws -> [Float] {
        var embeddingBuffer = [Float](repeating: 0, count: 4096)
        var embeddingDim: Int = 0

        let result = embeddingBuffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_embed(
                handle,
                text,
                bufferPtr.baseAddress,
                bufferPtr.count,
                &embeddingDim,
                normalize
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.embeddingFailed(error.isEmpty ? "Unknown error" : error)
        }

        return Array(embeddingBuffer.prefix(embeddingDim))
    }

    public func ragQuery(_ query: String, topK: Int = 5) throws -> String {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_rag_query(
                handle,
                query,
                bufferPtr.baseAddress,
                bufferPtr.count,
                topK
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.completionFailed(error.isEmpty ? "Unknown error" : error)
        }

        return String(cString: buffer)
    }

    public func reset() {
        cactus_reset(handle)
    }

    public func stop() {
        cactus_stop(handle)
    }


    public static func setTelemetryToken(_ token: String) {
        cactus_set_telemetry_token(token)
    }

    public static func setProKey(_ key: String) {
        cactus_set_pro_key(key)
    }


    private func serializeMessages(_ messages: [Message]) throws -> String {
        let dicts = messages.map { $0.toDict() }
        guard let data = try? JSONSerialization.data(withJSONObject: dicts),
              let json = String(data: data, encoding: .utf8) else {
            throw CactusError.completionFailed("Failed to serialize messages")
        }
        return json
    }

    private func serializeTools(_ tools: [[String: Any]]?) throws -> String? {
        guard let tools = tools else { return nil }
        guard let data = try? JSONSerialization.data(withJSONObject: tools),
              let json = String(data: data, encoding: .utf8) else {
            throw CactusError.completionFailed("Failed to serialize tools")
        }
        return json
    }

    private func parseJSON(_ string: String) -> [String: Any]? {
        guard let data = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }
}


private class TokenCallbackContext {
    let callback: (String, UInt32) -> Void
    init(callback: @escaping (String, UInt32) -> Void) {
        self.callback = callback
    }
}

private func tokenCallbackBridge(token: UnsafePointer<CChar>?, tokenId: UInt32, userData: UnsafeMutableRawPointer?) {
    guard let token = token, let userData = userData else { return }
    let context = Unmanaged<TokenCallbackContext>.fromOpaque(userData).takeUnretainedValue()
    let tokenString = String(cString: token)
    context.callback(tokenString, tokenId)
}


#if os(iOS) || os(macOS) || os(tvOS) || os(watchOS) || os(visionOS)
@available(iOS 13.0, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
#endif
public extension Cactus {

    func complete(
        messages: [Message],
        options: CompletionOptions = .default,
        tools: [[String: Any]]? = nil
    ) async throws -> CompletionResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.complete(messages: messages, options: options, tools: tools)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func completeStream(
        messages: [Message],
        options: CompletionOptions = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    _ = try self.complete(messages: messages, options: options) { token, _ in
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func transcribe(
        audioPath: String,
        prompt: String? = nil,
        options: TranscriptionOptions = .default
    ) async throws -> TranscriptionResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.transcribe(audioPath: audioPath, prompt: prompt, options: options)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func embed(text: String, normalize: Bool = true) async throws -> [Float] {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.embed(text: text, normalize: normalize)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

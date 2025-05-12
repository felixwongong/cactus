import Foundation
import Combine
import OSLog // For logging
import Cxx // Import the Swift C++ standard library adapter

// Import the C module from the binary framework.
// The actual name matches the `name` in `.binaryTarget` in Package.swift.
// This assumes the framework headers are correctly exposed (might need a module map
// or umbrella header within the xcframework if not automatically found).
// If using Swift C++ Interop, you might import specific C++ namespaces/classes.
@_implementationOnly import cactus

/// Main actor for interacting with the Cactus language model engine.
/// Provides methods for loading models, generating text, tokenization, etc.
/// Use `@StateObject` or `@ObservedObject` in SwiftUI views to react to state changes.
public actor CactusSession: ObservableObject {

    // MARK: - Published Properties (for UI binding)

    /// True if a model is currently being loaded.
    @MainActor @Published public private(set) var isLoading: Bool = false
    /// Progress of the current model loading operation (0.0 to 1.0).
    @MainActor @Published public private(set) var loadProgress: Float = 0.0
    /// True if the model is currently generating text.
    @MainActor @Published public private(set) var isPredicting: Bool = false
    /// Information about the currently loaded model (nil if no model is loaded).
    @MainActor @Published public private(set) var modelInfo: ModelInfo? = nil
    /// List of currently applied LoRA adapters.
    @MainActor @Published public private(set) var loadedLoraAdapters: [LoraAdapterInfo] = []
    /// True if Metal acceleration is enabled and active for the loaded model.
    @MainActor @Published public private(set) var isMetalEnabled: Bool = false
    /// Reason why Metal might be disabled (e.g., "Unsupported device").
    @MainActor @Published public private(set) var reasonNoMetal: String? = nil

    // MARK: - Combine Subjects (alternative for non-UI updates)

    /// Publishes loading progress updates.
    public let progressSubject = PassthroughSubject<Float, Never>()
    /// Publishes generated tokens during streaming completion.
    public let tokenSubject = PassthroughSubject<TokenResult, Never>()

    // MARK: - Private State

    // Hold the C++ context object directly
    private var cactusContext: cactus.cactus_context?

    // Logger instance
    private static let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "CactusKit", category: "CactusSession")

    // Task handles for cancellation support
    private var loadTask: Task<Void, Error>?
    private var completionTask: Task<Void, Never>?

    // Store the parameters used to load the model
    private var currentLoadParams: ModelLoadParams? = nil

    // MARK: - Initialization & Cleanup

    public init() {
        Self.logger.info("CactusSession initialized (actor)")
        // Initialize the C++ context object
        self.cactusContext = cactus.cactus_context()
        // TODO: Consider llama_backend_init() call here if needed globally
    }

    deinit {
        Self.logger.info("CactusSession deinit")
        // Cancel any ongoing tasks
        loadTask?.cancel()
        completionTask?.cancel()
        // Ensure the C++ context is freed
        freeContext()
        // TODO: Consider llama_backend_free() here if needed globally
    }

    /// Frees the underlying C++ context and associated resources.
    private func freeContext() {
        if cactusContext != nil {
            Self.logger.info("Freeing C++ context")
            // C++ object is automatically deallocated when set to nil thanks to ARC/
            // Swift C++ interop lifetime management (verify this behavior).
            // If explicit deletion is needed (e.g., custom destructor):
            // self.cactusContext?.call_explicit_delete_or_cleanup()
            self.cactusContext = nil
            // Reset state (must be done on MainActor for @Published vars)
            Task { await MainActor.run { self.resetPublishedState() } }
            currentLoadParams = nil
        }
    }

    @MainActor
    private func resetPublishedState() {
         modelInfo = nil
         loadedLoraAdapters = []
         isMetalEnabled = false
         reasonNoMetal = nil
         isLoading = false
         isPredicting = false
         loadProgress = 0.0
    }

    // MARK: - Static Metadata Fetching

    /// Retrieves model metadata without fully loading the model.
    public static func getModelMetadata(modelPath: String, skipKeys: [String] = []) async throws -> [String: String] {
        logger.info("Fetching metadata for: \(modelPath)")
        var params = lm_gguf_init_params(no_alloc: false, ctx: nil)
        guard let cPath = modelPath.cString(using: .utf8) else {
            throw CactusError.invalidModelPath("Invalid UTF-8 path")
        }

        // Run gguf interaction in a detached task
        return try await Task.detached { () -> [String: String] in
            let ggufCtx = lm_gguf_init_from_file(cPath, params)
            if ggufCtx == nil {
                logger.error("Failed to init gguf context for metadata.")
                throw CactusError.modelLoadFailed("Failed to open model file for metadata")
            }
            defer { lm_gguf_free(ggufCtx) }

            var metadata = [String: String]()
            let nKv = lm_gguf_get_n_kv(ggufCtx)
            let skipSet = Set(skipKeys)

            for i in 0..<nKv {
                guard let keyPtr = lm_gguf_get_key(ggufCtx, i) else { continue }
                let key = String(cString: keyPtr)
                if skipSet.contains(key) { continue }

                // CXX-INTEROP-TODO: Check if lm_gguf_kv_to_str returns std::string
                // Need to bridge std::string to Swift String
                // Assuming it returns a C string for now:
                 guard let valuePtr = lm_gguf_kv_to_str_c(ggufCtx, i) else { continue } // Assuming hypothetical C wrapper
                 metadata[key] = String(cString: valuePtr)
                 // Assuming C wrapper allocated, need hypothetical free:
                 // lm_gguf_free_string(valuePtr)
            }
            logger.info("Metadata fetched successfully.")
            return metadata
        }.value
    }

    // MARK: - Core Public Methods (Actor Isolated)

    public func loadModel(params: ModelLoadParams) async throws {
        // Actor isolation ensures only one loadModel call runs at a time.
        guard !isLoading else {
            Self.logger.warning("Load attempt ignored, already loading.")
            throw CactusError.modelLoadFailed("Already loading") // Or just return
        }

        guard let cactusContext = self.cactusContext else {
             Self.logger.error("Load attempt failed: C++ context is nil.")
            throw CactusError.invalidContext
        }

        // Reset state before loading
        freeContext() // This nils out the old context
        self.cactusContext = cactus.cactus_context() // Create a fresh C++ context
        guard let newCactusContext = self.cactusContext else { // Ensure creation succeeded
             Self.logger.error("Failed to create new C++ context.")
            throw CactusError.invalidContext
        }

        await MainActor.run { // Update published properties on MainActor
            self.isLoading = true
            self.loadProgress = 0.0
        }
        progressSubject.send(0.0)
        currentLoadParams = params
        Self.logger.info("Starting model load: \(params.modelPath)")

        // Defer cleanup runs even if an error is thrown within the Task
        defer {
            Task { await MainActor.run { self.isLoading = false } }
        }

        // Create and run the loading task
        let task = Task.detached(priority: .userInitiated) { [weak self] in
            // We need a non-isolated reference inside the detached task.
            // The actor guarantees exclusive access when we call methods later.
            // But we need to pass the context reference *into* the task.
            // This is tricky. Let's assume we pass needed data explicitly or the C++ call is self-contained.
            // A safer pattern might involve a non-actor helper class managing the C++ ptr.
            // For now, proceed with caution, assuming C++ calls are okay from detached task
            // if they don't involve yielding/re-entering the actor.

            // --- Parameter Bridging (Swift -> C++) --- 
            var cParams: cactus.common_params = try self?.bridgeLoadParams(params) ?? { throw CactusError.invalidContext }()

            // --- Progress Callback Setup --- 
            // This is complex with actors and detached tasks. A Combine subject or
            // a delegate pattern might be safer than passing `self` via Unmanaged.
            // For simplicity, placeholder - REAL IMPLEMENTATION NEEDS CARE.
            if params.reportLoadProgress {
                 // CXX-INTEROP-TODO: Define C callback trampoline and manage user data (self reference) safely
                // cParams.progress_callback = my_c_progress_callback
                // cParams.progress_callback_user_data = Unmanaged.passUnretained(self!).toOpaque() // Risky
            }

            // --- C++ Call --- 
            Self.logger.debug("Calling C++ cactus_context.loadModel...")
            // Use the captured reference
            let loadSuccess = newCactusContext.loadModel(&cParams) // Pass by reference/pointer if needed
            Self.logger.debug("C++ loadModel returned: \(loadSuccess)")

            // CXX-INTEROP-TODO: Free any manually managed C strings from bridging
            // e.g., cParams.chat_template?.deallocate()

            try Task.checkCancellation() // Check cancellation *after* blocking C++ call

            if !loadSuccess {
                Self.logger.error("C++ context.loadModel returned false")
                // CXX-INTEROP-TODO: Get specific error from C++ context if possible
                throw CactusError.modelLoadFailed("context.loadModel returned false")
            }
            Self.logger.info("C++ context loaded model successfully")

            // --- Post-Load Info Fetching (from C/C++ context) --- 
            // CXX-INTEROP-TODO: Implement these calls using direct C++ interop
            let metalStatus = false // placeholder: newCactusContext.is_metal_enabled() ??
            let noMetalReason: String? = nil // placeholder: bridge from std::string?

            guard let modelPtr = newCactusContext.model else {
                 Self.logger.error("Model pointer is null after load.")
                 throw CactusError.modelLoadFailed("Model pointer null after load")
            }

            let fetchedModelInfo = try self?.fetchModelInfo(modelPtr: modelPtr) ?? { throw CactusError.invalidContext }()
            let fetchedLoras = self?.fetchLoadedLoras(context: newCactusContext) ?? []

            // --- Update Actor State (Dispatching back to actor's executor implicitly) ---
            // No need for explicit MainActor.run here as we are updating the actor's own state
            // *after* awaiting the Task.detached result.
            // However, the @Published properties MUST be updated on MainActor.
            await self?.updatePostLoadState(success: true,
                                             metalEnabled: metalStatus,
                                             reasonNoMetal: noMetalReason,
                                             modelInfo: fetchedModelInfo,
                                             loras: fetchedLoras)

        } // End Task.detached

        self.loadTask = task

        // Await the task completion and handle errors/cancellation
        do {
            try await task.value
        } catch is CancellationError {
            Self.logger.info("Model load cancelled.")
            await self.updatePostLoadState(success: false) // Update state on cancellation
            freeContext() // Clean up C++ context if load was cancelled mid-way
            throw CactusError.cancelled
        } catch let error as CactusError {
             Self.logger.error("Model load failed: \(error.localizedDescription)")
             await self.updatePostLoadState(success: false)
             freeContext()
             throw error
        } catch {
            Self.logger.error("Model load failed with unexpected error: \(error.localizedDescription)")
            await self.updatePostLoadState(success: false)
            freeContext()
            throw CactusError.underlyingError(error.localizedDescription)
        }
    } // End loadModel

    /// Updates actor state after load attempt, ensuring UI updates are on MainActor.
    private func updatePostLoadState(success: Bool, metalEnabled: Bool? = nil, reasonNoMetal: String? = nil, modelInfo: ModelInfo? = nil, loras: [LoraAdapterInfo]? = nil) async {
        if !success {
            await MainActor.run { // Reset published state on failure
                 self.resetPublishedState()
            }
        } else {
             await MainActor.run { // Update published state on success
                 self.isMetalEnabled = metalEnabled ?? false
                 self.reasonNoMetal = reasonNoMetal
                 self.modelInfo = modelInfo
                 self.loadedLoraAdapters = loras ?? []
                 self.loadProgress = 1.0 // Mark as complete
             }
             progressSubject.send(1.0)
             Self.logger.info("Model load complete and actor state updated.")
        }
    }

    /// Generates text completion asynchronously based on the prompt and parameters.
    public func complete(params: CompletionParams) -> AsyncThrowingStream<TokenResult, Error> {
        // Actor isolation prevents concurrent calls to complete on the same instance.
        AsyncThrowingStream { continuation in
            // Run the generation logic within a Task managed by the actor.
            Task {
                guard let cactusContext = self.cactusContext else {
                    continuation.finish(throwing: CactusError.invalidContext)
                    return
                }
                guard !isPredicting else {
                    Self.logger.warning("Completion attempt ignored: Already predicting.")
                    continuation.finish() // Finish immediately if already predicting
                    return
                }

                await MainActor.run { self.isPredicting = true }
                Self.logger.info("Starting completion for prompt: \(params.prompt.prefix(50))...")

                // Store task handle for cancellation
                 self.completionTask = Task.detached(priority: .userInitiated) { [weak self] in // Detached task for C++ loop
                     var predictionError: Error? = nil
                     var stopReasonString = "unknown"
                     let streamEndedPrematurely = true // Track if loop finishes naturally

                     defer {
                         // Ensure state is reset and continuation finished, even on errors/cancellation
                         Task { [weak self] in
                             await self?.finishCompletionStream(continuation: continuation,
                                                               error: predictionError,
                                                               prematurely: streamEndedPrematurely)
                         }
                     }

                     do {
                         guard let cactusContext = cactusContext else { throw CactusError.invalidContext }

                         // --- Parameter Bridging & Setup --- 
                         // CXX-INTEROP-TODO: Bridge CompletionParams to cactusContext.params.sampling
                         // This likely involves modifying the C++ context's internal state.
                         try self?.setupCompletionParams(cactusContext: cactusContext, params: params)

                         // Rewind context state if needed (assuming C++ method)
                         cactusContext.rewind()
                         // Load prompt into C++ context
                         cactusContext.loadPrompt()
                         // Begin the completion process in C++ context
                         cactusContext.beginCompletion()
                         Self.logger.debug("C++ context prepared for completion loop.")

                         // --- Generation Loop (Pull-based) --- 
                         while cactusContext.has_next_token && !Task.isCancelled {
                            // Call C++ method to get next token and probabilities
                            // CXX-INTEROP-TODO: Handle potential C++ exceptions from doCompletion
                            let cppTokenOutput: cactus.completion_token_output = cactusContext.doCompletion()

                            // CXX-INTEROP-TODO: Bridge cactus.completion_token_output to Swift TokenResult
                            let swiftTokenResult = self?.bridgeTokenOutput(cppTokenOutput, context: cactusContext) ?? TokenResult(content: "", stop: true, probabilities: nil) // Placeholder

                            // Yield result to the stream
                            continuation.yield(swiftTokenResult)
                            // Optionally publish via subject (on main thread)
                            Task { await MainActor.run { self?.tokenSubject.send(swiftTokenResult) } }

                            // Check C++ internal stopping conditions
                            if cactusContext.stopped_eos || cactusContext.stopped_word || cactusContext.stopped_limit {
                                stopReasonString = cactusContext.stopped_eos ? "eos" : (cactusContext.stopped_word ? "stop_word" : "limit")
                                break // Exit loop naturally based on C++ state
                            }
                         } // End while loop

                        try Task.checkCancellation() // Check cancellation after loop
                        streamEndedPrematurely = false // Loop finished normally
                        Self.logger.info("Completion loop finished normally. Stop reason: \(stopReasonString)")

                     } catch is CancellationError {
                         Self.logger.info("Completion task cancelled.")
                         predictionError = CactusError.cancelled
                         stopReasonString = "cancelled"
                     } catch let error as CactusError {
                         Self.logger.error("Completion failed: \(error.localizedDescription)")
                         predictionError = error
                         stopReasonString = "error"
                     } catch {
                         Self.logger.error("Completion failed with unexpected error: \(error.localizedDescription)")
                         predictionError = CactusError.underlyingError(error.localizedDescription)
                         stopReasonString = "error"
                     }
                     // CXX-INTEROP-TODO: Get final completion text and timings from cactusContext
                     // let finalText = bridgeStdString(cactusContext.generated_text)
                     // let finalTimings = bridgeTimings(llama_perf_context(cactusContext.ctx))
                     // Could yield a final CompletionResult here if needed, or just finish.

                } // End Task.detached

                // Handle cleanup when the stream continuation is terminated externally
                continuation.onTermination = { @Sendable [weak self] _ in
                    Self.logger.info("Completion stream terminated externally.")
                    self?.completionTask?.cancel() // Cancel the background task
                    // Ensure state is reset on termination
                     Task { [weak self] in
                         await self?.finishCompletionStream(continuation: continuation, error: CactusError.cancelled, prematurely: true)
                    }
                }
            } // End Task attached to actor
        } // End AsyncThrowingStream
    } // End complete

    /// Helper to finish the stream and update actor state
    private func finishCompletionStream(continuation: AsyncThrowingStream<TokenResult, Error>.Continuation, error: Error?, prematurely: Bool) async {
        await MainActor.run { self.isPredicting = false }
        if prematurely {
             continuation.finish(throwing: error)
        } else {
            // If loop finished normally, finish without error
            continuation.finish(throwing: error) // Still pass error if one occurred before natural finish
        }
        self.completionTask = nil // Clear task handle
    }

    /// Stops the current completion task.
    public func stopCompletion() {
        Self.logger.info("Stop completion requested.")
        completionTask?.cancel()
        // State is reset in the defer/onTermination blocks of the completion task
    }

    public func tokenize(text: String) throws -> [CactusToken] {
        guard let cactusContext = self.cactusContext, let ctx = cactusContext.ctx else {
            throw CactusError.invalidContext
        }
        // CXX-INTEROP-TODO: Get vocab pointer safely from context or model
        guard let modelPtr = cactusContext.model, let vocab = llama_model_get_vocab(modelPtr) else {
            throw CactusError.invalidContext
        }
        Self.logger.debug("Tokenizing text: \(text.prefix(50))...")

        let maxTokens = text.utf8.count + 16
        var tokenBuffer = [CactusToken](repeating: 0, count: maxTokens)
        var nTokens: Int32 = 0

        text.withCString { cText in
            nTokens = llama_tokenize(
                vocab,      // Use vocab from C++ context
                cText,
                Int32(text.utf8.count), // Pass correct length
                &tokenBuffer,
                Int32(maxTokens),
                true, // add_special - true usually needed unless specific instructions
                false // parse_special
            )
        }

        if nTokens < 0 {
            Self.logger.error("Tokenization failed with code: \(nTokens)")
            throw CactusError.tokenizationFailed
        } else {
            let result = Array(tokenBuffer.prefix(Int(nTokens)))
            Self.logger.info("Tokenization successful: \(nTokens) tokens")
            return result
        }
    }

    public func detokenize(tokens: [CactusToken]) throws -> String {
        guard let cactusContext = self.cactusContext, let ctx = cactusContext.ctx else {
             throw CactusError.invalidContext
        }
        guard let modelPtr = cactusContext.model, let vocab = llama_model_get_vocab(modelPtr) else {
            throw CactusError.invalidContext
        }
        guard !tokens.isEmpty else { return "" }
        Self.logger.debug("Detokenizing \(tokens.count) tokens...")

        var resultString = ""
        for token in tokens {
             // CXX-INTEROP-TODO: Determine max piece length needed
            let bufferSize = 64 // Start with a reasonable guess
            var pieceBuffer = [CChar](repeating: 0, count: bufferSize)

            let nChars = llama_token_to_piece(
                 vocab, // Use vocab from C++ context
                 token,
                 &pieceBuffer,
                 Int32(bufferSize),
                 0, // lstrip (usually 0 unless handling prefixes)
                 true // special (usually true to see special tokens)
            )

            if nChars < 0 {
                Self.logger.warning("Buffer size \(bufferSize) too small for token \(token)? Retrying might be needed.")
                // Handle error or maybe resize and retry if nChars indicates required size
                 continue // Skip token for now
            } else if nChars > 0 {
                resultString += String(cString: pieceBuffer)
            } // If nChars is 0, it's an empty piece
        }

        Self.logger.info("Detokenization successful.")
        return resultString
    }

    // MARK: - Placeholder Implementations (Require C++ Interop Bridging)

    public func generateEmbedding(text: String, params: EmbeddingParams = EmbeddingParams()) async throws -> EmbeddingResult {
        guard let cactusContext = self.cactusContext else { throw CactusError.invalidContext }
        guard cactusContext.params.embedding else { // Check if loaded in embedding mode
            throw CactusError.embeddingFailed("Model not loaded in embedding mode")
        }
        Self.logger.info("Generating embedding for text: \(text.prefix(50))...")

        // --- C++ Call (on background thread) ---
        let embeddings = try await Task.detached(priority: .userInitiated) { () -> std.vector<Float> in
            // CXX-INTEROP-TODO: Bridge EmbeddingParams to modify cactusContext.params if needed
            // Need to ensure the cactusContext reference is safe here.
            // This might require passing necessary data or using a non-actor helper.
            // Assuming direct call is safe for now:
            var tempParams = cactusContext.params // Or create specific params
            // tempParams.embd_normalize = ...
            let resultVec: std.vector<Float> = cactusContext.getEmbedding(&tempParams)
            // CXX-INTEROP-TODO: Check for errors if getEmbedding can fail
            return resultVec
        }.value

        // CXX-INTEROP-TODO: Convert std::vector<Float> to Swift [Float]
        let swiftEmbeddings = Array(embeddings) // Implicit conversion?
        Self.logger.info("Embedding generation successful.")
        return EmbeddingResult(values: swiftEmbeddings)
    }

    public func applyLoraAdapters(adapters: [LoraAdapterParams]) async throws {
         guard let cactusContext = self.cactusContext else { throw CactusError.invalidContext }
         guard !adapters.isEmpty else { return }
         Self.logger.info("Applying \(adapters.count) LoRA adapter(s)...")

         // --- C++ Call (on background thread) ---
         let result = try await Task.detached { () -> Int32 in
             // CXX-INTEROP-TODO: Bridge [LoraAdapterParams] to std::vector<common_adapter_lora_info>
             var cLoraInfoVec = std.vector<common_adapter_lora_info>()
             for adapter in adapters {
                 // Need safe conversion from String to std::string/const char*
                 // common_adapter_lora_info info;
                 // info.path = std::string(adapter.path)
                 // info.scale = adapter.scale
                 // cLoraInfoVec.push_back(info)
             }
             // CXX-INTEROP-TODO: Ensure vector lifetime if C++ takes pointer
             return cactusContext.applyLoraAdapters(cLoraInfoVec)
         }.value

         if result != 0 {
              Self.logger.error("Failed to apply LoRA adapters, code: \(result)")
              throw CactusError.loraApplyFailed
         } else {
              // Refresh loaded adapters list
              let fetchedLoras = self.fetchLoadedLoras(context: cactusContext)
               await MainActor.run { self.loadedLoraAdapters = fetchedLoras }
              Self.logger.info("LoRA adapters applied successfully.")
         }
    }

    public func removeAllLoraAdapters() async throws {
         guard let cactusContext = self.cactusContext else { throw CactusError.invalidContext }
         Self.logger.info("Removing all LoRA adapters...")

         try await Task.detached { // Run on background thread
              // CXX-INTEROP-TODO: Check if removeLoraAdapters can throw/fail
              cactusContext.removeLoraAdapters()
         }.value

         await MainActor.run { self.loadedLoraAdapters = [] }
         Self.logger.info("LoRA adapters removed.")
    }

    public func formatChat(params: ChatFormatParams) throws -> FormattedChatResult {
         guard let cactusContext = self.cactusContext else { throw CactusError.invalidContext }
         Self.logger.debug("Formatting chat messages...")

         // CXX-INTEROP-TODO: Bridge params and call C++ method
         let messagesStdString = std.string(params.messagesJson)
         let templateStdString = std.string(params.chatTemplate ?? "")

         var formattedPromptStdString: std.string
         // Need to determine which C++ method to call based on useJinja
         if params.useJinja {
             // CXX-INTEROP-TODO: Call getFormattedChatWithJinja, bridge params & results
             // let chatParams = cactusContext.getFormattedChatWithJinja(...)
             // Bridge common_chat_params back to FormattedChatResult
             throw CactusError.chatFormattingFailed // Placeholder
         } else {
             formattedPromptStdString = cactusContext.getFormattedChat(messagesStdString, templateStdString)
             // Need to get other fields if non-Jinja produces them
             let swiftPrompt = String(formattedPromptStdString) // Bridge std::string
             Self.logger.debug("Chat formatting successful (non-Jinja).")
             // Return partial result for now
             return FormattedChatResult(prompt: swiftPrompt, chatFormat: 0, grammar: nil, grammarLazy: nil, preservedTokens: nil, additionalStops: nil)
         }
    }

    public func saveSession(params: SessionSaveParams) throws {
        guard let cactusContext = self.cactusContext, let ctx = cactusContext.ctx else {
             throw CactusError.invalidContext
        }
        Self.logger.info("Saving session to: \(params.path)")

        // CXX-INTEROP-TODO: Bridge tokensToExclude ([Int32]) to const llama_token*
        let tokensToExcludePtr: UnsafePointer<llama_token>? = nil
        let tokenCount = params.tokensToExclude?.count ?? 0

        var success = false
        params.path.withCString { cPath in
            // Ensure internal token buffer is up-to-date if needed (depends on C++ API usage)
            // let currentTokens = cactusContext.embd // Assuming this holds relevant tokens
             // let tokenPtr = UnsafePointer<llama_token>(currentTokens) // Get pointer

            success = llama_state_save_file(
                 ctx,
                 cPath,
                 nil, // Which tokens? API seems to require the tokens *being* saved
                 0, // Maybe count of tokens being saved?
                // Assuming API wants current context tokens: Pass currentTokens pointer and count
                 // tokensToExcludePtr, // Tokens to *exclude*? API is ambiguous.
                 // Int32(tokenCount),
                 // Int32(params.targetSize) // What is size parameter for?
            )
        }

        if !success {
            Self.logger.error("Session save failed.")
            throw CactusError.sessionSaveFailed
        } else {
             Self.logger.info("Session saved successfully.")
        }
    }

    public func loadSession(params: SessionLoadParams) throws {
        guard let cactusContext = self.cactusContext, let ctx = cactusContext.ctx else {
             throw CactusError.invalidContext
        }
        Self.logger.info("Loading session from: \(params.path)")

        var n_token_count_out: Int = 0
        var success = false

        // We need a buffer to potentially receive the loaded tokens
        let tokenCapacity = cactusContext.n_ctx // Max capacity is context size
        var tokenBuffer = [llama_token](repeating: 0, count: Int(tokenCapacity))

        params.path.withCString { cPath in
             success = llama_state_load_file(
                 ctx,
                 cPath,
                 &tokenBuffer,
                 tokenCapacity,
                 &n_token_count_out
             )
        }

        if !success {
             Self.logger.error("Session load failed.")
             throw CactusError.sessionLoadFailed
        } else {
             // CXX-INTEROP-TODO: Update cactusContext internal state (e.g., embd vector) with loaded tokens
             // let loadedTokens = Array(tokenBuffer.prefix(n_token_count_out))
             // cactusContext.embd = std::vector<llama_token>(loadedTokens)
             Self.logger.info("Session loaded successfully, token count: \(n_token_count_out).")
        }
    }

    public func benchmark(params: BenchmarkParams) async throws -> String {
         guard let cactusContext = self.cactusContext else { throw CactusError.invalidContext }
         Self.logger.info("Starting benchmark...")

         let resultString = try await Task.detached { () -> String in
             // CXX-INTEROP-TODO: Bridge params and call C++ method
             let resultStdString: std.string = cactusContext.bench(
                 Int32(params.ppBatchSize),
                 Int32(params.tgBatchSize),
                 Int32(params.promptLength),
                 Int32(params.generationLength)
                 // Thread count might be implicit or need setting elsewhere
             )
             // CXX-INTEROP-TODO: Check for errors if bench can fail
             return String(resultStdString) // Bridge std::string to Swift String
         }.value

         Self.logger.info("Benchmark finished.")
         return resultString
    }

    public func invalidate() {
        Self.logger.info("Invalidating session.")
        freeContext()
    }

    // MARK: - Private Bridging Helpers (Placeholders)

    /// Bridges Swift ModelLoadParams to C++ common_params.
    private func bridgeLoadParams(_ params: ModelLoadParams) throws -> cactus.common_params {
        var cParams = cactus.common_params()
        Self.logger.debug("Bridging load parameters...")

        // CXX-INTEROP-TODO: Implement full bridging logic here
        // Handle path resolution (assets vs files)
         var resolvedPath = params.modelPath
         if params.isModelAsset { /* ... resolve path ... */ }
         guard let cModelPath = resolvedPath.cString(using: .utf8) else {
             throw CactusError.invalidModelPath("Invalid UTF-8 path")
         }
         // Assuming common_params has a std::string or copies the C string
         cParams.model = std.string(cModelPath)

         // Basic types
         if let ctxSize = params.contextSize { cParams.n_ctx = UInt32(ctxSize) }
         cParams.n_gpu_layers = Int32(params.gpuLayers ?? -1)
         cParams.use_mlock = params.useMlock
         cParams.use_mmap = params.useMmap
         cParams.embedding = params.embeddingMode
         cParams.flash_attn = params.flashAttention
         cParams.cpuparams.n_threads = UInt32(params.threadCount)
         cParams.n_batch = UInt32(params.batchSize)
         cParams.n_ubatch = UInt32(params.ubatchSize ?? params.batchSize)

         // Enums
         // cParams.pooling_type = llama_pooling_type(rawValue: params.poolingType.rawValue)

         // Strings (needs std::string conversion)
         // cParams.chat_template = std.string(params.chatTemplate ?? "")

         // Vectors (needs conversion to std::vector)
         // cParams.lora_adapters = std::vector<common_adapter_lora_info>(...)

        Self.logger.debug("Load parameter bridging complete.")
        return cParams
    }

    /// Sets up sampling parameters in the C++ context.
    private func setupCompletionParams(cactusContext: cactus.cactus_context, params: CompletionParams) throws {
         Self.logger.debug("Setting up completion parameters...")
         // CXX-INTEROP-TODO: Access and modify cactusContext.params.sampling
         // cactusContext.params.sampling.temp = Float(params.temperature)
         // cactusContext.params.sampling.top_k = Int32(params.topK)
         // ... etc for all sampling params ...
         // cactusContext.params.sampling.grammar = std::string(params.grammar ?? "")
         // Handle JSON schema -> grammar conversion if needed by calling C++ helper

         // Set prompt and other top-level params
         // cactusContext.params.prompt = std::string(params.prompt)
         // cactusContext.params.n_predict = Int32(params.maxPredictedTokens)
         // cactusContext.params.antiprompt = std::vector<std::string>(params.stopWords.map { std::string($0) })

         // Apply logit bias
         // cactusContext.params.sampling.logit_bias.clear()
         // for (token, bias) in params.logitBias ?? [:] {
         //     cactusContext.params.sampling.logit_bias[llama_token(token)] = llama_logit_bias(token: llama_token(token), bias: Float(bias))
         // }

         // Re-initialize sampler if params changed significantly
         // if !cactusContext.initSampling() { throw CactusError.predictionFailed("Failed to re-initialize sampling") }
         Self.logger.debug("Completion parameters setup complete.")
    }

    /// Bridges C++ completion_token_output to Swift TokenResult.
    private func bridgeTokenOutput(_ cppOutput: cactus.completion_token_output, context: cactus.cactus_context) throws -> TokenResult {
        // CXX-INTEROP-TODO: Implement bridging
        let content = String(cppOutput.tok) // Placeholder - need token_to_piece
        let stop = false // Placeholder - need info from C++
        var probs: [TokenProbability]? = nil
        if !cppOutput.probs.empty() {
            probs = []
            // for cppProb in cppOutput.probs {
            //     let tokenString = String(cppProb.tok) // Placeholder - need token_to_piece
            //     probs?.append(TokenProbability(tokenString: tokenString, probability: Double(cppProb.prob)))
            // }
        }
        return TokenResult(content: content, stop: stop, probabilities: probs)
    }

    /// Fetches ModelInfo from the C++ llama_model pointer.
    private func fetchModelInfo(modelPtr: UnsafeMutablePointer<llama_model>) throws -> ModelInfo {
         Self.logger.debug("Fetching model info from C++ pointer...")
         // CXX-INTEROP-TODO: Bridge description string (needs buffer/free?)
         let desc = "N/A"
         let modelSize = llama_model_size(modelPtr)
         let nEmbd = llama_model_n_embd(modelPtr)
         let nParams = llama_model_n_params(modelPtr)
         // CXX-INTEROP-TODO: Fetch metadata map
         let metadata: [String: String] = [:]
         // CXX-INTEROP-TODO: Fetch vocab type
         let vocabType = "N/A"

         return ModelInfo(description: desc,
                          size: Int(modelSize),
                          embeddingDim: Int(nEmbd),
                          paramCount: Int(nParams),
                          metadata: metadata,
                          vocabType: vocabType)
    }

    /// Fetches loaded LoRA info from the C++ context.
    private func fetchLoadedLoras(context: cactus.cactus_context) -> [LoraAdapterInfo] {
        Self.logger.debug("Fetching loaded LoRAs...")
        // CXX-INTEROP-TODO: Call getLoadedLoraAdapters and bridge result
        // let cppLoras: std::vector<common_adapter_lora_info> = context.getLoadedLoraAdapters()
        var swiftLoras: [LoraAdapterInfo] = []
        // for cppLora in cppLoras {
        //     swiftLoras.append(LoraAdapterInfo(path: String(cppLora.path), scale: cppLora.scale))
        // }
        return swiftLoras
    }

    // MARK: - Private Callback Helper Class (If needed for C Callbacks)
    // Currently unused due to pull-based completion, might be needed for logging.
    private class CallbackData { // Needs to be Sendable if used across threads
        // ... state needed by C callback ...
    }
}

// MARK: - Placeholder C Function Signatures (Replace with actual definitions)
// These are assumed based on llama.cpp and the Objective-C bridge.
// You MUST verify these against the headers in your cactus.xcframework.

// Initialization & Cleanup
@_silgen_name("cactus_init_from_params")
private func cactus_init_from_params(_ params: common_params) -> OpaquePointer?
@_silgen_name("cactus_free")
private func cactus_free(_ ctx: OpaquePointer)

// Info Getters
@_silgen_name("cactus_is_metal_enabled")
private func cactus_is_metal_enabled(_ ctx: OpaquePointer) -> Bool
@_silgen_name("cactus_reason_no_metal")
private func cactus_reason_no_metal(_ ctx: OpaquePointer) -> UnsafePointer<CChar>?
@_silgen_name("cactus_model_desc")
private func cactus_model_desc(_ ctx: OpaquePointer) -> UnsafePointer<CChar>?
@_silgen_name("cactus_model_size")
private func cactus_model_size(_ ctx: OpaquePointer) -> Int64
@_silgen_name("cactus_model_n_embd")
private func cactus_model_n_embd(_ ctx: OpaquePointer) -> Int32
@_silgen_name("cactus_model_n_params")
private func cactus_model_n_params(_ ctx: OpaquePointer) -> Int64

// LoRA Getters (Placeholders)
@_silgen_name("cactus_get_lora_count")
private func cactus_get_lora_count(_ ctx: OpaquePointer) -> Int32
@_silgen_name("cactus_get_lora_path")
private func cactus_get_lora_path(_ ctx: OpaquePointer, _ index: Int32) -> UnsafePointer<CChar>?
@_silgen_name("cactus_get_lora_scale")
private func cactus_get_lora_scale(_ ctx: OpaquePointer, _ index: Int32) -> Float

// Completion
// Assumed C callback type: typedef bool (*cactus_token_callback)(void* token_data, const char* stop_reason, void* user_data);
// Stop reason might be part of token_data instead.
@_silgen_name("cactus_complete")
private func cactus_complete(_ ctx: OpaquePointer, _ params: llama_sampling_params, _ callback: @convention(c) (UnsafeMutablePointer<cactus_token_data>?, UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Bool, _ userData: UnsafeMutableRawPointer?) -> Int32

// Tokenization
@_silgen_name("cactus_tokenize")
private func cactus_tokenize(_ ctx: OpaquePointer, _ text: UnsafePointer<CChar>, _ tokens: UnsafeMutablePointer<CactusToken>, _ max_tokens: Int32, _ add_bos: Bool, _ special: Bool) -> Int32
@_silgen_name("cactus_detokenize") // Placeholder - detokenization is complex
private func cactus_detokenize(_ ctx: OpaquePointer, _ tokens: UnsafePointer<CactusToken>, _ count: Int32) -> UnsafeMutablePointer<CChar>?
@_silgen_name("cactus_token_to_piece") // More likely API
private func cactus_token_to_piece(_ ctx: OpaquePointer, _ token: CactusToken) -> UnsafePointer<CChar>?

// Embedding
@_silgen_name("cactus_embed")
private func cactus_embed(_ ctx: OpaquePointer, _ text: UnsafePointer<CChar>, _ embeddings_out: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>, _ count_out: UnsafeMutablePointer<Int32>, _ params: embedding_params) -> Int32
@_silgen_name("cactus_free_embeddings")
private func cactus_free_embeddings(_ embeddings: UnsafeMutablePointer<Float>?)

// LoRA Application
@_silgen_name("cactus_apply_loras")
private func cactus_apply_loras(_ ctx: OpaquePointer, _ lora_infos: UnsafeMutablePointer<common_adapter_lora_info>, _ count: Int32) -> Int32
@_silgen_name("cactus_remove_all_loras")
private func cactus_remove_all_loras(_ ctx: OpaquePointer) -> Int32

// Chat Formatting
@_silgen_name("cactus_format_chat")
private func cactus_format_chat(_ ctx: OpaquePointer, _ params: cactus_chat_format_params, _ result_out: UnsafeMutablePointer<cactus_formatted_chat_result>) -> Int32
@_silgen_name("cactus_free_formatted_chat_result")
private func cactus_free_formatted_chat_result(_ result: UnsafeMutablePointer<cactus_formatted_chat_result>)

// Session Save/Load
@_silgen_name("cactus_save_session")
private func cactus_save_session(_ ctx: OpaquePointer, _ path: UnsafePointer<CChar>, _ tokens_to_exclude: UnsafePointer<CactusToken>?, _ count: Int32, _ size: Int32) -> Int32
@_silgen_name("cactus_load_session")
private func cactus_load_session(_ ctx: OpaquePointer, _ path: UnsafePointer<CChar>, _ tokens_out: UnsafeMutablePointer<CactusToken>?, _ max_count: Int32, _ size: Int32) -> Int32

// Benchmark
@_silgen_name("cactus_bench")
private func cactus_bench(_ ctx: OpaquePointer, _ pp: Int32, _ tg: Int32, _ pl: Int32, _ nr: Int32, _ n_threads: Int32) -> UnsafeMutablePointer<CChar>?

// Utility (assumed)
@_silgen_name("cactus_free_string")
private func cactus_free_string(_ ptr: UnsafeMutablePointer<CChar>?)

// Placeholder C structs (replace with actual definitions from C headers if possible,
// otherwise Swift needs equivalent struct definitions)
// These must match the layout expected by the C functions!
struct common_params { /* Define fields matching C++ */ var model: UnsafePointer<CChar>? = nil; var chat_template: UnsafePointer<CChar>? = nil; var n_ctx: UInt32 = 0; var n_gpu_layers: Int32 = 0; var use_mlock: Bool = false; var use_mmap: Bool = true; var embedding: Bool = false; var rope_freq_base: Float = 0; var rope_freq_scale: Float = 0; var flash_attn: Bool = false; var n_batch: UInt32 = 0; var n_ubatch: UInt32 = 0; var cpuparams: cpu_params = cpu_params(); var progress_callback: (@convention(c) (Float, UnsafeMutableRawPointer?) -> Bool)? = nil; var progress_callback_user_data: UnsafeMutableRawPointer? = nil }
struct cpu_params { var n_threads: UInt32 = 0 }
struct llama_sampling_params { var n_probs: UInt32 = 0; var top_k: Int32 = 0; var top_p: Float = 0; var min_p: Float = 0; var typ_p: Float = 0; var temp: Float = 0; var penalty_last_n: Int32 = 0; var penalty_repeat: Float = 0; var penalty_freq: Float = 0; var penalty_present: Float = 0; var mirostat: Int32 = 0; var mirostat_tau: Float = 0; var mirostat_eta: Float = 0; var seed: UInt32 = 0; var ignore_eos: Bool = false; var grammar: UnsafePointer<CChar>? = nil }
struct common_adapter_lora_info { var path: UnsafePointer<CChar>? = nil; var scale: Float = 0 }
struct cactus_token_data { var content: UnsafePointer<CChar>!; var stop: Bool = false; var n_probs: Int32 = 0 /* ... probs array ... */ }
struct embedding_params { /* ... */ }
struct cactus_chat_format_params { /* Define fields matching C++ */ }
struct cactus_formatted_chat_result { var prompt: UnsafeMutablePointer<CChar>!; var format: Int32 = 0; var grammar: UnsafeMutablePointer<CChar>?; var grammar_lazy: Bool = false /* ... other fields ... */ } 
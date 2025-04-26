import Foundation

/// Main entry point for Cactus Swift functionality
public enum Cactus {
    /// Context parameters for initializing a model
    public struct ContextParams {
        /// Path to the model file
        public let modelPath: String
        
        /// Whether the model is bundled with the app
        public let isModelAsset: Bool
        
        /// LoRA adapter file
        public let lora: String?
        
        /// Multiple LoRA adapters
        public let loraList: [LoraAdapter]?
        
        /// Number of layers to offload to GPU
        public let gpuLayers: Int
        
        /// Number of threads to use
        public let threads: Int
        
        /// Size of context window
        public let contextSize: Int
        
        /// Seed for random number generation
        public let seed: Int
        
        /// Whether to use mmap
        public let useMmap: Bool
        
        /// Whether to use mlock
        public let useMlock: Bool
        
        /// Type for key cache
        public let cacheTypeK: CacheType?
        
        /// Type for value cache
        public let cacheTypeV: CacheType?
        
        /// Pooling type for embeddings
        public let poolingType: PoolingType?
        
        /// Whether to use batch processing
        public let batch: Bool
        
        public init(
            modelPath: String,
            isModelAsset: Bool = false,
            lora: String? = nil,
            loraList: [LoraAdapter]? = nil,
            gpuLayers: Int = 0,
            threads: Int = 0,
            contextSize: Int = 2048,
            seed: Int = 0,
            useMmap: Bool = true,
            useMlock: Bool = false,
            cacheTypeK: CacheType? = nil,
            cacheTypeV: CacheType? = nil,
            poolingType: PoolingType? = nil,
            batch: Bool = true
        ) {
            self.modelPath = modelPath
            self.isModelAsset = isModelAsset
            self.lora = lora
            self.loraList = loraList
            self.gpuLayers = gpuLayers
            self.threads = threads
            self.contextSize = contextSize
            self.seed = seed
            self.useMmap = useMmap
            self.useMlock = useMlock
            self.cacheTypeK = cacheTypeK
            self.cacheTypeV = cacheTypeV
            self.poolingType = poolingType
            self.batch = batch
        }
    }
    
    /// Type for model caches
    public enum CacheType: String {
        case f16 = "f16"
        case f32 = "f32"
        case q8_0 = "q8_0"
        case q4_0 = "q4_0"
        case q4_1 = "q4_1"
        case iq4_nl = "iq4_nl"
        case q5_0 = "q5_0"
        case q5_1 = "q5_1"
    }
    
    /// Type of pooling for embeddings
    public enum PoolingType: String {
        case none = "none"
        case mean = "mean"
        case cls = "cls"
        case last = "last"
        case rank = "rank"
    }
    
    /// LoRA adapter configuration
    public struct LoraAdapter {
        /// Path to the LoRA adapter file
        public let path: String
        
        /// Scaling factor for the adapter
        public let scaled: Double?
        
        public init(path: String, scaled: Double? = nil) {
            self.path = path
            self.scaled = scaled
        }
    }
    
    /// Initialize a Llama model context
    /// - Parameters:
    ///   - params: Parameters for context initialization
    ///   - progressHandler: Optional handler for initialization progress
    /// - Returns: Initialized LlamaContext
    public static func initLlama(params: ContextParams, progressHandler: ((Double) -> Void)? = nil) async throws -> LlamaContext {
        // Generate a random ID (for compatibility with the React implementation)
        let contextId = Int.random(in: 0..<100000)
        
        // Create context reference
        let contextRef = cactus_context_create()
        
        // Setup common parameters
        // In a real implementation, we would convert these parameters to C++ equivalents
        // and pass them to the context during model loading
        
        // Load the model
        let modelLoaded = params.modelPath.withCString { cStr in
            cactus_context_load_model(contextRef, cStr) == 1
        }
        
        if !modelLoaded {
            cactus_context_destroy(contextRef)
            throw CactusError.modelLoadFailed
        }
        
        // Now that model is loaded, we can retrieve model information
        
        // Get model type
        let modelType: String
        if let cModelType = cactus_context_get_model_type(contextRef) {
            modelType = String(cString: cModelType)
        } else {
            modelType = "unknown"
        }
        
        // Get model parameters
        let nParams = cactus_context_get_n_params(contextRef)
        let nLayers = cactus_context_get_n_layers(contextRef)
        let contextSize = cactus_context_get_context_size(contextRef)
        let embeddingSize = cactus_context_get_embedding_size(contextRef)
        
        // Get chat template support
        let supportsLlamaChat = cactus_context_has_llama_chat(contextRef)
        let supportsMinja = cactus_context_has_minja(contextRef)
        
        // Determine GPU usage
        let usingGPU = params.gpuLayers > 0
        let reasonNoGPU = usingGPU ? "" : "GPU layers not requested"
        
        // Create model info
        let modelInfo = ModelInfo(
            type: modelType,
            nParams: Int(nParams),
            nLayers: nLayers,
            contextSize: contextSize,
            embeddingSize: embeddingSize,
            chatTemplates: ChatTemplates(llamaChat: supportsLlamaChat, minja: supportsMinja)
        )
        
        // Return the context
        return LlamaContext(
            id: contextId,
            gpu: usingGPU,
            reasonNoGPU: reasonNoGPU,
            model: modelInfo,
            contextRef: contextRef
        )
    }
    
    /// Release all loaded Llama contexts
    public static func releaseAllLlama() async throws {
        // TODO: Implement this when we have a proper context tracking mechanism
    }
    
    /// Error types for Cactus operations
    public enum CactusError: Error {
        case modelLoadFailed
        case invalidOperation
        case completionFailed
        case sessionOperationFailed
    }
} 
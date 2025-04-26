# Cactus Swift

A Swift wrapper for the Cactus library, providing a modern, type-safe API for working with LLM models on Apple platforms.

## Overview

Cactus Swift provides a Swift interface to the Cactus C++ library, allowing you to:

- Load and run LLM models
- Generate text completions with streaming support
- Create embeddings for text
- Manage model sessions and state
- Format chat conversations
- Download models from remote sources

## Current Implementation Status

**IMPORTANT**: This Swift wrapper is currently a demonstration implementation with placeholder/mock C++ bridge functions. The Swift API is complete and follows best practices, but the underlying C++ bridge needs to be updated to match the actual Cactus C++ API.

The API is designed to be compatible with the React Native implementation in structure and functionality, making it easy to create cross-platform applications.

## Usage

### Basic Initialization

```swift
// Initialize with a model
let context = try await Cactus.initLlama(
    params: Cactus.ContextParams(
        modelPath: "/path/to/model.gguf",
        gpuLayers: 32,
        threads: 4,
        contextSize: 2048
    )
)
```

### Text Completion

```swift
// Perform a completion with streaming
let result = try await context.completion(
    params: CompletionParams(
        prompt: "Once upon a time,",
        maxTokens: 128,
        temperature: 0.7
    )
) { tokenData in
    // Process each token as it's generated
    print(tokenData.token, terminator: "")
}

// The full generated text is available in the result
print("\nFull text: \(result.text)")
print("Tokens used: \(result.usage.totalTokens)")
```

### Chat Completion

```swift
// Create chat messages
let messages = [
    ChatMessage(role: .system, content: "You are a helpful assistant."),
    ChatMessage(role: .user, content: "Hello, can you tell me a joke?")
]

// Perform chat completion
let result = try await context.completion(
    params: CompletionParams(
        messages: messages,
        jinja: true,
        maxTokens: 256
    )
) { tokenData in
    print(tokenData.token, terminator: "")
}
```

### Embeddings

```swift
// Generate embeddings for text
let embedding = try await context.embedding(
    text: "Hello, world!",
    params: EmbeddingParams(normalize: true)
)

// Use the embedding vector
print("Embedding dimension: \(embedding.embedding.count)")
```

### Model Download

```swift
// Download a model
let downloader = ModelDownloader()
let modelPath = try await downloader.downloadModel(
    options: ModelDownloader.DownloadOptions(
        modelURL: URL(string: "https://example.com/model.gguf")!,
        modelFolderName: "my-model"
    )
) { progress in
    print("Download: \(Int(progress.percentCompleted))%")
}

// Use the downloaded model
let context = try await Cactus.initLlama(
    params: Cactus.ContextParams(modelPath: modelPath.path)
)
```

## Integration Guide

To integrate Cactus Swift into your project:

1. Add the Swift package or framework to your project
2. Make sure the C++ Cactus library is available (via xcframework)
3. Import and use the APIs as shown in the examples

## Development and Contribution

To contribute to the development of Cactus Swift:

1. Update the C++ bridge implementation in `CactusBridge.mm` to match the actual Cactus C++ API
2. Maintain Swift API compatibility
3. Add tests and examples for new functionality
4. Submit pull requests with detailed descriptions of changes

## License

This project is licensed under [LICENSE TBD].

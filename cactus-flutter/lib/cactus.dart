/// The main library for the Cactus Flutter plugin, providing on-device AI model inference
/// capabilities by interfacing with a native `cactus` backend (typically C++ based on llama.cpp).
///
/// This library exports all the necessary classes and functions to:
/// - Initialize and manage an AI model context ([CactusContext]).
/// - Configure model loading and inference parameters ([CactusInitParams], [CactusCompletionParams]).
/// - Perform text and chat completions ([CactusContext.completion], [CactusCompletionResult]).
/// - Generate text embeddings ([CactusContext.embedding]).
/// - Tokenize and detokenize text ([CactusContext.tokenize], [CactusContext.detokenize]).
/// - Manage chat conversations ([ChatMessage]).
/// - Download models from URLs ([downloadModel], also handled internally by [CactusContext.init]).
///
/// ## Getting Started
///
/// 1.  **Add the dependency** to your `pubspec.yaml`.
/// 2.  **Ensure the native `cactus` library** (e.g., `libcactus.so` on Android, `cactus.framework` on iOS)
///     is correctly included in your Flutter project and linked.
/// 3.  **Initialize a context**:
///     ```dart
///     import 'package:cactus/cactus.dart';
///
///     Future<void> main() async {
///       CactusContext? context;
///       try {
///         final initParams = CactusInitParams(
///           // Provide either modelPath for a local file or modelUrl to download
///           modelPath: '/path/to/your/model.gguf',
///           // modelUrl: 'https://example.com/your_model.gguf',
///           // modelFilename: 'downloaded_model.gguf', // if using modelUrl
///           contextSize: 2048, // Example context size
///           gpuLayers: 1,      // Example: offload some layers to GPU if supported
///           onInitProgress: (progress, status, isError) {
///             print('Init Progress: ${progress ?? "N/A"} - $status (Error: $isError)');
///           },
///         );
///         context = await CactusContext.init(initParams);
///         print('Cactus context initialized!');
///
///         // Now use the context for completion, embedding, etc.
///         final completionParams = CactusCompletionParams(
///           messages: [ChatMessage(role: 'user', content: 'Hello, world!')],
///           temperature: 0.7,
///           maxPredictedTokens: 100,
///         );
///         final result = await context.completion(completionParams);
///         print('Completion result: ${result.text}');
///
///       } catch (e) {
///         print('Error initializing or using Cactus: $e');
///       } finally {
///         context?.free(); // Crucial: always free the context when done!
///       }
///     }
///     ```
///
/// ## Key Components:
///
/// - [CactusContext]: The main class for interacting with a loaded AI model.
///   Manages the native model instance and provides methods for inference operations.
/// - [CactusInitParams]: Specifies parameters for initializing the [CactusContext],
///   such as model path/URL, context size, GPU layers, and chat templates.
/// - [CactusCompletionParams]: Defines parameters for controlling text or chat generation,
///   including the input messages, sampling settings (temperature, topK, topP, etc.),
///   stopping conditions, grammar, and streaming callbacks.
/// - [CactusCompletionResult]: Encapsulates the output of a completion operation,
///   including the generated text and metadata about how generation stopped.
/// - [ChatMessage]: Represents a single message in a chat sequence, with a `role`
///   (e.g., 'system', 'user', 'assistant') and `content`.
/// - [downloadModel]: A utility function for downloading model files, also used
///   internally by [CactusContext.init] when a `modelUrl` is provided.
///
/// Remember to always call [CactusContext.free] to release native resources
/// when you are finished with a context instance.
library cactus;

export 'chat.dart';
export 'init_params.dart';
export 'completion.dart';
export 'context.dart';
export 'model_downloader.dart'; 

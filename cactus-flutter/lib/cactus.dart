/// The main library for the Cactus Flutter plugin, providing on-device AI model inference.
///
/// This library exports all the necessary classes and functions to initialize a model,
/// perform text and chat completions, generate embeddings, and manage the model context.
///
/// Key components:
/// - [CactusContext]: Manages the loaded model and inference operations.
/// - [CactusInitParams]: Parameters for initializing the model context.
/// - [CactusCompletionParams]: Parameters for controlling text/chat generation.
/// - [CactusCompletionResult]: The output of a completion operation.
/// - [ChatMessage]: Represents a message in a chat sequence.
/// - [downloadModel]: Utility function for downloading models (also handled internally).
library cactus;

export 'chat.dart';
export 'init_params.dart';
export 'completion.dart';
export 'context.dart';
export 'model_downloader.dart'; 

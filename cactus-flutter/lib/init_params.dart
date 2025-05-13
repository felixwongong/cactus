// import 'dart:io';

// import 'package:path_provider/path_provider.dart'; 

/// Callback type for monitoring the progress of context initialization and model downloads.
///
/// [progress] is a value between 0.0 and 1.0 indicating download progress, or null if progress is indeterminate.
/// [statusMessage] provides a textual description of the current status.
/// [isError] is true if the status message represents an error.
typedef OnInitProgress = void Function(double? progress, String statusMessage, bool isError);

/// Parameters for initializing a [CactusContext].
class CactusInitParams {
  /// Path to the model file on the local device. 
  /// Either [modelPath] or [modelUrl] must be provided.
  final String? modelPath; 
  /// URL from which to download the model file.
  /// If provided, the model will be downloaded to the application's documents directory.
  /// Either [modelPath] or [modelUrl] must be provided.
  final String? modelUrl;
  /// Optional filename to use when saving a model downloaded via [modelUrl].
  /// If null, the filename is inferred from the URL.
  final String? modelFilename; 

  /// Optional custom chat template string (e.g., Jinja2 format).
  /// If null or empty, [defaultChatMLTemplate] from `chat.dart` is used.
  final String? chatTemplate; 
  /// The context size for the model (number of tokens).
  /// Defaults to 512.
  final int nCtx;
  /// The batch size for prompt processing.
  /// Defaults to 512.
  final int nBatch;
  /// The ubatch size.
  /// Defaults to 512.
  final int nUbatch;
  /// Number of GPU layers to offload. 0 for CPU-only inference.
  /// Defaults to 0.
  final int nGpuLayers;
  /// Number of threads to use for computation.
  /// Defaults to 4.
  final int nThreads;
  /// Whether to use mmap for model loading if possible.
  /// Defaults to true.
  final bool useMmap;
  /// Whether to use mlock to prevent the model from being swapped out.
  /// Defaults to false.
  final bool useMlock;
  /// Whether the model is primarily for embedding generation.
  /// Defaults to false.
  final bool embedding;
  /// Pooling type for embedding, if [embedding] is true.
  /// (0: unspecified, 1: none, 2: mean, 3: cls, 4: last, 5: rank)
  /// Defaults to 0 (unspecified by this library, relies on native default).
  final int poolingType; 
  /// Whether to normalize embeddings, if [embedding] is true.
  /// (0: false, 1: true)
  /// Defaults to 1 (true).
  final int embdNormalize; 
  /// Whether to use Flash Attention if available.
  /// Defaults to false.
  final bool flashAttn;
  /// The K cache type for the model (e.g., "f16", "q8_0"). 
  /// Native side determines default if null.
  final String? cacheTypeK;
  /// The V cache type for the model (e.g., "f16", "q8_0").
  /// Native side determines default if null.
  final String? cacheTypeV;
  
  /// Callback for receiving progress updates during initialization and model download.
  final OnInitProgress? onInitProgress; 

  /// Creates parameters for [CactusContext] initialization.
  ///
  /// Throws an [ArgumentError] if neither [modelPath] nor [modelUrl] is provided,
  /// or if both are provided.
  CactusInitParams({
    this.modelPath,
    this.modelUrl,
    this.modelFilename,
    this.chatTemplate,
    this.nCtx = 512,
    this.nBatch = 512,
    this.nUbatch = 512,
    this.nGpuLayers = 0, 
    this.nThreads = 4,   
    this.useMmap = true,
    this.useMlock = false,
    this.embedding = false,
    this.poolingType = 0, 
    this.embdNormalize = 1, 
    this.flashAttn = false,
    this.cacheTypeK,
    this.cacheTypeV,
    this.onInitProgress,
  }) {
    if (modelPath == null && modelUrl == null) {
      throw ArgumentError('Either modelPath or modelUrl must be provided.');
    }
    if (modelPath != null && modelUrl != null) {
      throw ArgumentError('Cannot provide both modelPath and modelUrl. Choose one.');
    }
  }
} 
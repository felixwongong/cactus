import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io'; 

import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';

import './ffi_bindings.dart' as bindings;
import './init_params.dart';
import './completion.dart';
import './model_downloader.dart';
import './chat.dart';

// --- Custom Exception Classes ---

/// Base exception for Cactus-related errors.
class CactusException implements Exception {
  final String message;
  final dynamic underlyingError;

  CactusException(this.message, [this.underlyingError]);

  @override
  String toString() {
    if (underlyingError != null) {
      return 'CactusException: $message (Caused by: $underlyingError)';
    }
    return 'CactusException: $message';
  }
}

/// Exception thrown during [CactusContext] initialization.
class CactusInitializationException extends CactusException {
  CactusInitializationException(String message, [dynamic underlyingError])
      : super('Initialization failed: $message', underlyingError);
}

/// Exception thrown if a provided model path is invalid or the model cannot be accessed.
class CactusModelPathException extends CactusInitializationException {
  CactusModelPathException(String message, [dynamic underlyingError])
      : super('Model path error: $message', underlyingError);
}

/// Exception thrown during a completion operation.
class CactusCompletionException extends CactusException {
  CactusCompletionException(String message, [dynamic underlyingError])
      : super('Completion failed: $message', underlyingError);
}

/// Exception thrown for general errors during Cactus operations like tokenization or embedding.
class CactusOperationException extends CactusException {
  CactusOperationException(String operation, String message, [dynamic underlyingError])
      : super('$operation failed: $message', underlyingError);
}

// --- End Custom Exception Classes ---

/// Internal callback management, not part of public API.
/// Stores the currently active token callback for a completion operation.
CactusTokenCallback? _currentOnNewTokenCallback;

/// Internal FFI dispatcher for token callbacks, not part of public API.
///
/// This static function is invoked by the native C code for each new token.
/// It then calls the Dart callback stored in [_currentOnNewTokenCallback].
///
/// [tokenC] is a C string (Pointer<Utf8>) containing the new token.
/// Returns `true` to continue generation, `false` to stop.
@pragma('vm:entry-point')
bool _staticTokenCallbackDispatcher(Pointer<Utf8> tokenC) {
  if (_currentOnNewTokenCallback != null) {
    try {
      final token = tokenC.toDartString();
      return _currentOnNewTokenCallback!(token);
    } catch (e) {
      // Stop generation if the Dart callback throws an error.
      print('Error in token callback: $e');
      return false;
    }
  }
  // Continue generation if no Dart callback is set (should not happen if streaming).
  return true;
}

/// Manages a loaded AI model instance and provides methods for interaction
/// with the underlying `cactus` native library.
///
/// Use [CactusContext.init] to create and initialize a context with model parameters.
/// This involves loading the model (from a local path or URL) and setting up
/// the native inference engine.
///
/// Once initialized, the context can be used for:
/// - [tokenize]: Converting text to model-specific tokens.
/// - [detokenize]: Converting tokens back to text.
/// - [embedding]: Generating numerical vector representations of text.
/// - [completion]: Performing text or chat completion.
///
/// It is crucial to call [free] when the context is no longer needed to release
/// the native resources (model, KV cache, etc.) and avoid memory leaks.
class CactusContext {
  /// Internal handle to the native `cactus_context_opaque`. Not for public use.
  final bindings.CactusContextHandle _handle;

  // Private constructor. Users should use the static `CactusContext.init()` method.
  CactusContext._(this._handle);

  /// Initializes a new [CactusContext] with the given [params].
  ///
  /// This is an asynchronous operation that involves:
  /// 1. Resolving the model path:
  ///    - If [CactusInitParams.modelUrl] is provided, the model is downloaded
  ///      to the application's documents directory (if not already present).
  ///      The [CactusInitParams.modelFilename] can be used to specify the local filename.
  ///    - If [CactusInitParams.modelPath] is provided, it's used directly.
  /// 2. Initializing the native `cactus` context with the model and other parameters
  ///    from [params] (e.g., context size, GPU layers, chat template).
  ///
  /// This can be a long-running operation, especially if a model needs to be
  /// downloaded or if the model is large. Use [CactusInitParams.onInitProgress]
  /// to monitor progress updates (download percentage, status messages).
  ///
  /// Throws:
  /// - [CactusModelPathException] if the model path is invalid, the file doesn't exist,
  ///   or download fails.
  /// - [CactusInitializationException] if native context initialization fails for other reasons.
  /// - [ArgumentError] if [params] are invalid (e.g., neither modelPath nor modelUrl provided).
  static Future<CactusContext> init(CactusInitParams params) async {
    params.onInitProgress?.call(null, 'Initialization started.', false);

    String effectiveModelPath;

    if (params.modelUrl != null) {
      params.onInitProgress?.call(null, 'Resolving model path from URL...', false);
      try {
        final Directory appDocDir = await getApplicationDocumentsDirectory();
        String filename = params.modelFilename ?? params.modelUrl!.split('/').last;
        if (filename.contains('?')) {
          filename = filename.split('?').first;
        }
        if (filename.isEmpty) {
          filename = "downloaded_model.gguf";
        }
        effectiveModelPath = '${appDocDir.path}/$filename';

        final modelFile = File(effectiveModelPath);
        final bool fileExists = await modelFile.exists();

        params.onInitProgress?.call(
            null,
            fileExists
                ? 'Model found at $effectiveModelPath.'
                : 'Model not found locally. Preparing to download from ${params.modelUrl} to $effectiveModelPath.',
            false);

        if (!fileExists) {
          await downloadModel(
            params.modelUrl!,
            effectiveModelPath,
            onProgress: (progress, status) {
              params.onInitProgress?.call(progress, status, false);
            },
          );
          params.onInitProgress?.call(1.0, 'Model download complete.', false);
        }
      } catch (e) {
        final msg = 'Error during model download/path resolution: $e';
        params.onInitProgress?.call(null, msg, true);
        throw CactusModelPathException(msg, e);
      }
    } else if (params.modelPath != null) {
      effectiveModelPath = params.modelPath!;
      params.onInitProgress?.call(null, 'Using provided model path: $effectiveModelPath', false);
      if (!await File(effectiveModelPath).exists()) {
        final msg = 'Provided modelPath does not exist: $effectiveModelPath';
        params.onInitProgress?.call(null, msg, true);
        throw CactusModelPathException(msg);
      }
    } else {
      const msg = 'No valid model source (URL or path) provided.';
      params.onInitProgress?.call(null, msg, true);
      throw ArgumentError(msg);
    }

    params.onInitProgress?.call(null, 'Initializing native context with model: $effectiveModelPath', false);
    
    final cParams = calloc<bindings.CactusInitParamsC>();
    Pointer<Utf8> modelPathC = nullptr;
    Pointer<Utf8> chatTemplateForC = nullptr;
    Pointer<Utf8> cacheTypeKC = nullptr;
    Pointer<Utf8> cacheTypeVC = nullptr;
    Pointer<NativeFunction<Void Function(Float)>> progressCallbackC = nullptr;

    try {
      modelPathC = effectiveModelPath.toNativeUtf8(allocator: calloc);
      
      // Use user-provided template or the default Jinja template
      final String templateToUse = (params.chatTemplate != null && params.chatTemplate!.isNotEmpty)
          ? params.chatTemplate!
          : defaultChatMLTemplate;
      chatTemplateForC = templateToUse.toNativeUtf8(allocator: calloc);
      
      cacheTypeKC = params.cacheTypeK?.toNativeUtf8(allocator: calloc) ?? nullptr;
      cacheTypeVC = params.cacheTypeV?.toNativeUtf8(allocator: calloc) ?? nullptr;

      cParams.ref.model_path = modelPathC;
      cParams.ref.chat_template = chatTemplateForC;
      cParams.ref.n_ctx = params.contextSize;
      cParams.ref.n_batch = params.batchSize;
      cParams.ref.n_ubatch = params.ubatchSize;
      cParams.ref.n_gpu_layers = params.gpuLayers;
      cParams.ref.n_threads = params.threads;
      cParams.ref.use_mmap = params.useMmap;
      cParams.ref.use_mlock = params.useMlock;
      cParams.ref.embedding = params.generateEmbeddings;
      cParams.ref.pooling_type = params.poolingType;
      cParams.ref.embd_normalize = params.normalizeEmbeddings;
      cParams.ref.flash_attn = params.useFlashAttention;
      cParams.ref.cache_type_k = cacheTypeKC;
      cParams.ref.cache_type_v = cacheTypeVC;
      cParams.ref.progress_callback = progressCallbackC;

      final bindings.CactusContextHandle handle = bindings.initContext(cParams);

      if (handle == nullptr) {
        const msg = 'Failed to initialize native cactus context. Handle was null. Check native logs for details.';
        params.onInitProgress?.call(null, msg, true);
        throw CactusInitializationException(msg);
      }
      
      final context = CactusContext._(handle);
      params.onInitProgress?.call(1.0, 'CactusContext initialized successfully.', false);
      return context;
    } catch (e) {
      final msg = 'Error during native context initialization: $e';
      params.onInitProgress?.call(null, msg, true);
      if (e is CactusException) rethrow;
      throw CactusInitializationException(msg, e);
    } finally {
      if (modelPathC != nullptr) calloc.free(modelPathC);
      if (chatTemplateForC != nullptr) calloc.free(chatTemplateForC);
      if (cacheTypeKC != nullptr) calloc.free(cacheTypeKC);
      if (cacheTypeVC != nullptr) calloc.free(cacheTypeVC);
      calloc.free(cParams);
    }
  }

  /// Releases the native resources associated with this context.
  ///
  /// This method **must** be called when the context is no longer needed to free
  /// the memory occupied by the model and other native structures. Failure to do so
  /// will result in memory leaks.
  void free() {
    bindings.freeContext(_handle);
  }

  /// Converts the given [text] into a list of tokens according to the loaded model's tokenizer.
  ///
  /// Tokens are numerical representations used by the model.
  ///
  /// [text] The input string to tokenize.
  /// Returns a list of integer token IDs.
  /// Returns an empty list if the input text is empty or tokenization fails.
  /// Throws [CactusOperationException] if a native error occurs.
  List<int> tokenize(String text) {
    if (text.isEmpty) return [];

    Pointer<Utf8> textC = nullptr;
    try {
      textC = text.toNativeUtf8(allocator: calloc);
      final cTokenArray = bindings.tokenize(_handle, textC);
      
      if (cTokenArray.tokens == nullptr || cTokenArray.count == 0) {
        bindings.freeTokenArray(cTokenArray); // Still need to free the struct itself
        return [];
      }
      final dartTokens = List<int>.generate(cTokenArray.count, (i) => cTokenArray.tokens[i]);
      bindings.freeTokenArray(cTokenArray); // Frees cTokenArray.tokens and the struct
      return dartTokens;
    } catch (e) {
        throw CactusOperationException("Tokenization", "Native error during tokenization.", e);
    }
    finally {
      if (textC != nullptr) calloc.free(textC);
    }
  }

  /// Converts a list of [tokens] back into a string using the model's vocabulary.
  ///
  /// [tokens] A list of integer token IDs.
  /// Returns the detokenized string.
  /// Returns an empty string if the input list is empty or detokenization fails.
  /// Throws [CactusOperationException] if a native error occurs.
  String detokenize(List<int> tokens) {
    if (tokens.isEmpty) return "";

    Pointer<Int32> tokensCPtr = nullptr;
    Pointer<Utf8> resultCPtr = nullptr;
    try {
      tokensCPtr = calloc<Int32>(tokens.length);
      for (int i = 0; i < tokens.length; i++) {
        tokensCPtr[i] = tokens[i];
      }

      resultCPtr = bindings.detokenize(_handle, tokensCPtr, tokens.length);
      if (resultCPtr == nullptr) {
        return "";
      }
      final resultString = resultCPtr.toDartString();
      bindings.freeString(resultCPtr); // Important: free the C string
      resultCPtr = nullptr; // Avoid double free in finally
      return resultString;
    } catch (e) {
        throw CactusOperationException("Detokenization", "Native error during detokenization.", e);
    }
    finally {
      if (tokensCPtr != nullptr) calloc.free(tokensCPtr);
      // resultCPtr is freed within try if not null, or was null.
    }
  }

  /// Generates an embedding (a list of float values representing the semantic meaning)
  /// for the given [text].
  ///
  /// The model must have been initialized with [CactusInitParams.generateEmbeddings] set to true.
  /// The nature of the embedding (e.g., pooling strategy, normalization) is determined by
  /// [CactusInitParams.poolingType] and [CactusInitParams.normalizeEmbeddings].
  ///
  /// [text] The input string to embed.
  /// Returns a list of double-precision floating-point values representing the embedding.
  /// Returns an empty list if the input text is empty or embedding generation fails.
  /// Throws [CactusOperationException] if a native error occurs or if embedding mode was not enabled.
  List<double> embedding(String text) {
    if (text.isEmpty) return [];

    Pointer<Utf8> textC = nullptr;
    try {
      textC = text.toNativeUtf8(allocator: calloc);
      final cFloatArray = bindings.embedding(_handle, textC);

      if (cFloatArray.values == nullptr || cFloatArray.count == 0) {
        bindings.freeFloatArray(cFloatArray); // Free the struct
        // Check if context was initialized for embeddings. This is a best guess.
        // A more robust check would be if the native layer could return a specific error code.
        // For now, assume empty result might mean not initialized for embeddings or actual empty result.
        // Consider logging a warning or having params.generateEmbeddings available on the context.
        return [];
      }
      final dartEmbeddings = List<double>.generate(cFloatArray.count, (i) => cFloatArray.values[i]);
      bindings.freeFloatArray(cFloatArray); // Frees cFloatArray.values and the struct
      return dartEmbeddings;
    } catch (e) {
        throw CactusOperationException("Embedding generation", "Native error during embedding generation.", e);
    }
    finally {
      if (textC != nullptr) calloc.free(textC);
    }
  }

  /// Performs text or chat completion based on the provided [params].
  ///
  /// This is an asynchronous operation.
  ///
  /// It constructs a prompt from [CactusCompletionParams.messages] using a
  /// default ChatML-like format if a custom `chat_template` was not provided
  /// during [CactusContext.init]. If a `chat_template` was provided at init,
  /// the native layer handles prompt formatting.
  ///
  /// For streaming results (receiving tokens as they are generated), provide an
  /// [CactusCompletionParams.onNewToken] callback.
  ///
  /// [params] Parameters controlling the completion, including messages, sampling settings,
  /// grammar, and streaming callback.
  ///
  /// Returns a [Future] that completes with a [CactusCompletionResult] containing
  /// the generated text and other metadata.
  ///
  /// Throws [CactusCompletionException] if the native completion call fails or if
  /// an error occurs during parameter setup.
  Future<CactusCompletionResult> completion(CactusCompletionParams params) async {
    Pointer<bindings.CactusCompletionParamsC> cCompParams = nullptr;
    Pointer<bindings.CactusCompletionResultC> cResult = nullptr;
    Pointer<Utf8> promptC = nullptr;
    Pointer<Utf8> grammarC = nullptr;
    Pointer<Pointer<Utf8>> stopSequencesC = nullptr;

    try {
      // --- Prompt Formatting: Always send messages as JSON ---
      // Assumes the native side (if using Jinja or another template) can parse this.
      final List<Map<String, String>> messagesJson = params.messages.map((m) => m.toJson()).toList();
      final String formattedPrompt = jsonEncode(messagesJson);
      // --- End Prompt Formatting ---

      cCompParams = calloc<bindings.CactusCompletionParamsC>();
      cResult = calloc<bindings.CactusCompletionResultC>();
      promptC = formattedPrompt.toNativeUtf8(allocator: calloc);
      grammarC = params.grammar?.toNativeUtf8(allocator: calloc) ?? nullptr;

      if (params.stopSequences != null && params.stopSequences!.isNotEmpty) {
        stopSequencesC = calloc<Pointer<Utf8>>(params.stopSequences!.length);
        for (int i = 0; i < params.stopSequences!.length; i++) {
          stopSequencesC[i] = params.stopSequences![i].toNativeUtf8(allocator: calloc);
        }
      }

      Pointer<NativeFunction<Bool Function(Pointer<Utf8>)>> tokenCallbackC = nullptr;
      _currentOnNewTokenCallback = params.onNewToken; 
      if (params.onNewToken != null) {
        tokenCallbackC = Pointer.fromFunction<Bool Function(Pointer<Utf8>)>(_staticTokenCallbackDispatcher, false);
      }

      cCompParams.ref.prompt = promptC;
      cCompParams.ref.n_predict = params.maxPredictedTokens;
      cCompParams.ref.n_threads = params.threads;
      cCompParams.ref.seed = params.seed;
      cCompParams.ref.temperature = params.temperature;
      cCompParams.ref.top_k = params.topK;
      cCompParams.ref.top_p = params.topP;
      cCompParams.ref.min_p = params.minP;
      cCompParams.ref.typical_p = params.typicalP;
      cCompParams.ref.penalty_last_n = params.penaltyLastN;
      cCompParams.ref.penalty_repeat = params.penaltyRepeat;
      cCompParams.ref.penalty_freq = params.penaltyFreq;
      cCompParams.ref.penalty_present = params.penaltyPresent;
      cCompParams.ref.mirostat = params.mirostat;
      cCompParams.ref.mirostat_tau = params.mirostatTau;
      cCompParams.ref.mirostat_eta = params.mirostatEta;
      cCompParams.ref.ignore_eos = params.ignoreEos;
      cCompParams.ref.n_probs = params.nProbs;
      cCompParams.ref.stop_sequences = stopSequencesC ?? nullptr;
      cCompParams.ref.stop_sequence_count = params.stopSequences?.length ?? 0;
      cCompParams.ref.grammar = grammarC;
      cCompParams.ref.token_callback = tokenCallbackC ?? nullptr;
      
      final status = bindings.completion(_handle, cCompParams, cResult);

      if (status != 0) {
        final msg = 'Native completion call failed with status: $status. Check native logs.';
        throw CactusCompletionException(msg);
      }

      final result = CactusCompletionResult(
        text: cResult.ref.text.toDartString(),
        tokensPredicted: cResult.ref.tokens_predicted,
        tokensEvaluated: cResult.ref.tokens_evaluated,
        truncated: cResult.ref.truncated,
        stoppedEos: cResult.ref.stopped_eos,
        stoppedWord: cResult.ref.stopped_word,
        stoppedLimit: cResult.ref.stopped_limit,
        stoppingWord: cResult.ref.stopping_word.toDartString(),
      );

      return result;
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusCompletionException("Error during completion setup or execution.", e);
    }
    finally {
      _currentOnNewTokenCallback = null; 

      if (promptC != nullptr) calloc.free(promptC);
      if (grammarC != nullptr) calloc.free(grammarC);
      if (stopSequencesC != nullptr) {
        for (int i = 0; i < (params.stopSequences?.length ?? 0); i++) {
          if (stopSequencesC[i] != nullptr) calloc.free(stopSequencesC[i]);
        }
        calloc.free(stopSequencesC);
      }
      if (cResult != nullptr) {
        bindings.freeCompletionResultMembers(cResult); 
        calloc.free(cResult);
      }
      if (cCompParams != nullptr) calloc.free(cCompParams);
    }
  }

  /// Asynchronously requests the current completion operation to stop.
  ///
  /// This provides a way to interrupt a long-running generation.
  /// The actual stopping is handled by the native side and may not be immediate,
  /// as the native code needs to reach a point where it checks for the interrupt flag.
  ///
  /// This method is non-blocking.
  void stopCompletion() {
    bindings.stopCompletion(_handle);
  }
} 
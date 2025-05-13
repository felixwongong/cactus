import 'dart:async';
import 'dart:ffi';
import 'dart:io'; 
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';

import 'src/ffi_bindings.dart' as bindings;

// Default ChatML template. Users can override this via CactusInitParams.chatTemplate.
const String _defaultChatMLTemplate = """
{% for message in messages %}
  {% if message.role == 'system' %}
    {{ '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}
  {% elif message.role == 'user' %}
    {{ '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
  {% elif message.role == 'assistant' %}
    {{ '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
  {{ '<|im_start|>assistant\\n' }}
{% endif %}
""";

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({required this.role, required this.content});

  Map<String, String> toJson() => {
    'role': role,
    'content': content,
  };
}

/// Callback type for initialization progress updates.
/// [progress]: Download progress (0.0 to 1.0), null if not applicable (e.g., file exists or general status).
/// [statusMessage]: A descriptive message of the current state.
/// [isError]: True if the statusMessage represents an error.
typedef OnInitProgress = void Function(double? progress, String statusMessage, bool isError);

class CactusInitParams {
  /// Option 1: Provide a direct path to an existing model file.
  final String? modelPath; 

  /// Option 2: Provide a URL to download the model from.
  final String? modelUrl;
  /// Optional: Filename to save the downloaded model as. 
  /// If null and modelUrl is provided, a filename will be derived from the URL.
  final String? modelFilename; 

  final String? chatTemplate; 
  final int nCtx;
  final int nBatch;
  final int nUbatch;
  final int nGpuLayers;
  final int nThreads;
  final bool useMmap;
  final bool useMlock;
  final bool embedding;
  final int poolingType; 
  final int embdNormalize;
  final bool flashAttn;
  final String? cacheTypeK;
  final String? cacheTypeV;
  
  /// Callback for progress updates during initialization (including download).
  final OnInitProgress? onInitProgress; 
  // Removed direct progressCallback for C, as onInitProgress covers download and general init status.

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

bool Function(String)? _currentOnNewTokenCallback;

@pragma('vm:entry-point') 
bool _staticTokenCallbackDispatcher(Pointer<Utf8> tokenC) {
  if (_currentOnNewTokenCallback != null) {
    try {
      return _currentOnNewTokenCallback!(tokenC.toDartString());
    } catch (e) {
      print("Error in Dart onNewToken callback dispatcher: $e");
      return false; 
    }
  }
  return true; 
}

class CactusContext {
  final bindings.CactusContextHandle _handle;
  // _progressNativeCallable might not be needed if C-level progress callback is removed
  // NativeCallable<Void Function(Float)>? _progressNativeCallable; 
  final String _chatTemplate; // Will always be populated (either user's or default)

  CactusContext._(this._handle, String? userProvidedTemplate) : 
    _chatTemplate = (userProvidedTemplate != null && userProvidedTemplate.isNotEmpty) 
                      ? userProvidedTemplate 
                      : _defaultChatMLTemplate;

  static Future<CactusContext> init(CactusInitParams params) async {
    params.onInitProgress?.call(null, 'Initialization started.', false);

    String effectiveModelPath;

    if (params.modelUrl != null) {
      params.onInitProgress?.call(null, 'Resolving model path from URL...', false);
      try {
        final Directory appDocDir = await getApplicationDocumentsDirectory();
        String filename = params.modelFilename ?? params.modelUrl!.split('/').last;
        // Ensure filename is valid and doesn't contain query parameters if URL had them
        if (filename.contains('?')) {
          filename = filename.split('?').first;
        }
        if (filename.isEmpty) {
           filename = "downloaded_model.gguf"; // Fallback filename
        }
        effectiveModelPath = '${appDocDir.path}/$filename';

        final modelFile = File(effectiveModelPath);
        final bool fileExists = await modelFile.exists();

        params.onInitProgress?.call(null, 
          fileExists 
            ? 'Model found at $effectiveModelPath.' 
            : 'Model not found locally. Preparing to download from ${params.modelUrl} to $effectiveModelPath.',
          false
        );

        if (!fileExists) {
          await downloadModel(
            params.modelUrl!,
            effectiveModelPath,
            onProgress: (progress, status) { // downloadModel's existing onProgress
              params.onInitProgress?.call(progress, status, false);
            },
          );
          params.onInitProgress?.call(1.0, 'Model download complete.', false);
        }
      } catch (e) {
        params.onInitProgress?.call(null, 'Error during model download/path resolution: $e', true);
        throw Exception('Failed to prepare model from URL: $e');
      }
    } else if (params.modelPath != null) {
      effectiveModelPath = params.modelPath!;
      params.onInitProgress?.call(null, 'Using provided model path: $effectiveModelPath', false);
      // Optionally, verify if this file exists and is accessible
      if (!await File(effectiveModelPath).exists()) {
        final msg = 'Provided modelPath does not exist: $effectiveModelPath';
        params.onInitProgress?.call(null, msg, true);
        throw ArgumentError(msg);
      }
    } else {
      // This case should be prevented by the constructor check, but as a safeguard:
      const msg = 'No valid model source (URL or path) provided.';
      params.onInitProgress?.call(null, msg, true);
      throw ArgumentError(msg);
    }

    params.onInitProgress?.call(null, 'Initializing native context with model: $effectiveModelPath', false);
    final cParams = calloc<bindings.CactusInitParamsC>();
    final modelPathC = effectiveModelPath.toNativeUtf8(allocator: calloc);
    // Pass the user's template to C if they provided one, otherwise C gets nullptr for chat_template.
    // The Dart side (_chatTemplate field) will use the default if userProvided is null.
    final chatTemplateForC = (params.chatTemplate != null && params.chatTemplate!.isNotEmpty) 
                              ? params.chatTemplate!.toNativeUtf8(allocator: calloc) 
                              : nullptr;
    final cacheTypeKC = params.cacheTypeK?.toNativeUtf8(allocator: calloc) ?? nullptr;
    final cacheTypeVC = params.cacheTypeV?.toNativeUtf8(allocator: calloc) ?? nullptr;

    // C-level progress callback for llama.cpp internal loading is removed from this simplified API.
    // The onInitProgress callback handles download progress and general status updates from Dart side.
    Pointer<NativeFunction<Void Function(Float)>> progressCallbackC = nullptr;

    try {
      cParams.ref.model_path = modelPathC;
      cParams.ref.chat_template = chatTemplateForC; // C side gets user's or null
      cParams.ref.n_ctx = params.nCtx;
      cParams.ref.n_batch = params.nBatch;
      cParams.ref.n_ubatch = params.nUbatch;
      cParams.ref.n_gpu_layers = params.nGpuLayers;
      cParams.ref.n_threads = params.nThreads;
      cParams.ref.use_mmap = params.useMmap;
      cParams.ref.use_mlock = params.useMlock;
      cParams.ref.embedding = params.embedding;
      cParams.ref.pooling_type = params.poolingType;
      cParams.ref.embd_normalize = params.embdNormalize;
      cParams.ref.flash_attn = params.flashAttn;
      cParams.ref.cache_type_k = cacheTypeKC;
      cParams.ref.cache_type_v = cacheTypeVC;
      cParams.ref.progress_callback = progressCallbackC; // Always nullptr now

      final handle = bindings.initContext(cParams);

      if (handle == nullptr) {
        const msg = 'Failed to initialize native cactus context. Check native logs for details.';
        params.onInitProgress?.call(null, msg, true);
        throw Exception(msg);
      }
      // Initialize CactusContext with user's template (or null, constructor handles default)
      final context = CactusContext._(handle, params.chatTemplate); 
      // context._progressNativeCallable = null; // No longer used
      params.onInitProgress?.call(1.0, 'CactusContext initialized successfully.', false);
      return context;
    } catch(e) {
      final msg = 'Error during native context initialization: $e';
      params.onInitProgress?.call(null, msg, true);
      rethrow; // Rethrow after reporting progress
    } finally {
      calloc.free(modelPathC);
      if (chatTemplateForC != nullptr) calloc.free(chatTemplateForC);
      if (cacheTypeKC != nullptr) calloc.free(cacheTypeKC);
      if (cacheTypeVC != nullptr) calloc.free(cacheTypeVC);
      calloc.free(cParams);
    }
  }
  void free() {
    bindings.freeContext(_handle);
    // _progressNativeCallable?.close(); // No longer used
    // _progressNativeCallable = null;
  }

  List<int> tokenize(String text) {
    if (text.isEmpty) return [];

    final textC = text.toNativeUtf8(allocator: calloc);
    try {
      final cTokenArray = bindings.tokenize(_handle, textC);
      if (cTokenArray.tokens == nullptr || cTokenArray.count == 0) {
        bindings.freeTokenArray(cTokenArray);
        return [];
      }
      final dartTokens = List<int>.generate(cTokenArray.count, (i) => cTokenArray.tokens[i]);
      bindings.freeTokenArray(cTokenArray);
      return dartTokens;
    } finally {
      calloc.free(textC);
    }
  }

  String detokenize(List<int> tokens) {
    if (tokens.isEmpty) return "";

    final tokensCPtr = calloc<Int32>(tokens.length);
    for (int i = 0; i < tokens.length; i++) {
      tokensCPtr[i] = tokens[i];
    }

    try {
      final resultCPtr = bindings.detokenize(_handle, tokensCPtr, tokens.length);
      if (resultCPtr == nullptr) {
        return ""; 
      }
      final resultString = resultCPtr.toDartString();
      bindings.freeString(resultCPtr); 
      return resultString;
    } finally {
      calloc.free(tokensCPtr);
    }
  }

  List<double> embedding(String text) {
    if (text.isEmpty) return [];

    final textC = text.toNativeUtf8(allocator: calloc);
    try {
      final cFloatArray = bindings.embedding(_handle, textC);
      if (cFloatArray.values == nullptr || cFloatArray.count == 0) {
        bindings.freeFloatArray(cFloatArray);
        return [];
      }
      final dartEmbeddings = List<double>.generate(cFloatArray.count, (i) => cFloatArray.values[i]);
      bindings.freeFloatArray(cFloatArray);
      return dartEmbeddings;
    } finally {
      calloc.free(textC);
    }
  }

  Future<CactusCompletionResult> completion(CactusCompletionParams params) async {
    // _chatTemplate is guaranteed to be non-empty by the constructor logic
    // (either user-provided or the default).
    // Thus, the previous check for `_chatTemplate == null || _chatTemplate!.isEmpty` is no longer needed here.

    StringBuffer promptBuffer = StringBuffer();
    // The formatting logic here directly uses the _chatTemplate field, which might be user-defined
    // or the internal default. However, this current implementation hardcodes the ChatML structure.
    // For a truly flexible templating system based on _chatTemplate string, a mini-parser/renderer
    // would be needed here. For now, it effectively applies the *logic* of the default ChatML.
    for (var message in params.messages) {
      if (message.role == 'system' || message.role == 'user' || message.role == 'assistant') {
        promptBuffer.write('<|im_start|>');
        promptBuffer.write(message.role);
        promptBuffer.write('\\n'); // Note: If _chatTemplate were a real template, these would be part of it
        promptBuffer.write(message.content);
        promptBuffer.write('<|im_end|>\\n');
      } else {
        print("Warning: Unknown role '${message.role}' in ChatMessage list. Skipping.");
      }
    }
    promptBuffer.write('<|im_start|>assistant\\n');
    
    final String formattedPrompt = promptBuffer.toString();

    final cCompParams = calloc<bindings.CactusCompletionParamsC>();
    final cResult = calloc<bindings.CactusCompletionResultC>();
    final promptC = formattedPrompt.toNativeUtf8(allocator: calloc); 
    final grammarC = params.grammar?.toNativeUtf8(allocator: calloc) ?? nullptr;

    Pointer<Pointer<Utf8>> stopSequencesC = nullptr;
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
    } else {
      tokenCallbackC = nullptr;
    }

    try {
      cCompParams.ref.prompt = promptC;
      cCompParams.ref.n_predict = params.nPredict;
      cCompParams.ref.n_threads = params.nThreads;
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
      cCompParams.ref.stop_sequences = stopSequencesC;
      cCompParams.ref.stop_sequence_count = params.stopSequences?.length ?? 0;
      cCompParams.ref.grammar = grammarC;
      cCompParams.ref.token_callback = tokenCallbackC;

      final status = bindings.completion(_handle, cCompParams, cResult);

      if (status != 0) {
        throw Exception('Native completion call failed with status: $status. Check native logs.');
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
    } finally {
      _currentOnNewTokenCallback = null; 

      calloc.free(promptC);
      if (grammarC != nullptr) calloc.free(grammarC);
      if (stopSequencesC != nullptr) {
        for (int i = 0; i < (params.stopSequences?.length ?? 0); i++) {
          calloc.free(stopSequencesC[i]);
        }
        calloc.free(stopSequencesC);
      }
      bindings.freeCompletionResultMembers(cResult); 
      calloc.free(cCompParams);
      calloc.free(cResult);
    }
  }

  void stopCompletion() {
    bindings.stopCompletion(_handle);
  }
}

class CactusCompletionParams {
  final List<ChatMessage> messages; 
  final int nPredict;
  final int nThreads;
  final int seed;
  final double temperature;
  final int topK;
  final double topP;
  final double minP;
  final double typicalP;
  final int penaltyLastN;
  final double penaltyRepeat;
  final double penaltyFreq;
  final double penaltyPresent;
  final int mirostat;
  final double mirostatTau;
  final double mirostatEta;
  final bool ignoreEos;
  final int nProbs;
  final List<String>? stopSequences;
  final String? grammar;
  final bool Function(String token)? onNewToken;

  CactusCompletionParams({
    required this.messages, 
    this.nPredict = -1, 
    this.nThreads = 0, 
    this.seed = -1, 
    this.temperature = 0.8,
    this.topK = 20,
    this.topP = 0.95,
    this.minP = 0.05,
    this.typicalP = 1.0,
    this.penaltyLastN = 64,
    this.penaltyRepeat = 1.1,
    this.penaltyFreq = 0.0,
    this.penaltyPresent = 0.0,
    this.mirostat = 0,
    this.mirostatTau = 5.0,
    this.mirostatEta = 0.1,
    this.ignoreEos = false,
    this.nProbs = 0,
    this.stopSequences,
    this.grammar,
    this.onNewToken,
  });
}

class CactusCompletionResult {
  final String text;
  final int tokensPredicted;
  final int tokensEvaluated;
  final bool truncated;
  final bool stoppedEos;
  final bool stoppedWord;
  final bool stoppedLimit;
  final String stoppingWord;

  CactusCompletionResult({
    required this.text,
    required this.tokensPredicted,
    required this.tokensEvaluated,
    required this.truncated,
    required this.stoppedEos,
    required this.stoppedWord,
    required this.stoppedLimit,
    required this.stoppingWord,
  });

  @override
  String toString() {
    return 'CactusCompletionResult(text: ${text.substring(0, (text.length > 50) ? 50 : text.length)}..., tokensPredicted: $tokensPredicted, stoppedEos: $stoppedEos)';
  }
}


Future<void> downloadModel(
  String url,
  String filePath,
  {void Function(double? progress, String statusMessage)? onProgress}
) async {
  onProgress?.call(null, 'Starting download for: ${filePath.split('/').last}');
  final File modelFile = File(filePath);

  try {
    final httpClient = HttpClient(); 
    final request = await httpClient.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode == 200) {
      final IOSink fileSink = modelFile.openWrite();
      final totalBytes = response.contentLength; 
      int receivedBytes = 0;

      onProgress?.call(0.0, 'Connected. Receiving data...');

      await for (var chunk in response) {
        fileSink.add(chunk);
        receivedBytes += chunk.length;
        if (totalBytes != -1 && totalBytes != 0) {
          final progress = receivedBytes / totalBytes;
          onProgress?.call(
            progress,
            'Downloading: ${(progress * 100).toStringAsFixed(1)}% ' 
            '(${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB / ${(totalBytes / (1024 * 1024)).toStringAsFixed(2)}MB)'
          );
        } else {
          onProgress?.call(
            null, 
            'Downloading: ${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB received'
          );
        }
      }
      await fileSink.flush();
      await fileSink.close();
      
      onProgress?.call(1.0, 'Download complete. Saving file...');
      onProgress?.call(1.0, 'Model saved successfully to $filePath');
    } else {
      throw Exception(
          'Failed to download model. Status code: ${response.statusCode}');
    }
    httpClient.close(); 
  } catch (e) {
    onProgress?.call(null, 'Error during download: $e');
    rethrow; 
  }
}

import 'dart:async';
import 'dart:ffi';
import 'dart:io'; 
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:http/http.dart' as http_pkg; 

import 'src/ffi_bindings.dart' as bindings;

class CactusInitParams {
  final String modelPath;
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
  final void Function(double progress)? progressCallback;

  CactusInitParams({
    required this.modelPath,
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
    this.progressCallback,
  });
}

bool Function(String)? _currentOnNewTokenCallback;


@pragma('vm:entry-point') // AOT compilation hint
bool _staticTokenCallbackDispatcher(Pointer<Utf8> tokenC) {
  if (_currentOnNewTokenCallback != null) {
    try {
      return _currentOnNewTokenCallback!(tokenC.toDartString());
    } catch (e) {
      print("Error in static onNewToken dispatcher: $e");
      return false; 
    }
  }
  return true; 
}

class CactusContext {
  final bindings.CactusContextHandle _handle;
  NativeCallable<Void Function(Float)>? _progressNativeCallable;

  CactusContext._(this._handle);

  static Future<CactusContext> init(CactusInitParams params) async {
    if (params.modelPath.isEmpty) {
      throw ArgumentError('modelPath cannot be empty.');
    }

    final cParams = calloc<bindings.CactusInitParamsC>();
    final modelPathC = params.modelPath.toNativeUtf8(allocator: calloc);
    final chatTemplateC = params.chatTemplate?.toNativeUtf8(allocator: calloc) ?? nullptr;
    final cacheTypeKC = params.cacheTypeK?.toNativeUtf8(allocator: calloc) ?? nullptr;
    final cacheTypeVC = params.cacheTypeV?.toNativeUtf8(allocator: calloc) ?? nullptr;

    NativeCallable<Void Function(Float)>? progressNativeCallable;
    Pointer<NativeFunction<Void Function(Float)>> progressCallbackC = nullptr;

    if (params.progressCallback != null) {
      // Ensure a static/top-level function is used for the callback.
      // Example: progressNativeCallable = NativeCallable<Void Function(Float)>.isolateLocal(staticProgressCallbackHandler, exceptionalReturn: Void());
      // where staticProgressCallbackHandler would then call params.progressCallback via a SendPort or similar mechanism if needed.
      // For simplicity, if params.progressCallback is already static, it might be directly usable after wrapping.
      // This part is highly dependent on the exact callback mechanism desired.
      // Direct assignment as below is only safe if the Dart function is truly static and matches the signature.
      // progressCallbackC = Pointer.fromFunction<Void Function(Float)>(params.progressCallback!, exceptionalReturn: Void());
      // The above is generally not recommended for non-trivial callbacks due to isolate and GC issues.
      // Using NativeCallable is safer but requires the Dart function to be static or top-level.
      // Placeholder - this would require a static dispatcher: 
      // progressNativeCallable = NativeCallable<Void Function(Float)>.isolateLocal(_staticProgressDispatcher, exceptionalReturn: Void());
      // progressCallbackC = progressNativeCallable.nativeFunction;
      // A proper implementation would likely involve Isolate communication to pass the Dart closure.
    }

    try {
      cParams.ref.model_path = modelPathC;
      cParams.ref.chat_template = chatTemplateC;
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
      cParams.ref.progress_callback = progressCallbackC;

      final handle = bindings.initContext(cParams);

      if (handle == nullptr) {
        throw Exception('Failed to initialize native cactus context.');
      }
      final context = CactusContext._(handle);
      context._progressNativeCallable = progressNativeCallable;
      return context;
    } finally {
      calloc.free(modelPathC);
      if (chatTemplateC != nullptr) calloc.free(chatTemplateC);
      if (cacheTypeKC != nullptr) calloc.free(cacheTypeKC);
      if (cacheTypeVC != nullptr) calloc.free(cacheTypeVC);
      calloc.free(cParams);
    }
  }

  void free() {
    bindings.freeContext(_handle);
    _progressNativeCallable?.close();
    _progressNativeCallable = null;
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
      final dartTokens = <int>[];
      for (int i = 0; i < cTokenArray.count; i++) {
        dartTokens.add(cTokenArray.tokens[i]);
      }
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
      final dartEmbeddings = <double>[];
      for (int i = 0; i < cFloatArray.count; i++) {
        dartEmbeddings.add(cFloatArray.values[i]);
      }
      bindings.freeFloatArray(cFloatArray);
      return dartEmbeddings;
    } finally {
      calloc.free(textC);
    }
  }

  Future<CactusCompletionResult> completion(CactusCompletionParams params) async {
    final cCompParams = calloc<bindings.CactusCompletionParamsC>();
    final cResult = calloc<bindings.CactusCompletionResultC>();
    final promptC = params.prompt.toNativeUtf8(allocator: calloc);
    final grammarC = params.grammar?.toNativeUtf8(allocator: calloc) ?? nullptr;

    Pointer<Pointer<Utf8>> stopSequencesC = nullptr;
    if (params.stopSequences != null && params.stopSequences!.isNotEmpty) {
      stopSequencesC = calloc<Pointer<Utf8>>(params.stopSequences!.length);
      for (int i = 0; i < params.stopSequences!.length; i++) {
        stopSequencesC[i] = params.stopSequences![i].toNativeUtf8(allocator: calloc);
      }
    }

    Pointer<NativeFunction<Bool Function(Pointer<Utf8>)>> tokenCallbackC = nullptr;

    if (params.onNewToken != null) {
      _currentOnNewTokenCallback = params.onNewToken; // Store the callback
      tokenCallbackC = Pointer.fromFunction<Bool Function(Pointer<Utf8>)>(_staticTokenCallbackDispatcher, false);
    } else {
      _currentOnNewTokenCallback = null;
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
        throw Exception('Native completion call failed with status: $status');
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
  final String prompt;
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
    required this.prompt,
    this.nPredict = -1, 
    this.nThreads = 0, 
    this.seed = -1, 
    this.temperature = 0.8,
    this.topK = 40,
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
      final List<int> bytes = [];
      final totalBytes = response.contentLength;
      int receivedBytes = 0;

      onProgress?.call(0.0, 'Connected. Receiving data...');

      await for (var chunk in response) {
        bytes.addAll(chunk);
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
      
      onProgress?.call(1.0, 'Download complete. Saving file...');
      await modelFile.writeAsBytes(bytes);
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

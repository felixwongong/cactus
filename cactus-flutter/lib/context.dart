import 'dart:async';
import 'dart:ffi';
import 'dart:io'; 
// import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';

import './ffi_bindings.dart' as bindings;
import './chat.dart';
import './init_params.dart';
import './completion.dart';
import './model_downloader.dart';

bool Function(String)? _currentOnNewTokenCallback;

@pragma('vm:entry-point') 
bool _staticTokenCallbackDispatcher(Pointer<Utf8> tokenC) {
  if (_currentOnNewTokenCallback != null) {
    try {
      return _currentOnNewTokenCallback!(tokenC.toDartString());
    } catch (e) {
      // print("Error in Dart onNewToken callback dispatcher: $e");
      return false; 
    }
  }
  return true; 
}

class CactusContext {
  final bindings.CactusContextHandle _handle;
  // final String _chatTemplate; 

  CactusContext._(this._handle, String? userProvidedTemplate) 
    // : _chatTemplate = (userProvidedTemplate != null && userProvidedTemplate.isNotEmpty) 
    //                   ? userProvidedTemplate 
    //                   : defaultChatMLTemplate
    ;

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
            onProgress: (progress, status) { 
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
      if (!await File(effectiveModelPath).exists()) {
        final msg = 'Provided modelPath does not exist: $effectiveModelPath';
        params.onInitProgress?.call(null, msg, true);
        throw ArgumentError(msg);
      }
    } else {
      const msg = 'No valid model source (URL or path) provided.';
      params.onInitProgress?.call(null, msg, true);
      throw ArgumentError(msg);
    }

    params.onInitProgress?.call(null, 'Initializing native context with model: $effectiveModelPath', false);
    final cParams = calloc<bindings.CactusInitParamsC>();
    final modelPathC = effectiveModelPath.toNativeUtf8(allocator: calloc);
    final chatTemplateForC = (params.chatTemplate != null && params.chatTemplate!.isNotEmpty) 
                              ? params.chatTemplate!.toNativeUtf8(allocator: calloc) 
                              : nullptr;
    final cacheTypeKC = params.cacheTypeK?.toNativeUtf8(allocator: calloc) ?? nullptr;
    final cacheTypeVC = params.cacheTypeV?.toNativeUtf8(allocator: calloc) ?? nullptr;

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
      final context = CactusContext._(handle, params.chatTemplate); 
      params.onInitProgress?.call(1.0, 'CactusContext initialized successfully.', false);
      return context;
    } catch(e) {
      final msg = 'Error during native context initialization: $e';
      params.onInitProgress?.call(null, msg, true);
      rethrow; 
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
    

    StringBuffer promptBuffer = StringBuffer();
    for (var message in params.messages) {
      if (message.role == 'system' || message.role == 'user' || message.role == 'assistant') {
        promptBuffer.write('<|im_start|>');
        promptBuffer.write(message.role);
        promptBuffer.write('\\n');
        promptBuffer.write(message.content);
        promptBuffer.write('<|im_end|>\\n');
      } else {
        // print("Warning: Unknown role '${message.role}' in ChatMessage list. Skipping.");
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
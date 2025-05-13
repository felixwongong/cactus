import 'dart:ffi';
import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';

final class CactusContextOpaque extends Opaque {}
typedef CactusContextHandle = Pointer<CactusContextOpaque>;

// ignore: camel_case_types
final class CactusInitParamsC extends Struct {
  // ignore: non_constant_identifier_names
  external Pointer<Utf8> model_path;
  // ignore: non_constant_identifier_names
  external Pointer<Utf8> chat_template; 

  // ignore: non_constant_identifier_names
  @Int32()
  external int n_ctx;
  // ignore: non_constant_identifier_names
  @Int32()
  external int n_batch;
  // ignore: non_constant_identifier_names
  @Int32()
  external int n_ubatch;
  // ignore: non_constant_identifier_names
  @Int32()
  external int n_gpu_layers;
  // ignore: non_constant_identifier_names
  @Int32()
  external int n_threads;
  // ignore: non_constant_identifier_names
  @Bool()
  external bool use_mmap;
  // ignore: non_constant_identifier_names
  @Bool()
  external bool use_mlock;
  @Bool()
  external bool embedding;
  // ignore: non_constant_identifier_names
  @Int32()
  external int pooling_type;
  // ignore: non_constant_identifier_names
  @Int32()
  external int embd_normalize;
  // ignore: non_constant_identifier_names
  @Bool()
  external bool flash_attn;
  // ignore: non_constant_identifier_names
  external Pointer<Utf8> cache_type_k;
  // ignore: non_constant_identifier_names
  external Pointer<Utf8> cache_type_v;

  // ignore: non_constant_identifier_names
  external Pointer<NativeFunction<Void Function(Float)>> progress_callback;
}

// ignore: camel_case_types
final class CactusCompletionParamsC extends Struct {
  external Pointer<Utf8> prompt;
  // ignore: non_constant_identifier_names
  @Int32()
  external int n_predict;
  // ignore: non_constant_identifier_names
  @Int32()
  external int n_threads;
  @Int32()
  external int seed;
  @Double()
  external double temperature;
  // ignore: non_constant_identifier_names
  @Int32()
  external int top_k;
  // ignore: non_constant_identifier_names
  @Double()
  external double top_p;
  // ignore: non_constant_identifier_names
  @Double()
  external double min_p;
  // ignore: non_constant_identifier_names
  @Double()
  external double typical_p;
  // ignore: non_constant_identifier_names
  @Int32()
  external int penalty_last_n;
  // ignore: non_constant_identifier_names
  @Double()
  external double penalty_repeat;
  // ignore: non_constant_identifier_names
  @Double()
  external double penalty_freq;
  // ignore: non_constant_identifier_names
  @Double()
  external double penalty_present;
  @Int32()
  external int mirostat;
  // ignore: non_constant_identifier_names
  @Double()
  external double mirostat_tau;
  // ignore: non_constant_identifier_names
  @Double()
  external double mirostat_eta;
  // ignore: non_constant_identifier_names
  @Bool()
  external bool ignore_eos;
  // ignore: non_constant_identifier_names
  @Int32()
  external int n_probs;
  // ignore: non_constant_identifier_names
  external Pointer<Pointer<Utf8>> stop_sequences; 
  // ignore: non_constant_identifier_names
  @Int32()
  external int stop_sequence_count;
  external Pointer<Utf8> grammar;

  // ignore: non_constant_identifier_names
  external Pointer<NativeFunction<Bool Function(Pointer<Utf8>)>> token_callback;
}

// ignore: camel_case_types
final class CactusTokenArrayC extends Struct {
  external Pointer<Int32> tokens;
  @Int32()
  external int count;
}

// ignore: camel_case_types
final class CactusFloatArrayC extends Struct {
  // ignore: non_constant_identifier_names
  external Pointer<Float> values; 
  @Int32()
  external int count;
}

// ignore: camel_case_types
final class CactusCompletionResultC extends Struct {
  external Pointer<Utf8> text;
  // ignore: non_constant_identifier_names
  @Int32()
  external int tokens_predicted;
  // ignore: non_constant_identifier_names
  @Int32()
  external int tokens_evaluated;
  @Bool()
  external bool truncated;
  // ignore: non_constant_identifier_names
  @Bool()
  external bool stopped_eos;
  // ignore: non_constant_identifier_names
  @Bool()
  external bool stopped_word;
  // ignore: non_constant_identifier_names
  @Bool()
  external bool stopped_limit;
  // ignore: non_constant_identifier_names
  external Pointer<Utf8> stopping_word;
}

typedef InitContextCNative = CactusContextHandle Function(Pointer<CactusInitParamsC> params);
typedef InitContextDart = CactusContextHandle Function(Pointer<CactusInitParamsC> params);

typedef FreeContextCNative = Void Function(CactusContextHandle handle);
typedef FreeContextDart = void Function(CactusContextHandle handle);

typedef CompletionCNative = Int32 Function(CactusContextHandle handle, Pointer<CactusCompletionParamsC> params, Pointer<CactusCompletionResultC> result);
typedef CompletionDart = int Function(CactusContextHandle handle, Pointer<CactusCompletionParamsC> params, Pointer<CactusCompletionResultC> result);

typedef StopCompletionCNative = Void Function(CactusContextHandle handle);
typedef StopCompletionDart = void Function(CactusContextHandle handle);

typedef TokenizeCNative = CactusTokenArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);
typedef TokenizeDart = CactusTokenArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);

typedef DetokenizeCNative = Pointer<Utf8> Function(CactusContextHandle handle, Pointer<Int32> tokens, Int32 count);
typedef DetokenizeDart = Pointer<Utf8> Function(CactusContextHandle handle, Pointer<Int32> tokens, int count);

typedef EmbeddingCNative = CactusFloatArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);
typedef EmbeddingDart = CactusFloatArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);

typedef FreeStringCNative = Void Function(Pointer<Utf8> str);
typedef FreeStringDart = void Function(Pointer<Utf8> str);

typedef FreeTokenArrayCNative = Void Function(CactusTokenArrayC arr);
typedef FreeTokenArrayDart = void Function(CactusTokenArrayC arr);

typedef FreeFloatArrayCNative = Void Function(CactusFloatArrayC arr);
typedef FreeFloatArrayDart = void Function(CactusFloatArrayC arr);

typedef FreeCompletionResultMembersCNative = Void Function(Pointer<CactusCompletionResultC> result);
typedef FreeCompletionResultMembersDart = void Function(Pointer<CactusCompletionResultC> result);

String _getLibraryPath() {
  const String libName = 'cactus'; 
  if (Platform.isMacOS || Platform.isIOS) {
    
    return '$libName.framework/$libName'; 
  }
  if (Platform.isAndroid) {
    return 'lib$libName.so';
  }
  throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
}

final DynamicLibrary cactusLib = DynamicLibrary.open(_getLibraryPath());

final initContext = cactusLib
    .lookup<NativeFunction<InitContextCNative>>('cactus_init_context_c')
    .asFunction<InitContextDart>();

final freeContext = cactusLib
    .lookup<NativeFunction<FreeContextCNative>>('cactus_free_context_c')
    .asFunction<FreeContextDart>();

final completion = cactusLib
    .lookup<NativeFunction<CompletionCNative>>('cactus_completion_c')
    .asFunction<CompletionDart>();

final stopCompletion = cactusLib
    .lookup<NativeFunction<StopCompletionCNative>>('cactus_stop_completion_c')
    .asFunction<StopCompletionDart>();

final tokenize = cactusLib
    .lookup<NativeFunction<TokenizeCNative>>('cactus_tokenize_c')
    .asFunction<TokenizeDart>();

final detokenize = cactusLib
    .lookup<NativeFunction<DetokenizeCNative>>('cactus_detokenize_c')
    .asFunction<DetokenizeDart>();

final embedding = cactusLib
    .lookup<NativeFunction<EmbeddingCNative>>('cactus_embedding_c')
    .asFunction<EmbeddingDart>();

final freeString = cactusLib
    .lookup<NativeFunction<FreeStringCNative>>('cactus_free_string_c')
    .asFunction<FreeStringDart>();

final freeTokenArray = cactusLib
    .lookup<NativeFunction<FreeTokenArrayCNative>>('cactus_free_token_array_c')
    .asFunction<FreeTokenArrayDart>();

final freeFloatArray = cactusLib
    .lookup<NativeFunction<FreeFloatArrayCNative>>('cactus_free_float_array_c')
    .asFunction<FreeFloatArrayDart>();

final freeCompletionResultMembers = cactusLib
    .lookup<NativeFunction<FreeCompletionResultMembersCNative>>('cactus_free_completion_result_members_c')
    .asFunction<FreeCompletionResultMembersDart>();

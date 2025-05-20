/// Parameters for specifying a vocoder model, typically used within [VocoderLoadParams].
class VocoderModelParams {
  /// Local path to the vocoder model file.
  final String path;
  // Add other fields like url, hf_repo, hf_file if needed for FFI-based downloading in the future.

  VocoderModelParams({required this.path});
}

/// Parameters for initializing the vocoder component within a [CactusContext]
/// via the `CactusContext.loadVocoder` method.
class VocoderLoadParams {
  /// Vocoder model details.
  final VocoderModelParams modelParams;

  /// Path to a speaker embedding file (optional).
  /// If provided, allows for multi-speaker TTS if the model supports it.
  final String? speakerFile;

  /// Whether to use guide tokens (specific to some vocoder architectures).
  /// Defaults to false.
  final bool useGuideTokens;

  VocoderLoadParams({
    required this.modelParams,
    this.speakerFile,
    this.useGuideTokens = false,
  });
}

/// Parameters for a speech synthesis request via `CactusContext.synthesizeSpeech`.
class SynthesizeSpeechParams {
  /// The text to synthesize into speech.
  final String textInput;

  /// The local file path where the output WAV audio file will be saved.
  final String outputWavPath;

  /// Optional speaker ID. The format and availability of speaker IDs depend on
  /// the loaded TTS model and speaker embedding file (if used).
  final String? speakerId;

  SynthesizeSpeechParams({
    required this.textInput,
    required this.outputWavPath,
    this.speakerId,
  });
} 
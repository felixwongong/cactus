import './chat.dart';

/// Parameters for a text or chat completion request.
class CactusCompletionParams {
  /// A list of [ChatMessage] objects representing the conversation history.
  /// Required for chat-style completions.
  final List<ChatMessage> messages; 

  /// The maximum number of tokens to predict. 
  /// A value of -1 means predict indefinitely (until EOS or other stopping conditions).
  /// Defaults to -1.
  final int nPredict;
  /// Number of threads to use for this specific completion. 
  /// If 0, the number of threads from [CactusInitParams] is used.
  /// Defaults to 0.
  final int nThreads;
  /// The random seed for generation. A value of -1 means use a random seed.
  /// Defaults to -1.
  final int seed;
  /// Controls randomness in sampling. Lower values make the model more deterministic.
  /// Defaults to 0.8.
  final double temperature;
  /// Limits sampling to the top K most probable tokens.
  /// Defaults to 20.
  final int topK;
  /// Nucleus sampling: limits sampling to tokens with cumulative probability >= topP.
  /// Defaults to 0.95.
  final double topP;
  /// Minimum probability for a token to be considered in sampling.
  /// Defaults to 0.05.
  final double minP;
  /// Typical P sampling parameter.
  /// Defaults to 1.0.
  final double typicalP;
  /// Number of recent tokens to consider for penalty_repeat.
  /// Defaults to 64.
  final int penaltyLastN;
  /// Penalty applied to repeated tokens. Higher values discourage repetition.
  /// Defaults to 1.1.
  final double penaltyRepeat;
  /// Penalty applied based on token frequency in the context.
  /// Defaults to 0.0.
  final double penaltyFreq;
  /// Penalty applied to tokens already present in the context.
  /// Defaults to 0.0.
  final double penaltyPresent;
  /// Mirostat sampling mode (0: disabled, 1: Mirostat, 2: Mirostat 2.0).
  /// Defaults to 0.
  final int mirostat;
  /// Mirostat target entropy.
  /// Defaults to 5.0.
  final double mirostatTau;
  /// Mirostat learning rate.
  /// Defaults to 0.1.
  final double mirostatEta;
  /// Whether to ignore the End-Of-Sequence (EOS) token.
  /// Defaults to false.
  final bool ignoreEos;
  /// Number of probabilities to return for the top N tokens (if supported by model).
  /// Defaults to 0.
  final int nProbs;
  /// A list of sequences that, when generated, will cause completion to stop.
  final List<String>? stopSequences;
  /// GBNF grammar to constrain the output to a specific format.
  final String? grammar;
  /// Callback function that receives each new token as it is generated.
  /// Return `false` from the callback to stop the generation early.
  final bool Function(String token)? onNewToken;

  /// Creates parameters for a completion request.
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

/// Represents the result of a text or chat completion operation.
class CactusCompletionResult {
  /// The generated text.
  final String text;
  /// The number of tokens predicted during the completion.
  final int tokensPredicted;
  /// The number of prompt tokens evaluated before generation started.
  final int tokensEvaluated;
  /// True if the generation was truncated due to context length limits.
  final bool truncated;
  /// True if generation stopped because the End-Of-Sequence (EOS) token was generated.
  final bool stoppedEos;
  /// True if generation stopped because one of the [CactusCompletionParams.stopSequences] was encountered.
  final bool stoppedWord;
  /// True if generation stopped due to reaching the prediction limit ([CactusCompletionParams.nPredict]).
  final bool stoppedLimit;
  /// The specific word from [CactusCompletionParams.stopSequences] that caused generation to stop, if [stoppedWord] is true.
  /// Otherwise, this will be an empty string.
  final String stoppingWord;

  /// Creates a new [CactusCompletionResult].
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
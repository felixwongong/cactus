import './chat.dart';

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
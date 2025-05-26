import 'dart:convert'; // For BenchResult.fromJson
import 'dart:math'; // For FormattedChatResult.toString min()

/// Information about a LoRA adapter.
class LoraAdapterInfo {
  final String path;
  final double scale;

  LoraAdapterInfo({required this.path, required this.scale});

  @override
  String toString() => 'LoraAdapterInfo(path: $path, scale: $scale)';
}

/// Result of a benchmark run.
class BenchResult {
  final String modelDesc;
  final int modelSize; // bytes
  final int modelNParams;
  final double ppAvg; // prompt processing speed (tokens/sec)
  final double ppStd;
  final double tgAvg; // text generation speed (tokens/sec)
  final double tgStd;

  BenchResult({
    required this.modelDesc,
    required this.modelSize,
    required this.modelNParams,
    required this.ppAvg,
    required this.ppStd,
    required this.tgAvg,
    required this.tgStd,
  });

  factory BenchResult.fromJson(List<dynamic> json) {
    if (json.length < 7) {
      throw FormatException("Invalid JSON array length for BenchResult");
    }
    return BenchResult(
      modelDesc: json[0] as String,
      modelSize: (json[1] as num).toInt(),
      modelNParams: (json[2] as num).toInt(),
      ppAvg: (json[3] as num).toDouble(),
      ppStd: (json[4] as num).toDouble(),
      tgAvg: (json[5] as num).toDouble(),
      tgStd: (json[6] as num).toDouble(),
    );
  }

  @override
  String toString() {
    return 'BenchResult(modelDesc: $modelDesc, modelSize: $modelSize, modelNParams: $modelNParams, ppAvg: $ppAvg, ppStd: $ppStd, tgAvg: $tgAvg, tgStd: $tgStd)';
  }
}

/// Result of advanced chat formatting, potentially using Jinja templates.
class FormattedChatResult {
  final String prompt;
  final String? grammar; // GBNF grammar string, if generated

  FormattedChatResult({required this.prompt, this.grammar});

  @override
  String toString() => 'FormattedChatResult(prompt: ${prompt.substring(0, min(50, prompt.length))}..., grammar: $grammar)';
}

/// Metadata about the loaded model.
class ModelMeta {
  final String description;
  final int size; // bytes
  final int nParams;

  ModelMeta({required this.description, required this.size, required this.nParams});

  @override
  String toString() => 'ModelMeta(description: $description, size: $size, nParams: $nParams)';
} 
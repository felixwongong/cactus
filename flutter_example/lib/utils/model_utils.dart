import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:cactus_flutter/cactus_flutter.dart';

/// The name of the model file
const String modelName = 'SmolLM-135.gguf';

/// The URL to download the model from
const String modelUrl = 'https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf';

/// Initialize a Llama context with the given model
/// 
/// Downloads the model if it doesn't exist and initializes the context
/// [onProgress] is called with the download/loading progress (0.0-1.0)
Future<LlamaContext> initLlamaContext({
  required Function(double progress) onProgress,
}) async {
  final appDocDir = await getApplicationDocumentsDirectory();
  final modelDirectory = '${appDocDir.path}/local-models/';
  final fullModelPath = '$modelDirectory$modelName';
  
  // Check if model exists
  final modelFile = File(fullModelPath);
  final modelExists = await modelFile.exists();
  
  if (!modelExists) {
    print('Model is not downloaded, downloading to $fullModelPath...');
    
    // Create directory if it doesn't exist
    final dir = Directory(modelDirectory);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    
    // Download the model
    await downloadModelIfNotExists(
      modelUrl: modelUrl,
      modelFolderName: 'local-models',
      onProgress: (progress) {
        onProgress(progress / 100); // Convert to 0.0-1.0 range
      },
    );
  }
  
  // Check if model exists after download attempt
  if (!await File(fullModelPath).exists()) {
    throw Exception('Model is not downloaded');
  }
  
  // Initialize the Llama context
  return await initLlama(
    ContextParams(
      model: fullModelPath,
      useMlock: true,
      nCtx: 2048,
      nGpuLayers: Platform.isIOS ? 99 : 0,
    ),
    onProgress: onProgress,
  );
} 
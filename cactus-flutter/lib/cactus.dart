library cactus;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math'; // Added for Random

import 'package:path_provider/path_provider.dart'; // Added for getTemporaryDirectory

/// The main library for the Cactus Flutter plugin, providing on-device AI model inference
/// capabilities by interfacing with a native `cactus` backend (typically C++ based on llama.cpp).
///
/// This library exports all the necessary classes and functions to:
/// - Initialize and manage an AI model context ([CactusContext]).
/// - Configure model loading and inference parameters ([CactusInitParams], [CactusCompletionParams]).
/// - Perform text and chat completions ([CactusContext.completion], [CactusCompletionResult]).
/// - Generate text embeddings ([CactusContext.embedding]).
/// - Tokenize and detokenize text ([CactusContext.tokenize], [CactusContext.detokenize]).
/// - Manage chat conversations ([ChatMessage]).
/// - Download models from URLs ([downloadModel], also handled internally by [CactusContext.init]).
///
/// ## Getting Started
///
/// 1.  **Add the dependency** to your `pubspec.yaml`.
/// 2.  **Ensure the native `cactus` library** (e.g., `libcactus.so` on Android, `cactus.framework` on iOS)
///     is correctly included in your Flutter project and linked.
/// 3.  **Initialize a context**:
///     ```dart
///     import 'package:cactus/cactus.dart';
///
///     Future<void> main() async {
///       CactusContext? context;
///       try {
///         final initParams = CactusInitParams(
///           // Provide either modelPath for a local file or modelUrl to download
///           modelPath: '/path/to/your/model.gguf',
///           // modelUrl: 'https://example.com/your_model.gguf',
///           // modelFilename: 'downloaded_model.gguf', // if using modelUrl
///           contextSize: 2048, // Example context size
///           gpuLayers: 1,      // Example: offload some layers to GPU if supported
///           onInitProgress: (progress, status, isError) {
///             print('Init Progress: ${progress ?? "N/A"} - $status (Error: $isError)');
///           },
///         );
///         context = await CactusContext.init(initParams);
///         print('Cactus context initialized!');
///
///         // Now use the context for completion, embedding, etc.
///         final completionParams = CactusCompletionParams(
///           messages: [ChatMessage(role: 'user', content: 'Hello, world!')],
///           temperature: 0.7,
///           maxPredictedTokens: 100,
///         );
///         final result = await context.completion(completionParams);
///         print('Completion result: ${result.text}');
///
///       } catch (e) {
///         print('Error initializing or using Cactus: $e');
///       } finally {
///         context?.free(); // Crucial: always free the context when done!
///       }
///     }
///     ```
///
/// ## Key Components:
///
/// - [CactusContext]: The main class for interacting with a loaded AI model.
///   Manages the native model instance and provides methods for inference operations.
/// - [CactusInitParams]: Specifies parameters for initializing the [CactusContext],
///   such as model path/URL, context size, GPU layers, and chat templates.
/// - [CactusCompletionParams]: Defines parameters for controlling text or chat generation,
///   including the input messages, sampling settings (temperature, topK, topP, etc.),
///   stopping conditions, grammar, and streaming callbacks.
/// - [CactusCompletionResult]: Encapsulates the output of a completion operation,
///   including the generated text and metadata about how generation stopped.
/// - [ChatMessage]: Represents a single message in a chat sequence, with a `role`
///   (e.g., 'system', 'user', 'assistant') and `content`.
/// - [downloadModel]: A utility function for downloading model files, also used
///   internally by [CactusContext.init] when a `modelUrl` is provided.
///
/// Remember to always call [CactusContext.free] to release native resources
/// when you are finished with a context instance.

export 'chat.dart';
export 'init_params.dart';
export 'completion.dart';
export './context.dart';
export './model_downloader.dart';
export './tts_params.dart'; 
export './structs.dart';

// --- New Image Downloader Utility ---

/// Type definition for the progress callback used by [downloadImage].
///
/// [progress] is a value between 0.0 and 1.0 indicating download progress.
/// It can be null if the total size is unknown.
/// [statusMessage] provides a textual description of the current download status.
typedef ImageDownloadProgressCallback = void Function(double? progress, String statusMessage);

/// Downloads an image file from the given [url] to a specified or temporary [filePath].
///
/// - [url]: The URL from which to download the image.
/// - [filePath]: Optional. The local path where the downloaded image file will be saved.
///   If null, the image will be saved to the application's temporary directory
///   with a filename derived from the URL or a UUID.
/// - [onProgress]: An optional [ImageDownloadProgressCallback] to monitor download progress.
///
/// Returns the local [String] path to the downloaded image file.
/// Throws an [Exception] if the download fails.
Future<String> downloadImage(
  String url,
  {String? filePath,
  ImageDownloadProgressCallback? onProgress}
) async {
  onProgress?.call(null, 'Starting image download from: $url');
  
  String effectiveFilePath;
  if (filePath != null && filePath.isNotEmpty) {
    effectiveFilePath = filePath;
    // Ensure directory exists if a full path is provided
    final dir = Directory(effectiveFilePath.substring(0, effectiveFilePath.lastIndexOf('/')));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
  } else {
    final Directory tempDir = await getTemporaryDirectory();
    String filename = url.split('/').last;
    if (filename.contains('?')) filename = filename.split('?').first;
    if (filename.isEmpty || !filename.contains('.')) {
      // Generate a UUID-based filename if it's not a valid image filename
      final uuid = Uuid();
      filename = '${uuid.v4()}.jpg'; // Assume jpg, or try to guess from headers later
    }
    effectiveFilePath = '${tempDir.path}/$filename';
  }

  final File imageFile = File(effectiveFilePath);

  try {
    final httpClient = HttpClient();
    final request = await httpClient.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode >= 200 && response.statusCode < 300) {
      final IOSink fileSink = imageFile.openWrite();
      final totalBytes = response.contentLength;
      int receivedBytes = 0;

      onProgress?.call(0.0, 'Connected. Receiving image data...');

      await for (var chunk in response) {
        fileSink.add(chunk);
        receivedBytes += chunk.length;
        if (totalBytes != -1 && totalBytes != 0) {
          final progress = receivedBytes / totalBytes;
          onProgress?.call(
            progress,
            'Downloading image: ${(progress * 100).toStringAsFixed(1)}% ' 
            '(${(receivedBytes / 1024).toStringAsFixed(1)}KB / ${(totalBytes / 1024).toStringAsFixed(1)}KB)'
          );
        } else {
          onProgress?.call(
            null, 
            'Downloading image: ${(receivedBytes / 1024).toStringAsFixed(1)}KB received'
          );
        }
      }
      await fileSink.flush();
      await fileSink.close();
      
      onProgress?.call(1.0, 'Image download complete. Saved to $effectiveFilePath');
      return effectiveFilePath;
    } else {
      String responseBody = await response.transform(utf8.decoder).join();
      if (responseBody.length > 200) responseBody = "${responseBody.substring(0,200)}...";
      throw Exception(
          'Failed to download image. Status: ${response.statusCode}. Response: $responseBody');
    }
  } catch (e) {
    onProgress?.call(null, 'Error during image download: $e');
    // Attempt to delete partial file on error
    if (await imageFile.exists()) {
      try {
        await imageFile.delete();
      } catch (deleteErr) {
        // Log delete error, but rethrow original download error
        print("Error deleting partial image file: $deleteErr");
      }
    }
    rethrow;
  } finally {
    // HttpClient is not explicitly closed here as it might be reused by the app.
    // If it were created with autoUncompress = false or other custom settings,
    // and only for this download, then client.close(force: true) would be appropriate.
  }
}

// Helper class for UUID generation (simplified, assumes crypto support)
// In a real app, prefer a robust UUID package like 'uuid'.
class Uuid {
  final _random = Random.secure();
  String v4() {
    final bytes = List<int>.generate(16, (i) => _random.nextInt(256));
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant
    return bytes.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
  }
} 

import 'dart:async';
import 'dart:io';
import 'dart:convert';

/// Type definition for the progress callback used by [downloadModel].
///
/// [progress] is a value between 0.0 and 1.0 indicating download progress.
/// It can be null if the total size is unknown (e.g., some HTTP responses).
/// [statusMessage] provides a textual description of the current download status
/// (e.g., "Downloading: 50.5%", "Download complete").
typedef ModelDownloadProgressCallback = void Function(double? progress, String statusMessage);

/// Downloads a model file from the given [url] to the specified [filePath].
///
/// This function is used internally by [CactusContext.init] if a `modelUrl` is provided
/// in [CactusInitParams], but can also be used directly if manual download
/// management is needed (e.g., for downloading models to custom locations or
/// with custom UI for progress).
///
/// - [url]: The URL from which to download the model.
/// - [filePath]: The local path where the downloaded model file will be saved.
///   The directory for this path should exist, or the function might fail.
/// - [onProgress]: An optional [ModelDownloadProgressCallback] to monitor download progress.
///   It receives the download progress (0.0 to 1.0, or null if indeterminate)
///   and a status message.
///
/// Throws an [Exception] if the download fails (e.g., network error, non-200 status code,
/// file system error).
Future<void> downloadModel(
  String url,
  String filePath,
  {ModelDownloadProgressCallback? onProgress}
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
      // The message below might be redundant if onInitProgress in CactusContext already says this.
      // Consider if it's needed when called from CactusContext.
      // params.onInitProgress?.call(1.0, 'Model saved successfully to $filePath', false);
      // For standalone use, this is fine:
      onProgress?.call(1.0, 'Model saved successfully to $filePath');
    } else {
      // Attempt to read the response body for more error details if it's small.
      String responseBody = await response.transform(utf8.decoder).join();
      if (responseBody.length > 200) responseBody = "${responseBody.substring(0,200)}..."; // Truncate
      throw Exception(
          'Failed to download model. Status code: ${response.statusCode}. Response: $responseBody');
    }
    httpClient.close();
  } catch (e) {
    onProgress?.call(null, 'Error during download: $e');
    rethrow;
  }
} 
import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:io'; 
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart'; 

import 'package:cactus_flutter/cactus_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  CactusContext? _cactusContext;
  final TextEditingController _promptController = TextEditingController();
  String _completionResult = '';
  bool _isLoading = true; 
  String _initError = '';
  String _statusMessage = 'Initializing...'; 
  double? _downloadProgress; 

  final String _modelUrl =
      'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-IQ3_M.gguf';
  final String _modelFilename = 'SmolLM2-135M-Instruct-IQ3_M.gguf';

  @override
  void initState() {
    super.initState();
    _initializeCactus();
  }

  Future<void> _initializeCactus() async {
    setState(() {
      _isLoading = true;
      _initError = '';
      _statusMessage = 'Preparing to load model...';
      _downloadProgress = null; // Reset progress
    });

    try {
      setState(() {
        _statusMessage = 'Getting application documents directory...';
      });
      final Directory appDocDir = await getApplicationDocumentsDirectory();
      final String modelPath = '${appDocDir.path}/$_modelFilename';
      final File modelFile = File(modelPath);

      setState(() {
        _statusMessage = 'Checking for existing model: $_modelFilename';
      });

      if (!await modelFile.exists()) {
        print('Model not found locally. Starting download from $_modelUrl to $modelPath');
        setState(() {
          _statusMessage = 'Starting download: $_modelFilename';
          _downloadProgress = 0.0; // Indicate start of download
        });

        final httpClient = HttpClient();
        final request = await httpClient.getUrl(Uri.parse(_modelUrl));
        final response = await request.close();

        if (response.statusCode == 200) {
          final List<int> bytes = [];
          final totalBytes = response.contentLength;
          int receivedBytes = 0;

          await for (var chunk in response) {
            bytes.addAll(chunk);
            receivedBytes += chunk.length;
            if (totalBytes != -1 && totalBytes != 0) {
              setState(() {
                _downloadProgress = receivedBytes / totalBytes;
                _statusMessage =
                    'Downloading: ${( (_downloadProgress ?? 0) * 100).toStringAsFixed(1)}% '
                    '(${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB / ${(totalBytes / (1024 * 1024)).toStringAsFixed(2)}MB)';
              });
            } else {
              // Fallback if contentLength is not available
              setState(() {
                _downloadProgress = null; // Indeterminate
                _statusMessage =
                    'Downloading: ${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB received';
              });
            }
          }
          
          setState(() {
            _statusMessage = 'Download complete. Saving file...';
             _downloadProgress = 1.0; // Visually complete before saving
          });
          await modelFile.writeAsBytes(bytes);
          print('Model downloaded and saved successfully.');
          setState(() {
            _statusMessage = 'Model saved. Initializing Cactus...';
          });

        } else {
          throw Exception(
              'Failed to download model. Status code: ${response.statusCode}');
        }
        httpClient.close();
      } else {
        print('Model already exists at $modelPath');
        setState(() {
          _statusMessage = 'Model found locally. Initializing Cactus...';
        });
      }

      final params = CactusInitParams(modelPath: modelPath);
      _cactusContext = await CactusContext.init(params);
      print('CactusContext initialized successfully.');
      setState(() {
         _statusMessage = ''; 
         _isLoading = false; // Crucial: set isLoading to false on success
      });

    } catch (e) {
      print("Error initializing CactusContext: $e");
      setState(() {
        _initError = "Failed to initialize/load model: $e";
        _statusMessage = ''; // Clear status on error
        _isLoading = false;
      });
    } 
    // Removed finally block for isLoading = false as it's handled in success/error paths
  }

  @override
  void dispose() {
    _cactusContext?.free();
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _runCompletion() async {
    if (_cactusContext == null || _promptController.text.isEmpty) {
      setState(() {
        _completionResult = _cactusContext == null
            ? "CactusContext not initialized."
            : "Please enter a prompt.";
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _completionResult = '';
    });

    try {
      final params = CactusCompletionParams(prompt: _promptController.text);
      final result = await _cactusContext!.completion(params);
      setState(() {
        _completionResult = result.text;
      });
    } catch (e) {
      setState(() {
        _completionResult = "Error during completion: $e";
      });
      print("Error during completion: $e");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    const spacerSmall = SizedBox(height: 10);

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Cactus Flutter Example'),
        ),
        body: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(10),
            child: Column(
              children: [
                if (_initError.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Text(
                      _initError,
                      style: const TextStyle(color: Colors.red, fontSize: 16),
                      textAlign: TextAlign.center,
                    ),
                  ),
                if (_isLoading)
                  Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // Show LinearProgressIndicator if download has started, else Circular
                        if (_downloadProgress != null) 
                          LinearProgressIndicator(
                            value: _downloadProgress,
                            minHeight: 10, // Make it a bit thicker
                          )
                        else 
                          const CircularProgressIndicator(),
                        const SizedBox(height: 20), // Increased spacing
                        Text(
                          _statusMessage,
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 16, color: Colors.grey[700]),
                        ),
                      ],
                    ),
                  )
                else if (_cactusContext == null && _initError.isEmpty)
                  const Center(
                      child: Text(
                          "Initializing Cactus... (Ensure model path is correct)"))
                else if (_cactusContext != null) ...[
                  const Text(
                    'Enter a prompt and press "Generate" to get a completion from the model.',
                    style: TextStyle(fontSize: 16),
                    textAlign: TextAlign.center,
                  ),
                  spacerSmall,
                  TextField(
                    controller: _promptController,
                    decoration: const InputDecoration(
                      labelText: 'Prompt',
                      border: OutlineInputBorder(),
                    ),
                    minLines: 1,
                    maxLines: 5,
                  ),
                  spacerSmall,
                  ElevatedButton(
                    onPressed: _isLoading ? null : _runCompletion,
                    child: _isLoading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2))
                        : const Text('Generate'),
                  ),
                  spacerSmall,
                  if (_completionResult.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.all(8.0),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(4.0),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text("Completion Result:",
                              style: TextStyle(
                                  fontSize: 16, fontWeight: FontWeight.bold)),
                          spacerSmall,
                          Text(_completionResult,
                              style: const TextStyle(fontSize: 14)),
                        ],
                      ),
                    ),
                ]
              ],
            ),
          ),
        ),
      ),
    );
  }
}

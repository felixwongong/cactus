import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:convert'; 
import 'dart:io'; 
import 'package:path_provider/path_provider.dart'; 

import 'package:cactus_flutter/cactus_flutter.dart';

class ChatMessage {
  final String role;
  final String content;
  ChatMessage({required this.role, required this.content});

  Map<String, String> toJson() => {
    'role': role,
    'content': content,
  };
}

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
  List<ChatMessage> _chatMessages = []; 
  bool _isLoading = true;
  String _initError = '';
  String _statusMessage = 'Initializing...'; 
  double? _downloadProgress; 

  final ScrollController _scrollController = ScrollController(); 

  final String _modelUrl =
      'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-IQ3_M.gguf';
  final String _modelFilename = 'SmolLM2-135M-Instruct-IQ3_M.gguf';

  final String _chatMLTemplate = """
{% for message in messages %}
  {% if message.role == 'system' %}
    {{ '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
  {% elif message.role == 'user' %}
    {{ '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
  {% elif message.role == 'assistant' %}
    {{ '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
  {{ '<|im_start|>assistant\n' }}
{% endif %}
""";

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
      _downloadProgress = null; 
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
        print('Model not found locally. Starting download using library function.');
        await downloadModel( 
          _modelUrl,
          modelPath,
          onProgress: (progress, status) {
            setState(() {
              _downloadProgress = progress;
              _statusMessage = status;
            });
          },
        );
        setState(() {
          _statusMessage = 'Model ready. Initializing Cactus...';
        });
      } else {
        print('Model already exists at $modelPath');
        setState(() {
          _statusMessage = 'Model found locally. Initializing Cactus...';
        });
      }

      final params = CactusInitParams(
        modelPath: modelPath,
        chatTemplate: _chatMLTemplate, 
      );
      _cactusContext = await CactusContext.init(params);
      print('CactusContext initialized successfully.');
      setState(() {
         _statusMessage = ''; 
         _isLoading = false; 
      });

    } catch (e) {
      print("Error during initialization or model download: $e");
      setState(() {
        _initError = "Failed to initialize/load model: $e";
        _statusMessage = ''; 
        _isLoading = false;
      });
    } 
  }

  @override
  void dispose() {
    _cactusContext?.free();
    _promptController.dispose();
    _scrollController.dispose(); 
    super.dispose();
  }

  Future<void> _sendMessage() async {
    final userInput = _promptController.text.trim();
    if (userInput.isEmpty) return;

    if (_cactusContext == null) {
      setState(() {
        _chatMessages.add(ChatMessage(role: 'system', content: 'Error: CactusContext not initialized.'));
      });
      return;
    }

    setState(() {
      _chatMessages.add(ChatMessage(role: 'user', content: userInput));
      _isLoading = true;
    });
    _promptController.clear();
    _scrollToBottom();

    try {
      final List<Map<String, String>> messagesForJson = 
          _chatMessages.where((m) => m.role == 'user' || m.role == 'assistant').map((m) => m.toJson()).toList();

      final String jsonPrompt = jsonEncode(messagesForJson);
      
      final completionParams = CactusCompletionParams(
        prompt: jsonPrompt,
        stopSequences: ['<|im_end|>'], 
      );
      final result = await _cactusContext!.completion(completionParams);

      String assistantResponse = result.text; // Get the raw response

      // Check if the response stopped due to the <|im_end|> token 
      // and if the token is present at the end of the response.
      if (result.stoppedWord && 
          result.stoppingWord == '<|im_end|>' && 
          assistantResponse.endsWith('<|im_end|>')) {
        // Remove the <|im_end|> token from the end of the string
        assistantResponse = assistantResponse.substring(0, assistantResponse.length - '<|im_end|>'.length);
      }

      setState(() {
        // Use the potentially trimmed response, and also trim any general whitespace
        _chatMessages.add(ChatMessage(role: 'assistant', content: assistantResponse.trim()));
      });
    } catch (e) {
      setState(() {
        _chatMessages.add(ChatMessage(role: 'system', content: "Error during completion: $e"));
      });
      print("Error during completion: $e");
    } finally {
      setState(() {
        _isLoading = false;
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    const spacerSmall = SizedBox(height: 10);

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Cactus Flutter Chat'), 
        ),
        body: Column( 
          children: [
            if (_isLoading && _chatMessages.isEmpty) 
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      if (_downloadProgress != null) 
                        LinearProgressIndicator(
                          value: _downloadProgress,
                          minHeight: 10,
                        )
                      else 
                        const CircularProgressIndicator(),
                      const SizedBox(height: 20),
                      Text(
                        _statusMessage,
                        textAlign: TextAlign.center,
                        style: TextStyle(fontSize: 16, color: Colors.grey[700]),
                      ),
                    ],
                  ),
                ),
              )
            else if (_initError.isNotEmpty)
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text(
                  _initError,
                  style: const TextStyle(color: Colors.red, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              ),
            
            Expanded(
              child: ListView.builder(
                controller: _scrollController,
                padding: const EdgeInsets.all(8.0),
                itemCount: _chatMessages.length,
                itemBuilder: (context, index) {
                  final message = _chatMessages[index];
                  bool isUser = message.role == 'user';
                  bool isSystem = message.role == 'system';
                  return Align(
                    alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                    child: Container(
                      margin: const EdgeInsets.symmetric(vertical: 4.0),
                      padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8.0),
                      decoration: BoxDecoration(
                        color: isSystem ? Colors.red[100] : (isUser ? Colors.blue[100] : Colors.green[100]),
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      child: Text(message.content, style: TextStyle(color: isSystem ? Colors.red[900] : Colors.black)),
                    ),
                  );
                },
              ),
            ),

            // Input area
            if (_cactusContext != null && _initError.isEmpty) // Only show input if initialized
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _promptController,
                        decoration: const InputDecoration(
                          hintText: 'Type your message...',
                          border: OutlineInputBorder(),
                        ),
                        onSubmitted: (_) => _sendMessage(), // Send on submit
                        minLines: 1,
                        maxLines: 3,
                      ),
                    ),
                    spacerSmall,
                    IconButton(
                      icon: _isLoading ? const SizedBox(width:24, height:24, child:CircularProgressIndicator(strokeWidth: 2,)) : const Icon(Icons.send),
                      onPressed: _isLoading ? null : _sendMessage, // Disable while loading previous response
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

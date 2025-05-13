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
  bool _isStreamingComplete = false;

  final ScrollController _scrollController = ScrollController(); 

  final String _modelUrl =
      'https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf';
  final String _modelFilename = 'SmolLM2-360M-Instruct.gguf';

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

      // ---- DEBUG: Print model path and existence ----
      print("Checking for model at path: $modelPath");
      bool fileExists = await modelFile.exists();
      print("Model file exists before download check: $fileExists");
      // ---- END DEBUG ----

      setState(() {
        _statusMessage = 'Checking for existing model: $_modelFilename';
      });

      if (!fileExists) { // Use the pre-checked value
        print('Model not found locally. Starting download using library function for: $_modelFilename from $_modelUrl'); // More explicit
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
      _chatMessages.add(ChatMessage(role: 'assistant', content: '')); 
      _isLoading = true;
      _isStreamingComplete = false;
    });
    _promptController.clear();
    _scrollToBottom();

    try {
      final List<Map<String, String>> messagesForHistory = 
          _chatMessages
            .where((m) => (m.role == 'user' || m.role == 'assistant') && m.content.isNotEmpty) 
            .take(_chatMessages.length - (_chatMessages.last.role == 'assistant' ? 2 : 1)) // Take all processed messages
            .map((m) => m.toJson()).toList();
      
      // Ensure we correctly select messages for the prompt
      // This should include the latest user message that triggered this call.
      final List<Map<String,String>> messagesToFormat = [
        ...messagesForHistory,
        if (_chatMessages.last.role == 'assistant') // If an empty assistant message was just added
          _chatMessages[_chatMessages.length - 2].toJson() // This is the user's current message
        else // Should not happen if logic is user input -> empty assistant -> call
          _chatMessages.last.toJson(), 
      ];
      
      // Apply ChatML template logic
      StringBuffer formattedPromptBuffer = StringBuffer();
      for (var messageData in messagesToFormat) {
        String? role = messageData['role'];
        String? content = messageData['content'];

        if (role != null && content != null) {
          if (role == 'system' || role == 'user' || role == 'assistant') {
            formattedPromptBuffer.write('<|im_start|>');
            formattedPromptBuffer.write(role);
            formattedPromptBuffer.write('\n');
            formattedPromptBuffer.write(content);
            formattedPromptBuffer.write('<|im_end|>\n');
          }
        }
      }
      // Add the generation prompt for the assistant
      formattedPromptBuffer.write('<|im_start|>assistant\n');

      final String finalFormattedPrompt = formattedPromptBuffer.toString();
      // For debugging, print the formatted prompt
      // print("Formatted Prompt:\n$finalFormattedPrompt");
      
      final completionParams = CactusCompletionParams(
        prompt: finalFormattedPrompt, // Use the new formatted prompt
        stopSequences: ['<|im_end|>'], 
        temperature: 0.7,
        topK: 10,
        topP: 0.9,
        onNewToken: (String token) {
          print("Streamed token: '${token}' (length: ${token.length})");

          // Only process if we are actively loading/streaming for the current message
          if (!_isLoading || _isStreamingComplete) { 
            print("Ignoring token: _isLoading=$_isLoading, _isStreamingComplete=$_isStreamingComplete");
            // Still return false if it was a stop token to ensure C++ side stops, 
            // but don't return false if we are just ignoring due to isLoading being false.
            return !(token == '<|im_end|>' && _isStreamingComplete); // essentially, if already complete, tell C to stop if it asks.
          }

          if (token == '<|im_end|>') {
            setState(() { // SetState to ensure UI reflects loading state if changed by this
              _isStreamingComplete = true;
            });
            print("Stop token <|im_end|> received in stream. Returning false to C++.");
            return false; // Signal C++ to stop
          }
          
          if (token.isNotEmpty) {
            setState(() {
              if (_chatMessages.isNotEmpty && _chatMessages.last.role == 'assistant') {
                final lastMessage = _chatMessages.last;
                _chatMessages[_chatMessages.length - 1] = ChatMessage(
                  role: lastMessage.role, 
                  content: lastMessage.content + token,
                );
                _scrollToBottom(); 
              } 
            });
          }
          return true; // Continue streaming
        },
      );

      final result = await _cactusContext!.completion(completionParams);
      _isStreamingComplete = true;

      print("Completion finished. Stop reason: ${result.stoppedWord}, ${result.stoppedEos}, ${result.stoppedLimit}");
      print("Final result.text from C++: '${result.text}'");

      String finalCleanText = result.text;
      if (finalCleanText.endsWith('<|im_end|>')) {
        finalCleanText = finalCleanText.substring(0, finalCleanText.length - '<|im_end|>'.length);
      }
      
      setState(() {
        if (_chatMessages.isNotEmpty && _chatMessages.last.role == 'assistant') {
          _chatMessages[_chatMessages.length - 1] = ChatMessage(
            role: 'assistant',
            content: finalCleanText.trim(),
          );
        }
      });

    } catch (e) {
      setState(() {
        if (_chatMessages.isNotEmpty && _chatMessages.last.role == 'assistant') {
           final lastMessage = _chatMessages.last;
           _chatMessages[_chatMessages.length - 1] = ChatMessage(
             role: lastMessage.role,
             content: "Error: $e",
           );
        } else {
           _chatMessages.add(ChatMessage(role: 'system', content: "Error during completion: $e"));
        }
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

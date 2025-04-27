import 'package:flutter/material.dart';
import 'package:cactus_flutter/cactus_flutter.dart';
import 'components/header.dart';
import 'components/message.dart';
import 'components/message_field.dart';
import 'utils/constants.dart';
import 'utils/model_utils.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // State variables
  final List<Message> _messages = [];
  String _currentMessage = '';
  bool _isGenerating = false;
  bool _isModelLoaded = false;
  double _downloadProgress = 0.0;
  LlamaContext? _context;

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  @override
  void dispose() {
    _releaseModel();
    super.dispose();
  }

  // Initialize the LLM model
  Future<void> _initializeModel() async {
    try {
      final context = await initLlamaContext(
        onProgress: (progress) {
          setState(() {
            _downloadProgress = progress;
          });
        },
      );

      setState(() {
        _context = context;
        _isModelLoaded = true;
      });
    } catch (e) {
      print('Error initializing model: $e');
    }
  }

  // Release the model when done
  Future<void> _releaseModel() async {
    if (_context != null) {
      await _context!.release();
      _context = null;
    }
  }

  // Handle sending a message
  Future<void> _handleSendMessage() async {
    if (_currentMessage.isEmpty || _isGenerating) return;

    final userMessage = Message(role: 'user', content: _currentMessage);
    
    setState(() {
      _messages.add(userMessage);
      _currentMessage = '';
      _isGenerating = true;
    });

    await _getLLMcompletion(_messages);
  }

  // Get completion from the LLM
  Future<void> _getLLMcompletion(List<Message> messages) async {
    if (_context == null) {
      print('Model not yet loaded');
      return;
    }

    // Convert our Message objects to ChatMessage objects for the Cactus API
    final chatMessages = messages.map((msg) => 
      ChatMessage(role: msg.role, content: msg.content)
    ).toList();

    String llmResponse = '';
    
    // Add an empty placeholder for the assistant's response
    setState(() {
      _messages.add(Message(role: 'assistant', content: llmResponse));
    });

    try {
      await _context!.completion(
        CompletionParams(
          messages: chatMessages,
          nPredict: 512,
          stop: stopWords,
        ),
        callback: (data) {
          if (data.token.isNotEmpty) {
            llmResponse += data.token;
            
            // Update the last message (assistant's response)
            setState(() {
              _messages[_messages.length - 1] = Message(
                role: 'assistant', 
                content: llmResponse
              );
            });
          }
        },
      );
    } catch (e) {
      print('Error during completion: $e');
    } finally {
      setState(() {
        _isGenerating = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      body: SafeArea(
        child: Column(
          children: [
            const Header(),
            Expanded(
              child: _isModelLoaded
                  ? _buildChatArea()
                  : _buildLoadingView(),
            ),
            MessageField(
              message: _currentMessage,
              setMessage: (text) {
                setState(() {
                  _currentMessage = text;
                });
              },
              handleSendMessage: _handleSendMessage,
              isGenerating: _isGenerating,
            ),
          ],
        ),
      ),
    );
  }

  // Build the chat messages area
  Widget _buildChatArea() {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _messages.length,
      itemBuilder: (context, index) {
        return MessageBubble(message: _messages[index]);
      },
    );
  }

  // Build the loading view
  Widget _buildLoadingView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(
            value: _downloadProgress > 0 ? _downloadProgress : null,
          ),
          const SizedBox(height: 16),
          Text(
            'Hold tight... setting up the model (${(_downloadProgress * 100).toStringAsFixed(1)}%)',
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
} 
import 'package:flutter/material.dart';

/// Input field for entering messages
class MessageField extends StatelessWidget {
  final String message;
  final Function(String) setMessage;
  final VoidCallback handleSendMessage;
  final bool isGenerating;

  const MessageField({
    super.key,
    required this.message,
    required this.setMessage,
    required this.handleSendMessage,
    required this.isGenerating,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 1,
            blurRadius: 3,
            offset: const Offset(0, -1),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: TextEditingController(text: message)
                ..selection = TextSelection.fromPosition(
                  TextPosition(offset: message.length),
                ),
              onChanged: setMessage,
              decoration: const InputDecoration(
                hintText: 'Type a message...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.all(Radius.circular(20)),
                ),
                contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              ),
              enabled: !isGenerating,
              onSubmitted: (_) => handleSendMessage(),
            ),
          ),
          const SizedBox(width: 10),
          ElevatedButton(
            onPressed: isGenerating ? null : handleSendMessage,
            style: ElevatedButton.styleFrom(
              shape: const CircleBorder(),
              padding: const EdgeInsets.all(12),
            ),
            child: Icon(
              Icons.send,
              color: isGenerating ? Colors.grey : Theme.of(context).colorScheme.primary,
            ),
          ),
        ],
      ),
    );
  }
} 
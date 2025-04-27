import 'package:flutter/material.dart';
import 'package:cactus_flutter/cactus_flutter.dart';

/// A message in the chat
class Message {
  final String role;
  final String content;

  Message({required this.role, required this.content});
}

/// A bubble widget to display a message
class MessageBubble extends StatelessWidget {
  final Message message;

  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == 'user';

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.8,
        ),
        decoration: BoxDecoration(
          color: isUser 
              ? Theme.of(context).colorScheme.primary 
              : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          message.content,
          style: TextStyle(
            color: isUser 
                ? Theme.of(context).colorScheme.onPrimary 
                : Colors.black,
            fontStyle: isUser ? FontStyle.normal : FontStyle.italic,
          ),
        ),
      ),
    );
  }
} 
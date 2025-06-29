import 'dart:convert';
import 'dart:async';
import 'chat.dart';
import './types.dart';

typedef ToolFunction = Future<dynamic> Function(Map<String, dynamic> args);

class FunctionDefinition {
  final String name;
  final String description;
  final Map<String, dynamic> parameters;

  FunctionDefinition({
    required this.name,
    required this.description,
    required this.parameters,
  });

  Map<String, dynamic> toJson() => {
        'name': name,
        'description': description,
        'parameters': parameters,
      };
}

class ToolDefinition {
  final FunctionDefinition function;

  ToolDefinition({required this.function});

  Map<String, dynamic> toJson() => {
        'type': 'function',
        'function': function.toJson(),
      };
}

class ToolCall {
  final String id;
  final String name;
  final Map<String, dynamic> arguments;

  ToolCall({required this.id, required this.name, required this.arguments});
}

class ToolExecutionResult {
  final bool toolCalled;
  final String? toolName;
  final Map<String, dynamic>? toolInput;
  final dynamic toolOutput;
  final String? error;

  ToolExecutionResult._({
    required this.toolCalled,
    this.toolName,
    this.toolInput,
    this.toolOutput,
    this.error,
  });

  factory ToolExecutionResult.called({
    required String name,
    required Map<String, dynamic> input,
    dynamic output,
  }) => ToolExecutionResult._(
        toolCalled: true,
        toolName: name,
        toolInput: input,
        toolOutput: output,
      );

  factory ToolExecutionResult.error({
    required String name,
    required Map<String, dynamic> input,
    required String error,
  }) => ToolExecutionResult._(
        toolCalled: true,
        toolName: name,
        toolInput: input,
        error: error,
      );

  factory ToolExecutionResult.notCalled() => ToolExecutionResult._(toolCalled: false);
}

/// Tools collection for function calling
class Tools {
  final Map<String, ToolFunction> _functions = {};
  final Map<String, ToolDefinition> _definitions = {};

  /// Add a tool function
  void addTool(String name, ToolFunction function, ToolDefinition definition) {
    _functions[name] = function;
    _definitions[name] = definition;
  }

  /// Add a simple tool with automatic parameter inference
  void addSimpleTool({
    required String name,
    required String description,
    required ToolFunction function,
    Map<String, dynamic>? parameters,
  }) {
    final definition = ToolDefinition(
      function: FunctionDefinition(
        name: name,
        description: description,
        parameters: parameters ?? {
          'type': 'object',
          'properties': {},
        },
      ),
    );
    addTool(name, function, definition);
  }

  /// Get all tool definitions
  List<ToolDefinition> getDefinitions() => _definitions.values.toList();

  /// Get tool definitions as JSON
  String getDefinitionsJson() {
    return jsonEncode(_definitions.values.map((def) => def.toJson()).toList());
  }

  /// Execute a tool function
  Future<ToolExecutionResult> executeTool(String name, Map<String, dynamic> args) async {
    if (!_functions.containsKey(name)) {
      return ToolExecutionResult.error(
        name: name,
        input: args,
        error: 'Tool "$name" not found',
      );
    }

    try {
      final result = await _functions[name]!(args);
      return ToolExecutionResult.called(
        name: name,
        input: args,
        output: result,
      );
    } catch (e) {
      return ToolExecutionResult.error(
        name: name,
        input: args,
        error: e.toString(),
      );
    }
  }

  /// Check if a tool exists
  bool hasTool(String name) => _functions.containsKey(name);

  /// Get all tool names
  List<String> getToolNames() => _functions.keys.toList();

  /// Clear all tools
  void clear() {
    _functions.clear();
    _definitions.clear();
  }
}

/// Tool calling utilities
class ToolCalling {
  /// Inject tools into the first message (system message)
  static List<ChatMessage> injectToolsIntoMessages(
    List<ChatMessage> messages,
    Tools tools,
  ) {
    if (tools.getDefinitions().isEmpty) return messages;

    final toolsText = '''

You have access to the following functions. Use them if required:

${tools.getDefinitionsJson()}''';

    final List<ChatMessage> result = [...messages];
    
    if (result.isNotEmpty && result.first.role == 'system') {
      // Append to existing system message
      result[0] = ChatMessage(
        role: 'system',
        content: result.first.content + toolsText,
      );
    } else {
      // Insert new system message
      result.insert(0, ChatMessage(
        role: 'system',
        content: 'You are a helpful assistant.$toolsText',
      ));
    }

    return result;
  }

  /// Parse tool calls from completion output
  static List<ToolCall> parseToolCalls(String text) {
    final List<ToolCall> toolCalls = [];
    
    // Look for function call patterns
    final functionCallPattern = RegExp(
      r'<functioncall>\s*({[^}]+})\s*</functioncall>',
      caseSensitive: false,
    );
    
    final matches = functionCallPattern.allMatches(text);
    
    for (final match in matches) {
      try {
        final functionJson = match.group(1);
        if (functionJson != null) {
          final parsed = jsonDecode(functionJson) as Map<String, dynamic>;
          
          if (parsed.containsKey('name')) {
            toolCalls.add(ToolCall(
              id: 'call_${DateTime.now().millisecondsSinceEpoch}',
              name: parsed['name'],
              arguments: Map<String, dynamic>.from(parsed['arguments'] ?? {}),
            ));
          }
        }
      } catch (e) {
        // Ignore malformed tool calls
        continue;
      }
    }
    
    // Also try OpenAI-style tool calls
    final openaiPattern = RegExp(
      r'```json\s*({[^}]*"name"[^}]*})\s*```',
      caseSensitive: false,
    );
    
    final openaiMatches = openaiPattern.allMatches(text);
    
    for (final match in openaiMatches) {
      try {
        final toolJson = match.group(1);
        if (toolJson != null) {
          final parsed = jsonDecode(toolJson) as Map<String, dynamic>;
          
          if (parsed.containsKey('name')) {
            toolCalls.add(ToolCall(
              id: 'call_${DateTime.now().millisecondsSinceEpoch}',
              name: parsed['name'],
              arguments: Map<String, dynamic>.from(parsed['arguments'] ?? {}),
            ));
          }
        }
      } catch (e) {
        continue;
      }
    }
    
    return toolCalls;
  }

  /// Parse and execute tools from completion result
  static Future<ToolExecutionResult> parseAndExecuteTool(
    CactusCompletionResult result,
    Tools tools,
  ) async {
    final toolCalls = parseToolCalls(result.text);
    
    if (toolCalls.isEmpty) {
      return ToolExecutionResult.notCalled();
    }
    
    // Execute the first tool call
    final toolCall = toolCalls.first;
    return await tools.executeTool(toolCall.name, toolCall.arguments);
  }

  /// Update messages with tool call and response
  static List<ChatMessage> updateMessagesWithToolCall(
    List<ChatMessage> messages,
    String toolName,
    Map<String, dynamic> toolInput,
    dynamic toolOutput,
  ) {
    final List<ChatMessage> result = [...messages];
    
    // Add assistant message with tool call
    result.add(ChatMessage(
              role: 'assistant',
      content: '''<functioncall>
${jsonEncode({
  'name': toolName,
  'arguments': toolInput,
})}
</functioncall>''',
    ));
    
    // Add tool response
    final outputText = toolOutput is String ? toolOutput : jsonEncode(toolOutput);
    result.add(ChatMessage(
      role: 'user', // Tool responses are treated as user messages
      content: '''<functionresponse>
$outputText
</functionresponse>

Please continue with your response.''',
    ));
    
    return result;
  }
} 
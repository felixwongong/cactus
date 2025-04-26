import Foundation

/// Utility for formatting chat messages
public enum ChatFormatting {
    /// Format chat messages into a prompt using a template
    /// - Parameters:
    ///   - messages: Array of chat messages
    ///   - template: Optional template to use (nil = model default)
    ///   - useJinja: Whether to use Jinja templating
    /// - Returns: Formatted chat text
    public static func formatChat(
        messages: [ChatMessage],
        template: String? = nil,
        useJinja: Bool = false
    ) -> String {
        // This is a simple implementation of common chat templates
        // For complex formatting and proper template rendering,
        // we would use the C++ implementation
        
        if !messages.isEmpty {
            if useJinja {
                // Complex Jinja formatting would require the C++ implementation
                // This is just a simplified placeholder
                return formatLlamaChat(messages)
            } else if let template = template {
                // Custom template
                return applyCustomTemplate(messages: messages, template: template)
            } else {
                // Default to Llama chat format
                return formatLlamaChat(messages)
            }
        }
        
        return ""
    }
    
    /// Format messages using the standard Llama chat format
    /// - Parameter messages: Array of chat messages
    /// - Returns: Formatted chat text
    private static func formatLlamaChat(_ messages: [ChatMessage]) -> String {
        var formattedChat = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
        
        for message in messages {
            let role = message.role.rawValue
            let content = message.content
            
            formattedChat += "<|im_start|>\(role)\n\(content)\n<|im_end|>\n"
        }
        
        formattedChat += "<|im_start|>assistant\n"
        return formattedChat
    }
    
    /// Apply a custom template to messages
    /// - Parameters:
    ///   - messages: Array of chat messages
    ///   - template: Template string with placeholders
    /// - Returns: Formatted chat text
    private static func applyCustomTemplate(messages: [ChatMessage], template: String) -> String {
        // Very basic template support for demonstration
        // Real implementation would use the C++ template engine
        
        var result = template
        
        // Replace system message
        if let systemMessage = messages.first(where: { $0.role == .system }) {
            result = result.replacingOccurrences(of: "{{system}}", with: systemMessage.content)
        }
        
        // Replace message sequence
        var messagesText = ""
        for message in messages where message.role != .system {
            let roleStr = message.role.rawValue
            messagesText += "{{#if \(roleStr)}}\(message.content){{/if}}\n"
        }
        
        result = result.replacingOccurrences(of: "{{messages}}", with: messagesText)
        
        return result
    }
    
    /// Format a chat for completion with additional parameters
    /// - Parameters:
    ///   - messages: Array of chat messages
    ///   - template: Optional template to use
    ///   - useJinja: Whether to use Jinja templating
    ///   - responseFormat: Optional response format
    ///   - tools: Optional tools configuration
    /// - Returns: Formatted chat and metadata
    public static func formatChatForCompletion(
        messages: [ChatMessage],
        template: String? = nil,
        useJinja: Bool = false,
        responseFormat: CompletionResponseFormat? = nil,
        tools: [Any]? = nil
    ) -> (text: String, metadata: [String: Any]) {
        // Format the chat
        let formattedChat = formatChat(
            messages: messages,
            template: template,
            useJinja: useJinja
        )
        
        // Create metadata with formatting info
        var metadata: [String: Any] = [
            "formatted_with_jinja": useJinja
        ]
        
        if let template = template {
            metadata["template"] = template
        }
        
        return (formattedChat, metadata)
    }
} 
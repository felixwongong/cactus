# Cactus

Cactus is a high-level interface layer built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp) designed to facilitate Large Language Model (LLM) inference in C++ applications.

## Overview

Cactus provides a streamlined API for common LLM operations including:

- Model loading and management
- Context management and token handling
- Text generation and completion
- Chat formatting with templates (including OpenAI-compatible formats)
- Token sampling with various strategies
- LoRA adapter support
- Embedding generation
- Performance benchmarking

## Key Features

- **Simplified API**: Clean interface to the powerful llama.cpp library
- **Chat Template Support**: Format chat conversations with built-in templates
- **LoRA Support**: Easy application and management of LoRA adapters
- **Flexible Sampling**: Various sampling strategies for text generation
- **Embeddings**: Generate embeddings from text
- **Benchmarking**: Built-in performance testing
- **Logging**: Comprehensive logging system with different verbosity levels

## Usage

### Basic Example

```cpp
#include "cactus.h"

int main() {
    // Initialize parameters
    common_params params;
    params.model = "path/to/model.gguf";
    
    // Create context
    cactus::cactus_context context;
    if (!context.loadModel(params)) {
        return 1;
    }
    
    // Initialize sampling
    context.initSampling();
    
    // Set prompt
    params.prompt = "Hello, I am a";
    
    // Load prompt and begin completion
    context.loadPrompt();
    context.beginCompletion();
    
    // Generate text
    while (context.has_next_token) {
        auto token = context.doCompletion();
        // Process token as needed
    }
    
    // Get generated text
    std::cout << context.generated_text << std::endl;
    
    return 0;
}
```

### Chat Example

```cpp
#include "cactus.h"
#include <string>

int main() {
    // Initialize parameters
    common_params params;
    params.model = "path/to/model.gguf";
    params.chat_template = "chatml"; // or other template
    
    // Create context
    cactus::cactus_context context;
    if (!context.loadModel(params)) {
        return 1;
    }
    
    // Initialize sampling
    context.initSampling();
    
    // Format a chat with Jinja template
    std::string messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ])";
    
    auto chat_params = context.getFormattedChatWithJinja(messages, "", "", "", false, "");
    
    // Set prompt
    params.prompt = chat_params.prompt;
    
    // Load prompt and begin completion
    context.loadPrompt();
    context.beginCompletion();
    
    // Generate response
    while (context.has_next_token) {
        auto token = context.doCompletion();
        // Process token as needed
    }
    
    // Get generated text
    std::cout << context.generated_text << std::endl;
    
    return 0;
}
```

## API Documentation

See [cactus.h](cactus.h) for the complete API reference.

## Dependencies

- llama.cpp (core inference engine)
- ggml (tensor library used by llama.cpp)
- gguf (model format library)

## Building

Cactus is built as part of the llama.cpp project. See [llama.cpp build instructions](https://github.com/ggerganov/llama.cpp#build) for details.

## License

Cactus follows the same license as llama.cpp.

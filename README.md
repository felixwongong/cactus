![Logo](assets/banner.jpg)

Cactus is a lightweight, high-performance framework for running AI models on mobile phones. cactus has unified and consistent APIs across React-Natiive, Android/Kotlin, Android/Java, iOS/Swift, iOS/Objective-C++, and Flutter/Dart. For now, leverages GGML backends to support any GGUF model already compatible with Llama.cpp, while we focus on broadly supporting every moblie app development platform, as well as upcoming features like MCP, phone tool use, thinking, prompt-enhancement, higher-level APIs. We are backed by YCombinator, Oxford Seed Fund and Google For Startups. Contributors with any of the above experiences are welcome! However, feel free to submit cool example apps you built with cactus, issues or tests!

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Applications                         │
└───────────────┬─────────────────┬───────────────────────┘
                │                 │                
┌───────────────┼─────────────────┼───────────────────────┐
│ ┌─────────────▼─────┐ ┌─────────▼───────┐ ┌─────────────┐│
│ │     React API      │ │   Flutter API   │ │  Native APIs ││
│ └───────────────────┘ └─────────────────┘ └─────────────┘│
│                Platform Bindings                          │
└───────────────┬─────────────────┬───────────────────────┘
                │                 │                
┌───────────────▼─────────────────▼───────────────────────┐
│                   Cactus Core (C++)                      │
└───────────────┬─────────────────┬───────────────────────┘
                │                 │                
┌───────────────▼─────┐ ┌─────────▼───────────────────────┐
│   llama.cpp Core     │ │    GGML/GGUF Model Format       │
└─────────────────────┘ └─────────────────────────────────┘
```
- **Features**:
  - Model download from HuggingFace 
  - Text completion and chat completion
  - Streaming token generation 
  - Embedding generation
  - JSON mode with schema validation
  - Chat templates with Jinja2 support
  - Low memory footprint
  - Battery-efficient inference
  - Background processing

## Benchmarks 


## Platform Support

- **React/React Native** (shipped)
  - TypeScript API
  - Async/Promise based interface
  - Event system for token streaming

- **Android** (currently testing)
  - API 24+ (Android 7.0+)
  - ARM64 and x86_64 architectures
  - JNI interface for Java/Kotlin integration

- **iOS** (in development)
  - iOS 13.0+
  - ARM64 architecture
  - Metal acceleration for Apple Silicon
  - Swift/Objective-C API

- **Flutter** (in development)
  - Platform channel communication
  - Dart API with native performance
  - iOS and Android support

## Getting Started

### Installation

#### React Native

```bash
npm install @cactus/react-native
# or
yarn add @cactus/react-native

# In your ios folder
npx pod-install
```

#### Android

Add to your `build.gradle`:

```gradle
dependencies {
    implementation 'com.cactuscompute:cactus-android:x.y.z'
}
```

#### iOS

Add to your `Podfile`:

```ruby
pod 'Cactus', '^0.0.3'
```

#### Flutter

```bash
flutter pub add cactus_flutter
```

### Basic Usage Example

#### C++ (Native)

```cpp
#include "cactus.h"

int main() {
    // Initialize parameters
    cactus::common_params params;
    params.model = "models/llama-2-7b-chat.gguf";
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.n_threads = 4;
    
    // Create context and load model
    cactus::cactus_context ctx;
    if (!ctx.loadModel(params)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // Set up completion parameters
    params.prompt = "Explain quantum computing in simple terms";
    params.n_predict = 512;
    params.sampling.temp = 0.7f;
    params.sampling.top_k = 40;
    params.sampling.top_p = 0.95f;
    
    // Generate completion
    ctx.loadPrompt();
    ctx.beginCompletion();
    
    std::string result;
    while (true) {
        auto token_output = ctx.doCompletion();
        if (!ctx.has_next_token) break;
        std::cout << ctx.generated_text;
        result += ctx.generated_text;
    }
    
    return 0;
}
```

#### React Native (JavaScript/TypeScript)

```typescript
import { initLlama, LlamaContext } from '@cactus/react-native';

// Load model
const context = await initLlama({
  model: 'models/llama-2-7b-chat.gguf',
  n_ctx: 2048,
  n_batch: 512,
  n_threads: 4
});

// Generate completion
const result = await context.completion({
  prompt: 'Explain quantum computing in simple terms',
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  n_predict: 512
}, (token) => {
  // Process each token
  process.stdout.write(token.token);
});

// Clean up
await context.release();
```

#### Kotlin (Android)

```kotlin
import com.cactus.LlamaContext

// Load model
val llamaContext = LlamaContext.createContext(
    applicationContext,
    "models/llama-2-7b-chat.gguf",
    LlamaContextParams(
        nCtx = 2048,
        nBatch = 512, 
        nThreads = 4
    )
)

// Set up completion
val result = llamaContext.completion(
    CompletionParams(
        prompt = "Explain quantum computing in simple terms",
        temperature = 0.7f,
        topK = 40,
        topP = 0.95f,
        nPredict = 512
    )
) { token ->
    // Stream tokens as they're generated
    print(token.text)
}

// Clean up
llamaContext.release()
```

#### Swift (iOS)

```swift
import Cactus

// Load model
let context = try CactusContext(
    modelPath: "models/llama-2-7b-chat.gguf",
    contextParams: ContextParams(
        contextSize: 2048,
        batchSize: 512,
        threadCount: 4
    )
)

// Generate completion
try context.completion(
    params: CompletionParams(
        prompt: "Explain quantum computing in simple terms",
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxTokens: 512
    )
) { token in
    // Process each token as it's generated
    print(token.text, terminator: "")
}

// Clean up
context.release()
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Cactus for React Native

A lightweight, high-performance framework for running AI models on mobile devices with React Native.

## Installation

```bash
# Using npm
npm install react-native-fs
npm install cactus-react-native

# Using yarn
yarn add react-native-fs
yarn add cactus-react-native

# For iOS, install pods if not on Expo
npx pod-install
```

## Basic Usage

### Initialize a Model

```typescript
import { initLlama, LlamaContext } from 'cactus-react-native';

// Initialize the model
const context = await initLlama({
  model: 'models/llama-2-7b-chat.gguf', // Path to your model
  n_ctx: 2048,                          // Context size
  n_batch: 512,                         // Batch size for prompt processing
  n_threads: 4                          // Number of threads to use
});
```

### Text Completion

```typescript
// Generate text completion
const result = await context.completion({
  prompt: "Explain quantum computing in simple terms",
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  n_predict: 512
}, (token) => {
  // Process each token as it's generated
  console.log(token.token);
});

// Clean up when done
await context.release();
```

### Chat Completion

```typescript
// Chat messages following OpenAI format
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is machine learning?" }
];

// Generate chat completion
const result = await context.completion({
  messages: messages,
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  n_predict: 512
}, (token) => {
  // Process each token
  console.log(token.token);
});
```

## Advanced Features

### JSON Mode with Schema Validation

```typescript
// Define a JSON schema
const schema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "number" },
    hobbies: { 
      type: "array",
      items: { type: "string" }
    }
  },
  required: ["name", "age"]
};

// Generate JSON-structured output
const result = await context.completion({
  prompt: "Generate a profile for a fictional person",
  response_format: {
    type: "json_schema",
    json_schema: {
      schema: schema,
      strict: true
    }
  },
  temperature: 0.7,
  n_predict: 512
});

// The result will be valid JSON according to the schema
const jsonData = JSON.parse(result.text);
```

### Working with Embeddings

```typescript
// Generate embeddings for text
const embedding = await context.embedding("This is a sample text", {
  pooling_type: "mean" // Options: "none", "mean", "cls", "last", "rank"
});

console.log(`Embedding dimensions: ${embedding.embedding.length}`);
// Use the embedding for similarity comparison, clustering, etc.
```

### Session Management

```typescript
// Save the current session state
const tokenCount = await context.saveSession("session.bin", { tokenSize: 1024 });
console.log(`Saved session with ${tokenCount} tokens`);

// Load a saved session
const loadResult = await context.loadSession("session.bin");
console.log(`Loaded session: ${loadResult.success}`);
```

### Working with LoRA Adapters

```typescript
// Apply LoRA adapters to the model
await context.applyLoraAdapters([
  { path: "models/lora_adapter.bin", scaled: 0.8 }
]);

// Get currently loaded adapters
const loadedAdapters = await context.getLoadedLoraAdapters();

// Remove all LoRA adapters
await context.removeLoraAdapters();
```

### Model Benchmarking

```typescript
// Benchmark the model performance
const benchResult = await context.bench(
  32,  // pp: prompt processing tests
  32,  // tg: token generation tests
  512, // pl: prompt length
  5    // nr: number of runs
);

console.log(`Average token generation speed: ${benchResult.tgAvg} tokens/sec`);
console.log(`Model size: ${benchResult.modelSize} bytes`);
```

### Native Logging

```typescript
import { addNativeLogListener, toggleNativeLog } from 'cactus-react-native';

// Enable native logging
await toggleNativeLog(true);

// Add a listener for native logs
const logListener = addNativeLogListener((level, text) => {
  console.log(`[${level}] ${text}`);
});

// Remove the listener when no longer needed
logListener.remove();
```

## Error Handling

```typescript
try {
  const context = await initLlama({
    model: 'models/non-existent-model.gguf',
    n_ctx: 2048,
    n_threads: 4
  });
} catch (error) {
  console.error('Failed to initialize model:', error);
}
```

## Best Practices

1. **Model Management**
   - Store models in the app's document directory
   - Consider model size when targeting specific devices
   - Smaller models like SmolLM (135M) work well on most devices

2. **Performance Optimization**
   - Adjust `n_threads` based on the device's capabilities
   - Use a smaller `n_ctx` for memory-constrained devices
   - Consider INT8 or INT4 quantized models for better performance

3. **Battery Efficiency**
   - Release the model context when not in use
   - Process inference in smaller batches
   - Consider background processing for long generations

4. **Memory Management**
   - Always call `context.release()` when done with a model
   - Use `releaseAllLlama()` when switching between multiple models

## Example App

For a complete working example, check out the [React Native example app](https://github.com/cactus-compute/cactus/tree/main/examples/react-example) in the repository.

This example demonstrates:
- Loading and initializing models
- Building a chat interface
- Streaming responses
- Proper resource management

## License

This project is licensed under the Apache 2.0 License.

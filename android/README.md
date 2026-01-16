# Cactus for Android

Run AI models on-device with a simple Kotlin API.

## Building

```bash
cd android
./build.sh
```

Build outputs (in `android/build/lib/`):

| File | Description |
|------|-------------|
| `libcactus.so` | Shared library with JNI bindings |

Source file to copy:

| File | Location |
|------|----------|
| `Cactus.kt` | `android/Cactus.kt` |

## Integration

1. Copy `libcactus.so` to `app/src/main/jniLibs/arm64-v8a/`
2. Copy `Cactus.kt` to your project (e.g., `app/src/main/java/com/cactus/`)

## Usage

### Basic Completion

```kotlin
import com.cactus.Cactus
val model = Cactus.create("/path/to/model")
val result = model.complete("What is the capital of France?")
model.close()
```

### Chat Messages

```kotlin
Cactus.create(modelPath).use { model ->
    val result = model.complete(
        messages = listOf(
            Cactus.Message.system("You are a helpful assistant."),
            Cactus.Message.user("What is 2 + 2?")
        )
    )
}
```

### Completion Options

```kotlin
val options = Cactus.CompletionOptions(
    temperature = 0.7f,
    topP = 0.9f,
    topK = 40,
    maxTokens = 256,
    stopSequences = listOf("\n\n")
)

val result = model.complete("Write a haiku:", options)
```

### Streaming Tokens

```kotlin
val result = model.complete(
    messages = listOf(Cactus.Message.user("Tell me a story")),
    callback = Cactus.TokenCallback { token, tokenId ->
        print(token)
    }
)
```

### Audio Transcription

```kotlin
val result = model.transcribe("/path/to/audio.wav")
val pcmData: ByteArray = ... // 16kHz mono PCM
val result = model.transcribe(pcmData)
```

### Embeddings

```kotlin
val embedding = model.embed("Hello, world!")
```

### RAG

```kotlin
val model = Cactus.create(
    modelPath = "/path/to/model",
    corpusDir = "/path/to/documents"
)

val result = model.complete("What does the documentation say about X?")
```

## API Reference

### Cactus

```kotlin
companion object {
    fun create(modelPath: String, corpusDir: String? = null): Cactus
    fun setTelemetryToken(token: String)
    fun setProKey(key: String)
}

fun complete(prompt: String, options: CompletionOptions = CompletionOptions()): CompletionResult
fun complete(messages: List<Message>, options: CompletionOptions = CompletionOptions(), tools: List<Map<String, Any>>? = null, callback: TokenCallback? = null): CompletionResult

fun transcribe(audioPath: String, prompt: String? = null, language: String? = null, translate: Boolean = false): TranscriptionResult
fun transcribe(pcmData: ByteArray, prompt: String? = null, language: String? = null, translate: Boolean = false): TranscriptionResult

fun embed(text: String, normalize: Boolean = true): FloatArray
fun ragQuery(query: String, topK: Int = 5): String

fun reset()
fun stop()
fun close()
```

### CompletionResult

```kotlin
data class CompletionResult(
    val text: String,
    val functionCalls: List<Map<String, Any>>?,
    val promptTokens: Int,
    val completionTokens: Int,
    val timeToFirstToken: Double,
    val totalTime: Double,
    val prefillTokensPerSecond: Double,
    val decodeTokensPerSecond: Double,
    val confidence: Double,
    val needsCloudHandoff: Boolean
)
```

### Message

```kotlin
data class Message(val role: String, val content: String) {
    companion object {
        fun system(content: String): Message
        fun user(content: String): Message
        fun assistant(content: String): Message
    }
}
```

### CompletionOptions

```kotlin
data class CompletionOptions(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val maxTokens: Int = 512,
    val stopSequences: List<String> = emptyList(),
    val confidenceThreshold: Float = 0f
)
```

## Requirements

- Android API 24+
- arm64-v8a architecture only

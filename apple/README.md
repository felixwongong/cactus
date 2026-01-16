# Cactus for Swift Multiplatform

Run AI models on-device with a simple Swift API on iOS, macOS, and Android.

## Building

```bash
cd apple
./build.sh
```

Build outputs (in `apple/`):

| File | Description |
|------|-------------|
| `cactus-ios.xcframework/` | iOS framework (device + simulator) |
| `cactus-macos.xcframework/` | macOS framework |
| `libcactus-device.a` | Static library for iOS device |
| `libcactus-simulator.a` | Static library for iOS simulator |

For Android, build `libcactus.so` from the `android/` directory.

## Integration

### iOS/macOS: XCFramework (Recommended)

1. Drag `cactus-ios.xcframework` (or `cactus-macos.xcframework`) into your Xcode project
2. Ensure "Embed & Sign" is selected in "Frameworks, Libraries, and Embedded Content"
3. Copy `Cactus.swift` into your project

### iOS/macOS: Static Library

1. Add `libcactus-device.a` (or `libcactus-simulator.a`) to "Link Binary With Libraries"
2. Create a folder with `cactus_ffi.h` and `module.modulemap`, add to Build Settings:
   - "Header Search Paths" → path to folder
   - "Import Paths" (Swift) → path to folder
3. Copy `Cactus.swift` into your project

### Android (Swift SDK)

Requires [Swift SDK for Android](https://www.swift.org/documentation/articles/swift-sdk-for-android-getting-started.html).

1. Copy files to your Swift project:
   - `libcactus.so` → your library path
   - `cactus_ffi.h` → your include path
   - `module.android.modulemap` → rename to `module.modulemap` in include path
   - `Cactus.swift` → your sources

2. Build with Swift SDK for Android:
```bash
swift build --swift-sdk aarch64-unknown-linux-android28 \
    -Xswiftc -I/path/to/include \
    -Xlinker -L/path/to/lib \
    -Xlinker -lcactus
```

3. Bundle `libcactus.so` with your APK in `jniLibs/arm64-v8a/`

## Usage

### Basic Completion

```swift
import Foundation

let model = try Cactus(modelPath: "/path/to/model")
let result = try model.complete("What is the capital of France?")
```

### Chat Messages

```swift
let result = try model.complete(messages: [
    .system("You are a helpful assistant."),
    .user("What is 2 + 2?")
])
```

### Completion Options

```swift
let options = Cactus.CompletionOptions(
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    maxTokens: 256,
    stopSequences: ["\n\n"]
)

let result = try model.complete("Write a haiku:", options: options)
```

### Streaming Tokens

```swift
let result = try model.complete(
    messages: [.user("Tell me a story")],
    onToken: { token, tokenId in
        print(token, terminator: "")
        fflush(stdout)
    }
)
```

### Async/Await

```swift
let result = try await model.complete(messages: [.user("Hello!")])

for try await token in model.completeStream(messages: [.user("Tell me a joke")]) {
    print(token, terminator: "")
}
```

### Audio Transcription

```swift
// From file
let result = try model.transcribe(audioPath: "/path/to/audio.wav")

// From PCM data
let pcmData: Data = ... // 16kHz mono PCM
let result = try model.transcribe(pcmData: pcmData)
```

### Embeddings

```swift
let embedding = try model.embed(text: "Hello, world!")
```

### RAG (Retrieval-Augmented Generation)

```swift
let model = try Cactus(
    modelPath: "/path/to/model",
    corpusDir: "/path/to/documents"
)

let result = try model.complete("What does the documentation say about X?")
```

## API Reference

### Cactus

```swift
init(modelPath: String, corpusDir: String? = nil) throws

func complete(_ prompt: String, options: CompletionOptions = .default) throws -> CompletionResult
func complete(messages: [Message], options: CompletionOptions = .default, tools: [[String: Any]]? = nil, onToken: ((String, UInt32) -> Void)? = nil) throws -> CompletionResult

func transcribe(audioPath: String, prompt: String? = nil, options: TranscriptionOptions = .default) throws -> TranscriptionResult
func transcribe(pcmData: Data, prompt: String? = nil, options: TranscriptionOptions = .default) throws -> TranscriptionResult

func embed(text: String, normalize: Bool = true) throws -> [Float]
func ragQuery(_ query: String, topK: Int = 5) throws -> String

func reset()  // Clear KV cache
func stop()   // Stop generation

static func setTelemetryToken(_ token: String)
static func setProKey(_ key: String)
```

### CompletionResult

```swift
struct CompletionResult {
    let text: String                   
    let functionCalls: [[String: Any]]? 
    let promptTokens: Int
    let completionTokens: Int
    let timeToFirstToken: Double  
    let totalTime: Double   
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let confidence: Double  
    let needsCloudHandoff: Bool
}
```

### Message

```swift
struct Message {
    static func system(_ content: String) -> Message
    static func user(_ content: String) -> Message
    static func assistant(_ content: String) -> Message
}
```

### CompletionOptions

```swift
struct CompletionOptions {
    var temperature: Float = 0.7
    var topP: Float = 0.9
    var topK: Int = 40
    var maxTokens: Int = 512
    var stopSequences: [String] = []
    var confidenceThreshold: Float = 0.0

    static let `default` = CompletionOptions()
}
```

## Requirements

**Apple Platforms:**
- iOS 14.0+ / macOS 13.0+ / tvOS 14.0+ / watchOS 7.0+
- Xcode 14.0+
- Swift 5.7+

**Android:**
- Swift 6.0+ with [Swift SDK for Android](https://www.swift.org/documentation/articles/swift-sdk-for-android-getting-started.html)
- Android NDK 27d+
- Android API 28+ / arm64-v8a

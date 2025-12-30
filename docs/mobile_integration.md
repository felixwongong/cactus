# Mobile Integration Guide

Integrate Cactus into iOS, macOS, and Android applications using the native C FFI.

## Prerequisites

```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
```

## iOS / macOS

### Build

```bash
cactus build --apple
```

**Output:**
- `apple/libcactus-device.a` - iOS device static library
- `apple/libcactus-simulator.a` - iOS simulator static library
- `apple/cactus-ios.xcframework` - iOS universal framework
- `apple/cactus-macos.xcframework` - macOS framework

### Xcode Setup

1. Drag `cactus-ios.xcframework` into your Xcode project
2. Add to "Frameworks, Libraries, and Embedded Content"
3. Create a bridging header with:
   ```c
   #include "cactus_ffi.h"
   ```

### Swift Wrapper

```swift
import Foundation

class CactusModel {
    private var model: OpaquePointer?

    init?(modelPath: String, contextSize: Int = 2048) {
        model = cactus_init(modelPath, contextSize, nil)
        guard model != nil else { return nil }
    }

    func complete(messages: String, options: String? = nil) -> String? {
        guard let model = model else { return nil }
        var response = [CChar](repeating: 0, count: 8192)
        let result = cactus_complete(
            model, messages, &response, response.count,
            options, nil, nil, nil
        )
        guard result > 0 else { return nil }
        return String(cString: response)
    }

    func embed(text: String) -> [Float]? {
        guard let model = model else { return nil }
        var embeddings = [Float](repeating: 0, count: 2048)
        var dim: Int = 0
        let result = cactus_embed(model, text, &embeddings, embeddings.count * 4, &dim, true)
        guard result == 0 else { return nil }
        return Array(embeddings.prefix(dim))
    }

    func reset() {
        guard let model = model else { return }
        cactus_reset(model)
    }

    func stop() {
        guard let model = model else { return }
        cactus_stop(model)
    }

    deinit {
        if let model = model {
            cactus_destroy(model)
        }
    }
}
```

### Usage

```swift
// Load from app bundle
let modelPath = Bundle.main.path(forResource: "my-model", ofType: nil)!

if let cactus = CactusModel(modelPath: modelPath) {
    let response = cactus.complete(messages: """
        [{"role": "user", "content": "Hello!"}]
        """)
    print(response ?? "Error")
}

// Or from documents directory
let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
let modelURL = docsURL.appendingPathComponent("my-model")
```

### Streaming Responses

```swift
// Define callback type
typealias TokenCallback = @convention(c) (UnsafePointer<CChar>?, UInt32, UnsafeMutableRawPointer?) -> Void

let callback: TokenCallback = { token, tokenId, userData in
    if let token = token {
        print(String(cString: token), terminator: "")
    }
}

// Use with completion
var response = [CChar](repeating: 0, count: 8192)
cactus_complete(model, messages, &response, response.count, nil, nil, callback, nil)
```

## Android

### Build

```bash
cactus build --android
```

**Output:**
- `android/libcactus.so` - Shared library (arm64-v8a)
- `android/libcactus.a` - Static library

### Project Setup

1. Copy the shared library:
   ```
   app/src/main/jniLibs/arm64-v8a/libcactus.so
   ```

2. Copy headers to your JNI include path:
   ```
   app/src/main/cpp/cactus_ffi.h
   ```

### Kotlin Wrapper

```kotlin
// CactusNative.kt
class CactusNative {
    companion object {
        init { System.loadLibrary("cactus") }
    }

    external fun init(modelPath: String, contextSize: Long, corpusDir: String?): Long
    external fun complete(
        model: Long,
        messagesJson: String,
        responseBuffer: ByteArray,
        optionsJson: String?,
        toolsJson: String?
    ): Int
    external fun embed(
        model: Long,
        text: String,
        embeddingsBuffer: FloatArray,
        normalize: Boolean
    ): Int
    external fun destroy(model: Long)
    external fun reset(model: Long)
    external fun stop(model: Long)
    external fun getLastError(): String?
}

// CactusModel.kt
class CactusModel(modelPath: String, contextSize: Int = 2048) : AutoCloseable {
    private val native = CactusNative()
    private val handle: Long = native.init(modelPath, contextSize.toLong(), null)

    init {
        require(handle != 0L) { "Failed to init model: ${native.getLastError()}" }
    }

    fun complete(messages: String, options: String? = null): String {
        val buffer = ByteArray(8192)
        val result = native.complete(handle, messages, buffer, options, null)
        require(result > 0) { "Completion failed: ${native.getLastError()}" }
        return buffer.decodeToString().trimEnd('\u0000')
    }

    fun embed(text: String, normalize: Boolean = true): FloatArray {
        val buffer = FloatArray(2048)
        val dim = native.embed(handle, text, buffer, normalize)
        require(dim > 0) { "Embedding failed: ${native.getLastError()}" }
        return buffer.copyOf(dim)
    }

    fun reset() = native.reset(handle)
    fun stop() = native.stop(handle)

    override fun close() = native.destroy(handle)
}
```

### JNI Bridge (C++)

Create `app/src/main/cpp/cactus_jni.cpp`:

```cpp
#include <jni.h>
#include "cactus_ffi.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_CactusNative_init(JNIEnv* env, jobject, jstring modelPath,
                                    jlong contextSize, jstring corpusDir) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    const char* corpus = corpusDir ? env->GetStringUTFChars(corpusDir, nullptr) : nullptr;

    cactus_model_t model = cactus_init(path, (size_t)contextSize, corpus);

    env->ReleaseStringUTFChars(modelPath, path);
    if (corpus) env->ReleaseStringUTFChars(corpusDir, corpus);

    return (jlong)model;
}

JNIEXPORT jint JNICALL
Java_com_example_CactusNative_complete(JNIEnv* env, jobject, jlong model,
                                        jstring messagesJson, jbyteArray responseBuffer,
                                        jstring optionsJson, jstring toolsJson) {
    const char* messages = env->GetStringUTFChars(messagesJson, nullptr);
    const char* options = optionsJson ? env->GetStringUTFChars(optionsJson, nullptr) : nullptr;
    const char* tools = toolsJson ? env->GetStringUTFChars(toolsJson, nullptr) : nullptr;

    jsize bufferSize = env->GetArrayLength(responseBuffer);
    char* buffer = new char[bufferSize];

    int result = cactus_complete((cactus_model_t)model, messages, buffer, bufferSize,
                                  options, tools, nullptr, nullptr);

    if (result > 0) {
        env->SetByteArrayRegion(responseBuffer, 0, result, (jbyte*)buffer);
    }

    delete[] buffer;
    env->ReleaseStringUTFChars(messagesJson, messages);
    if (options) env->ReleaseStringUTFChars(optionsJson, options);
    if (tools) env->ReleaseStringUTFChars(toolsJson, tools);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_example_CactusNative_embed(JNIEnv* env, jobject, jlong model,
                                     jstring text, jfloatArray embeddingsBuffer,
                                     jboolean normalize) {
    const char* txt = env->GetStringUTFChars(text, nullptr);
    jsize bufferSize = env->GetArrayLength(embeddingsBuffer);
    float* buffer = new float[bufferSize];
    size_t dim = 0;

    int result = cactus_embed((cactus_model_t)model, txt, buffer,
                               bufferSize * sizeof(float), &dim, normalize);

    if (result == 0) {
        env->SetFloatArrayRegion(embeddingsBuffer, 0, (jsize)dim, buffer);
    }

    delete[] buffer;
    env->ReleaseStringUTFChars(text, txt);

    return result == 0 ? (jint)dim : -1;
}

JNIEXPORT void JNICALL
Java_com_example_CactusNative_destroy(JNIEnv*, jobject, jlong model) {
    cactus_destroy((cactus_model_t)model);
}

JNIEXPORT void JNICALL
Java_com_example_CactusNative_reset(JNIEnv*, jobject, jlong model) {
    cactus_reset((cactus_model_t)model);
}

JNIEXPORT void JNICALL
Java_com_example_CactusNative_stop(JNIEnv*, jobject, jlong model) {
    cactus_stop((cactus_model_t)model);
}

JNIEXPORT jstring JNICALL
Java_com_example_CactusNative_getLastError(JNIEnv* env, jobject) {
    const char* error = cactus_get_last_error();
    return error ? env->NewStringUTF(error) : nullptr;
}

}
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)

add_library(cactus_jni SHARED cactus_jni.cpp)

target_link_libraries(cactus_jni
    ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libcactus.so
    log
)
```

### Usage

```kotlin
// From app files directory
val modelPath = "${context.filesDir}/my-model"

CactusModel(modelPath).use { model ->
    // Chat completion
    val response = model.complete("""[{"role":"user","content":"Hello!"}]""")
    Log.d("Cactus", response)

    // Embeddings
    val embedding = model.embed("Hello world")
    Log.d("Cactus", "Embedding dim: ${embedding.size}")
}
```

## Model Deployment

### Bundle with App

**iOS:**
- Add model folder to Xcode project
- Access via `Bundle.main.path(forResource:ofType:)`

**Android:**
- Place in `assets/` folder
- Copy to internal storage on first launch:
  ```kotlin
  context.assets.open("my-model").use { input ->
      File(context.filesDir, "my-model").outputStream().use { output ->
          input.copyTo(output)
      }
  }
  ```

### Download at Runtime

Use your preferred HTTP client to download model weights, then load from the download location.

## API Reference

See [cactus_ffi.h](../cactus/ffi/cactus_ffi.h) and [cactus_engine.md](./cactus_engine.md) for the full C API documentation.
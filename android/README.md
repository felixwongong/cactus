# Cactus for Android (Kotlin/Java)

A lightweight, high-performance framework for running AI models natively on Android devices.

## Installation

Add the dependency to your module's `build.gradle.kts` (or `build.gradle`) file.

**Important: Credentials Required for GitHub Packages**

Accessing the Cactus Android library via GitHub Packages now requires authentication. You'll need to use a GitHub Personal Access Token (PAT) with the `read:packages` scope.

**Steps to Get and Use Credentials:**

1.  **Generate a GitHub PAT:**
    *   Go to your GitHub [Developer settings](https://github.com/settings/tokens).
    *   Click "Generate new token" (select classic or fine-grained).
    *   Give your token a descriptive name (e.g., `cactus-android-dependency`).
    *   Select the `read:packages` scope.
    *   Click "Generate token" and **copy the token immediately**. You won't be able to see it again.

2.  **Store Your Credentials Securely:**
    Do **not** hardcode your PAT directly into your `settings.gradle.kts` file. Instead, use one of the following methods:

    *   **Using `local.properties` (Recommended for local development):**
        1.  Create or open the `local.properties` file in your Android project's root directory (the same directory as `gradle.properties`). If it's not there, create it.
        2.  Add the following lines, replacing `YOUR_GITHUB_USERNAME` with your GitHub username and `YOUR_PAT` with the token you generated:
            ```properties
            gpr.user=YOUR_GITHUB_USERNAME
            gpr.key=YOUR_PAT
            ```
        3.  Ensure `local.properties` is listed in your project's `.gitignore` file to prevent committing your credentials.

    *   **Using Environment Variables (Recommended for CI/CD or shared environments):**
        Set the following environment variables in your build environment:
        *   `GPR_USER`: Your GitHub username.
        *   `GPR_KEY`: Your GitHub PAT.

        The `settings.gradle.kts` file is configured to read these from `local.properties` first, then fall back to environment variables.

**1. Add Repository to `settings.gradle.kts`:**

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS) // Optional but recommended
    repositories {
        google()
        mavenCentral()
        // Add GitHub Packages repository for Cactus
        maven {
            name = "GitHubPackagesCactusCompute"
            url = uri("https://maven.pkg.github.com/cactus-compute/cactus")
            credentials {
                username = project.findProperty("gpr.user") as String? ?: System.getenv("GPR_USER")
                password = project.findProperty("gpr.key") as String? ?: System.getenv("GPR_KEY")
            }
        }
    }
}
```

**2. Add Dependency to Module's `build.gradle.kts`:**

```kotlin
// app/build.gradle.kts
dependencies {
    // Replace x.y.z with the desired version (e.g., 0.0.1)
    implementation("io.github.cactus-compute:cactus-android:x.y.z")

    // Other dependencies...
}
```

Sync your project with Gradle files in Android Studio.

## Basic Usage

### Initialize a Model

Make sure to handle potential exceptions during initialization.

```kotlin
import com.cactus.android.LlamaContext
import com.cactus.android.LlamaInitParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import android.util.Log

// In an Activity, ViewModel, or background coroutine scope

var llamaContext: LlamaContext? = null

suspend fun initializeModel() {
    try {
        // Ensure this runs off the main thread
        llamaContext = withContext(Dispatchers.IO) {
            LlamaContext.create(
                params = LlamaInitParams(
                    modelPath = "path/to/your/model.gguf", // Path accessible by the app
                    nCtx = 2048,         // Context size
                    nBatch = 512,        // Batch size for prompt processing
                    nThreads = 4,        // Number of threads
                    embedding = true     // Enable embedding generation if needed
                    // Add other parameters like nGpuLayers, rope settings etc. if needed
                ),
                loadProgressCallback = { progress ->
                    Log.i("LlamaInit", "Model Load Progress: ${(progress * 100).toInt()}%")
                    // Update UI or log progress
                }
            )
        }
        Log.i("LlamaInit", "Model loaded successfully!")
    } catch (e: Exception) {
        Log.e("LlamaInit", "Failed to initialize model", e)
        // Handle initialization error (e.g., show message to user)
    }
}

// Remember to release the context when done
fun cleanupModel() {
    llamaContext?.close() // Safe to call close() multiple times
    llamaContext = null
    Log.i("LlamaCleanup", "Model released.")
}

// Call initializeModel() when needed, e.g., in onCreate or a button click handler (using lifecycleScope)
// Call cleanupModel() when the component is destroyed, e.g., in onDestroy
```

### Text Completion

```kotlin
import com.cactus.android.LlamaCompletionParams
import com.cactus.android.PartialCompletionCallback
import kotlinx.coroutines.launch
import androidx.lifecycle.lifecycleScope // Example using lifecycle scope

// Assuming llamaContext is initialized and not null

lifecycleScope.launch(Dispatchers.IO) { // Run completion off the main thread
    try {
        val result = llamaContext?.complete(
            prompt = "Explain quantum computing in simple terms",
            params = LlamaCompletionParams(
                temperature = 0.7f,
                topK = 40,
                topP = 0.95f,
                nPredict = 512
                // Add other parameters like stop words, grammar, etc.
            ),
            partialCompletionCallback = object : PartialCompletionCallback {
                override fun onPartialCompletion(partialResultMap: Map<String, Any?>): Boolean {
                    val token = partialResultMap["token"] as? String ?: ""
                    // Process each token as it's generated (e.g., update UI on main thread)
                    Log.d("LlamaCompletion", "Token: $token")
                    // Return true to continue, false to stop generation early
                    return true
                }
            }
        )

        // Process the final result (contains full text, timings, stop reason, etc.)
        Log.i("LlamaCompletion", "Final Text: ${result?.text}")
        Log.i("LlamaCompletion", "Timings: ${result?.timings}")

    } catch (e: Exception) {
        Log.e("LlamaCompletion", "Completion failed", e)
    }
}
```

### Chat Completion (using Jinja Formatting)

```kotlin
import com.cactus.android.LlamaCompletionParams
import kotlinx.serialization.encodeToString // Requires kotlinx-serialization dependency
import kotlinx.serialization.json.Json

// Define message structure (or use a library)
@kotlinx.serialization.Serializable
data class ChatMessage(val role: String, val content: String)

// ... inside a coroutine scope ...

val messages = listOf(
  ChatMessage(role = "system", content = "You are a helpful assistant."),
  ChatMessage(role = "user", content = "What is machine learning?")
)
// Convert messages list to JSON string
val messagesJson = Json.encodeToString(messages)

try {
    // 1. Format the chat using the model's template (optional but recommended)
    val formatted = llamaContext?.getFormattedChatWithJinja(messagesJson)
    val promptForCompletion = formatted?.prompt ?: // Fallback if formatting fails or is skipped

    // 2. Run completion with the formatted prompt
    val result = llamaContext?.complete(
        prompt = promptForCompletion,
        params = LlamaCompletionParams(
             temperature = 0.7f,
             topK = 40,
             topP = 0.95f,
             nPredict = 512,
             chatFormat = formatted?.chatFormat ?: -1, // Pass format info if available
             grammar = formatted?.grammar,             // Pass grammar if available
             // Pass other formatted results like additional_stops if needed
             stop = formatted?.additionalStops?.toTypedArray()
        ),
        partialCompletionCallback = { partialResultMap ->
             val token = partialResultMap["token"] as? String ?: ""
             Log.d("LlamaChat", "Token: $token")
             true // Continue generation
        }
    )
    Log.i("LlamaChat", "Final Response: ${result?.text}")

} catch (e: Exception) {
    Log.e("LlamaChat", "Chat completion failed", e)
}
```

## Advanced Features

### JSON Mode with Schema Validation

```kotlin
import com.cactus.android.LlamaCompletionParams

// ... inside a coroutine scope ...

// Define your JSON schema as a String
val schemaJsonString = """
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "number" },
    "hobbies": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["name", "age"]
}
"""

try {
    // 1. Format the chat using Jinja, providing the schema
    val formatted = llamaContext?.getFormattedChatWithJinja(
        messagesJson = "[{\"role\":\"user\", \"content\":\"Generate a profile for a fictional person.\"}]",
        jsonSchema = schemaJsonString
    )

    // 2. Run completion using the generated prompt and grammar
    val result = llamaContext?.complete(
        prompt = formatted?.prompt ?: "Generate JSON:", // Provide fallback prompt
        params = LlamaCompletionParams(
            temperature = 0.7f,
            nPredict = 512,
            grammar = formatted?.grammar // Use grammar generated by formatting step
        )
        // No streaming callback needed usually for JSON mode
    )

    // The result.text should contain valid JSON conforming to the schema
    Log.i("LlamaJson", "JSON Output: ${result?.text}")
    // Parse the JSON string using a library like kotlinx.serialization or Gson

} catch (e: Exception) {
    Log.e("LlamaJson", "JSON completion failed", e)
}
```

### Working with Embeddings

```kotlin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// ... inside a coroutine scope, assuming llamaContext is initialized with embedding=true ...

try {
    val embeddingResult = withContext(Dispatchers.IO) {
        llamaContext?.embedding("This is a sample text.")
    }

    if (embeddingResult != null) {
        Log.i("LlamaEmbedding", "Embedding dimensions: ${embeddingResult.embedding.size}")
        // Use embeddingResult.embedding (List<Float>) for similarity, clustering, etc.
    } else {
        Log.w("LlamaEmbedding", "Failed to generate embedding.")
    }

} catch (e: Exception) {
    Log.e("LlamaEmbedding", "Embedding generation failed", e)
}
```

### Session Management

```kotlin
// ... inside a coroutine scope ...

val sessionFilePath = context.filesDir.path + "/mysession.bin" // Example path

try {
    // Save session state
    val tokensSaved = withContext(Dispatchers.IO) {
        llamaContext?.saveSession(sessionFilePath)
    }
    Log.i("LlamaSession", "Saved session with $tokensSaved tokens.")

    // ... later, perhaps after app restart ...

    // Load session state (ensure context is initialized with compatible model/params)
    val loadResult = withContext(Dispatchers.IO) {
        llamaContext?.loadSession(sessionFilePath)
    }
    Log.i("LlamaSession", "Loaded session. Tokens: ${loadResult?.tokensLoaded}, Prompt: ${loadResult?.prompt}")

} catch (e: Exception) {
    Log.e("LlamaSession", "Session operation failed", e)
}
```

### Working with LoRA Adapters

```kotlin
import com.cactus.android.LoraAdapterInfo

// ... inside a coroutine scope ...

val adapters = listOf(
    LoraAdapterInfo(path = "path/to/your/lora_adapter.bin", scale = 0.8f)
)

try {
    // Apply LoRA adapters
    withContext(Dispatchers.IO) {
        llamaContext?.applyLoraAdapters(adapters)
    }
    Log.i("LlamaLoRA", "Applied LoRA adapters.")

    // Get currently loaded adapters
    val loadedAdapters = llamaContext?.getLoadedLoraAdapters()
    Log.i("LlamaLoRA", "Loaded adapters: $loadedAdapters")

    // Remove all LoRA adapters
    withContext(Dispatchers.IO) {
        llamaContext?.removeLoraAdapters()
    }
    Log.i("LlamaLoRA", "Removed LoRA adapters.")

} catch (e: Exception) {
    Log.e("LlamaLoRA", "LoRA operation failed", e)
}
```

### Model Benchmarking

```kotlin
// ... inside a coroutine scope ...

try {
    val benchmarkJsonResult = withContext(Dispatchers.IO) {
        llamaContext?.bench(
            pp = 32,  // Prompt processing tests
            tg = 32,  // Token generation tests
            pl = 512, // Prompt length
            nr = 5    // Number of runs
        )
    }
    Log.i("LlamaBench", "Benchmark Result (JSON): $benchmarkJsonResult")
    // Parse the JSON string to get detailed metrics

} catch (e: Exception) {
    Log.e("LlamaBench", "Benchmarking failed", e)
}
```

### Native Logging

```kotlin
import com.cactus.android.LogCallback

// Set up logging (e.g., in Application class or early in MainActivity)
LlamaContext.setupLog(object : LogCallback {
    override fun onLog(level: Int, message: String) {
        // Route native logs to Android's Logcat or your logging framework
        when (level) {
            // Define mapping based on native log levels if available
            // e.g., Log.VERBOSE, Log.DEBUG, Log.INFO, Log.WARN, Log.ERROR
            else -> Log.d("LlamaNativeLog", "[Level $level] $message")
        }
    }
})

// To disable the callback later:
// LlamaContext.unsetLog()
```

## Error Handling

Most methods can throw exceptions (`IllegalStateException`, `RuntimeException`, potentially `IOException` for file operations). Use standard Kotlin `try-catch` blocks to handle errors gracefully. Check logs for details, especially when interacting with native code.

## Best Practices

1.  **Model Management**
    *   Store models in the app's internal storage (`context.filesDir`) or external storage (requesting permissions if necessary).
    *   Download models securely. Consider using Android's `DownloadManager`.
    *   Provide mechanisms for users to manage downloaded models.

2.  **Threading**
    *   **Never** run `LlamaContext.create`, `complete`, `embedding`, `loadSession`, `saveSession`, `applyLoraAdapters`, `bench` on the **main thread**. Use coroutines (`Dispatchers.IO`), Threads, or WorkManager.
    *   Callbacks (`LoadProgressCallback`, `PartialCompletionCallback`, `LogCallback`) might be called from different threads. Ensure UI updates are posted back to the main thread (e.g., using `withContext(Dispatchers.Main)`).

3.  **Performance Optimization**
    *   Adjust `nThreads` in `LlamaInitParams` based on the device (e.g., `Runtime.getRuntime().availableProcessors() / 2`).
    *   Use appropriate context (`nCtx`) and batch sizes (`nBatch`).
    *   Prefer quantized models (Q4_K_M, Q5_K_M, etc.) for significant performance gains and reduced memory usage on mobile.

4.  **Battery Efficiency**
    *   Call `llamaContext.close()` as soon as you are finished with the context to free native resources. Use Kotlin's `use` block for safety where possible (though context creation is often longer-lived).
    *   Avoid keeping models loaded unnecessarily in the background.

5.  **Memory Management**
    *   Explicitly call `close()` on `LlamaContext` instances. Relying on `finalize` is not recommended.
    *   Be mindful of the model size and the device's available RAM.

## Example App

For a complete working example using the Android library, check out the `test-app` module within the `android` directory of the main Cactus repository: [android/test-app](https://github.com/cactus-compute/cactus/tree/main/android/test-app)

## License

This project is licensed under the MIT License.
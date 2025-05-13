![Logo](repo-assets/banner.jpg)

[![Email][gmail-shield]][gmail-url]
[![Discord][discord-shield]][discord-url]
[![Design Docs][docs-shield]][docs-url]
![License](https://img.shields.io/github/license/cactus-compute/cactus?style=for-the-badge)
[![Stars][stars-shield]][github-url]
[![Forks][forks-shield]][github-url]


[gmail-shield]: https://img.shields.io/badge/Gmail-red?style=for-the-badge&logo=gmail&logoColor=white
[gmail-url]: founders@cactuscompute.com

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-blue.svg?style=for-the-badge&logo=linkedin&colorB=blue
[linkedin-url]: https://www.linkedin.com/company/106281696

[discord-shield]: https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white
[discord-url]: https://discord.gg/j4SS7Nwr

[docs-shield]: https://img.shields.io/badge/Design_Docs-009485?style=for-the-badge&logo=readthedocs&logoColor=white
[docs-url]: https://deepwiki.com/cactus-compute/cactus

[website-shield]: https://img.shields.io/badge/Website-black?style=for-the-badge&logo=safari&logoColor=white
[website-url]: https://cactuscompute.com

[stars-shield]: https://img.shields.io/github/stars/cactus-compute/cactus?style=for-the-badge&color=yellow
[forks-shield]: https://img.shields.io/github/forks/cactus-compute/cactus?style=for-the-badge&color=blue
[issues-shield]: https://img.shields.io/github/issues/cactus-compute/cactus?style=for-the-badge
[prs-shield]: https://img.shields.io/github/issues-pr/cactus-compute/cactus?style=for-the-badge
[github-url]: https://github.com/cactus-compute/cactus

Cactus is a lightweight, high-performance framework for running AI models on mobile phones. Cactus has unified and consistent APIs across 
- React-Native
- Android/Kotlin
- Android/Java
- iOS/Swift
- iOS/Objective-C++
- Flutter/Dart

Cactus currently leverages GGML backends to support any GGUF model already compatible with [![Llama.cpp](https://img.shields.io/badge/Llama.cpp-000000?style=flat&logo=github&logoColor=white)](https://github.com/ggerganov/llama.cpp), while we focus on broadly supporting every moblie app development platform, as well as upcoming features like:

- MCP
- phone tool use
- thinking
- prompt-enhancement
- higher-level APIs

Contributors with any of the above experiences are welcome! Feel free to submit cool example apps you built with Cactus, issues or tests! 

Cactus Models coming soon.

## Table of Contents

- [Technical Architecture](#technical-architecture)
- [Features](#features)
- [Benchmarks](#benchmarks)
- [Getting Started](#getting-started)
  - [Flutter](#flutter-in-development)
  - [React Native](#react-native-shipped)
  - [Android](#android-currently-testing)
  - [Swift](#ios-in-developement)
  - [C++ (Raw backend)](#c-raw-backend)
- [License](#license)

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Applications                        │
└───────────────┬─────────────────┬───────────────────────┘
                │                 │                
┌───────────────┼─────────────────┼───────────────────────-┐
│ ┌─────────────▼─────┐ ┌─────────▼───────┐ ┌─────────────┐|
│ │     React API     │ │   Flutter API   │ │  Native APIs│|
│ └───────────────────┘ └─────────────────┘ └─────────────┘|
│                Platform Bindings                         │
└───────────────┬─────────────────┬───────────────────────-┘
                │                 │                
┌───────────────▼─────────────────▼───────────────────────┐
│                   Cactus Core (C++) / llama.rn API      │
└───────────────┬─────────────────┬───────────────────────┘
                │                 │                
┌───────────────▼─────┐ ┌─────────▼───────────────────────┐
│   Llama.cpp Core    │ │    GGML/GGUF Model Format       │
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

we created a demo chat app we use for benchmarking:

[![Download App](https://img.shields.io/badge/Download_iOS_App-grey?style=for-the-badge&logo=apple&logoColor=white)](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
[![Download App](https://img.shields.io/badge/Download_Android_App-grey?style=for-the-badge&logo=android&logoColor=white)](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share)

These are day-to-day usage scores, NOT a controlled environment.

| Device                       | Gemma 3 1B INT8 (toks/sec) | Qwen 2.5 1.5B INT8 (toks/sec) |
|------------------------------|----------------------------|-------------------------------|
| iPhone 16 Pro Max            | 45                         | 28                            |
| iPhone 16 Pro                | N/A                        | 28                            |
| iPhone 16                    | N/A                        | 27                            |
| iPhone 15 Pro Max            | N/A                        | 23                            |
| iPhone 15 Pro                | N/A                        | 23                            |
| iPhone 15                    | N/A                        | 23                            |
| iPhone 13 Pro                | 30                         | N/A                           |
| iPhone 12 mini               | 21                         | N/A                           |
| Galaxy S25 Ultra             | 25                         | N/A                           |
| Galaxy S24+                  | 20                         | N/A                           |
| Galaxy S22 Ultra             | 16                         | N/A                           |
| Galaxy S21                   | 14                         | N/A                           |
| Galaxy A14                   | 6                          | N/A                           |
| Google Pixel 8               | 14                         | N/A                           |
| Google Pixel 6a              | 14                         | N/A                           |
| Oneplus 13                   | 34                         | N/A                           |
| Oneplus 12                   | 23                         | N/A                           |
| Oneplus Nord CE Lite         | 10                         | N/A                           |
| Xiaomi Redmi k70 Ultra       | 19                         | N/A                           |
| Moto G62 5G (Gran's Phone)   | 6                          | N/A                           |
| Huawei P60 Lite (Gran's phone)| N/A                        | N/A                           |

## Examples
We have ready-to-run-and-deploy examples [here](https://github.com/cactus-compute/cactus/tree/main/examples), you can simply copy, modify and deploy! And reach out if stuck or need hand-holding.

## Getting Started

### ✅ Flutter (Dart)

Full setup and API details are available in the [Flutter README](cactus-flutter/README.md).

**1. Add Dependency:**
Add `cactus` to your `pubspec.yaml`:
```yaml
dependencies:
  cactus: ^0.0.2
```
Then run `flutter pub get`.

**2. Basic Usage (Dart):**
```dart
import 'package:cactus/cactus.dart';

CactusContext? cactusContext;

Future<void> initializeAndRun() async {
  try {
    // Initialize from a URL (will be downloaded)
    final initParams = CactusInitParams(
      modelUrl: 'YOUR_MODEL_URL_HERE', // e.g., https://huggingface.co/.../phi-2.Q4_K_M.gguf
      nCtx: 512,
      nThreads: 4,
      onInitProgress: (progress, message, isError) {
        print('Init Progress: $message (${progress != null ? (progress * 100).toStringAsFixed(1) + '%' : 'N/A'})');
      },
    );
    cactusContext = await CactusContext.init(initParams);

    // Perform chat completion
    final messages = [
      ChatMessage(role: 'system', content: 'You are a helpful AI assistant.'),
      ChatMessage(role: 'user', content: 'Explain quantum computing in simple terms.'),
    ];
    final completionParams = CactusCompletionParams(
      messages: messages,
      temperature: 0.7,
      onNewToken: (token) {
        print(token); // Stream tokens
        return true; // Continue generation
      },
    );
    final result = await cactusContext!.completion(completionParams);
    print('Generated Text: ${result.text}');

  } catch (e) {
    print('Error: $e');
  } finally {
    cactusContext?.free();
  }
}
```

### ✅ React Native (TypeScript/JavaScript)

```bash
npm install react-native-fs
npm install cactus-react-native
# or
yarn add react-native-fs
yarn add cactus-react-native

# For iOS, install pods if not on Expo
npx pod-install
```
```typescript
import { initLlama, LlamaContext } from 'cactus-react-native';

// Load model
const context = await initLlama({
  model: 'models/llama-2-7b-chat.gguf', // Path to your model
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

For more detailed documentation and examples, see the [React Native README](cactus-react/README.md).

### ✅ Android (Kotlin/Java)

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
    implementation("io.github.cactus-compute:cactus-android:0.0.1")
}
```

**3. Basic Usage (Kotlin):**

```kotlin
import com.cactus.android.LlamaContext
import com.cactus.android.LlamaInitParams
import com.cactus.android.LlamaCompletionParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// In an Activity, ViewModel, or coroutine scope

suspend fun runInference() {
    var llamaContext: LlamaContext? = null
    try {
        // Initialize (off main thread)
        llamaContext = withContext(Dispatchers.IO) {
            LlamaContext.create(
                params = LlamaInitParams(
                    modelPath = "path/to/your/model.gguf",
                    nCtx = 2048, nThreads = 4
                )
            )
        }

        // Complete (off main thread)
        val result = withContext(Dispatchers.IO) {
            llamaContext?.complete(
                prompt = "Explain quantum computing in simple terms",
                params = LlamaCompletionParams(temperature = 0.7f, nPredict = 512)
            ) { partialResultMap ->
                val token = partialResultMap["token"] as? String ?: ""
                print(token) // Process stream on background thread
                true // Continue generation
            }
        }
        println("\nFinal text: ${result?.text}")

    } catch (e: Exception) {
        // Handle errors
        println("Error: ${e.message}")
    } finally {
        // Clean up (off main thread)
        withContext(Dispatchers.IO) {
             llamaContext?.close()
        }
    }
}
```

For more detailed documentation and examples, see the [Android README](cactus-android/README.md).

### 🚧 Swift (in developement)

```

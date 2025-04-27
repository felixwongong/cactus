# Contributing to Cactus

Thank you for your interest in contributing to Cactus! This document provides detailed guidelines for contributing to the Cactus codebase, which is a high-performance cross-platform LLM inference library.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Codebase Architecture](#codebase-architecture)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Platform-Specific Guidelines](#platform-specific-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Benchmarking](#benchmarking)
- [Common Issues](#common-issues)

## Development Environment Setup

### Requirements

- CMake 3.24+
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Python 3.8+ (for testing and scripts)
- Platform-specific tools:
  - Android: Android Studio, NDK r25+
  - iOS: Xcode 14+, CocoaPods
  - React: Node.js 18+, Yarn
  - Flutter: Flutter SDK 3.0+

### Setting up the development environment

1. **Clone the repository with submodules**:
   ```bash
   git clone https://github.com/cactuscompute/cactus.git
   cd cactus
   ```

2. **Build the core C++ library**:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make -j$(nproc)
   ```

3. **For Android development**:
   ```bash
   cd android
   ./gradlew assembleDebug
   ```

4. **For iOS development**:
   ```bash
   cd ios
   pod install
   open Cactus.xcworkspace
   ```

5. **For React Native development**:
   ```bash
   cd react
   yarn install
   yarn build
   ```

## Codebase Architecture

Cactus is organized into several key components:

### Core C++ Library (`/cactus`)

- **cactus.h/cpp**: Main API implementation
- **llama-*.h/cpp**: Modified llama.cpp components for mobile
- **ggml*.h/cpp**: Tensor computation library
- **common.h/cpp**: Utility functions and parameter handling

### Platform Bindings

- **android/**: JNI interface for Android
- **ios/**: Objective-C/Swift bindings
- **react/**: TypeScript interface for React Native
- **flutter/**: Dart bindings for Flutter

### Examples

- **examples/**: Contains example applications for each platform

## Pull Request Process

1. **Fork the repository** and create your branch from `main`.
2. **Make your changes** following the coding standards below.
3. **Add tests** that cover your changes.
4. **Update documentation** including code comments and markdown files.
5. **Run the test suite** and ensure all tests pass.
6. **Submit a pull request** to the `main` branch.
7. **Code review process**:
   - At least one core maintainer must approve
   - CI checks must pass
   - All review comments must be addressed

## Coding Standards

### C++ Code

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with these exceptions:
  - Use 4-space indentation instead of 2-space
  - Use `snake_case` for function and variable names
  - Use `PascalCase` for class names
- Keep code modular with clear separation of concerns
- Always check for memory leaks
- Optimize for mobile performance

### Performance Considerations

- Avoid dynamic memory allocations in performance-critical paths
- Use SIMD optimizations where appropriate
- Consider memory usage patterns for mobile devices
- Profile your code before and after changes

### Code Example

```cpp
// Good example
bool cactus_context::loadModel(common_params &params) {
    // Clear initialization state
    is_load_interrupted = false;
    loading_progress = 0.0f;
    
    // Set up model loading parameters
    llama_init = common_init_from_params(params);
    model = llama_init.model.get();
    ctx = llama_init.context.get();
    
    if (model == nullptr) {
        LOG_ERROR("Failed to load model: %s", params.model.c_str());
        return false;
    }
    
    // Initialize chat templates
    templates = common_chat_templates_init(model, params.chat_template);
    n_ctx = llama_n_ctx(ctx);
    
    return true;
}
```

## Testing Requirements

### Core Library Testing

- Add unit tests for all new functionality
- Ensure backward compatibility with existing code
- Test on multiple platforms (desktop, Android, iOS)
- Test with different model sizes and formats

### Mobile Testing

- Test on at least one physical Android and iOS device
- Test memory usage with different model sizes
- Verify background processing behavior
- Test with interruptions (app minimized, network changes, etc.)

## Platform-Specific Guidelines

### Android Development

- Follow Android platform conventions for JNI code
- Test on multiple API levels (minimum API 24)
- Ensure proper memory management and callbacks
- Support both ARM64 and x86_64 architectures

Example JNI pattern:
```cpp
JNIEXPORT jlong JNICALL
Java_com_cactus_LlamaContext_initContext(
    JNIEnv *env,
    jobject thiz,
    jstring model_path_str,
    /* other parameters */
) {
    // Convert Java parameters to C++
    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);
    
    // Create C++ object
    auto *ctx = new cactus::cactus_context();
    
    // Configure context with parameters
    cactus::common_params params;
    params.model = model_path_chars;
    
    // Load model
    bool success = ctx->loadModel(params);
    
    // Release Java strings
    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    
    return success ? reinterpret_cast<jlong>(ctx) : 0;
}
```

### iOS Development

- Provide both Objective-C and Swift interfaces
- Support Metal acceleration for Apple Silicon
- Test on multiple iOS versions (minimum iOS 13)
- Ensure proper memory usage in limited environments

### React/JavaScript Development

- Use TypeScript for all code
- Follow React Native best practices
- Provide proper typings and documentation
- Ensure compatibility with React Native's JavaScript thread

## Documentation Guidelines

- Document all public APIs with detailed descriptions
- Include parameter explanations and return value details
- Add example code snippets for common use cases
- Update README.md and platform-specific documentation
- Document performance implications and memory requirements

## Benchmarking

When making performance-related changes, include benchmarks:

1. Use the `bench()` functionality to measure tokens per second
2. Test with at least three different model sizes
3. Compare before and after your changes
4. Document performance improvements in your PR

## Common Issues

### Memory Management

- **Issue**: Memory leaks in long-running contexts  
  **Solution**: Ensure all resources are properly freed in destructors

- **Issue**: Excessive memory usage during model loading  
  **Solution**: Use memory mapping (`use_mmap`) when appropriate

### Cross-Platform Compatibility

- **Issue**: JNI crashes on specific Android devices  
  **Solution**: Always check JNI references for NULL before using

- **Issue**: iOS Metal acceleration not working  
  **Solution**: Verify Metal library paths and device compatibility

### Threading Limitations

- **Issue**: Deadlocks in multi-threaded environments  
  **Solution**: Avoid sharing context objects between threads

## License

By contributing to Cactus, you agree that your contributions will be licensed under the project's MIT License as found in the [LICENSE](LICENSE) file.

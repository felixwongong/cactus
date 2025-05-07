# Add consumer ProGuard rules here for libraries that use this library.
# These rules will be applied to the consuming app.
# It is generally recommended NOT to obfuscate or remove public API classes and methods
# from your library, as consumers rely on them.

-keep class com.cactus.android.LlamaContext { *; }
-keepclassmembers class com.cactus.android.LlamaContext {
    native <methods>;
}

# If you have other public API classes from your library, uncomment and add them below:
# -keep class com.cactus.android.LlamaInitParams { *; }
# -keep class com.cactus.android.LlamaCompletionParams { *; }
# -keep class com.cactus.android.YourOtherPublicClass { *; }

# Example:
# -keep class com.cactus.android.** { *; } # Keep everything in the package 
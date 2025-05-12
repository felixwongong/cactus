#ifndef JNI_HELPERS_H
#define JNI_HELPERS_H

#include <jni.h>
#include <string>
#include <vector>
#include <map> // Using std::map for simplicity, could use unordered_map

// Include llama.h for llama_token definition
#include "llama.h"

// --- Forward Declarations for Helper Functions ---

// Basic Conversions (Java -> C++)
std::string javaStringToCppString(JNIEnv* env, jstring javaStr);
std::vector<std::string> javaStringArrayToCppVector(JNIEnv* env, jobjectArray stringArray);
std::vector<float> javaFloatArrayToCppVector(JNIEnv* env, jfloatArray floatArray);
std::vector<int> javaIntArrayToCppVector(JNIEnv* env, jintArray intArray);
std::map<llama_token, float> javaMapTokenFloatToCppMap(JNIEnv* env, jobject mapObject); // For logit bias


// Basic Conversions (C++ -> Java)
jstring cppStringToJavaString(JNIEnv* env, const std::string& cppStr);
jobjectArray cppVectorToJavaStringArray(JNIEnv* env, const std::vector<std::string>& vec);
jintArray cppVectorToJavaIntArray(JNIEnv* env, const std::vector<int>& vec);
jfloatArray cppVectorToJavaFloatArray(JNIEnv* env, const std::vector<float>& vec); // For embeddings


// Java Object Creation
jobject createJavaHashMap(JNIEnv* env, int initialCapacity = 16); // Default capacity
jobject createJavaArrayList(JNIEnv* env, int initialCapacity = 10);


// Populating Java HashMap (from C++)
void putJavaObjectInMap(JNIEnv* env, jobject map, const char* key, jobject value); // Generic put
void putJavaStringInMap(JNIEnv* env, jobject map, const char* key, const char* value);
void putJavaIntInMap(JNIEnv* env, jobject map, const char* key, jint value);
void putJavaLongInMap(JNIEnv* env, jobject map, const char* key, jlong value);
void putJavaDoubleInMap(JNIEnv* env, jobject map, const char* key, jdouble value);
void putJavaBooleanInMap(JNIEnv* env, jobject map, const char* key, jboolean value);
// putJavaMapInMap and putJavaListInMap are covered by putJavaObjectInMap


// Populating Java ArrayList (from C++)
void addJavaObjectToList(JNIEnv* env, jobject list, jobject value); // Generic add
void addJavaStringToList(JNIEnv* env, jobject list, const char* value);
void addJavaIntToList(JNIEnv* env, jobject list, jint value);
void addJavaDoubleToList(JNIEnv* env, jobject list, jdouble value);
// addJavaMapToList and addJavaListToList are covered by addJavaObjectToList


// Exception Handling
void jniThrowNativeException(JNIEnv* env, const char* className, const char* msg);
bool checkAndClearException(JNIEnv* env, const char* functionName); // Helper to log exceptions

// Callback Handling Structure (Example - adjust as needed for Kotlin interface)
struct NativeCallbackContext {
    JavaVM* jvm;
    jobject callbackObjectRef; // Global ref to the Kotlin callback object
    jmethodID progressMethodId; // Cached method ID for progress callback
    jmethodID partialCompletionMethodId; // Cached method ID for partial completion
    jmethodID logMethodId; // Cached method ID for logging
};

#endif // JNI_HELPERS_H 
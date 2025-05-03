#include "jni-helpers.h"
#include <android/log.h>
#include <stdexcept> // Include for exception types if needed, though we throw Java ones
#include <vector>

// Add include for llama_token type
#include "llama.h"

#define TAG "JNI_HELPERS"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)

// --- Helper Implementations ---

// --- Basic Conversions (Java -> C++) ---

std::string javaStringToCppString(JNIEnv* env, jstring javaStr) {
    if (!javaStr) {
        return "";
    }
    const char* cStr = env->GetStringUTFChars(javaStr, nullptr);
    if (!cStr) {
        checkAndClearException(env, "GetStringUTFChars");
        return ""; // Failed to get chars
    }
    std::string cppStr = cStr;
    env->ReleaseStringUTFChars(javaStr, cStr);
    return cppStr;
}

std::vector<std::string> javaStringArrayToCppVector(JNIEnv* env, jobjectArray stringArray) {
    std::vector<std::string> vec;
    if (!stringArray) return vec;

    jsize len = env->GetArrayLength(stringArray);
    vec.reserve(len);
    for (jsize i = 0; i < len; ++i) {
        jstring jStr = (jstring)env->GetObjectArrayElement(stringArray, i);
        if (jStr) {
            vec.push_back(javaStringToCppString(env, jStr));
            env->DeleteLocalRef(jStr); // Clean up local reference
        } else {
             checkAndClearException(env, "GetObjectArrayElement");
             vec.push_back(""); // Add empty string for null elements
        }
    }
    return vec;
}

std::vector<float> javaFloatArrayToCppVector(JNIEnv* env, jfloatArray floatArray) {
    std::vector<float> vec;
    if (!floatArray) return vec;

    jsize len = env->GetArrayLength(floatArray);
    jfloat* elements = env->GetFloatArrayElements(floatArray, nullptr);
    if (!elements) {
        checkAndClearException(env, "GetFloatArrayElements");
        return vec;
    }
    vec.assign(elements, elements + len);
    env->ReleaseFloatArrayElements(floatArray, elements, JNI_ABORT); // Use JNI_ABORT for read-only access
    return vec;
}

std::vector<int> javaIntArrayToCppVector(JNIEnv* env, jintArray intArray) {
    std::vector<int> vec;
    if (!intArray) return vec;

    jsize len = env->GetArrayLength(intArray);
    jint* elements = env->GetIntArrayElements(intArray, nullptr);
    if (!elements) {
        checkAndClearException(env, "GetIntArrayElements");
        return vec;
    }
    vec.assign(elements, elements + len);
    env->ReleaseIntArrayElements(intArray, elements, JNI_ABORT);
    return vec;
}

// --- Basic Conversions (C++ -> Java) ---

jstring cppStringToJavaString(JNIEnv* env, const std::string& cppStr) {
    return env->NewStringUTF(cppStr.c_str());
}

jobjectArray cppVectorToJavaStringArray(JNIEnv* env, const std::vector<std::string>& vec) {
    jclass stringClass = env->FindClass("java/lang/String");
    if (!stringClass) return nullptr;

    jobjectArray stringArray = env->NewObjectArray(vec.size(), stringClass, nullptr);
    if (!stringArray) {
        checkAndClearException(env, "NewObjectArray");
        env->DeleteLocalRef(stringClass);
        return nullptr;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        jstring jStr = cppStringToJavaString(env, vec[i]);
        if (!jStr) { // Handle potential error in string conversion
            // Clean up partially created array?
            env->DeleteLocalRef(stringClass);
            env->DeleteLocalRef(stringArray);
            return nullptr;
        }
        env->SetObjectArrayElement(stringArray, i, jStr);
        env->DeleteLocalRef(jStr); // Delete local ref created by cppStringToJavaString
    }

    env->DeleteLocalRef(stringClass);
    return stringArray;
}

jintArray cppVectorToJavaIntArray(JNIEnv* env, const std::vector<int>& vec) {
    jintArray intArray = env->NewIntArray(vec.size());
    if (!intArray) {
        checkAndClearException(env, "NewIntArray");
        return nullptr;
    }
    // JNI critical access might be faster for large arrays if no blocking JNI calls are made
    env->SetIntArrayRegion(intArray, 0, vec.size(), reinterpret_cast<const jint*>(vec.data()));
    if (checkAndClearException(env, "SetIntArrayRegion")) {
        env->DeleteLocalRef(intArray);
        return nullptr;
    }
    return intArray;
}

jfloatArray cppVectorToJavaFloatArray(JNIEnv* env, const std::vector<float>& vec) {
    jfloatArray floatArray = env->NewFloatArray(vec.size());
    if (!floatArray) {
        checkAndClearException(env, "NewFloatArray");
        return nullptr;
    }
    env->SetFloatArrayRegion(floatArray, 0, vec.size(), vec.data());
    if (checkAndClearException(env, "SetFloatArrayRegion")) {
        env->DeleteLocalRef(floatArray);
        return nullptr;
    }
    return floatArray;
}

// --- Java Object Creation ---

jobject createJavaHashMap(JNIEnv* env, int initialCapacity) {
    jclass mapClass = env->FindClass("java/util/HashMap");
    if (!mapClass) return nullptr;
    jmethodID init = env->GetMethodID(mapClass, "<init>", "(I)V"); // Constructor with capacity
    if (!init) { // Fallback to default constructor if capacity one not found
        init = env->GetMethodID(mapClass, "<init>", "()V");
    }
    if (!init) {
        checkAndClearException(env, "HashMap constructor");
        env->DeleteLocalRef(mapClass);
        return nullptr;
    }
    jobject hashMap = env->NewObject(mapClass, init, initialCapacity);
    env->DeleteLocalRef(mapClass);
    return hashMap;
}

jobject createJavaArrayList(JNIEnv* env, int initialCapacity) {
    jclass listClass = env->FindClass("java/util/ArrayList");
    if (!listClass) return nullptr;
    jmethodID init = env->GetMethodID(listClass, "<init>", "(I)V"); // Constructor with capacity
     if (!init) { // Fallback to default constructor
        init = env->GetMethodID(listClass, "<init>", "()V");
    }
    if (!init) {
        checkAndClearException(env, "ArrayList constructor");
        env->DeleteLocalRef(listClass);
        return nullptr;
    }
    jobject arrayList = env->NewObject(listClass, init, initialCapacity);
    env->DeleteLocalRef(listClass);
    return arrayList;
}

// --- Populating Java HashMap ---

void putJavaObjectInMap(JNIEnv* env, jobject map, const char* key, jobject value) {
    if (!map || !key || !value) return;

    jclass mapClass = env->GetObjectClass(map); // Use GetObjectClass for existing objects
    if (!mapClass) return;
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    if (!putMethod) {
        checkAndClearException(env, "Map.put method");
        env->DeleteLocalRef(mapClass);
        return;
    }

    jstring jKey = env->NewStringUTF(key);
    if (!jKey) {
        checkAndClearException(env, "NewStringUTF for key");
        env->DeleteLocalRef(mapClass);
        return;
    }

    env->CallObjectMethod(map, putMethod, (jobject)jKey, value);
    checkAndClearException(env, "Map.put call");

    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(mapClass); // Clean up class ref obtained from GetObjectClass
}

void putJavaStringInMap(JNIEnv* env, jobject map, const char* key, const char* value) {
    if (!value) return; // Don't put null strings
    jstring jValue = env->NewStringUTF(value);
    if (!jValue) {
         checkAndClearException(env, "NewStringUTF for value");
         return;
    }
    putJavaObjectInMap(env, map, key, (jobject)jValue);
    env->DeleteLocalRef(jValue); // Clean up local ref for value
}

void putJavaIntInMap(JNIEnv* env, jobject map, const char* key, jint value) {
    jclass integerClass = env->FindClass("java/lang/Integer");
    if (!integerClass) return;
    jmethodID valueOf = env->GetStaticMethodID(integerClass, "valueOf", "(I)Ljava/lang/Integer;");
    if (!valueOf) {
        checkAndClearException(env, "Integer.valueOf(int)");
        env->DeleteLocalRef(integerClass);
        return;
    }
    jobject jValue = env->CallStaticObjectMethod(integerClass, valueOf, value);
    if (!jValue) {
         checkAndClearException(env, "Integer.valueOf call");
         env->DeleteLocalRef(integerClass);
         return;
    }

    putJavaObjectInMap(env, map, key, jValue);

    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(integerClass);
}

void putJavaLongInMap(JNIEnv* env, jobject map, const char* key, jlong value) {
    jclass longClass = env->FindClass("java/lang/Long");
    if (!longClass) return;
    jmethodID valueOf = env->GetStaticMethodID(longClass, "valueOf", "(J)Ljava/lang/Long;");
    if (!valueOf) {
         checkAndClearException(env, "Long.valueOf(long)");
         env->DeleteLocalRef(longClass);
         return;
    }
    jobject jValue = env->CallStaticObjectMethod(longClass, valueOf, value);
    if (!jValue) {
         checkAndClearException(env, "Long.valueOf call");
         env->DeleteLocalRef(longClass);
         return;
    }

    putJavaObjectInMap(env, map, key, jValue);

    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(longClass);
}

void putJavaDoubleInMap(JNIEnv* env, jobject map, const char* key, jdouble value) {
    jclass doubleClass = env->FindClass("java/lang/Double");
    if (!doubleClass) return;
    jmethodID valueOf = env->GetStaticMethodID(doubleClass, "valueOf", "(D)Ljava/lang/Double;");
     if (!valueOf) {
         checkAndClearException(env, "Double.valueOf(double)");
         env->DeleteLocalRef(doubleClass);
         return;
    }
    jobject jValue = env->CallStaticObjectMethod(doubleClass, valueOf, value);
    if (!jValue) {
         checkAndClearException(env, "Double.valueOf call");
         env->DeleteLocalRef(doubleClass);
         return;
    }

    putJavaObjectInMap(env, map, key, jValue);

    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(doubleClass);
}

void putJavaBooleanInMap(JNIEnv* env, jobject map, const char* key, jboolean value) {
    jclass booleanClass = env->FindClass("java/lang/Boolean");
    if (!booleanClass) return;
    jmethodID valueOf = env->GetStaticMethodID(booleanClass, "valueOf", "(Z)Ljava/lang/Boolean;");
     if (!valueOf) {
         checkAndClearException(env, "Boolean.valueOf(boolean)");
         env->DeleteLocalRef(booleanClass);
         return;
    }
    jobject jValue = env->CallStaticObjectMethod(booleanClass, valueOf, value);
    if (!jValue) {
         checkAndClearException(env, "Boolean.valueOf call");
         env->DeleteLocalRef(booleanClass);
         return;
    }

    putJavaObjectInMap(env, map, key, jValue);

    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(booleanClass);
}


// --- Populating Java ArrayList ---

void addJavaObjectToList(JNIEnv* env, jobject list, jobject value) {
     if (!list || !value) return;

    jclass listClass = env->GetObjectClass(list);
    if (!listClass) return;
    jmethodID addMethod = env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z"); // ArrayList.add returns boolean
    if (!addMethod) {
         checkAndClearException(env, "List.add method");
         env->DeleteLocalRef(listClass);
         return;
    }

    env->CallBooleanMethod(list, addMethod, value);
    checkAndClearException(env, "List.add call");

    env->DeleteLocalRef(listClass);
}

void addJavaStringToList(JNIEnv* env, jobject list, const char* value) {
    if (!value) return;
    jstring jValue = env->NewStringUTF(value);
    if (!jValue) {
         checkAndClearException(env, "NewStringUTF for value");
         return;
    }
    addJavaObjectToList(env, list, (jobject)jValue);
    env->DeleteLocalRef(jValue);
}

void addJavaIntToList(JNIEnv* env, jobject list, jint value) {
    jclass integerClass = env->FindClass("java/lang/Integer");
    if (!integerClass) return;
    jmethodID valueOf = env->GetStaticMethodID(integerClass, "valueOf", "(I)Ljava/lang/Integer;");
    if (!valueOf) {
         checkAndClearException(env, "Integer.valueOf(int)");
         env->DeleteLocalRef(integerClass);
         return;
    }
    jobject jValue = env->CallStaticObjectMethod(integerClass, valueOf, value);
    if (!jValue) {
         checkAndClearException(env, "Integer.valueOf call");
         env->DeleteLocalRef(integerClass);
         return;
    }

    addJavaObjectToList(env, list, jValue);

    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(integerClass);
}

void addJavaDoubleToList(JNIEnv* env, jobject list, jdouble value) {
     jclass doubleClass = env->FindClass("java/lang/Double");
    if (!doubleClass) return;
    jmethodID valueOf = env->GetStaticMethodID(doubleClass, "valueOf", "(D)Ljava/lang/Double;");
    if (!valueOf) {
         checkAndClearException(env, "Double.valueOf(double)");
         env->DeleteLocalRef(doubleClass);
         return;
    }
    jobject jValue = env->CallStaticObjectMethod(doubleClass, valueOf, value);
    if (!jValue) {
         checkAndClearException(env, "Double.valueOf call");
         env->DeleteLocalRef(doubleClass);
         return;
    }

    addJavaObjectToList(env, list, jValue);

    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(doubleClass);
}


// --- Exception Handling ---

void jniThrowNativeException(JNIEnv* env, const char* className, const char* msg) {
    jclass exClass = env->FindClass(className);
    if (exClass != nullptr) {
        env->ThrowNew(exClass, msg);
        // No need to DeleteLocalRef after ThrowNew, it's handled
    } else {
        // Fallback if the specified exception class isn't found
        jclass rteClass = env->FindClass("java/lang/RuntimeException");
        if (rteClass != nullptr) {
            std::string fullMsg = "Failed to find exception class ";
            fullMsg += className;
            fullMsg += ", original message: ";
            fullMsg += msg;
            env->ThrowNew(rteClass, fullMsg.c_str());
        }
        // If even RuntimeException isn't found, we're in deep trouble.
        // JNI functions will likely start failing.
    }
}

bool checkAndClearException(JNIEnv* env, const char* functionName) {
    if (env->ExceptionCheck()) {
        LOGE("JNI Exception occurred in %s", functionName);
        env->ExceptionDescribe(); // Prints the exception stack trace to logcat
        env->ExceptionClear();
        return true;
    }
    return false;
}

// --- Logit Bias Helper (Example) ---
// TODO: Implement javaMapTokenFloatToCppMap if needed for logit bias
std::map<llama_token, float> javaMapTokenFloatToCppMap(JNIEnv* env, jobject mapObject) {
    std::map<llama_token, float> cppMap;
    if (!mapObject) return cppMap;

    jclass mapClass = env->FindClass("java/util/Map");
    if (!mapClass) return cppMap;
    jmethodID entrySetMethod = env->GetMethodID(mapClass, "entrySet", "()Ljava/util/Set;");
    if (!entrySetMethod) { checkAndClearException(env, "Map.entrySet"); env->DeleteLocalRef(mapClass); return cppMap; }

    jobject entrySet = env->CallObjectMethod(mapObject, entrySetMethod);
    if (!entrySet) { checkAndClearException(env, "entrySet call"); env->DeleteLocalRef(mapClass); return cppMap; }

    jclass setClass = env->GetObjectClass(entrySet);
    jmethodID iteratorMethod = env->GetMethodID(setClass, "iterator", "()Ljava/util/Iterator;");
    if (!iteratorMethod) { checkAndClearException(env, "Set.iterator"); env->DeleteLocalRef(mapClass); env->DeleteLocalRef(entrySet); env->DeleteLocalRef(setClass); return cppMap; }

    jobject iterator = env->CallObjectMethod(entrySet, iteratorMethod);
    if (!iterator) { checkAndClearException(env, "iterator call"); env->DeleteLocalRef(mapClass); env->DeleteLocalRef(entrySet); env->DeleteLocalRef(setClass); return cppMap; }

    jclass iteratorClass = env->GetObjectClass(iterator);
    jmethodID hasNextMethod = env->GetMethodID(iteratorClass, "hasNext", "()Z");
    jmethodID nextMethod = env->GetMethodID(iteratorClass, "next", "()Ljava/lang/Object;");
    if (!hasNextMethod || !nextMethod) { checkAndClearException(env, "Iterator methods"); /* cleanup */ return cppMap; }

    jclass entryClass = env->FindClass("java/util/Map$Entry");
    if (!entryClass) { /* cleanup */ return cppMap; }
    jmethodID getKeyMethod = env->GetMethodID(entryClass, "getKey", "()Ljava/lang/Object;");
    jmethodID getValueMethod = env->GetMethodID(entryClass, "getValue", "()Ljava/lang/Object;");
    if (!getKeyMethod || !getValueMethod) { checkAndClearException(env, "Entry methods"); /* cleanup */ return cppMap; }

    jclass integerClass = env->FindClass("java/lang/Integer");
    jmethodID intValueMethod = env->GetMethodID(integerClass, "intValue", "()I");
    jclass floatClass = env->FindClass("java/lang/Float"); // Assuming bias is Float
    jmethodID floatValueMethod = env->GetMethodID(floatClass, "floatValue", "()F");
    if (!integerClass || !intValueMethod || !floatClass || !floatValueMethod) { checkAndClearException(env, "Number classes/methods"); /* cleanup */ return cppMap; }


    while (env->CallBooleanMethod(iterator, hasNextMethod)) {
        jobject entry = env->CallObjectMethod(iterator, nextMethod);
        jobject keyObj = env->CallObjectMethod(entry, getKeyMethod);
        jobject valueObj = env->CallObjectMethod(entry, getValueMethod);

        if (env->IsInstanceOf(keyObj, integerClass) && env->IsInstanceOf(valueObj, floatClass)) {
            llama_token token = (llama_token)env->CallIntMethod(keyObj, intValueMethod);
            float bias = env->CallFloatMethod(valueObj, floatValueMethod);
            cppMap[token] = bias;
        }

        env->DeleteLocalRef(entry);
        env->DeleteLocalRef(keyObj);
        env->DeleteLocalRef(valueObj);
    }

    // Extensive cleanup
    env->DeleteLocalRef(mapClass);
    env->DeleteLocalRef(entrySet);
    env->DeleteLocalRef(setClass);
    env->DeleteLocalRef(iterator);
    env->DeleteLocalRef(iteratorClass);
    env->DeleteLocalRef(entryClass);
    env->DeleteLocalRef(integerClass);
    env->DeleteLocalRef(floatClass);

    return cppMap;
} 
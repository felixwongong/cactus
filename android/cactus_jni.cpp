#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "cactus_ffi.h"

static constexpr size_t DEFAULT_BUFFER_SIZE = 65536;

static const char* jstring_to_cstr(JNIEnv* env, jstring str) {
    if (str == nullptr) return nullptr;
    return env->GetStringUTFChars(str, nullptr);
}

static void release_jstring(JNIEnv* env, jstring str, const char* cstr) {
    if (str != nullptr && cstr != nullptr) {
        env->ReleaseStringUTFChars(str, cstr);
    }
}

struct TokenCallbackContext {
    JNIEnv* env;
    jobject callback;
    jmethodID method;
};

static void token_callback_bridge(const char* token, uint32_t token_id, void* user_data) {
    if (!user_data) return;
    auto* ctx = static_cast<TokenCallbackContext*>(user_data);
    jstring jtoken = ctx->env->NewStringUTF(token);
    ctx->env->CallVoidMethod(ctx->callback, ctx->method, jtoken, static_cast<jint>(token_id));
    ctx->env->DeleteLocalRef(jtoken);
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_cactus_Cactus_nativeInit(JNIEnv* env, jobject, jstring modelPath, jstring corpusDir) {
    const char* path = jstring_to_cstr(env, modelPath);
    const char* corpus = jstring_to_cstr(env, corpusDir);
    jlong handle = reinterpret_cast<jlong>(cactus_init(path, corpus));
    release_jstring(env, modelPath, path);
    release_jstring(env, corpusDir, corpus);
    return handle;
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) {
        cactus_destroy(reinterpret_cast<cactus_model_t>(handle));
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeReset(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) {
        cactus_reset(reinterpret_cast<cactus_model_t>(handle));
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeStop(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) {
        cactus_stop(reinterpret_cast<cactus_model_t>(handle));
    }
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeComplete(JNIEnv* env, jobject, jlong handle,
                                       jstring messagesJson, jstring optionsJson,
                                       jstring toolsJson, jobject callback) {
    if (handle == 0) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    const char* messages = jstring_to_cstr(env, messagesJson);
    const char* options = jstring_to_cstr(env, optionsJson);
    const char* tools = jstring_to_cstr(env, toolsJson);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    TokenCallbackContext* ctx = nullptr;
    cactus_token_callback cb = nullptr;

    if (callback != nullptr) {
        jclass callbackClass = env->GetObjectClass(callback);
        jmethodID method = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;I)V");
        if (method != nullptr) {
            ctx = new TokenCallbackContext{env, callback, method};
            cb = token_callback_bridge;
        }
    }

    int result = cactus_complete(
        reinterpret_cast<cactus_model_t>(handle),
        messages,
        buffer.data(),
        buffer.size(),
        options,
        tools,
        cb,
        ctx
    );

    delete ctx;

    release_jstring(env, messagesJson, messages);
    release_jstring(env, optionsJson, options);
    release_jstring(env, toolsJson, tools);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeTranscribe(JNIEnv* env, jobject, jlong handle,
                                         jstring audioPath, jstring prompt,
                                         jstring optionsJson, jbyteArray pcmData) {
    if (handle == 0) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    const char* path = jstring_to_cstr(env, audioPath);
    const char* promptStr = jstring_to_cstr(env, prompt);
    const char* options = jstring_to_cstr(env, optionsJson);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    const uint8_t* pcmBuffer = nullptr;
    size_t pcmSize = 0;
    jbyte* pcmBytes = nullptr;

    if (pcmData != nullptr) {
        pcmSize = env->GetArrayLength(pcmData);
        pcmBytes = env->GetByteArrayElements(pcmData, nullptr);
        pcmBuffer = reinterpret_cast<const uint8_t*>(pcmBytes);
    }

    int result = cactus_transcribe(
        reinterpret_cast<cactus_model_t>(handle),
        path,
        promptStr,
        buffer.data(),
        buffer.size(),
        options,
        nullptr,
        nullptr,
        pcmBuffer,
        pcmSize
    );

    if (pcmBytes != nullptr) {
        env->ReleaseByteArrayElements(pcmData, pcmBytes, JNI_ABORT);
    }

    release_jstring(env, audioPath, path);
    release_jstring(env, prompt, promptStr);
    release_jstring(env, optionsJson, options);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jfloatArray JNICALL
Java_com_cactus_Cactus_nativeEmbed(JNIEnv* env, jobject, jlong handle,
                                    jstring text, jboolean normalize) {
    if (handle == 0) {
        return nullptr;
    }

    const char* textStr = jstring_to_cstr(env, text);

    std::vector<float> buffer(4096);
    size_t embeddingDim = 0;

    int result = cactus_embed(
        reinterpret_cast<cactus_model_t>(handle),
        textStr,
        buffer.data(),
        buffer.size(),
        &embeddingDim,
        normalize == JNI_TRUE
    );

    release_jstring(env, text, textStr);

    if (result < 0 || embeddingDim == 0) {
        return nullptr;
    }

    jfloatArray jarray = env->NewFloatArray(static_cast<jsize>(embeddingDim));
    env->SetFloatArrayRegion(jarray, 0, static_cast<jsize>(embeddingDim), buffer.data());

    return jarray;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeRagQuery(JNIEnv* env, jobject, jlong handle,
                                       jstring query, jint topK) {
    if (handle == 0) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    const char* queryStr = jstring_to_cstr(env, query);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    int result = cactus_rag_query(
        reinterpret_cast<cactus_model_t>(handle),
        queryStr,
        buffer.data(),
        buffer.size(),
        static_cast<size_t>(topK)
    );

    release_jstring(env, query, queryStr);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeGetLastError(JNIEnv* env, jobject) {
    const char* error = cactus_get_last_error();
    return env->NewStringUTF(error ? error : "");
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeSetTelemetryToken(JNIEnv* env, jobject, jstring token) {
    const char* tokenStr = jstring_to_cstr(env, token);
    if (tokenStr != nullptr) {
        cactus_set_telemetry_token(tokenStr);
        release_jstring(env, token, tokenStr);
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeSetProKey(JNIEnv* env, jobject, jstring key) {
    const char* keyStr = jstring_to_cstr(env, key);
    if (keyStr != nullptr) {
        cactus_set_pro_key(keyStr);
        release_jstring(env, key, keyStr);
    }
}

}

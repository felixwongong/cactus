#include <jni.h>
// #include <android/asset_manager.h>
// #include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstdlib>
#include <ctime>
#include <sys/sysinfo.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <stdexcept> // For exception handling

// Core library includes
#include "cactus.h"
#include "llama.h"
// #include "llama-impl.h" // Probably not needed directly
#include "ggml.h"
#include "common.h" // Needed for common_params, common_tokenize, etc.
#include "json-schema-to-grammar.h" // Needed for schema conversion

// JNI Helpers
#include "jni-helpers.h"

#define UNUSED(x) (void)(x)
#define TAG "CACTUS_ANDROID_JNI"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    TAG, __VA_ARGS__)

// Global map to hold context pointers, mapping jlong (from Kotlin) to C++ pointers
static std::unordered_map<jlong, cactus::cactus_context*> context_map;

// Global pointer for callback context (simplistic approach, might need improvement for multi-context)
static NativeCallbackContext* g_callback_context = nullptr;

// --- Forward declare the C++ callback function wrappers ---
static bool native_progress_callback(float progress, void * user_data);
static void native_log_callback(lm_ggml_log_level level, const char * text, void * user_data);

// --- Common Implementation Function ---
jlong internal_initContextNative(
    JNIEnv *env,
    jclass clazz,
    jstring model_path_str,
    jstring chat_template_str,
    jstring reasoning_format_str, // Assuming string for simplicity, could be enum int
    jboolean embedding,
    jint embd_normalize,
    jint n_ctx,
    jint n_batch,
    jint n_ubatch,
    jint n_threads,
    jint n_gpu_layers,
    jboolean flash_attn,
    jstring cache_type_k_str,
    jstring cache_type_v_str,
    jboolean use_mlock,
    jboolean use_mmap,
    jboolean vocab_only,
    jobject lora_list, // Assuming List<Map<String, Object>>: [{path: String, scaled: Float}, ...]
    jfloat rope_freq_base,
    jfloat rope_freq_scale,
    jint pooling_type,
    jobject load_progress_callback // Kotlin interface object
) {
    UNUSED(clazz); // Now unused in the internal function directly

    // 1. Convert Java types to C++ types
    std::string model_path = javaStringToCppString(env, model_path_str);
    std::string chat_template = javaStringToCppString(env, chat_template_str);
    std::string reasoning_format = javaStringToCppString(env, reasoning_format_str);
    std::string cache_type_k = javaStringToCppString(env, cache_type_k_str);
    std::string cache_type_v = javaStringToCppString(env, cache_type_v_str);

    // 2. Set up common_params
    common_params defaultParams;
    // Basic params
    defaultParams.model = model_path;
    defaultParams.chat_template = chat_template;
    defaultParams.embedding = embedding;
    defaultParams.n_ctx = n_ctx;
    defaultParams.n_batch = n_batch;
    defaultParams.n_ubatch = n_ubatch;
    defaultParams.n_gpu_layers = n_gpu_layers;
    defaultParams.flash_attn = flash_attn;
    defaultParams.use_mlock = use_mlock;
    defaultParams.use_mmap = use_mmap;
    defaultParams.vocab_only = vocab_only;
    if (vocab_only) defaultParams.warmup = false;

    // Enum/special params
    if (strcmp(reasoning_format.c_str(), "deepseek") == 0) {
        defaultParams.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    } else {
        defaultParams.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    }
    if (pooling_type != -1) defaultParams.pooling_type = static_cast<enum llama_pooling_type>(pooling_type);
    if (embd_normalize != -1) defaultParams.embd_normalize = embd_normalize;
    if (embedding) defaultParams.n_ubatch = defaultParams.n_batch; // Required for non-causal

    int max_threads = std::thread::hardware_concurrency();
    int default_n_threads = max_threads == 4 ? 2 : std::min(4, max_threads);
    defaultParams.cpuparams.n_threads = n_threads > 0 ? n_threads : default_n_threads;

    try {
        defaultParams.cache_type_k = cactus::kv_cache_type_from_str(cache_type_k);
        defaultParams.cache_type_v = cactus::kv_cache_type_from_str(cache_type_v);
    } catch (const std::runtime_error& e) {
        jniThrowNativeException(env, "java/lang/IllegalArgumentException", e.what());
        return -1;
    }

    defaultParams.rope_freq_base = rope_freq_base;
    defaultParams.rope_freq_scale = rope_freq_scale;

    // 3. Create cactus context instance
    auto llama = new cactus::cactus_context();
    llama->is_load_interrupted = false;
    llama->loading_progress = 0;

    // 4. Handle Callbacks (Progress)
    NativeCallbackContext* callback_ctx = nullptr;
    if (load_progress_callback != nullptr) {
        callback_ctx = new NativeCallbackContext();
        env->GetJavaVM(&callback_ctx->jvm);
        callback_ctx->callbackObjectRef = env->NewGlobalRef(load_progress_callback);

        defaultParams.progress_callback = native_progress_callback;
        defaultParams.progress_callback_user_data = callback_ctx;
    }

    // 5. Call the core C++ library function
    bool is_model_loaded = false;
    try {
        is_model_loaded = llama->loadModel(defaultParams);
    } catch (const std::exception& e) {
        LOGE("Exception during model loading: %s", e.what());
        if (callback_ctx) {
            env->DeleteGlobalRef(callback_ctx->callbackObjectRef);
            delete callback_ctx;
        }
        delete llama;
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return -1;
    }

    LOGI("[CACTUS] is_model_loaded %s", (is_model_loaded ? "true" : "false"));

    if (is_model_loaded) {
        // Check for unsupported embedding case
        if (embedding && llama_model_has_encoder(llama->model) && llama_model_has_decoder(llama->model)) {
             LOGE("[CACTUS] computing embeddings in encoder-decoder models is not supported");
            llama_free(llama->ctx);
            return -1;
        }

        // LoRA Adapters - Requires parsing lora_list (List<Map<String, Object>>)
        std::vector<common_adapter_lora_info> lora_adapters;
    if (lora_list != nullptr) {
            // TODO: Implement parsing of Java List<Map<String, Object>> into lora_adapters vector
            // Need jni-helpers for List iteration and Map reading
            // Example structure:
            // jsize list_size = env->CallIntMethod(lora_list, list_size_method_id);
            // for (jsize i = 0; i < list_size; ++i) {
            //    jobject map_obj = env->CallObjectMethod(lora_list, list_get_method_id, i);
            //    jstring path_jstr = (jstring) env->CallObjectMethod(map_obj, map_get_method_id, env->NewStringUTF("path"));
            //    jobject scaled_obj = env->CallObjectMethod(map_obj, map_get_method_id, env->NewStringUTF("scaled"));
            //    // Convert path_jstr to std::string
            //    // Convert scaled_obj (Float/Double) to float
            //    // Add to lora_adapters vector
            //    // Cleanup local refs
            // }
        }
        int lora_result = llama->applyLoraAdapters(lora_adapters);
        if (lora_result != 0) {
            LOGE("[Cactus] Failed to apply lora adapters");
             if (callback_ctx) {
                env->DeleteGlobalRef(callback_ctx->callbackObjectRef);
                delete callback_ctx;
             }
      llama_free(llama->ctx);
      return -1;
    }

        // Store context
        jlong context_ptr = reinterpret_cast<jlong>(llama->ctx);
        context_map[context_ptr] = llama;
        // Attach callback context if it exists
        if (callback_ctx) {
            // How to associate this callback context with the llama context?
            // Maybe add a `void* user_data` to cactus_context?
            // Or store it in a separate map keyed by context_ptr?
            // For now, assume it's somehow linked or globally accessible for the callback.
            // This needs refinement based on how callbacks will be managed.
        }
        return context_ptr;
    } else {
        if (callback_ctx) {
            env->DeleteGlobalRef(callback_ctx->callbackObjectRef);
            delete callback_ctx;
        }
        delete llama; // Destructor calls llama_free(ctx) if needed
        jniThrowNativeException(env, "java/lang/RuntimeException", "Model loading failed (unknown reason)");
        return -1; // Indicate failure
    }
}

// --- JNIEXPORT Functions --- (Now call the internal function)
extern "C" {

// --- Context Management ---    

JNIEXPORT jlong JNICALL
Java_com_cactus_android_LlamaContext_initContextNative_00024CactusAndroidLib_1release(
    JNIEnv *env,
    jclass clazz,
    jstring model_path_str,
    jstring chat_template_str,
    jstring reasoning_format_str,
    jboolean embedding,
    jint embd_normalize,
    jint n_ctx,
    jint n_batch,
    jint n_ubatch,
    jint n_threads,
    jint n_gpu_layers,
    jboolean flash_attn,
    jstring cache_type_k_str,
    jstring cache_type_v_str,
    jboolean use_mlock,
    jboolean use_mmap,
    jboolean vocab_only,
    jobject lora_list,
    jfloat rope_freq_base,
    jfloat rope_freq_scale,
    jint pooling_type,
    jobject load_progress_callback
) {
    // Call the common internal implementation
    return internal_initContextNative(env, clazz, model_path_str, chat_template_str, reasoning_format_str,
                                   embedding, embd_normalize, n_ctx, n_batch, n_ubatch, n_threads,
                                   n_gpu_layers, flash_attn, cache_type_k_str, cache_type_v_str,
                                   use_mlock, use_mmap, vocab_only, lora_list, rope_freq_base,
                                   rope_freq_scale, pooling_type, load_progress_callback);
}

JNIEXPORT jlong JNICALL
Java_com_cactus_android_LlamaContext_initContextNative_00024CactusAndroidLib_1debug(
    JNIEnv *env,
    jclass clazz,
    jstring model_path_str,
    jstring chat_template_str,
    jstring reasoning_format_str,
    jboolean embedding,
    jint embd_normalize,
    jint n_ctx,
    jint n_batch,
    jint n_ubatch,
    jint n_threads,
    jint n_gpu_layers,
    jboolean flash_attn,
    jstring cache_type_k_str,
    jstring cache_type_v_str,
    jboolean use_mlock,
    jboolean use_mmap,
    jboolean vocab_only,
    jobject lora_list,
    jfloat rope_freq_base,
    jfloat rope_freq_scale,
    jint pooling_type,
    jobject load_progress_callback
) {
    // Call the common internal implementation
    // Ensure parameter list EXACTLY matches the _release version's call
    return internal_initContextNative(env, clazz, model_path_str, chat_template_str, reasoning_format_str,
                                   embedding, embd_normalize, n_ctx, n_batch, n_ubatch, n_threads,
                                   n_gpu_layers, flash_attn, cache_type_k_str, cache_type_v_str,
                                   use_mlock, use_mmap, vocab_only, lora_list, rope_freq_base,
                                   rope_freq_scale, pooling_type, load_progress_callback);
}

JNIEXPORT void JNICALL
Java_com_cactus_android_LlamaContext_interruptLoad(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env); UNUSED(thiz);
    // This is tricky - the load happens within initContext.
    // We need a way to signal the cactus_context being created during load.
    // Maybe iterate through the map if only one load happens at a time?
    // Or the caller needs to provide the pointer *during* the load?
    // For now, assume we look it up *after* load, which isn't quite right.
    auto it = context_map.find(context_ptr);
    if (it != context_map.end()) {
        it->second->is_load_interrupted = true;
    } else {
        // If called during load, the context might not be in the map yet.
        LOGW("interruptLoad called for context not yet fully initialized or not found: %ld", context_ptr);
        // Maybe set a global flag? This is problematic for concurrent loads.
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_android_LlamaContext_freeContext(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env); UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it != context_map.end()) {
        cactus::cactus_context* llama = it->second;
        context_map.erase(it);
        // TODO: Clean up associated callback context if stored separately
        // if (g_callback_context && reinterpret_cast<jlong>(llama->ctx) == context_ptr) { // Example check
        //     env->DeleteGlobalRef(g_callback_context->callbackObjectRef);
        //     delete g_callback_context;
        //     g_callback_context = nullptr;
        // }
        llama_free(llama->ctx);
        LOGI("Freed context: %ld", context_ptr);
    } else {
        LOGW("Attempting to free non-existent or already freed context pointer: %ld", context_ptr);
    }
}

// --- Model Information ---    

JNIEXPORT jobject JNICALL
Java_com_cactus_android_LlamaContext_modelInfoNative(
    JNIEnv *env,
    jclass clazz,
    jstring model_path_str,
    jobjectArray skip_array // String[]
) {
    UNUSED(env); UNUSED(clazz);
    // 1. Convert Java types to C++
    std::string model_path = javaStringToCppString(env, model_path_str);
    std::vector<std::string> skip_vec = javaStringArrayToCppVector(env, skip_array);

    // 2. Call underlying gguf functions
    struct lm_gguf_init_params params = { false, NULL };
    struct lm_gguf_context * gguf_ctx = lm_gguf_init_from_file(model_path.c_str(), params);

    if (!gguf_ctx) {
        LOGE("%s: failed to load GGUF '%s'", __func__, model_path.c_str());
        jniThrowNativeException(env, "java/io/IOException", "Failed to load model file GGUF info");
        return nullptr;
    }

    // 3. Create Java HashMap to return
    jobject infoMap = createJavaHashMap(env);
    if (!infoMap) { // Check if HashMap creation failed
        lm_gguf_free(gguf_ctx);
        jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create HashMap for model info");
        return nullptr;
    }

    // 4. Populate the HashMap using helpers
    try {
        putJavaIntInMap(env, infoMap, "version", lm_gguf_get_version(gguf_ctx));
        putJavaLongInMap(env, infoMap, "alignment", (jlong)lm_gguf_get_alignment(gguf_ctx));
        putJavaLongInMap(env, infoMap, "data_offset", (jlong)lm_gguf_get_data_offset(gguf_ctx));

        const int n_kv = lm_gguf_get_n_kv(gguf_ctx);
        putJavaIntInMap(env, infoMap, "kv_count", n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char* key = lm_gguf_get_key(gguf_ctx, i);
            if (!key) continue; // Skip if key is null

            // Skip logic
            bool skipped = false;
            for (const auto& skip_entry : skip_vec) {
                if (skip_entry == key) {
                    skipped = true;
                    break;
                }
            }
            if (skipped) continue;

            // Convert value and put in map
            const std::string value = lm_gguf_kv_to_str(gguf_ctx, i);
            putJavaStringInMap(env, infoMap, key, value.c_str());
    }
    } catch (const std::exception& e) {
        lm_gguf_free(gguf_ctx);
        LOGE("Exception while populating modelInfo map: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return nullptr; // Return null instead of partially filled map
    }

    // 5. Cleanup C++ resources
    lm_gguf_free(gguf_ctx);

    // 6. Return the Java HashMap
    return infoMap; // jobject representing HashMap<String, Object>
}

JNIEXPORT jobject JNICALL
Java_com_cactus_android_LlamaContext_loadModelDetails(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;
    if (!llama->model) {
         jniThrowNativeException(env, "java/lang/IllegalStateException", "Model not loaded in context");
         return nullptr;
    }

    jobject result = createJavaHashMap(env);
    jobject meta = createJavaHashMap(env);
    jobject chat_templates = createJavaHashMap(env);
    jobject minja_templates = createJavaHashMap(env);

    if (!result || !meta || !chat_templates || !minja_templates) {
        jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create HashMaps for model details");
        // Clean up any maps that *were* created
        if (result) env->DeleteLocalRef(result);
        if (meta) env->DeleteLocalRef(meta);
        if (chat_templates) env->DeleteLocalRef(chat_templates);
        if (minja_templates) env->DeleteLocalRef(minja_templates);
        return nullptr;
    }

    try {
        // Basic Info
        char desc[1024];
        llama_model_desc(llama->model, desc, sizeof(desc));
        putJavaStringInMap(env, result, "desc", desc);
        putJavaDoubleInMap(env, result, "size", (jdouble)llama_model_size(llama->model)); // size_t -> double
        putJavaDoubleInMap(env, result, "nEmbd", (jdouble)llama_model_n_embd(llama->model)); // int -> double ? Maybe long?
        putJavaDoubleInMap(env, result, "nParams", (jdouble)llama_model_n_params(llama->model)); // uint64_t -> double

        // Metadata
    int count = llama_model_meta_count(llama->model);
    for (int i = 0; i < count; i++) {
        char key[256];
        llama_model_meta_key_by_index(llama->model, i, key, sizeof(key));
        char val[4096];
        llama_model_meta_val_str_by_index(llama->model, i, val, sizeof(val));
            putJavaStringInMap(env, meta, key, val);
    }
        putJavaObjectInMap(env, result, "metadata", meta);

        // Chat Template Info
        putJavaBooleanInMap(env, chat_templates, "isChatTemplateSupported", llama->validateModelChatTemplate(false, nullptr)); // Deprecated one
        putJavaBooleanInMap(env, chat_templates, "llamaChat", llama->validateModelChatTemplate(false, nullptr)); // Legacy name
        
        // Minja Templates
        putJavaBooleanInMap(env, minja_templates, "default", llama->validateModelChatTemplate(true, nullptr));
        putJavaBooleanInMap(env, minja_templates, "toolUse", llama->validateModelChatTemplate(true, "tool_use"));

        // Default Minja Caps
        auto default_caps_map = createJavaHashMap(env);
        if (default_caps_map && llama->templates) { // Check templates ptr
    auto default_tmpl = llama->templates.get()->template_default.get();
            if (default_tmpl) {
    auto default_tmpl_caps = default_tmpl->original_caps();
                 putJavaBooleanInMap(env, default_caps_map, "tools", default_tmpl_caps.supports_tools);
                 putJavaBooleanInMap(env, default_caps_map, "toolCalls", default_tmpl_caps.supports_tool_calls);
                 putJavaBooleanInMap(env, default_caps_map, "parallelToolCalls", default_tmpl_caps.supports_parallel_tool_calls);
                 putJavaBooleanInMap(env, default_caps_map, "toolResponses", default_tmpl_caps.supports_tool_responses);
                 putJavaBooleanInMap(env, default_caps_map, "systemRole", default_tmpl_caps.supports_system_role);
                 putJavaBooleanInMap(env, default_caps_map, "toolCallId", default_tmpl_caps.supports_tool_call_id);
                 putJavaObjectInMap(env, minja_templates, "defaultCaps", default_caps_map);
            } else { env->DeleteLocalRef(default_caps_map); } // clean up if tmpl null
        } else if (default_caps_map) { env->DeleteLocalRef(default_caps_map); } // clean up if map created but templates null
        
        // Tool Use Minja Caps
        auto tool_use_caps_map = createJavaHashMap(env);
        if (tool_use_caps_map && llama->templates) {
    auto tool_use_tmpl = llama->templates.get()->template_tool_use.get();
    if (tool_use_tmpl != nullptr) {
      auto tool_use_tmpl_caps = tool_use_tmpl->original_caps();
                putJavaBooleanInMap(env, tool_use_caps_map, "tools", tool_use_tmpl_caps.supports_tools);
                putJavaBooleanInMap(env, tool_use_caps_map, "toolCalls", tool_use_tmpl_caps.supports_tool_calls);
                putJavaBooleanInMap(env, tool_use_caps_map, "parallelToolCalls", tool_use_tmpl_caps.supports_parallel_tool_calls);
                putJavaBooleanInMap(env, tool_use_caps_map, "systemRole", tool_use_tmpl_caps.supports_system_role);
                putJavaBooleanInMap(env, tool_use_caps_map, "toolResponses", tool_use_tmpl_caps.supports_tool_responses);
                putJavaBooleanInMap(env, tool_use_caps_map, "toolCallId", tool_use_tmpl_caps.supports_tool_call_id);
                putJavaObjectInMap(env, minja_templates, "toolUseCaps", tool_use_caps_map);
             } else { env->DeleteLocalRef(tool_use_caps_map); } // clean up if tmpl null
        } else if (tool_use_caps_map) { env->DeleteLocalRef(tool_use_caps_map); } // clean up if map created but templates null

        putJavaObjectInMap(env, chat_templates, "minja", minja_templates);
        putJavaObjectInMap(env, result, "chatTemplates", chat_templates);

    } catch (const std::exception& e) {
        LOGE("Exception during loadModelDetails: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        // Clean up maps before returning null
        env->DeleteLocalRef(result); 
        env->DeleteLocalRef(meta); 
        env->DeleteLocalRef(chat_templates); 
        env->DeleteLocalRef(minja_templates);
        // Need to check/delete caps maps too if they were created
        return nullptr;
    }

    // Clean up intermediate maps (only keep 'result')
    env->DeleteLocalRef(meta);
    env->DeleteLocalRef(chat_templates);
    env->DeleteLocalRef(minja_templates);
    // Need to check/delete caps maps too if they were created and put

    return result;
}

// --- Chat Formatting ---    

JNIEXPORT jobject JNICALL
Java_com_cactus_android_LlamaContext_getFormattedChatWithJinja(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring messages_json_str,
    jstring chat_template_str,
    jstring json_schema_str,
    jstring tools_json_str,
    jboolean parallel_tool_calls,
    jstring tool_choice_str
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    // Convert inputs
    std::string messages_json = javaStringToCppString(env, messages_json_str);
    std::string chat_template = javaStringToCppString(env, chat_template_str);
    std::string json_schema = javaStringToCppString(env, json_schema_str);
    std::string tools_json = javaStringToCppString(env, tools_json_str);
    std::string tool_choice = javaStringToCppString(env, tool_choice_str);

    jobject result = createJavaHashMap(env);
    if (!result) {
        jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create HashMap for formatted chat");
        return nullptr;
    }

    try {
        common_chat_params formatted = llama->getFormattedChatWithJinja(
            messages_json,
            chat_template,
            json_schema,
            tools_json,
            parallel_tool_calls,
            tool_choice
        );

        // Populate result map
        putJavaStringInMap(env, result, "prompt", formatted.prompt.c_str());
        putJavaIntInMap(env, result, "chat_format", static_cast<int>(formatted.format));
        putJavaStringInMap(env, result, "grammar", formatted.grammar.c_str());
        putJavaBooleanInMap(env, result, "grammar_lazy", formatted.grammar_lazy);

        // Grammar Triggers (List<Map<String, Object>>)
        jobject grammar_triggers_list = createJavaArrayList(env, formatted.grammar_triggers.size());
        if (grammar_triggers_list) {
        for (const auto &trigger : formatted.grammar_triggers) {
                jobject trigger_map = createJavaHashMap(env, 3);
                if (trigger_map) {
                    putJavaIntInMap(env, trigger_map, "type", trigger.type);
                    putJavaStringInMap(env, trigger_map, "value", trigger.value.c_str());
                    putJavaIntInMap(env, trigger_map, "token", trigger.token); // Assuming llama_token fits in jint
                    addJavaObjectToList(env, grammar_triggers_list, trigger_map);
                    env->DeleteLocalRef(trigger_map); // Added map to list
                }
            }
            putJavaObjectInMap(env, result, "grammar_triggers", grammar_triggers_list);
            env->DeleteLocalRef(grammar_triggers_list); // Added list to result map
        }

        // Preserved Tokens (List<String>)
        jobject preserved_tokens_list = cppVectorToJavaStringArray(env, formatted.preserved_tokens);
        if (preserved_tokens_list) {
            putJavaObjectInMap(env, result, "preserved_tokens", preserved_tokens_list);
            env->DeleteLocalRef(preserved_tokens_list);
        }

        // Additional Stops (List<String>)
        jobject additional_stops_list = cppVectorToJavaStringArray(env, formatted.additional_stops);
        if (additional_stops_list) {
             putJavaObjectInMap(env, result, "additional_stops", additional_stops_list);
             env->DeleteLocalRef(additional_stops_list);
        }

    } catch (const std::exception &e) {
        LOGE("[Cactus] Error formatting chat with Jinja: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        env->DeleteLocalRef(result);
        return nullptr;
    }

    return result;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_android_LlamaContext_getFormattedChat(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring messages_json_str,
    jstring chat_template_str
) {
    UNUSED(thiz);
     auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    std::string messages_json = javaStringToCppString(env, messages_json_str);
    std::string chat_template = javaStringToCppString(env, chat_template_str);

    try {
        std::string formatted_chat = llama->getFormattedChat(messages_json, chat_template);
        return cppStringToJavaString(env, formatted_chat);
    } catch (const std::exception &e) {
        LOGE("[Cactus] Error formatting chat: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

// --- Session Management ---    

JNIEXPORT jobject JNICALL
Java_com_cactus_android_LlamaContext_loadSession(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring path_str
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    std::string path = javaStringToCppString(env, path_str);

    jobject result = createJavaHashMap(env, 2);
    if (!result) {
        jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create HashMap for session load result");
        return nullptr;
    }

    size_t n_token_count_out = 0;
    // Ensure embd vector has enough capacity *before* calling load
    // llama->params.n_ctx should be set during initContext
    if (llama->params.n_ctx == 0) { // Safety check
         jniThrowNativeException(env, "java/lang/IllegalStateException", "Context size (n_ctx) is zero, cannot load session");
         env->DeleteLocalRef(result);
         return nullptr;
    }
    try {
        // Resize based on context size, actual loaded tokens might be less.
    llama->embd.resize(llama->params.n_ctx);
        // llama_state_load_file expects a non-const pointer to the data.
        if (!llama_state_load_file(llama->ctx, path.c_str(), llama->embd.data(), llama->embd.capacity(), &n_token_count_out)) {
            jniThrowNativeException(env, "java/io/IOException", "Failed to load session file");
            env->DeleteLocalRef(result);
            return nullptr;
        }
        // Resize down to the actual number of tokens loaded.
    llama->embd.resize(n_token_count_out);

        // Convert loaded tokens back to string for the 'prompt'
    const std::string text = cactus::tokens_to_str(llama->ctx, llama->embd.cbegin(), llama->embd.cend());

        putJavaLongInMap(env, result, "tokens_loaded", (jlong)n_token_count_out);
        putJavaStringInMap(env, result, "prompt", text.c_str());

    } catch (const std::exception& e) {
        LOGE("Exception during loadSession: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        env->DeleteLocalRef(result);
        return nullptr;
    }

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_android_LlamaContext_saveSession(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring path_str,
    jint size_to_save // Max tokens to save, 0 or negative for all
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return -1;
    }
    cactus::cactus_context* llama = it->second;

    std::string path = javaStringToCppString(env, path_str);

    try {
        // llama_state_save_file expects a non-const pointer
        std::vector<llama_token> session_tokens = llama->embd; // Make a copy if needed? Docs are unclear if it modifies.
        int current_size = session_tokens.size();
        int save_size = (size_to_save > 0 && size_to_save <= current_size) ? size_to_save : current_size;

        if (save_size == 0) { // Don't try to save if no tokens
            LOGW("Save session called with 0 tokens to save.");
            return 0;
        }

        if (!llama_state_save_file(llama->ctx, path.c_str(), session_tokens.data(), save_size)) {
            jniThrowNativeException(env, "java/io/IOException", "Failed to save session file");
      return -1;
    }
        return save_size; // Return number of tokens actually saved
    } catch (const std::exception& e) {
        LOGE("Exception during saveSession: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return -1;
    }
}


// --- Completion ---    

// Helper to convert C++ token probs to Java List<Map<String, Object>>
jobject convertTokenProbsToJavaList(JNIEnv *env, cactus::cactus_context *llama, const std::vector<cactus::completion_token_output>& probs) {
    jobject resultList = createJavaArrayList(env, probs.size());
    if (!resultList) return nullptr;

    for (const auto &prob_output : probs) {
        jobject tokenResultMap = createJavaHashMap(env, 2);
        if (!tokenResultMap) continue; // Skip if map creation fails

        std::string tokenStr = cactus::tokens_to_output_formatted_string(llama->ctx, prob_output.tok);
        putJavaStringInMap(env, tokenResultMap, "content", tokenStr.c_str());

        jobject probsForTokenList = createJavaArrayList(env, prob_output.probs.size());
        if (probsForTokenList) {
            for (const auto &p : prob_output.probs) {
                jobject probResultMap = createJavaHashMap(env, 2);
                if (probResultMap) {
            std::string tokStr = cactus::tokens_to_output_formatted_string(llama->ctx, p.tok);
                    putJavaStringInMap(env, probResultMap, "tok_str", tokStr.c_str());
                    putJavaDoubleInMap(env, probResultMap, "prob", p.prob);
                    addJavaObjectToList(env, probsForTokenList, probResultMap);
                    env->DeleteLocalRef(probResultMap);
                }
            }
            putJavaObjectInMap(env, tokenResultMap, "probs", probsForTokenList);
            env->DeleteLocalRef(probsForTokenList);
        }
        addJavaObjectToList(env, resultList, tokenResultMap);
        env->DeleteLocalRef(tokenResultMap);
    }
    return resultList;
}


JNIEXPORT jobject JNICALL
Java_com_cactus_android_LlamaContext_doCompletionNative(
    JNIEnv *env,
    jclass clazz,
    jlong context_ptr,
    jstring prompt_str,
    jint chat_format, // Assuming passed as int matching common_chat_format enum
    jstring grammar_str, 
    // jstring json_schema_str, // Removed, handle schema->grammar conversion caller-side or inside Jinja formatting
    jboolean grammar_lazy,
    jobject grammar_triggers_list, // Java List<Map<String, Object>>
    jobject preserved_tokens_list, // Java List<String> (or maybe Set<Integer> if tokens are passed?)
    jfloat temperature,
    jint n_threads, // Overrides the init setting for this run
    jint n_predict,
    jint n_probs,
    jint penalty_last_n,
    jfloat penalty_repeat,
    jfloat penalty_freq,
    jfloat penalty_present,
    jfloat mirostat,
    jfloat mirostat_tau,
    jfloat mirostat_eta,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat xtc_threshold,
    jfloat xtc_probability,
    jfloat typical_p,
    jint seed,
    jobjectArray stop_array, // String[]
    jboolean ignore_eos,
    jobject logit_bias_map, // Map<Integer, Float> (Token ID -> Bias)
    jfloat   dry_multiplier,
    jfloat   dry_base,
    jint dry_allowed_length,
    jint dry_penalty_last_n,
    jfloat top_n_sigma,
    jobjectArray dry_sequence_breakers_array, // String[]
    jobject partial_completion_callback // Kotlin interface object
) {
    UNUSED(clazz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    if (llama->is_predicting) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Completion already in progress");
        return nullptr;
    }
    llama->is_predicting = true; // Set flag
    llama->is_interrupted = false; // Reset interruption flag

    // --- 1. Setup Parameters --- 
    try {
        llama->rewind(); // Reset generation state
        // llama_reset_timings(llama->ctx); // Reset timings if desired

        // Convert basic inputs
        llama->params.prompt = javaStringToCppString(env, prompt_str);
    llama->params.sampling.seed = (seed == -1) ? time(NULL) : seed;

        // Thread override
    int max_threads = std::thread::hardware_concurrency();
        int default_n_threads = max_threads == 4 ? 2 : std::min(4, max_threads);
    llama->params.cpuparams.n_threads = n_threads > 0 ? n_threads : default_n_threads;

    llama->params.n_predict = n_predict;
    llama->params.sampling.ignore_eos = ignore_eos;

        // Sampling params
    auto & sparams = llama->params.sampling;
    sparams.temp = temperature;
    sparams.penalty_last_n = penalty_last_n;
    sparams.penalty_repeat = penalty_repeat;
    sparams.penalty_freq = penalty_freq;
    sparams.penalty_present = penalty_present;
    sparams.mirostat = mirostat;
    sparams.mirostat_tau = mirostat_tau;
    sparams.mirostat_eta = mirostat_eta;
    sparams.top_k = top_k;
    sparams.top_p = top_p;
    sparams.min_p = min_p;
    sparams.typ_p = typical_p;
    sparams.n_probs = n_probs;
    sparams.xtc_threshold = xtc_threshold;
    sparams.xtc_probability = xtc_probability;
    sparams.dry_multiplier = dry_multiplier;
    sparams.dry_base = dry_base;
    sparams.dry_allowed_length = dry_allowed_length;
    sparams.dry_penalty_last_n = dry_penalty_last_n;
    sparams.top_n_sigma = top_n_sigma;

        // Grammar
        std::string grammar_cpp = javaStringToCppString(env, grammar_str);
        sparams.grammar.clear(); // Clear previous grammar
        if (!grammar_cpp.empty()) {
            sparams.grammar = grammar_cpp;
    }
    sparams.grammar_lazy = grammar_lazy;

        // Preserved Tokens (Assuming List<String> for now)
        sparams.preserved_tokens.clear();
        if (preserved_tokens_list != nullptr) {
             // TODO: Iterate Java List<String>, tokenize each, add single tokens to sparams.preserved_tokens (std::set)
             // std::vector<std::string> preserved_strs = javaStringListToCppVector(env, preserved_tokens_list);
             // for (const auto& token_str : preserved_strs) {
             //     auto ids = common_tokenize(llama->ctx, token_str, false, true);
             //     if (ids.size() == 1) sparams.preserved_tokens.insert(ids[0]);
             //     else LOGW("Preserved token '%s' maps to %zu tokens, skipping", token_str.c_str(), ids.size());
             // }
        }

        // Grammar Triggers (Assuming List<Map<String, Object>>)
        sparams.grammar_triggers.clear();
        if (grammar_triggers_list != nullptr) {
            // TODO: Iterate Java List<Map>, extract type/value/token, handle tokenization for WORD type, add to sparams.grammar_triggers
            // Check if WORD triggers are in preserved_tokens set.
        }

        // Logit Bias (Assuming Map<Integer, Float>)
    const llama_model * model = llama_get_model(llama->ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
        sparams.logit_bias.clear(); // Clear the vector first
        if (logit_bias_map != nullptr) {
            // Convert Java Map to C++ map
            std::map<llama_token, float> bias_map_cpp = javaMapTokenFloatToCppMap(env, logit_bias_map);
            // Iterate C++ map and add to the vector of structs
            for (const auto& pair : bias_map_cpp) {
                sparams.logit_bias.push_back({pair.first, pair.second});
            }
        }
        // Apply ignore_eos bias *after* loading map from Java
        if (ignore_eos) {
            // Check if EOS token already has a bias, update it if so, otherwise add it.
            bool eos_found = false;
            llama_token eos_tok = llama_vocab_eos(vocab);
            for (auto& bias_entry : sparams.logit_bias) {
                if (bias_entry.token == eos_tok) {
                    bias_entry.bias = -INFINITY;
                    eos_found = true;
                    break;
                }
            }
            if (!eos_found) {
                sparams.logit_bias.push_back({eos_tok, -INFINITY});
            }
        }

        // Stop words
        llama->params.antiprompt = javaStringArrayToCppVector(env, stop_array);

        // Dry sequence breakers
        sparams.dry_sequence_breakers = javaStringArrayToCppVector(env, dry_sequence_breakers_array);

    } catch (const std::exception& e) {
         LOGE("Exception during parameter setup for doCompletion: %s", e.what());
         llama->is_predicting = false;
         jniThrowNativeException(env, "java/lang/IllegalArgumentException", e.what());
         return nullptr;
    }

    // --- 2. Initialize Sampling & Load Prompt --- 
    if (!llama->initSampling()) {
        LOGE("Failed to initialize sampling");
        llama->is_predicting = false;
        jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to initialize sampling");
        return nullptr;
    }
    
    try {
    llama->beginCompletion();
    llama->loadPrompt();
    } catch (const std::exception& e) {
         LOGE("Exception during prompt loading: %s", e.what());
         llama->is_predicting = false;
         jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
         return nullptr;
    }

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    // --- 3. Setup Partial Completion Callback --- 
    NativeCallbackContext* completion_callback_ctx = nullptr;
    jmethodID partialCompletionMethodId = nullptr; // Store locally for the loop
    if (partial_completion_callback != nullptr) {
        // TODO: Refactor callback context handling.
        // This assumes a global or per-context callback setup done elsewhere (e.g., initContext)
        // We need the jvm, global ref to the callback object, and the method ID.
        // completion_callback_ctx = g_callback_context; // Example: retrieve context
        // partialCompletionMethodId = completion_callback_ctx->partialCompletionMethodId; // Example: retrieve ID
        // if (!completion_callback_ctx || !partialCompletionMethodId) {
        //     LOGE("Partial completion callback setup failed");
        //     // Decide whether to continue without callbacks or throw
        // }
    }

    // --- 4. Generation Loop --- 
    while (llama->has_next_token && !llama->is_interrupted) {
        cactus::completion_token_output token_with_probs;
        try {
             token_with_probs = llama->doCompletion();
        } catch (const std::exception& e) {
            LOGE("Exception during llama->doCompletion(): %s", e.what());
            // Continue? Break? Throw? For now, break the loop.
            llama->is_interrupted = true; // Mark as interrupted due to error
            break;
        }
        
        if (token_with_probs.tok == -1 || llama->incomplete) {
            continue;
        }
        const std::string token_text = common_token_to_piece(llama->ctx, token_with_probs.tok);

        // Stop string handling (mostly same as before)
        size_t pos = std::min(sent_count, llama->generated_text.size());
        const std::string str_test = llama->generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos = llama->findStoppingStrings(str_test, token_text.size(), cactus::STOP_FULL);
        
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            llama->generated_text.erase(llama->generated_text.begin() + pos + stop_pos, llama->generated_text.end());
            pos = std::min(sent_count, llama->generated_text.size()); // Recalculate pos
        } else {
            is_stop_full = false;
            stop_pos = llama->findStoppingStrings(str_test, token_text.size(), cactus::STOP_PARTIAL);
        }

        if (stop_pos == std::string::npos || (!llama->has_next_token && !is_stop_full && stop_pos > 0)) {
            const std::string to_send = llama->generated_text.substr(pos, std::string::npos);
            if (!to_send.empty()) {
            sent_count += to_send.size();

                // --- Send Partial Completion Callback --- 
                if (completion_callback_ctx && partialCompletionMethodId) {
                    JNIEnv* callbackEnv = nullptr;
                    bool attached = false;
                    int getEnvStat = completion_callback_ctx->jvm->GetEnv((void**)&callbackEnv, JNI_VERSION_1_6);
                    if (getEnvStat == JNI_EDETACHED) {
                        if (completion_callback_ctx->jvm->AttachCurrentThread(&callbackEnv, nullptr) == 0) {
                            attached = true;
                        } else {
                            LOGE("Failed to attach thread for partial completion callback");
                            callbackEnv = nullptr;
                        }
                    } else if (getEnvStat != JNI_OK) {
                         LOGE("Failed to get JNI env for partial completion callback");
                         callbackEnv = nullptr;
                    }

                    if (callbackEnv) {
                        jobject partialResultMap = createJavaHashMap(callbackEnv, 2);
                        if (partialResultMap) {
                            putJavaStringInMap(callbackEnv, partialResultMap, "token", to_send.c_str());

                            // Handle token probabilities if requested
            if (llama->params.sampling.n_probs > 0) {
                                // Calculate probs for the *sent* tokens
              const std::vector<llama_token> to_send_toks = common_tokenize(llama->ctx, to_send, false);
                                size_t probs_start_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
                                size_t probs_end_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
                                std::vector<cactus::completion_token_output> probs_output;
                                if (probs_start_pos < probs_end_pos) {
                                    probs_output.assign(llama->generated_token_probs.begin() + probs_start_pos, llama->generated_token_probs.begin() + probs_end_pos);
              }
                                sent_token_probs_index = probs_end_pos;

                                jobject probsList = convertTokenProbsToJavaList(callbackEnv, llama, probs_output);
                                if (probsList) {
                                    putJavaObjectInMap(callbackEnv, partialResultMap, "completion_probabilities", probsList);
                                    callbackEnv->DeleteLocalRef(probsList);
                                }
                            }
                            // Call the Kotlin callback method
                            callbackEnv->CallVoidMethod(completion_callback_ctx->callbackObjectRef, partialCompletionMethodId, partialResultMap);
                            checkAndClearException(callbackEnv, "partialCompletion callback");
                            callbackEnv->DeleteLocalRef(partialResultMap);
        }
                        if (attached) {
                            completion_callback_ctx->jvm->DetachCurrentThread();
                        }
                    }
                } // end if callback context valid
            } // end if !to_send.empty()
        } // end if send partial
    } // end while loop

    // --- 5. Finalize and Prepare Result --- 
    llama_perf_context_print(llama->ctx); // Print perf to logcat
    llama->is_predicting = false; // Reset flag

    jobject result = createJavaHashMap(env, 10); // Create final result map
    if (!result) {
        jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create final result HashMap");
        return nullptr;
    }

    try {
        // Base result text
        putJavaStringInMap(env, result, "text", llama->generated_text.c_str());

        // Parse tool calls if generation wasn't interrupted
        jobject toolCallsList = createJavaArrayList(env, 0);
        if (!llama->is_interrupted && toolCallsList) {
    std::string reasoningContent = "";
            std::string content = "";
        try {
            common_chat_msg message = common_chat_parse(llama->generated_text, static_cast<common_chat_format>(chat_format));
            if (!message.reasoning_content.empty()) {
                reasoningContent = message.reasoning_content;
                    putJavaStringInMap(env, result, "reasoning_content", reasoningContent.c_str());
            }
            content = message.content;
                 putJavaStringInMap(env, result, "content", content.c_str()); // Always put content, even if empty?

            for (const auto &tc : message.tool_calls) {
                    jobject toolCallMap = createJavaHashMap(env, 3);
                    if (toolCallMap) {
                        putJavaStringInMap(env, toolCallMap, "type", "function"); // Assuming always function
                        jobject functionMap = createJavaHashMap(env, 2);
                        if (functionMap) {
                            putJavaStringInMap(env, functionMap, "name", tc.name.c_str());
                            putJavaStringInMap(env, functionMap, "arguments", tc.arguments.c_str());
                            putJavaObjectInMap(env, toolCallMap, "function", functionMap);
                            env->DeleteLocalRef(functionMap);
                        }
                if (!tc.id.empty()) {
                            putJavaStringInMap(env, toolCallMap, "id", tc.id.c_str());
                }
                        addJavaObjectToList(env, toolCallsList, toolCallMap);
                        env->DeleteLocalRef(toolCallMap);
                    }
                }
                if (env->CallIntMethod(toolCallsList, env->GetMethodID(env->GetObjectClass(toolCallsList), "size", "()I")) > 0) {
                     putJavaObjectInMap(env, result, "tool_calls", toolCallsList);
            }

        } catch (const std::exception &e) {
                LOGW("Error parsing tool calls from generated text: %s", e.what());
                // Don't fail the whole completion, just skip tool calls
            }
            env->DeleteLocalRef(toolCallsList); // Delete if empty or after putting in map
        }
        else if (toolCallsList) { // Clean up if interrupted
             env->DeleteLocalRef(toolCallsList);
    }

        // Completion Probabilities (overall)
        jobject fullProbsList = convertTokenProbsToJavaList(env, llama, llama->generated_token_probs);
        if (fullProbsList) {
             putJavaObjectInMap(env, result, "completion_probabilities", fullProbsList);
             env->DeleteLocalRef(fullProbsList);
        }

        // Completion Stats
        putJavaIntInMap(env, result, "tokens_predicted", llama->num_tokens_predicted);
        putJavaIntInMap(env, result, "tokens_evaluated", llama->num_prompt_tokens);
        putJavaBooleanInMap(env, result, "truncated", llama->truncated);
        putJavaBooleanInMap(env, result, "stopped_eos", llama->stopped_eos);
        putJavaBooleanInMap(env, result, "stopped_word", llama->stopped_word);
        putJavaBooleanInMap(env, result, "stopped_limit", llama->stopped_limit);
        putJavaStringInMap(env, result, "stopping_word", llama->stopping_word.c_str());
        putJavaIntInMap(env, result, "tokens_cached", llama->n_past); // n_past is size_t

        // Timings
    const auto timings_token = llama_perf_context(llama -> ctx);
        jobject timingsResultMap = createJavaHashMap(env, 8);
        if (timingsResultMap) {
            putJavaIntInMap(env, timingsResultMap, "prompt_n", timings_token.n_p_eval);
            putJavaLongInMap(env, timingsResultMap, "prompt_ms", timings_token.t_p_eval_ms);
            if (timings_token.n_p_eval > 0) {
                putJavaDoubleInMap(env, timingsResultMap, "prompt_per_token_ms", (double)timings_token.t_p_eval_ms / timings_token.n_p_eval);
                putJavaDoubleInMap(env, timingsResultMap, "prompt_per_second", 1e3 / ((double)timings_token.t_p_eval_ms / timings_token.n_p_eval));
            } else {
                 putJavaDoubleInMap(env, timingsResultMap, "prompt_per_token_ms", 0.0);
                 putJavaDoubleInMap(env, timingsResultMap, "prompt_per_second", 0.0);
            }
            putJavaIntInMap(env, timingsResultMap, "predicted_n", timings_token.n_eval);
            putJavaLongInMap(env, timingsResultMap, "predicted_ms", timings_token.t_eval_ms);
             if (timings_token.n_eval > 0) {
                putJavaDoubleInMap(env, timingsResultMap, "predicted_per_token_ms", (double)timings_token.t_eval_ms / timings_token.n_eval);
                putJavaDoubleInMap(env, timingsResultMap, "predicted_per_second", 1e3 / ((double)timings_token.t_eval_ms / timings_token.n_eval));
            } else {
                 putJavaDoubleInMap(env, timingsResultMap, "predicted_per_token_ms", 0.0);
                 putJavaDoubleInMap(env, timingsResultMap, "predicted_per_second", 0.0);
            }
            putJavaObjectInMap(env, result, "timings", timingsResultMap);
            env->DeleteLocalRef(timingsResultMap);
        }

    } catch (const std::exception& e) {
         LOGE("Exception during final result processing: %s", e.what());
         jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
         env->DeleteLocalRef(result);
         return nullptr;
    }

    return result;
}


JNIEXPORT void JNICALL
Java_com_cactus_android_LlamaContext_stopCompletion(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env); UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it != context_map.end()) {
        it->second->is_interrupted = true;
    } else {
         LOGW("stopCompletion called on invalid context pointer: %ld", context_ptr);
    }
}

JNIEXPORT jboolean JNICALL
Java_com_cactus_android_LlamaContext_isPredicting(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env); UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it != context_map.end()) {
        return it->second->is_predicting;
}
    // Should we throw if context invalid?
    LOGW("isPredicting called on invalid context pointer: %ld", context_ptr);
    return false;
}

// --- Tokenization ---    

JNIEXPORT jobject JNICALL // Returns List<Integer>
Java_com_cactus_android_LlamaContext_tokenize(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring text_str,
    jboolean add_bos,       // Added parameter, common_tokenize needs it
    jboolean parse_special // Added parameter
) {
    UNUSED(thiz);
     auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    std::string text = javaStringToCppString(env, text_str);

    try {
        // Use common_tokenize, which handles special tokens based on parse_special
    const std::vector<llama_token> toks = common_tokenize(
        llama->ctx,
            text,
            add_bos,       // Add BOS token?
            parse_special  // Parse special tokens?
    );

        jobject resultList = createJavaArrayList(env, toks.size());
        if (!resultList) {
            jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create ArrayList for tokens");
            return nullptr;
        }
    for (const auto &tok : toks) {
            addJavaIntToList(env, resultList, tok);
    }
        return resultList;
    } catch (const std::exception& e) {
        LOGE("Exception during tokenize: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL
Java_com_cactus_android_LlamaContext_detokenize(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jintArray tokens_array // int[]
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    std::vector<int> tokens_vec = javaIntArrayToCppVector(env, tokens_array);
    // Convert vector<int> to vector<llama_token> (assuming direct cast is okay)
    std::vector<llama_token> llama_tokens(tokens_vec.begin(), tokens_vec.end());

    try {
        // Use cactus helper which likely handles utf-8 reconstruction better
        auto text = cactus::tokens_to_str(llama->ctx, llama_tokens.cbegin(), llama_tokens.cend());
        return cppStringToJavaString(env, text);
    } catch (const std::exception& e) {
         LOGE("Exception during detokenize: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

// --- Embeddings ---    

JNIEXPORT jboolean JNICALL
Java_com_cactus_android_LlamaContext_isEmbeddingEnabled(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env); UNUSED(thiz);
     auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        // Throw or return false? Returning false might be safer.
        LOGW("isEmbeddingEnabled called on invalid context pointer: %ld", context_ptr);
        return false;
    }
    return it->second->params.embedding;
}

JNIEXPORT jobject JNICALL // Returns Map<String, Object> { "embedding": List<Double>, "prompt_tokens": List<String> }
Java_com_cactus_android_LlamaContext_embedding(
    JNIEnv *env,
    jobject thiz,
        jlong context_ptr,
    jstring text_str,
    jint embd_normalize // Override normalize setting (-1 to use context default)
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    if (!llama->params.embedding) {
         jniThrowNativeException(env, "java/lang/IllegalStateException", "Embedding mode not enabled for this context");
         return nullptr;
    }

    std::string text = javaStringToCppString(env, text_str);

    jobject result = createJavaHashMap(env, 2);
     if (!result) {
        jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create HashMap for embedding result");
        return nullptr;
    }

    try {
        // Use a temporary params copy to potentially override normalization
        common_params embdParams = llama->params; // Copy existing params
        embdParams.embedding = true; // Ensure it's set
    if (embd_normalize != -1) {
      embdParams.embd_normalize = embd_normalize;
    }

    llama->rewind();
        llama_perf_context_reset(llama->ctx); // Reset timings for embedding

        llama->params.prompt = text; // Set prompt in the *main* context params
        llama->params.n_predict = 0; // No prediction needed

        if (!llama->initSampling()) { // Still need to init sampling?
            jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to initialize sampling for embedding");
            env->DeleteLocalRef(result);
            return nullptr;
    }

    llama->beginCompletion();
    llama->loadPrompt();
        // llama->doCompletion(); // Do we need one step? cactus->getEmbedding implies it might be done inside

    std::vector<float> embedding = llama->getEmbedding(embdParams);

        // Convert embedding to Java List<Double>
        jobject embeddingsList = createJavaArrayList(env, embedding.size());
        if (embeddingsList) {
    for (const auto &val : embedding) {
                addJavaDoubleToList(env, embeddingsList, (jdouble) val);
    }
            putJavaObjectInMap(env, result, "embedding", embeddingsList);
            env->DeleteLocalRef(embeddingsList);
        }

        // Convert prompt tokens (llama->embd) to Java List<String>
        jobject promptTokensList = createJavaArrayList(env, llama->embd.size());
        if (promptTokensList) {
    for (const auto &tok : llama->embd) {
                addJavaStringToList(env, promptTokensList, common_token_to_piece(llama->ctx, tok).c_str());
    }
            putJavaObjectInMap(env, result, "prompt_tokens", promptTokensList);
             env->DeleteLocalRef(promptTokensList);
        }

    } catch (const std::exception& e) {
         LOGE("Exception during embedding: %s", e.what());
         jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
         env->DeleteLocalRef(result);
         return nullptr;
    }

    return result;
}

// --- Benchmarking ---    

JNIEXPORT jstring JNICALL
Java_com_cactus_android_LlamaContext_bench(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jint pp,
    jint tg,
    jint pl,
    jint nr
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
    }
    cactus::cactus_context* llama = it->second;

    try {
    std::string result = llama->bench(pp, tg, pl, nr);
        return cppStringToJavaString(env, result);
    } catch (const std::exception& e) {
        LOGE("Exception during bench: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

// --- LoRA ---    

JNIEXPORT jint JNICALL
Java_com_cactus_android_LlamaContext_applyLoraAdapters(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jobject lora_adapters_list // Assuming List<Map<String, Object>>
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return -1;
    }
    cactus::cactus_context* llama = it->second;

    std::vector<common_adapter_lora_info> lora_adapters;
    if (lora_adapters_list != nullptr) {
         // TODO: Implement parsing of Java List<Map<String, Object>> into lora_adapters vector
    }

    try {
    return llama->applyLoraAdapters(lora_adapters);
    } catch (const std::exception& e) {
        LOGE("Exception applying LoRA: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return -1;
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_android_LlamaContext_removeLoraAdapters(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env); UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it != context_map.end()) {
        try {
            it->second->removeLoraAdapters();
        } catch (const std::exception& e) {
             LOGE("Exception removing LoRA: %s", e.what());
            jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        }
    } else {
         LOGW("removeLoraAdapters called on invalid context pointer: %ld", context_ptr);
    }
}

JNIEXPORT jobject JNICALL // Returns List<Map<String, Object>>
Java_com_cactus_android_LlamaContext_getLoadedLoraAdapters(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(thiz);
    auto it = context_map.find(context_ptr);
    if (it == context_map.end()) {
        jniThrowNativeException(env, "java/lang/IllegalStateException", "Context pointer invalid or freed");
        return nullptr;
}
    cactus::cactus_context* llama = it->second;

    try {
        auto loaded_lora_adapters = llama->getLoadedLoraAdapters();
        jobject resultList = createJavaArrayList(env, loaded_lora_adapters.size());
        if (!resultList) {
             jniThrowNativeException(env, "java/lang/RuntimeException", "Failed to create ArrayList for LoRA adapters");
             return nullptr;
        }
        for (const auto &la : loaded_lora_adapters) {
            jobject map = createJavaHashMap(env, 2);
            if (map) {
                putJavaStringInMap(env, map, "path", la.path.c_str());
                putJavaDoubleInMap(env, map, "scaled", la.scale); // scale is float
                addJavaObjectToList(env, resultList, map);
                env->DeleteLocalRef(map);
            }
        }
        return resultList;
    } catch (const std::exception& e) {
        LOGE("Exception getting loaded LoRA: %s", e.what());
        jniThrowNativeException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

// --- Logging ---    

// C++ function to be called by llama_log_set
static void native_log_callback(lm_ggml_log_level level, const char * text, void * user_data) {
    if (!user_data) return;
    NativeCallbackContext* cb_ctx = static_cast<NativeCallbackContext*>(user_data);

    JNIEnv *env;
    bool attached = false;
    int getEnvStat = cb_ctx->jvm->GetEnv((void**)&env, JNI_VERSION_1_6);

    if (getEnvStat == JNI_EDETACHED) {
        if (cb_ctx->jvm->AttachCurrentThread(&env, nullptr) == 0) {
            attached = true;
        } else {
            LOGE("Failed to attach thread for log callback");
            return;
        }
    } else if (getEnvStat != JNI_OK) {
         LOGE("Failed to get JNI env for log callback");
        return;
    }

    // Map lm_ggml_log_level to a string or int for Java/Kotlin
    const char* level_str;
    switch(level) {
        case LM_GGML_LOG_LEVEL_ERROR: level_str = "ERROR"; break;
        case LM_GGML_LOG_LEVEL_WARN:  level_str = "WARN"; break;
        case LM_GGML_LOG_LEVEL_INFO:  level_str = "INFO"; break;
        default: level_str = "DEBUG"; // Or VERBOSE?
    }

    jstring jLevel = env->NewStringUTF(level_str);
    jstring jText = env->NewStringUTF(text);

    // TODO: Find the correct method ID for the log callback in the Kotlin interface
    // if (cb_ctx->logMethodId && jLevel && jText) {
    //    env->CallVoidMethod(cb_ctx->callbackObjectRef, cb_ctx->logMethodId, jLevel, jText);
    //    checkAndClearException(env, "log callback");
    // } else {
    //    LOGE("Log callback method ID invalid or string creation failed");
    // }
    // Fallback: Print to android log if JNI call fails
    __android_log_print(ANDROID_LOG_INFO, TAG, "[%s] %s", level_str, text);

    if(jLevel) env->DeleteLocalRef(jLevel);
    if(jText) env->DeleteLocalRef(jText);

    if (attached) {
        cb_ctx->jvm->DetachCurrentThread();
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_android_LlamaContext_setupLog(JNIEnv *env, jclass clazz, jobject logCallback) {
    UNUSED(env); UNUSED(clazz);
    if (g_callback_context) {
        LOGW("Log callback already set up. Replacing.");
        // Clean up old one?
        env->DeleteGlobalRef(g_callback_context->callbackObjectRef);
        delete g_callback_context;
        g_callback_context = nullptr;
    }

    if (!logCallback) {
        LOGI("Disabling custom JNI log callback.");
         // llama_log_set(lm_ggml_log_callback_default, NULL); // Use llama's default if needed
        return; // Or maybe set back to default Android log?
    }

    g_callback_context = new NativeCallbackContext();
    env->GetJavaVM(&g_callback_context->jvm);
    g_callback_context->callbackObjectRef = env->NewGlobalRef(logCallback);

    // TODO: Find the method ID for the log callback method on the 'logCallback' object
    // jclass callbackClass = env->GetObjectClass(logCallback);
    // g_callback_context->logMethodId = env->GetMethodID(callbackClass, "onNativeLog", "(Ljava/lang/String;Ljava/lang/String;)V"); // Example signature
    // env->DeleteLocalRef(callbackClass);
    // if (!g_callback_context->logMethodId) {
    //    LOGE("Failed to find log callback method. Custom logging disabled.");
    //    env->DeleteGlobalRef(g_callback_context->callbackObjectRef);
    //    delete g_callback_context;
    //    g_callback_context = nullptr;
    //    return;
    // }

    llama_log_set(native_log_callback, g_callback_context);
    LOGI("Custom JNI log callback enabled.");
}

JNIEXPORT void JNICALL
Java_com_cactus_android_LlamaContext_unsetLog(JNIEnv *env, jclass clazz) {
    UNUSED(env); UNUSED(clazz);
    if (g_callback_context) {
         env->DeleteGlobalRef(g_callback_context->callbackObjectRef);
         delete g_callback_context;
         g_callback_context = nullptr;
         llama_log_set(nullptr, NULL); // Disable llama logging callback
         LOGI("Custom JNI log callback disabled.");
    } else {
         LOGI("Custom JNI log callback was not set.");
    }
    // Optionally set back to a default logger if desired:
    // llama_log_set(lm_ggml_log_callback_default, NULL);
}

// --- Utility --- 

// C++ function wrapper for progress callback
static bool native_progress_callback(float progress, void * user_data) {
    if (!user_data) return true; // Continue if no context
     NativeCallbackContext* cb_ctx = static_cast<NativeCallbackContext*>(user_data);

    JNIEnv *env;
    bool attached = false;
    int getEnvStat = cb_ctx->jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (getEnvStat == JNI_EDETACHED) {
        if (cb_ctx->jvm->AttachCurrentThread(&env, nullptr) == 0) attached = true;
        else { LOGE("Failed to attach thread for progress callback"); return false; /* Stop load */ }
    } else if (getEnvStat != JNI_OK) {
         LOGE("Failed to get JNI env for progress callback"); return false; /* Stop load */
    }

    bool continue_load = true;
    // TODO: Call the Kotlin progress callback method
    // if (cb_ctx->progressMethodId) {
    //    jboolean should_continue = env->CallBooleanMethod(cb_ctx->callbackObjectRef, cb_ctx->progressMethodId, (jint)(progress * 100));
    //    checkAndClearException(env, "progress callback");
    //    continue_load = (should_continue == JNI_TRUE);
    // } else {
    //    LOGW("Progress callback method ID invalid");
    // }

    if (attached) {
        cb_ctx->jvm->DetachCurrentThread();
    }
    // Need to check llama->is_load_interrupted as well? Or rely solely on callback return?
    auto it = context_map.find(reinterpret_cast<jlong>(cb_ctx)); // How to find the right context?
    // This linkage is broken without a way to tie user_data back to the cactus_context
    // if (it != context_map.end()) {
    //    if (it->second->is_load_interrupted) return false;
    // }
    return continue_load;
}


} // extern "C"

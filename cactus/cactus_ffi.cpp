#include "cactus_ffi.h"
#include "cactus.h"
#include "common.h"
#include "llama.h"

#include <string>
#include <vector>
#include <stdexcept>
#include <cstring> // For strdup, strlen
#include <cstdlib> // For malloc, free
#include <sstream> // For formatting token json
#include <iostream> // For potential debugging

// Helper to convert C string array to C++ vector of strings
static std::vector<std::string> c_str_array_to_vector(const char** arr, int count) {
    std::vector<std::string> vec;
    if (arr != nullptr) {
        for (int i = 0; i < count; ++i) {
            if (arr[i] != nullptr) {
                vec.push_back(arr[i]);
            }
        }
    }
    return vec;
}

// Helper to safely duplicate a C string (caller must free)
static char* safe_strdup(const std::string& str) {
    if (str.empty()) {
        // Return a pointer to an empty string literal (read-only) or allocated empty string
        // Let's allocate to be consistent with freeing non-empty strings
        char* empty_str = (char*)malloc(1);
        if (empty_str) empty_str[0] = '\0';
        return empty_str;
    }
    char* new_str = (char*)malloc(str.length() + 1);
    if (new_str) {
        std::strcpy(new_str, str.c_str());
    }
    return new_str; // Can be nullptr if malloc fails
}


extern "C" {

cactus_context_handle_t cactus_init_context_c(const cactus_init_params_c_t* params) {
    if (!params || !params->model_path) {
        // Log error: Invalid parameters
        return nullptr;
    }

    cactus::cactus_context* context = nullptr;
    try {
        context = new cactus::cactus_context();

        // --- Translate C params to C++ common_params --- 
        common_params cpp_params;
        cpp_params.model = params->model_path;
        if (params->chat_template) {
            cpp_params.chat_template = params->chat_template;
        }
        cpp_params.n_ctx = params->n_ctx;
        cpp_params.n_batch = params->n_batch;
        cpp_params.n_ubatch = params->n_ubatch;
        cpp_params.n_gpu_layers = params->n_gpu_layers;
        cpp_params.cpuparams.n_threads = params->n_threads;
        cpp_params.use_mmap = params->use_mmap;
        cpp_params.use_mlock = params->use_mlock;
        cpp_params.embedding = params->embedding;
        cpp_params.pooling_type = static_cast<enum llama_pooling_type>(params->pooling_type);
        cpp_params.embd_normalize = params->embd_normalize;
        cpp_params.flash_attn = params->flash_attn;
        if (params->cache_type_k) {
             try {
                  cpp_params.cache_type_k = cactus::kv_cache_type_from_str(params->cache_type_k);
             } catch (const std::exception& e) {
                 // Log warning about invalid cache type K
                 delete context;
                 return nullptr;
             }
        }
        if (params->cache_type_v) {
            try {
                cpp_params.cache_type_v = cactus::kv_cache_type_from_str(params->cache_type_v);
            } catch (const std::exception& e) {
                // Log warning about invalid cache type V
                delete context;
                return nullptr;
            }
        }
        // TODO: Add translation for LoRA, RoPE params if exposed in C struct

        // Progress callback - MORE COMPLEX - requires stable function pointer or trampoline
        // This simple version might crash if the Dart function disappears
        if (params->progress_callback) {
            cpp_params.progress_callback = [](float progress, void* user_data) {
                auto callback = reinterpret_cast<void (*)(float)>(user_data);
                callback(progress);
                // Return value indicates interruption - how to handle this from Dart?
                // For now, assume we don't interrupt loading via this callback.
                return true; 
            };
            cpp_params.progress_callback_user_data = reinterpret_cast<void*>(params->progress_callback);
        } else {
             cpp_params.progress_callback = nullptr;
             cpp_params.progress_callback_user_data = nullptr;
        }
        // ---------

        if (!context->loadModel(cpp_params)) {
            // loadModel logs errors internally
            delete context;
            return nullptr;
        }

        return reinterpret_cast<cactus_context_handle_t>(context);

    } catch (const std::exception& e) {
        // Log error: Exception during context creation
        std::cerr << "Error initializing context: " << e.what() << std::endl;
        if (context) delete context;
        return nullptr;
    } catch (...) {
        // Log error: Unknown exception
        if (context) delete context;
        return nullptr;
    }
}

void cactus_free_context_c(cactus_context_handle_t handle) {
    if (handle) {
        cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
        delete context;
    }
}

int cactus_completion_c(
    cactus_context_handle_t handle,
    const cactus_completion_params_c_t* params,
    cactus_completion_result_c_t* result // Output parameter
) {
    if (!handle || !params || !params->prompt || !result) {
        return -1; // Invalid arguments
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);

    // Ensure result is zero-initialized
    memset(result, 0, sizeof(cactus_completion_result_c_t));

    try {
        context->rewind();

        // --- Setup context params for this completion --- 
        context->params.prompt = params->prompt;
        if (params->n_threads > 0) {
             context->params.cpuparams.n_threads = params->n_threads;
        }
        context->params.n_predict = params->n_predict;
        context->params.sampling.seed = params->seed;
        context->params.sampling.temp = params->temperature;
        context->params.sampling.top_k = params->top_k;
        context->params.sampling.top_p = params->top_p;
        context->params.sampling.min_p = params->min_p;
        context->params.sampling.typ_p = params->typical_p;
        context->params.sampling.penalty_last_n = params->penalty_last_n;
        context->params.sampling.penalty_repeat = params->penalty_repeat;
        context->params.sampling.penalty_freq = params->penalty_freq;
        context->params.sampling.penalty_present = params->penalty_present;
        context->params.sampling.mirostat = params->mirostat;
        context->params.sampling.mirostat_tau = params->mirostat_tau;
        context->params.sampling.mirostat_eta = params->mirostat_eta;
        context->params.sampling.ignore_eos = params->ignore_eos;
        context->params.sampling.n_probs = params->n_probs;
        context->params.antiprompt = c_str_array_to_vector(params->stop_sequences, params->stop_sequence_count);
        if (params->grammar) {
             context->params.sampling.grammar = params->grammar;
        }
        // TODO: Add other sampling params like logit_bias, etc.
        // -----------

        if (!context->initSampling()) {
            // Log error
            return -2; // Sampling init failed
        }
        context->beginCompletion();
        context->loadPrompt();

        // --- Streaming loop --- 
        while (context->has_next_token && !context->is_interrupted) {
            const cactus::completion_token_output token_with_probs = context->doCompletion();

            if (token_with_probs.tok == -1 && !context->has_next_token) {
                 // End of stream or error signaled by doCompletion
                 break;
            }
            
            if (token_with_probs.tok != -1 && params->token_callback) {
                // Format token data (simple example: just the text)
                // A more complex implementation could create JSON here
                std::string token_text = common_token_to_piece(context->ctx, token_with_probs.tok);
                
                // Call the Dart callback
                bool continue_completion = params->token_callback(token_text.c_str());
                if (!continue_completion) {
                    context->is_interrupted = true; // Stop if callback returns false
                    break;
                }
            }
        }
        // ---------

        // --- Fill final result struct --- 
        result->text = safe_strdup(context->generated_text);
        result->tokens_predicted = context->num_tokens_predicted;
        result->tokens_evaluated = context->num_prompt_tokens;
        result->truncated = context->truncated;
        result->stopped_eos = context->stopped_eos;
        result->stopped_word = context->stopped_word;
        result->stopped_limit = context->stopped_limit;
        result->stopping_word = safe_strdup(context->stopping_word);
        // TODO: Populate timings if needed
        // ---------

        context->is_predicting = false;
        return 0; // Success

    } catch (const std::exception& e) {
        // Log error
        std::cerr << "Error during completion: " << e.what() << std::endl;
        context->is_predicting = false;
        context->is_interrupted = true; // Ensure state is cleaned up
        return -3; // Exception occurred
    } catch (...) {
        // Log error
        context->is_predicting = false;
        context->is_interrupted = true;
        return -4; // Unknown exception
    }
}

void cactus_stop_completion_c(cactus_context_handle_t handle) {
    if (handle) {
        cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
        context->is_interrupted = true;
    }
}

cactus_token_array_c_t cactus_tokenize_c(cactus_context_handle_t handle, const char* text) {
    cactus_token_array_c_t result = {nullptr, 0};
    if (!handle || !text) {
        return result;
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context->ctx) { // Need the llama_context
        return result;
    }

    try {
        std::vector<llama_token> tokens_vec = ::common_tokenize(context->ctx, text, false, true);
        if (!tokens_vec.empty()) {
            result.count = tokens_vec.size();
            result.tokens = (int32_t*)malloc(result.count * sizeof(int32_t));
            if (result.tokens) {
                // Copy data
                std::copy(tokens_vec.begin(), tokens_vec.end(), result.tokens);
            } else {
                result.count = 0; // Malloc failed
            }
        }
        return result;
    } catch (...) {
        // Log error
        return {nullptr, 0};
    }
}

char* cactus_detokenize_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count) {
    if (!handle || !tokens || count <= 0) {
        return safe_strdup(""); // Return empty string
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
     if (!context->ctx) {
        return safe_strdup("");
    }

    try {
        std::vector<llama_token> tokens_vec(tokens, tokens + count);
        std::string text = cactus::tokens_to_str(context->ctx, tokens_vec.cbegin(), tokens_vec.cend());
        // Print the intermediate C++ string for debugging
        std::cout << "[DEBUG cactus_detokenize_c] Intermediate std::string: [" << text << "]" << std::endl;
        return safe_strdup(text);
    } catch (...) {
        // Log error
        return safe_strdup("");
    }
}

cactus_float_array_c_t cactus_embedding_c(cactus_context_handle_t handle, const char* text) {
    cactus_float_array_c_t result = {nullptr, 0};
     if (!handle || !text) {
        return result;
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context->ctx || !context->params.embedding) { // Check if embedding mode is enabled
        // Log error: Not initialized for embeddings
        return result;
    }

    try {
        // Need to run a minimal inference pass to get embeddings
        context->rewind();
        context->params.prompt = text;
        context->params.n_predict = 0; // No prediction needed

        if (!context->initSampling()) { return result; }
        context->beginCompletion();
        context->loadPrompt();
        context->doCompletion(); // Evaluate the prompt

        // Dummy params for getEmbedding (it uses context's internal params)
        common_params dummy_embd_params;
        dummy_embd_params.embd_normalize = context->params.embd_normalize;

        std::vector<float> embedding_vec = context->getEmbedding(dummy_embd_params);

        if (!embedding_vec.empty()) {
            result.count = embedding_vec.size();
            result.values = (float*)malloc(result.count * sizeof(float));
            if (result.values) {
                std::copy(embedding_vec.begin(), embedding_vec.end(), result.values);
            } else {
                result.count = 0; // Malloc failed
            }
        }
        context->is_predicting = false;
        return result;

    } catch (...) {
        // Log error
        context->is_predicting = false;
        return {nullptr, 0};
    }
}

// --- Memory Freeing Functions ---

void cactus_free_string_c(char* str) {
    if (str) {
        free(str);
    }
}

void cactus_free_token_array_c(cactus_token_array_c_t arr) {
    if (arr.tokens) {
        free(arr.tokens);
    }
    // No need to zero out arr, caller owns it
}

void cactus_free_float_array_c(cactus_float_array_c_t arr) {
    if (arr.values) {
        free(arr.values);
    }
}

void cactus_free_completion_result_members_c(cactus_completion_result_c_t* result) {
    if (result) {
        cactus_free_string_c(result->text);
        cactus_free_string_c(result->stopping_word);
        result->text = nullptr; // Prevent double free
        result->stopping_word = nullptr;
    }
}

} // extern "C" 
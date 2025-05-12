#ifndef CACTUS_FFI_H
#define CACTUS_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Opaque handle ---
// Represents the C++ cactus::cactus_context
typedef struct cactus_context_opaque* cactus_context_handle_t;

// --- Structs for Parameters (Mirroring relevant parts of common_params) ---

typedef struct cactus_init_params_c {
    const char* model_path;
    const char* chat_template; // Optional

    // Matching common_params fields (add more as needed)
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_gpu_layers;
    int32_t n_threads; // Number of CPU threads
    bool use_mmap;
    bool use_mlock;
    bool embedding;     // Enable embedding mode
    int32_t pooling_type; // enum llama_pooling_type
    int32_t embd_normalize;
    bool flash_attn;
    const char* cache_type_k; // e.g., "f16"
    const char* cache_type_v; // e.g., "f16"
    // Add lora params if needed
    // Add rope params if needed

    // Callback for progress (optional)
    // Note: Passing function pointers requires care with Dart FFI NativeCallable
    void (*progress_callback)(float progress); // Simple example

} cactus_init_params_c_t;

typedef struct cactus_completion_params_c {
    const char* prompt;
    int32_t n_predict; // Max tokens to generate (-1 for infinite)
    int32_t n_threads; // Override context threads if > 0
    int32_t seed;
    double temperature;
    int32_t top_k;
    double top_p;
    double min_p;
    double typical_p;
    int32_t penalty_last_n;
    double penalty_repeat;
    double penalty_freq;
    double penalty_present;
    int32_t mirostat;
    double mirostat_tau;
    double mirostat_eta;
    bool ignore_eos;
    int32_t n_probs; // Number of probabilities to return per token
    const char** stop_sequences; // Null-terminated array of C strings
    int stop_sequence_count;
    const char* grammar; // Optional grammar

    // Callback for generated tokens (optional)
    // Called for each new piece of text generated.
    // `token_json` might contain token string, probs etc. (needs definition)
    // Return `false` from callback to stop completion early.
    bool (*token_callback)(const char* token_json);

} cactus_completion_params_c_t;


// --- Structs for Results ---

// Represents a list of tokens (int32_t array)
typedef struct cactus_token_array_c {
    int32_t* tokens;
    int32_t count;
} cactus_token_array_c_t;

// Represents a list of floats (float array)
typedef struct cactus_float_array_c {
    float* values;
    int32_t count;
} cactus_float_array_c_t;

// Represents the final result of a completion call
typedef struct cactus_completion_result_c {
    char* text; // Full generated text (caller must free using cactus_free_string_c)
    int32_t tokens_predicted;
    int32_t tokens_evaluated; // Prompt tokens
    bool truncated;
    bool stopped_eos;
    bool stopped_word;
    bool stopped_limit;
    char* stopping_word; // (caller must free using cactus_free_string_c)
    // Add timings if needed
} cactus_completion_result_c_t;


// --- Core API Functions ---

/**
 * @brief Initializes a cactus context with the given parameters.
 *
 * @param params Parameters for initialization.
 * @return A handle to the context, or NULL on failure. Caller must free with cactus_free_context_c.
 */
cactus_context_handle_t cactus_init_context_c(const cactus_init_params_c_t* params);

/**
 * @brief Frees the resources associated with a cactus context.
 *
 * @param handle The context handle returned by cactus_init_context_c.
 */
void cactus_free_context_c(cactus_context_handle_t handle);

/**
 * @brief Performs text completion based on the provided prompt and parameters.
 *        This is potentially a long-running operation.
 *        Tokens are streamed via the callback in params.
 *
 * @param handle The context handle.
 * @param params Completion parameters, including prompt and sampling settings.
 * @param result Output struct to store the final result details (text must be freed).
 * @return 0 on success, non-zero on failure.
 */
int cactus_completion_c(
    cactus_context_handle_t handle,
    const cactus_completion_params_c_t* params,
    cactus_completion_result_c_t* result // Output parameter
);

/**
 * @brief Requests the ongoing completion operation to stop.
 *        This sets an interrupt flag; completion does not stop instantly.
 *
 * @param handle The context handle.
 */
void cactus_stop_completion_c(cactus_context_handle_t handle);

/**
 * @brief Tokenizes the given text.
 *
 * @param handle The context handle.
 * @param text The text to tokenize.
 * @return A struct containing the tokens. Caller must free the `tokens` array using cactus_free_token_array_c.
 */
cactus_token_array_c_t cactus_tokenize_c(cactus_context_handle_t handle, const char* text);

/**
 * @brief Detokenizes the given sequence of tokens.
 *
 * @param handle The context handle.
 * @param tokens Pointer to the token IDs.
 * @param count Number of tokens.
 * @return The detokenized string. Caller must free using cactus_free_string_c.
 */
char* cactus_detokenize_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count);

/**
 * @brief Generates embeddings for the given text. Context must be initialized with embedding=true.
 *
 * @param handle The context handle.
 * @param text The text to embed.
 * @return A struct containing the embedding values. Caller must free the `values` array using cactus_free_float_array_c.
 */
cactus_float_array_c_t cactus_embedding_c(cactus_context_handle_t handle, const char* text);

// --- Memory Freeing Functions ---
// These MUST be called from Dart to free memory allocated by the C layer.

/** @brief Frees a string allocated by the C API. */
void cactus_free_string_c(char* str);

/** @brief Frees a token array allocated by the C API. */
void cactus_free_token_array_c(cactus_token_array_c_t arr);

/** @brief Frees a float array allocated by the C API. */
void cactus_free_float_array_c(cactus_float_array_c_t arr);

/** @brief Frees the members *within* a completion result struct (like text, stopping_word). */
void cactus_free_completion_result_members_c(cactus_completion_result_c_t* result);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // CACTUS_FFI_H 
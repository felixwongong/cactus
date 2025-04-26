#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to cactus_context
typedef void *CactusContextRef;

// Create a new cactus_context instance
CactusContextRef cactus_context_create(void);

// Load a model into the context. Returns 1 on success, 0 on failure.
int cactus_context_load_model(CactusContextRef ctx, const char *model_path);

// Destroy the cactus_context instance
void cactus_context_destroy(CactusContextRef ctx);

// --------- Model Information ---------

// Get model type
const char* cactus_context_get_model_type(CactusContextRef ctx);

// Get number of parameters in the model
int64_t cactus_context_get_n_params(CactusContextRef ctx);

// Get number of layers in the model
int cactus_context_get_n_layers(CactusContextRef ctx);

// Get context size (max tokens)
int cactus_context_get_context_size(CactusContextRef ctx);

// Get embedding size
int cactus_context_get_embedding_size(CactusContextRef ctx);

// Check if llama chat format is supported
bool cactus_context_has_llama_chat(CactusContextRef ctx);

// Check if minja (jinja) templates are supported
bool cactus_context_has_minja(CactusContextRef ctx);

// --------- Completion ---------

// Simplified struct for completion parameters
typedef struct {
    const char* prompt;           // Prompt text
    const char* messages_json;    // JSON string of messages
    const char* chat_template;    // Custom chat template
    bool jinja;                   // Whether to use Jinja templating
    int max_tokens;               // Maximum number of tokens to generate
    float temperature;            // Temperature for sampling
    float top_p;                  // Top-p for nucleus sampling
    int top_k;                    // Top-k for sampling
    float frequency_penalty;      // Frequency penalty
    float presence_penalty;       // Presence penalty
    bool logprobs;                // Whether to return token probabilities
    int top_logprobs;             // Number of tokens to return probabilities for
    const char* response_format;  // JSON string of response format
    const char* tools_json;       // JSON string of tools
} CactusCompletionParams;

// Struct for token data callback
typedef struct {
    const char* token;            // Token text
    const char* probs_json;       // JSON string of token probabilities
} CactusTokenData;

// Callback type for token generation
typedef void (*CactusTokenCallback)(CactusTokenData token_data, void* user_data);

// Perform completion
// Returns a JSON string with the completion result
char* cactus_context_completion(
    CactusContextRef ctx,
    CactusCompletionParams params,
    CactusTokenCallback callback,
    void* user_data
);

// Stop an ongoing completion
void cactus_context_stop_completion(CactusContextRef ctx);

// --------- Tokenization ---------

// Tokenize text into token IDs
// Returns a JSON string with the token IDs
char* cactus_context_tokenize(CactusContextRef ctx, const char* text);

// Convert token IDs back to text
// `tokens` should be a JSON array string of token IDs
char* cactus_context_detokenize(CactusContextRef ctx, const char* tokens);

// --------- Embeddings ---------

// Generate embeddings for text
// Returns a JSON string with the embedding vectors
char* cactus_context_embedding(
    CactusContextRef ctx,
    const char* text,
    bool normalize
);

// --------- Session Management ---------

// Load a session from a file
// Returns a JSON string with the session loading result
char* cactus_context_load_session(CactusContextRef ctx, const char* filepath);

// Save a session to a file
// Returns the number of tokens saved, or -1 on error
int cactus_context_save_session(
    CactusContextRef ctx,
    const char* filepath,
    int token_size
);

// --------- Memory Management ---------

// Free a string allocated by any of the functions that return strings
void cactus_free_string(char* str);

#ifdef __cplusplus
}
#endif 
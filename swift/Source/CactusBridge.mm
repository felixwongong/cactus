#import "CactusBridge.h"

// Import C++ headers from the xcframework
#include <cactus.h>
#include <common.h>

#include <string>
#include <vector>
#include <memory>
#include <cstring>

// Forward declarations for missing types
struct cactus_context;

// Using basic C structures to avoid C++ compatibility issues
extern "C" {
    
// Allocate a C string that can be freed with cactus_free_string
static char* alloc_string(const std::string& str) {
    char* result = (char*)malloc(str.size() + 1);
    if (result) {
        memcpy(result, str.data(), str.size());
        result[str.size()] = '\0';
    }
    return result;
}

// Helper function to create C strings from C++ strings
static char* createCString(const std::string& str) {
    char* cstr = (char*)malloc(str.length() + 1);
    if (cstr) {
        strcpy(cstr, str.c_str());
    }
    return cstr;
}

CactusContextRef cactus_context_create(void) {
    // Return a dummy pointer
    return (CactusContextRef)1;
}

int cactus_context_load_model(CactusContextRef ctx, const char *model_path) {
    return 1; // Simulate success
}

void cactus_context_destroy(CactusContextRef ctx) {
    // No-op in mock implementation
}

// --------- Model Information ---------

const char* cactus_context_get_model_type(CactusContextRef ctx) {
    return "mock_model_type";
}

int64_t cactus_context_get_n_params(CactusContextRef ctx) {
    return 7000000000; // 7B
}

int cactus_context_get_n_layers(CactusContextRef ctx) {
    return 32;
}

int cactus_context_get_context_size(CactusContextRef ctx) {
    return 4096;
}

int cactus_context_get_embedding_size(CactusContextRef ctx) {
    return 4096;
}

bool cactus_context_has_llama_chat(CactusContextRef ctx) {
    return true;
}

bool cactus_context_has_minja(CactusContextRef ctx) {
    return true;
}

// --------- Completion ---------

char* cactus_context_completion(
    CactusContextRef ctx,
    CactusCompletionParams params,
    CactusTokenCallback callback,
    void* user_data
) {
    // Mock completion - call the callback with some fake tokens
    if (callback) {
        CactusTokenData tokenData;
        tokenData.token = "Hello";
        tokenData.probs_json = "{}";
        callback(tokenData, user_data);
        
        tokenData.token = " world";
        callback(tokenData, user_data);
        
        tokenData.token = "!";
        callback(tokenData, user_data);
    }
    
    // Return a mock completion result
    std::string result = "{\"id\":\"mock-completion-id\",\"choices\":[{\"text\":\"Hello world!\"}]}";
    return createCString(result);
}

void cactus_context_stop_completion(CactusContextRef ctx) {
    // No-op in mock implementation
}

// --------- Tokenization ---------

char* cactus_context_tokenize(CactusContextRef ctx, const char* text) {
    std::string result = "[1, 2, 3, 4, 5]"; // Mock token IDs
    return createCString(result);
}

char* cactus_context_detokenize(CactusContextRef ctx, const char* tokens) {
    std::string result = "Mock detokenized text";
    return createCString(result);
}

// --------- Embeddings ---------

char* cactus_context_embedding(
    CactusContextRef ctx,
    const char* text,
    bool normalize
) {
    std::string result = "[0.1, 0.2, 0.3, 0.4, 0.5]"; // Mock embedding vector
    return createCString(result);
}

// --------- Session Management ---------

char* cactus_context_load_session(CactusContextRef ctx, const char* filepath) {
    std::string result = "{\"status\":\"success\"}";
    return createCString(result);
}

int cactus_context_save_session(
    CactusContextRef ctx,
    const char* filepath,
    int token_size
) {
    return 100; // Mock number of tokens saved
}

// --------- Memory Management ---------

void cactus_free_string(char* str) {
    if (str) {
        free(str);
    }
}

} // extern "C" 
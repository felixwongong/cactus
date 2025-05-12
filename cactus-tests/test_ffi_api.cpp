#include "test_ffi_api.h"
#include "../cactus/cactus_ffi.h"
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstring> 

void test_ffi_init_free_context() {
    std::cout << "Testing FFI context init/free..." << std::endl;
    cactus_init_params_c_t params_c = {}; 
    params_c.model_path = "../llm.gguf"; 
    params_c.n_ctx = 256; 
    params_c.n_batch = 512; 
    params_c.n_threads = 1;
    params_c.embedding = false;
    params_c.use_mmap = true;

    cactus_context_handle_t handle = cactus_init_context_c(&params_c);
    assert(handle != nullptr && "FFI: cactus_init_context_c failed");

    cactus_free_context_c(handle);
    cactus_free_context_c(nullptr);

    std::cout << "FFI context init/free test passed" << std::endl;
}

void test_ffi_tokenize_detokenize() {
    std::cout << "Testing FFI tokenize/detokenize..." << std::endl;

    // 1. Initialize context via FFI
    cactus_init_params_c_t init_params_c = {};
    init_params_c.model_path = "../llm.gguf";
    init_params_c.n_ctx = 512;
    init_params_c.n_batch = 512; 
    init_params_c.n_threads = 1;
    init_params_c.use_mmap = true;
    cactus_context_handle_t handle = cactus_init_context_c(&init_params_c);
    assert(handle != nullptr && "FFI: Context init failed for tokenize test");

    // 2. Tokenize via FFI
    const char* test_text = "Hello FFI.";
    cactus_token_array_c_t token_array = cactus_tokenize_c(handle, test_text);
    assert(token_array.tokens != nullptr && "FFI: Tokenization failed (null tokens)");
    assert(token_array.count > 0 && "FFI: Tokenization failed (zero count)");
    std::cout << "  FFI: Tokenized '" << test_text << "' into " << token_array.count << " tokens." << std::endl;

    // 3. Detokenize via FFI
    char* detokenized_text = cactus_detokenize_c(handle, token_array.tokens, token_array.count);
    assert(detokenized_text != nullptr && "FFI: Detokenization failed (null text)");

    // 4. Assert correctness - Expecting a leading space based on tokenizer behavior
    const char* expected_detokenized_text = " Hello FFI.";
    assert(strcmp(detokenized_text, expected_detokenized_text) == 0 && "FFI: Detokenized text does not match expected output");
    std::cout << "  FFI: Detokenized back to: '" << detokenized_text << "' (matches expected)" << std::endl;

    // 5. Clean up FFI-allocated memory FIRST
    cactus_free_string_c(detokenized_text);
    cactus_free_token_array_c(token_array); 

    // 6. Clean up context handle
    cactus_free_context_c(handle);

    std::cout << "FFI tokenize/detokenize test passed" << std::endl;
}

void test_ffi_completion_basic() {
    std::cout << "Testing FFI basic completion..." << std::endl;
    // 1. Init context
    cactus_init_params_c_t init_params_c = {};
    init_params_c.model_path = "../llm.gguf";
    init_params_c.n_ctx = 512;
    init_params_c.n_batch = 512; 
    init_params_c.n_threads = 1;
    init_params_c.use_mmap = true;
    cactus_context_handle_t handle = cactus_init_context_c(&init_params_c);
    assert(handle != nullptr && "FFI: Context init failed for completion test");

    // 2. Setup completion params
    cactus_completion_params_c_t comp_params_c = {};
    comp_params_c.prompt = "What is the capital of France?";
    comp_params_c.n_predict = 10; 
    comp_params_c.temperature = 0.1; 
    comp_params_c.seed = 1234;
    comp_params_c.token_callback = nullptr;

    // 3. Call completion
    cactus_completion_result_c_t result = {};
    int status = cactus_completion_c(handle, &comp_params_c, &result);
    assert(status == 0 && "FFI: cactus_completion_c failed");

    // 4. Check results (basic checks)
    assert(result.text != nullptr && "FFI: Completion result text is null");
    assert(strlen(result.text) > 0 && "FFI: Completion result text is empty");
    assert(result.tokens_predicted > 0 && "FFI: Completion predicted zero tokens");
    std::cout << "  FFI: Completion prompt: '" << comp_params_c.prompt << "'" << std::endl;
    std::cout << "  FFI: Completion result text (first ~50 chars): " << std::string(result.text).substr(0, 50) << "..." << std::endl;

    // 5. Clean up result memory
    cactus_free_completion_result_members_c(&result);

    // 6. Clean up context
    cactus_free_context_c(handle);

    std::cout << "FFI basic completion test passed" << std::endl;
}

void test_ffi_embedding_basic() {
    std::cout << "Testing FFI basic embedding..." << std::endl;
    // 1. Init context for embedding
    cactus_init_params_c_t init_params_c = {};
    init_params_c.model_path = "../llm.gguf";
    init_params_c.n_ctx = 512;
    init_params_c.n_batch = 512; 
    init_params_c.n_threads = 1;
    init_params_c.use_mmap = true;
    init_params_c.embedding = true; 
    init_params_c.pooling_type = 0; 

    cactus_context_handle_t handle = cactus_init_context_c(&init_params_c);
    assert(handle != nullptr && "FFI: Context init failed for embedding test");

    // 2. Generate embedding
    const char* embed_text = "Embed this.";
    cactus_float_array_c_t embedding_array = cactus_embedding_c(handle, embed_text);
    assert(embedding_array.values != nullptr && "FFI: Embedding failed (null values)");
    assert(embedding_array.count > 0 && "FFI: Embedding failed (zero count)");
    std::cout << "  FFI: Embedding vector size: " << embedding_array.count << std::endl;

    // 3. Clean up embedding memory
    cactus_free_float_array_c(embedding_array);

    // 4. Clean up context
    cactus_free_context_c(handle);

    std::cout << "FFI basic embedding test passed" << std::endl;
} 
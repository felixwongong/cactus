#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstring> 

#include "test_core_api.h"  
#include "test_ffi_api.h"  

// Helper function to check if a string contains another string
bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}


int main() {
    try {
        // Call core C++ API tests
        test_model_loading();
        test_basic_completion();
        test_chat_formatting();
        test_prompt_truncation();
        test_stopping_criteria();
        test_embedding_generation();
        test_benchmarking();
        test_jinja_chat_formatting();
        test_kv_cache_type();
        
        // Call FFI API tests
        test_ffi_init_free_context();
        test_ffi_tokenize_detokenize();
        test_ffi_completion_basic();
        test_ffi_embedding_basic();
        
        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
} 
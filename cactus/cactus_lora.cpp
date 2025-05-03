#include "cactus.h"
#include "common.h" // For common_set_adapter_lora
#include "llama.h"  // For llama_adapter_lora_init
#include <vector>
#include <string>

namespace cactus {

/**
 * @brief Applies LoRA adapters to the model
 * 
 * @param lora_adapters Vector of LoRA adapter information
 * @return 0 on success, negative on failure
 */
int cactus_context::applyLoraAdapters(std::vector<common_adapter_lora_info> lora_adapters) {
    // Ensure model and context are valid
    if (!ctx || !model) {
        LOG_ERROR("Context or model not initialized for applying LoRA adapters.");
        return -1;
    }

    // Initialize adapter pointers
    for (auto &la : lora_adapters) {
        // Check if path is valid
        if (la.path.empty()) {
            LOG_WARNING("Skipping LoRA adapter with empty path.");
            continue;
        }
        la.ptr = llama_adapter_lora_init(model, la.path.c_str());
        if (la.ptr == nullptr) {
            LOG_ERROR("Failed to initialize LoRA adapter '%s'\n", la.path.c_str());
            // Clean up already initialized adapters before returning?
            // For simplicity, just return error code.
            return -1;
        }
        LOG_INFO("Initialized LoRA adapter: %s, Scale: %f", la.path.c_str(), la.scale);
    }

    // Store the successfully initialized adapters
    this->lora = lora_adapters; 

    // Apply the set of adapters to the context
    common_set_adapter_lora(ctx, this->lora); // Pass the stored vector
    LOG_INFO("Applied %zu LoRA adapters.", this->lora.size());
    return 0;
}

/**
 * @brief Removes all LoRA adapters from the model
 */
void cactus_context::removeLoraAdapters() {
     // Ensure context is valid
    if (!ctx) {
        LOG_ERROR("Context not initialized, cannot remove LoRA adapters.");
        return;
    }
    this->lora.clear(); // Clear the internal list
    common_set_adapter_lora(ctx, this->lora); // Apply the empty list to the context
    LOG_INFO("Removed all LoRA adapters.");
}

/**
 * @brief Gets information about currently loaded LoRA adapters
 * 
 * @return Vector of LoRA adapter information
 */
std::vector<common_adapter_lora_info> cactus_context::getLoadedLoraAdapters() {
    // Simply return the currently stored list
    return this->lora;
}

} // namespace cactus 
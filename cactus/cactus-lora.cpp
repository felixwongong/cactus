#include "cactus.h"

/**
 * @file cactus-lora.cpp
 * @brief LoRA adapter functionality for the Cactus LLM interface
 * 
 * This file contains the implementation for handling LoRA adapters.
 */

namespace cactus {

/**
 * @brief Applies LoRA adapters to the model
 * 
 * @param lora Vector of LoRA adapter information
 * @return 0 on success, negative on failure
 */
int cactus_context::applyLoraAdapters(std::vector<common_adapter_lora_info> lora) {
    for (auto &la : lora) {
        la.ptr = llama_adapter_lora_init(model, la.path.c_str());
        if (la.ptr == nullptr) {
            LOG_ERROR("failed to apply lora adapter '%s'\n", la.path.c_str());
            return -1;
        }
    }
    this->lora = lora;
    common_set_adapter_lora(ctx, lora);
    return 0;
}

/**
 * @brief Removes all LoRA adapters from the model
 */
void cactus_context::removeLoraAdapters() {
    this->lora.clear();
    common_set_adapter_lora(ctx, this->lora); // apply empty list
}

/**
 * @brief Gets information about currently loaded LoRA adapters
 * 
 * @return Vector of LoRA adapter information
 */
std::vector<common_adapter_lora_info> cactus_context::getLoadedLoraAdapters() {
    return this->lora;
}

} // namespace cactus 
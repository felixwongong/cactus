#include "cactus.h"
#include "common.h" 
#include "llama.h" 
#include <vector>
#include <cstdio> 

namespace cactus {

/**
 * @brief Generates embeddings for the current prompt
 * 
 * @param embd_params Parameters for embedding generation (note: uses internal context params primarily)
 * @return Vector of embedding values
 */
std::vector<float> cactus_context::getEmbedding(common_params &embd_params) 
{
    // Ensure context and model are valid
    if (!ctx || !model) {
        LOG_ERROR("Context or model not initialized for embedding generation.");
        return {};
    }
    
    const int n_embd = llama_model_n_embd(model);
    
    if (!params.embedding) 
    {
        LOG_WARNING("Embedding mode not enabled for this context.");
        return std::vector<float>(n_embd, 0.0f); // Return zero vector consistent with original logic
    }

    float *data = nullptr; // Initialize to nullptr
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        // Get embeddings for the last token (assuming standard behavior)
        data = llama_get_embeddings(ctx);
    } else {
        // Get pooled embeddings for the sequence (assuming sequence ID 0)
        data = llama_get_embeddings_seq(ctx, 0);
    }

    if (!data) {
        LOG_WARNING("Failed to retrieve embeddings from llama context.");
        return std::vector<float>(n_embd, 0.0f);
    }

    // Create vector from the raw embedding data
    std::vector<float> embedding(data, data + n_embd);
    std::vector<float> out(n_embd); // Create output vector

    // Use the normalization setting from the *context's* parameters
    common_embd_normalize(embedding.data(), out.data(), n_embd, params.embd_normalize);
    return out;
}

} // namespace cactus 
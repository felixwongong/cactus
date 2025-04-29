#include "cactus.h"

/**
 * @file cactus-embedding.cpp
 * @brief Embedding generation functionality for the Cactus LLM interface
 * 
 * This file contains the implementation for generating embeddings from text.
 */

namespace cactus {

/**
 * @brief Generates embeddings for the current prompt
 * 
 * @param embd_params Parameters for embedding generation
 * @return Vector of embedding values
 */
std::vector<float> cactus_context::getEmbedding(common_params &embd_params)
{
    static const int n_embd = llama_model_n_embd(llama_get_model(ctx));
    if (!embd_params.embedding)
    {
        LOG_WARNING("embedding disabled, embedding: %s", embd_params.embedding);
        return std::vector<float>(n_embd, 0.0f);
    }
    float *data;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    printf("pooling_type: %d\n", pooling_type);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        data = llama_get_embeddings(ctx);
    } else {
        data = llama_get_embeddings_seq(ctx, 0);
    }

    if (!data) {
        return std::vector<float>(n_embd, 0.0f);
    }
    std::vector<float> embedding(data, data + n_embd), out(data, data + n_embd);
    common_embd_normalize(embedding.data(), out.data(), n_embd, embd_params.embd_normalize);
    return out;
}

} // namespace cactus 
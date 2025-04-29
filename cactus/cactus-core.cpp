#include "cactus.h"

/**
 * @file cactus-core.cpp
 * @brief Core functionality for the Cactus LLM interface
 * 
 * This file contains the implementation of core Cactus functionality,
 * including KV cache handling, context initialization, and model loading.
 */

namespace cactus {

/**
 * @brief List of supported KV cache types for quantization
 */
const std::vector<lm_ggml_type> kv_cache_types = {
    LM_GGML_TYPE_F32,
    LM_GGML_TYPE_F16,
    LM_GGML_TYPE_BF16,
    LM_GGML_TYPE_Q8_0,
    LM_GGML_TYPE_Q4_0,
    LM_GGML_TYPE_Q4_1,
    LM_GGML_TYPE_IQ4_NL,
    LM_GGML_TYPE_Q5_0,
    LM_GGML_TYPE_Q5_1,
};

/**
 * @brief Converts a string to a KV cache type
 * 
 * @param s String representation of the KV cache type
 * @return The corresponding lm_ggml_type
 * @throws std::runtime_error if the cache type is unsupported
 */
lm_ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (lm_ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

/**
 * @brief Destructor for cactus_context
 * 
 * Cleans up resources, including the sampling context
 */
cactus_context::~cactus_context() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }
}

/**
 * @brief Rewinds the context to start a new generation
 * 
 * Resets internal state to prepare for a new generation task
 */
void cactus_context::rewind() {
    is_interrupted = false;
    params.antiprompt.clear();
    params.sampling.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(params.n_ctx);
    generated_token_probs.clear();
    truncated = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    params.sampling.n_prev = n_ctx;
}

/**
 * @brief Initializes the sampling context
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool cactus_context::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }
    ctx_sampling = common_sampler_init(model, params.sampling);
    return ctx_sampling != nullptr;
}

/**
 * @brief Loads a language model
 * 
 * @param params_ Parameters for model loading and initialization
 * @return true if loading succeeded, false otherwise
 */
bool cactus_context::loadModel(common_params &params_)
{
    params = params_;
    llama_init = common_init_from_params(params);
    model = llama_init.model.get();
    ctx = llama_init.context.get();
    if (model == nullptr)
    {
        LOG_ERROR("unable to load model: %s", params_.model.c_str());
        return false;
    }
    templates = common_chat_templates_init(model, params.chat_template);
    n_ctx = llama_n_ctx(ctx);

    // We can uncomment for debugging or after this fix: https://github.com/ggerganov/llama.cpp/pull/11101
    // LOG_INFO("%s\n", common_params_get_system_info(params).c_str());

    return true;
}

} // namespace cactus 
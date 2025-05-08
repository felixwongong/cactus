#include "cactus.h"
#include "common.h" // For common_init_from_params, common_chat_templates_init, etc.
#include <stdexcept> // For runtime_error

namespace cactus {

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
        LOG_ERROR("unable to load model: %s", params.model.c_str());
        return false;
    }
    templates = common_chat_templates_init(model, params.chat_template);
    n_ctx = llama_n_ctx(ctx);

    return true;
}

/**
 * @brief Validates if a chat template exists and is valid
 * 
 * @param use_jinja Whether to use Jinja templates
 * @param name Name of the template to validate
 * @return true if template is valid, false otherwise
 */
bool cactus_context::validateModelChatTemplate(bool use_jinja, const char *name) const {
    const char * tmpl = llama_model_chat_template(model, name);
    if (tmpl == nullptr) {
      return false;
    }
    return common_chat_verify_template(tmpl, use_jinja);
}


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

lm_ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (lm_ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}


} // namespace cactus 
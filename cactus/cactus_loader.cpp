#include "ggml.h"  
#include "cactus.h"
#include "common.h"
#include "mtmd.h" 
#include <stdexcept> 

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
        LOG_ERROR("unable to load model: %s", params.model.path.c_str());
        return false;
    }
    templates = common_chat_templates_init(model, params.chat_template);
    n_ctx = llama_n_ctx(ctx);

    if (!params.mmproj.path.empty() && model != nullptr) {
        struct mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = params.mmproj_use_gpu;
        mtmd_params.n_threads = params.cpuparams.n_threads; 
        mtmd_params.verbosity = params.verbosity > 0 ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_ERROR; 
        ctx_mtmd = mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params);

        if (ctx_mtmd == nullptr) {
            LOG_ERROR("Failed to initialize mtmd_context with mmproj: %s", params.mmproj.path.c_str());
        } else {
            LOG_INFO("mtmd_context initialized successfully with mmproj: %s", params.mmproj.path.c_str());
        }
    } else if (!params.mmproj.path.empty() && model == nullptr) {
        LOG_ERROR("Cannot initialize mtmd_context because base model failed to load.");
    } else if (params.mmproj.path.empty() && !params.image.empty() && !params.no_mmproj) {
        LOG_WARNING("Image provided but no mmproj path specified. Multimodal processing will be skipped.");
    }

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
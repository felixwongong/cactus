#include "ggml.h"  
#include "cactus.h"
#include <stdexcept> 
#include "llama.h"
#include <vector>

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
        mtmd_params.verbosity = params.verbosity > 0 ? LM_GGML_LOG_LEVEL_INFO : LM_GGML_LOG_LEVEL_ERROR; 
        ctx_mtmd = mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params);

        if (ctx_mtmd == nullptr) {
            LOG_ERROR("Failed to initialize mtmd_context with mmproj: %s", params.mmproj.path.c_str());
        } else {
            LOG_INFO("mtmd_context initialized successfully with mmproj: %s", params.mmproj.path.c_str());

            if (ctx != nullptr) {
                LOG_INFO("Performing dummy multimodal decode to ensure graph allocator is prepared...");
                
                const int n_embd_model = llama_model_n_embd(model);
                const int n_dummy_img_tokens = 1;

                if (n_embd_model > 0) {
                    std::vector<float> dummy_embd_vec(n_dummy_img_tokens * n_embd_model, 0.0f);
                    
                    std::vector<llama_pos> dummy_pos_vec(n_dummy_img_tokens);
                    std::vector<int32_t> dummy_n_seq_id_vec(n_dummy_img_tokens);
                    std::vector<llama_seq_id*> dummy_seq_id_ptr_vec(n_dummy_img_tokens);
                    std::vector<llama_seq_id> dummy_seq_id_val_vec(n_dummy_img_tokens);

                    for(int i = 0; i < n_dummy_img_tokens; ++i) {
                        dummy_pos_vec[i] = i;
                        dummy_n_seq_id_vec[i] = 1;
                        dummy_seq_id_val_vec[i] = 0;
                        dummy_seq_id_ptr_vec[i] = &dummy_seq_id_val_vec[i];
                    }
                    
                    struct llama_batch dummy_batch = {
                        n_dummy_img_tokens,
                        nullptr,
                        dummy_embd_vec.data(),
                        dummy_pos_vec.data(),
                        dummy_n_seq_id_vec.data(),
                        dummy_seq_id_ptr_vec.data(),
                        nullptr
                    };

                    int decode_result = llama_decode(ctx, dummy_batch);
                    if (decode_result != 0) {
                        LOG_WARNING("Dummy multimodal decode finished with code %d. This might be expected during initial graph preparation if internal reservation occurs.", decode_result);
                    } else {
                        LOG_INFO("Dummy multimodal decode successful.");
                    }

                    llama_kv_self_clear(ctx);
                    LOG_INFO("KV cache cleared after dummy decode.");

                } else {
                    LOG_WARNING("Model embedding size is 0, skipping dummy multimodal decode.");
                }
            }
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
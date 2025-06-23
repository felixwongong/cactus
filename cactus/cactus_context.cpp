#include "cactus.h"
#include "common.h"
#include "json.hpp"

using json = nlohmann::ordered_json;

namespace cactus {

cactus_context::~cactus_context() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
    releaseMultimodal();
    releaseVocoder();
}

void cactus_context::rewind() {
    is_interrupted = false;
    is_predicting = false;
    params.antiprompt.clear();
    params.sampling.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(params.n_ctx);
    generated_token_probs.clear();
    truncated = false;
    context_full = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    embd.clear();
    next_token_uses_guide_token = true;
    guide_tokens.clear();
    mtmd_bitmap_past_hashes.clear();
    audio_tokens.clear();
    if (ctx_sampling) {
    }
}

bool cactus_context::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
    if (model) {
        ctx_sampling = common_sampler_init(model, params.sampling);
        if (ctx_sampling) {
             params.sampling.n_prev = n_ctx;
        }
    } else {
        LOG_ERROR("Cannot initialize sampling context: model is not loaded.");
        return false;
    }
    return ctx_sampling != nullptr;
}

void cactus_context::setGuideTokens(const std::vector<llama_token> &tokens) {
    guide_tokens = tokens;
}

void cactus_context::endCompletion() {
    is_predicting = false;
}

std::string cactus_context::generateResponse(const std::string &user_message, int max_tokens) {
    auto result = continueConversation(user_message, max_tokens);
    return result.text;
}

conversation_result cactus_context::continueConversation(const std::string &user_message, int max_tokens) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!model || !ctx) {
        LOG_ERROR("Model or context not initialized");
        return {"", std::chrono::milliseconds(0), std::chrono::milliseconds(0), 0};
    }

    bool is_first_message = !conversation_active || embd.empty();
    
    if (is_first_message) {
        // First message in conversation - use standard chat formatting
        json messages = json::array();
        messages.push_back({
            {"role", "user"},
            {"content", user_message}
        });
        
        std::string formatted_prompt;
        try {
            formatted_prompt = getFormattedChat(messages.dump(), "");
            last_chat_template = formatted_prompt;
        } catch (const std::exception& e) {
            LOG_ERROR("Chat template formatting failed: %s", e.what());
            formatted_prompt = "User: " + user_message + "\nAssistant: ";
            last_chat_template = formatted_prompt;
        }
        
        // Set up for generation
        params.prompt = formatted_prompt;
        params.n_predict = max_tokens;
        
        if (!initSampling()) {
            LOG_ERROR("Failed to initialize sampling");
            return {"", std::chrono::milliseconds(0), std::chrono::milliseconds(0), 0};
        }
        
        beginCompletion();
        loadPrompt();
        
        conversation_active = true;
    } else {
        // Continuing conversation - append only the new user message
        json messages = json::array();
        messages.push_back({
            {"role", "user"}, 
            {"content", user_message}
        });
        
        std::string user_part;
        try {
            user_part = getFormattedChat(messages.dump(), "");
            // Extract just the user message part by removing the assistant start
            size_t assistant_start = user_part.find("<|im_start|>assistant");
            if (assistant_start != std::string::npos) {
                user_part = user_part.substr(0, assistant_start) + "<|im_start|>assistant\n";
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Chat template formatting failed: %s", e.what());
            user_part = "\n<|im_start|>user\n" + user_message + "<|im_end|>\n<|im_start|>assistant\n";
        }
        
        // Tokenize only the new user message part
        std::vector<llama_token> new_tokens = common_tokenize(ctx, user_part, false, true);
        
        // Append to existing conversation
        embd.insert(embd.end(), new_tokens.begin(), new_tokens.end());
        
        // Set up generation parameters
        params.n_predict = max_tokens;
        n_remain = max_tokens;
        has_next_token = true;
        is_predicting = true;
        generated_text.clear();
        
        // Initialize sampling if needed
        if (!initSampling()) {
            LOG_ERROR("Failed to initialize sampling");
            return {"", std::chrono::milliseconds(0), std::chrono::milliseconds(0), 0};
        }
        
        // Accept the new tokens in the sampler
        for (auto token : new_tokens) {
            common_sampler_accept(ctx_sampling, token, false);
        }
    }
    
    // Generate the response with timing tracking
    bool first_token = true;
    std::chrono::high_resolution_clock::time_point first_token_time;
    int tokens_generated = 0;
    
    while (has_next_token && !is_interrupted) {
        auto token_output = doCompletion();
        if (token_output.tok == -1) break;
        
        if (first_token) {
            first_token_time = std::chrono::high_resolution_clock::now();
            first_token = false;
        }
        tokens_generated++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    endCompletion();
    
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto ttft = first_token ? std::chrono::milliseconds(0) : 
                std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
    
    LOG_VERBOSE("Generated response: %s (TTFT: %dms, Total: %dms, Tokens: %d)", 
                generated_text.c_str(), (int)ttft.count(), (int)total_time.count(), tokens_generated);
    
    return {generated_text, ttft, total_time, tokens_generated};
}

void cactus_context::clearConversation() {
    conversation_active = false;
    last_chat_template.clear();
    rewind();
    LOG_VERBOSE("Conversation cleared");
}

bool cactus_context::isConversationActive() const {
    return conversation_active;
}

} // namespace cactus
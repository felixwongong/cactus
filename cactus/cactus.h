#ifndef CACTUS_H
#define CACTUS_H

#include <sstream>
#include <iostream>
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-impl.h"
#include "sampling.h"
#if defined(__ANDROID__)
#include <android/log.h>
#endif

/**
 * @namespace cactus
 * @brief High-level interface for llama.cpp LLM inference
 * 
 * The cactus namespace provides a simplified API for working with llama.cpp
 * for Large Language Model inference. It handles model loading, text generation,
 * token sampling, chat formatting, and other common LLM operations.
 */
namespace cactus {

/**
 * @brief Converts a token to a formatted output string
 * 
 * Formats incomplete UTF-8 multibyte characters for output display
 * 
 * @param ctx The llama context
 * @param token The token to format
 * @return Formatted string representation of the token
 */
std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token);

/**
 * @brief Converts a range of tokens to a string
 * 
 * @param ctx The llama context
 * @param begin Iterator to the beginning of the token range
 * @param end Iterator to the end of the token range
 * @return String representation of the tokens
 */
std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end);

/**
 * @brief Converts a string to a KV cache type
 * 
 * @param s String representation of the KV cache type
 * @return The corresponding lm_ggml_type
 * @throws std::runtime_error if the cache type is unsupported
 */
lm_ggml_type kv_cache_type_from_str(const std::string & s);

/**
 * @enum stop_type
 * @brief Types of stopping criteria for text generation
 */
enum stop_type
{
    STOP_FULL,    /**< Full stop string found */
    STOP_PARTIAL, /**< Partial stop string found */
};

/**
 * @struct completion_token_output
 * @brief Structure to hold a completion token and its probabilities
 */
struct completion_token_output
{
    /**
     * @struct token_prob
     * @brief Token and probability pair
     */
    struct token_prob
    {
        llama_token tok; /**< The token */
        float prob;      /**< Probability of the token */
    };

    std::vector<token_prob> probs; /**< Probabilities of top tokens */
    llama_token tok;               /**< The selected token */
};

/**
 * @struct cactus_context
 * @brief Main context class for LLM operations
 * 
 * Manages the lifecycle of a language model, including loading, inference,
 * prompt formatting, and text generation.
 */
struct cactus_context {
    bool is_predicting = false;     /**< Whether prediction is in progress */
    bool is_interrupted = false;    /**< Whether generation has been interrupted */
    bool has_next_token = false;    /**< Whether there's another token to generate */
    std::string generated_text;     /**< The complete generated text */
    std::vector<completion_token_output> generated_token_probs; /**< Token probabilities */

    size_t num_prompt_tokens = 0;    /**< Number of tokens in the prompt */
    size_t num_tokens_predicted = 0; /**< Number of tokens predicted */
    size_t n_past = 0;               /**< Number of tokens already evaluated */
    size_t n_remain = 0;             /**< Number of tokens remaining to predict */

    std::vector<llama_token> embd;   /**< Current token embeddings */
    common_params params;            /**< Model and generation parameters */
    common_init_result llama_init;   /**< llama.cpp initialization result */

    llama_model *model = nullptr;    /**< Pointer to the llama model */
    float loading_progress = 0;      /**< Model loading progress (0-1) */
    bool is_load_interrupted = false; /**< Whether model loading was interrupted */

    llama_context *ctx = nullptr;    /**< llama context for generation */
    common_sampler *ctx_sampling = nullptr; /**< Sampler for token generation */
    common_chat_templates_ptr templates; /**< Chat templates for formatting */

    int n_ctx;                       /**< Context size */

    bool truncated = false;          /**< Whether prompt was truncated */
    bool stopped_eos = false;        /**< Stopped on EOS token */
    bool stopped_word = false;       /**< Stopped on stop word */
    bool stopped_limit = false;      /**< Stopped on token limit */
    std::string stopping_word;       /**< Word that triggered stopping */
    bool incomplete = false;         /**< Incomplete UTF-8 character */

    std::vector<common_adapter_lora_info> lora; /**< LoRA adapters */

    /**
     * @brief Destructor for cactus_context
     * 
     * Cleans up resources, including the sampling context
     */
    ~cactus_context();

    /**
     * @brief Rewinds the context to start a new generation
     * 
     * Resets internal state to prepare for a new generation task
     */
    void rewind();
    
    /**
     * @brief Initializes the sampling context
     * 
     * @return true if initialization succeeded, false otherwise
     */
    bool initSampling();
    
    /**
     * @brief Loads a language model
     * 
     * @param params_ Parameters for model loading and initialization
     * @return true if loading succeeded, false otherwise
     */
    bool loadModel(common_params &params_);
    
    /**
     * @brief Validates if a chat template exists and is valid
     * 
     * @param use_jinja Whether to use Jinja templates
     * @param name Name of the template to validate
     * @return true if template is valid, false otherwise
     */
    bool validateModelChatTemplate(bool use_jinja, const char *name) const;
    
    /**
     * @brief Formats a chat using Jinja templates
     * 
     * @param messages JSON string of chat messages
     * @param chat_template Optional custom chat template
     * @param json_schema JSON schema for validation
     * @param tools JSON string of tools available
     * @param parallel_tool_calls Whether to allow parallel tool calls
     * @param tool_choice Tool choice preference
     * @return Formatted chat parameters
     */
    common_chat_params getFormattedChatWithJinja(
      const std::string &messages,
      const std::string &chat_template,
      const std::string &json_schema,
      const std::string &tools,
      const bool &parallel_tool_calls,
      const std::string &tool_choice
    ) const;
    
    /**
     * @brief Formats a chat using standard templates
     * 
     * @param messages JSON string of chat messages
     * @param chat_template Optional custom chat template
     * @return Formatted prompt string
     */
    std::string getFormattedChat(
      const std::string &messages,
      const std::string &chat_template
    ) const;
    
    /**
     * @brief Truncates a prompt if it's too long for the context
     * 
     * @param prompt_tokens Tokens to truncate
     */
    void truncatePrompt(std::vector<llama_token> &prompt_tokens);
    
    /**
     * @brief Loads a prompt into the context
     * 
     * Tokenizes and prepares a prompt for inference
     */
    void loadPrompt();
    
    /**
     * @brief Begins the completion/generation process
     * 
     * Sets up internal state for token generation
     */
    void beginCompletion();
    
    /**
     * @brief Generates the next token
     * 
     * @return The generated token and its probabilities
     */
    completion_token_output nextToken();
    
    /**
     * @brief Searches for stopping strings in generated text
     * 
     * @param text The text to search in
     * @param last_token_size Size of the last token
     * @param type Type of stopping to check for
     * @return Position of the stop string if found, npos otherwise
     */
    size_t findStoppingStrings(const std::string &text, const size_t last_token_size, const stop_type type);
    
    /**
     * @brief Performs a single completion step
     * 
     * Generates the next token and updates generated text
     * @return The generated token and its probabilities
     */
    completion_token_output doCompletion();
    
    /**
     * @brief Generates embeddings for the current prompt
     * 
     * @param embd_params Parameters for embedding generation
     * @return Vector of embedding values
     */
    std::vector<float> getEmbedding(common_params &embd_params);
    
    /**
     * @brief Benchmarks the model performance
     * 
     * @param pp Prompt processing tokens
     * @param tg Text generation iterations
     * @param pl Parallel tokens to predict
     * @param nr Number of repetitions
     * @return JSON string with benchmark results
     */
    std::string bench(int pp, int tg, int pl, int nr);
    
    /**
     * @brief Applies LoRA adapters to the model
     * 
     * @param lora Vector of LoRA adapter information
     * @return 0 on success, negative on failure
     */
    int applyLoraAdapters(std::vector<common_adapter_lora_info> lora);
    
    /**
     * @brief Removes all LoRA adapters from the model
     */
    void removeLoraAdapters();
    
    /**
     * @brief Gets information about currently loaded LoRA adapters
     * 
     * @return Vector of LoRA adapter information
     */
    std::vector<common_adapter_lora_info> getLoadedLoraAdapters();
};

/** @var cactus_verbose
 *  @brief Flag controlling verbose logging
 */
extern bool cactus_verbose;

#if CACTUS_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
/**
 * @brief Logs verbose messages if verbose logging is enabled
 * 
 * @param MSG Message format string
 * @param ... Format arguments
 */
#define LOG_VERBOSE(MSG, ...)                                       \
    do                                                              \
    {                                                               \
        if (cactus_verbose)                                        \
        {                                                           \
            log("VERBOSE", __func__, __LINE__, MSG, ##__VA_ARGS__); \
        }                                                           \
    } while (0)
#endif

/**
 * @brief Logs error messages
 * 
 * @param MSG Message format string
 * @param ... Format arguments
 */
#define LOG_ERROR(MSG, ...) log("ERROR", __func__, __LINE__, MSG, ##__VA_ARGS__)

/**
 * @brief Logs warning messages
 * 
 * @param MSG Message format string
 * @param ... Format arguments
 */
#define LOG_WARNING(MSG, ...) log("WARNING", __func__, __LINE__, MSG, ##__VA_ARGS__)

/**
 * @brief Logs informational messages
 * 
 * @param MSG Message format string
 * @param ... Format arguments
 */
#define LOG_INFO(MSG, ...) log("INFO", __func__, __LINE__, MSG, ##__VA_ARGS__)

} // namespace cactus

#endif /* CACTUS_H */

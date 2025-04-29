#include "cactus.h"

/**
 * @file cactus-generation.cpp
 * @brief Text generation functionality for the Cactus LLM interface
 * 
 * This file contains implementations for text generation, completion,
 * and related functions like token prediction and handling.
 */

namespace cactus {

/**
 * @brief Truncates a prompt if it's too long for the context
 * 
 * @param prompt_tokens Tokens to truncate
 */
void cactus_context::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = n_ctx - params.n_keep;
    const int n_block_size = n_left / 2;
    const int erased_blocks = (prompt_tokens.size() - params.n_keep - n_block_size) / n_block_size;

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);

    new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

    LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens: %s, num_prompt_tokens: %d",
        n_ctx,
        params.n_keep,
        n_left,
        tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend()).c_str(),
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

/**
 * @brief Loads a prompt into the context
 * 
 * Tokenizes and prepares a prompt for inference
 */
void cactus_context::loadPrompt() {
    std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, params.prompt, true, true);
    num_prompt_tokens = prompt_tokens.size();

    // LOG tokens
    std::stringstream ss;
    ss << "\n" << __func__ << ": prompt_tokens = ";
    for (auto& token : prompt_tokens) {
        ss << token << " ";
    }
    LOG_INFO("%s\n", ss.str().c_str());

    if (params.n_keep < 0)
    {
        params.n_keep = (int)num_prompt_tokens;
    }
    params.n_keep = std::min(n_ctx - 4, params.n_keep);

    // if input prompt is too big, truncate like normal
    if (num_prompt_tokens >= (size_t) n_ctx)
    {
        truncatePrompt(prompt_tokens);
        num_prompt_tokens = prompt_tokens.size();

        LM_GGML_ASSERT(num_prompt_tokens < (size_t) n_ctx);
    }
    // push the prompt into the sampling context (do not apply grammar)
    for (auto & token : prompt_tokens)
    {
        common_sampler_accept(ctx_sampling, token, false);
    }

    // compare the evaluated prompt with the new prompt
    n_past = common_part(embd, prompt_tokens);

    embd = prompt_tokens;
    if (n_past == num_prompt_tokens)
    {
        // we have to evaluate at least 1 token to generate logits.
        n_past--;
    }

    // since #3228 we now have to manually manage the KV cache
    llama_kv_self_seq_rm(ctx, 0, n_past, -1);

    LOG_VERBOSE("prompt ingested, n_past: %d, cached: %s, to_eval: %s",
        n_past,
        tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past).c_str(),
        tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
    );

    has_next_token = true;
}

/**
 * @brief Begins the completion/generation process
 * 
 * Sets up internal state for token generation
 */
void cactus_context::beginCompletion() {
    // number of tokens to keep when resetting context
    n_remain = params.n_predict;
    llama_perf_context_reset(ctx);
    is_predicting = true;
}

/**
 * @brief Generates the next token
 * 
 * @return The generated token and its probabilities
 */
completion_token_output cactus_context::nextToken()
{
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)params.n_ctx)
    {
        // Shift context

        const int n_left    = n_past - params.n_keep - 1;
        const int n_discard = n_left/2;

        llama_kv_self_seq_rm (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
        llama_kv_self_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

        for (size_t i = params.n_keep + 1 + n_discard; i < embd.size(); i++)
        {
            embd[i - n_discard] = embd[i];
        }
        embd.resize(embd.size() - n_discard);

        n_past -= n_discard;

        LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens: %s",
            params.n_ctx,
            params.n_keep,
            n_left
        );
    }

    bool tg = true;
    while (n_past < embd.size())
    {
        int n_eval = (int)embd.size() - n_past;
        tg = n_eval == 1;
        if (n_eval > params.n_batch)
        {
            n_eval = params.n_batch;
        }
        if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval)))
        {
            LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd: %s",
                n_eval,
                n_past,
                params.cpuparams.n_threads,
                tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
            );
            has_next_token = false;
            return result;
        }
        n_past += n_eval;

        if(is_interrupted) {
            LOG_INFO("Decoding Interrupted");
            embd.resize(n_past);
            has_next_token = false;
            return result;
        }
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    if (params.n_predict == 0)
    {
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        return result;
    }

    {
        // out of user input, sample next token
        std::vector<llama_token_data> candidates;
        candidates.reserve(llama_vocab_n_tokens(vocab));

        result.tok = common_sampler_sample(ctx_sampling, ctx, -1);

        llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling);

        const int32_t n_probs = params.sampling.n_probs;

        // deprecated
        /*if (params.sampling.temp <= 0 && n_probs > 0)
        {
            // For llama_sample_token_greedy we need to sort candidates
            llama_sampler_init_softmax();

        }*/


        for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
        {
            result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
        }

        common_sampler_accept(ctx_sampling, result.tok, true);
        if (tg) {
            num_tokens_predicted++;
        }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    --n_remain;

    if (!embd.empty() && embd.back() == llama_vocab_eos(vocab))
    {
        // stopping_word = llama_token_to_piece(ctx, embd.back());
        has_next_token = false;
        stopped_eos = true;
        LOG_VERBOSE("eos token found", "");
        return result;
    }

    has_next_token = params.n_predict == -1 || n_remain != 0;
    return result;
}

/**
 * @brief Searches for stopping strings in generated text
 * 
 * @param text The text to search in
 * @param last_token_size Size of the last token
 * @param type Type of stopping to check for
 * @return Position of the stop string if found, npos otherwise
 */
size_t cactus_context::findStoppingStrings(const std::string &text, const size_t last_token_size,
                            const stop_type type)
{
    size_t stop_pos = std::string::npos;
    for (const std::string &word : params.antiprompt)
    {
        size_t pos;
        if (type == STOP_FULL)
        {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
            pos = text.find(word, from_pos);
        }
        else
        {
            pos = find_partial_stop_string(word, text);
        }
        if (pos != std::string::npos &&
            (stop_pos == std::string::npos || pos < stop_pos))
        {
            if (type == STOP_FULL)
            {
                stopping_word = word;
                stopped_word = true;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }
    return stop_pos;
}

/**
 * @brief Performs a single completion step
 * 
 * Generates the next token and updates generated text
 * @return The generated token and its probabilities
 */
completion_token_output cactus_context::doCompletion()
{
    const completion_token_output token_with_probs = nextToken();

    const std::string token_text = token_with_probs.tok == -1 ? "" : common_token_to_piece(ctx, token_with_probs.tok);
    generated_text += token_text;

    if (params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    // check if there is incomplete UTF-8 character at the end
    for (unsigned i = 1; i < 5 && i <= generated_text.size(); ++i) {
        unsigned char c = generated_text[generated_text.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            incomplete = i < 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            incomplete = i < 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            incomplete = i < 4;
        }
        // else 1-byte character or invalid byte
        break;
    }

    if (incomplete && !has_next_token)
    {
        has_next_token = true;
        n_remain++;
    }

    if (!has_next_token && n_remain == 0)
    {
        stopped_limit = true;
    }

    LOG_VERBOSE("next token, token: %s, token_text: %s, has_next_token: %d, n_remain: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
        common_token_to_piece(ctx, token_with_probs.tok),
        tokens_to_output_formatted_string(ctx, token_with_probs.tok).c_str(),
        has_next_token,
        n_remain,
        num_tokens_predicted,
        stopped_eos,
        stopped_word,
        stopped_limit,
        stopping_word.c_str()
    );
    return token_with_probs;
}

} // namespace cactus 
#include "cactus.h"
#include "common.h" // For common_sampler_sample, common_token_to_piece, common_tokenize, etc.
#include <algorithm> // For std::min
#include <vector>
#include <string>
#include <sstream> // For LOG_INFO
#include "llama.h" // For llama_kv_self_seq_rm, llama_decode, etc.

namespace cactus {

/**
 * @brief Truncates a prompt if it's too long for the context
 * 
 * @param prompt_tokens Tokens to truncate
 */
void cactus_context::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = n_ctx - params.n_keep;
    // Handle case where n_left might be zero or negative if n_keep >= n_ctx
    const int n_block_size = (n_left > 0) ? n_left / 2 : 0;
    // Avoid division by zero if n_block_size is 0
    const int erased_blocks = (n_block_size > 0) ? (prompt_tokens.size() - params.n_keep - n_block_size) / n_block_size : 0;

    // Ensure n_keep is not negative and within bounds
    int keep_count = std::max(0, params.n_keep);
    keep_count = std::min(keep_count, (int)prompt_tokens.size());

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + keep_count);

    // Calculate start index for the end part, ensuring it's valid
    size_t end_part_start_index = (size_t)keep_count + (size_t)std::max(0, erased_blocks * n_block_size);
    if (end_part_start_index < prompt_tokens.size()) {
        new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + end_part_start_index, prompt_tokens.end());
    }

    // Restore log call
    LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens_size: %zu",
        n_ctx,
        params.n_keep,
        n_left,
        // tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend()).c_str(), // Can be too verbose
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
    // Ensure n_keep allows for at least 4 tokens for generation if possible
    params.n_keep = std::min(n_ctx > 4 ? n_ctx - 4 : 0, params.n_keep);
    params.n_keep = std::max(0, params.n_keep); // Ensure n_keep is not negative

    // if input prompt is too big, truncate like normal
    if (num_prompt_tokens >= (size_t) n_ctx)
    {
        truncatePrompt(prompt_tokens);
        num_prompt_tokens = prompt_tokens.size();

        // This assertion might fail if n_ctx is very small and n_keep is 0.
        // Consider adjusting the logic or assertion based on minimum context requirements.
        LM_GGML_ASSERT(num_prompt_tokens < (size_t) n_ctx || n_ctx == 0);
    }
    // push the prompt into the sampling context (do not apply grammar)
    for (auto & token : prompt_tokens)
    {
        common_sampler_accept(ctx_sampling, token, false);
    }

    // compare the evaluated prompt with the new prompt
    n_past = common_part(embd, prompt_tokens);

    embd = prompt_tokens;
    // Ensure n_past doesn't exceed the current embedding size
    n_past = std::min(n_past, embd.size());

    if (n_past == num_prompt_tokens && n_past > 0) // Avoid making n_past negative if num_prompt_tokens is 0
    {
        // we have to evaluate at least 1 token to generate logits.
        n_past--;
    }

    // since #3228 we now have to manually manage the KV cache
    // Ensure n_past is valid before using it for KV cache manipulation
    if (n_past > 0) {
         llama_kv_self_seq_rm(ctx, 0, n_past, -1);
    }

    // Restore log call
    LOG_VERBOSE("prompt ingested, n_past: %d, cached_size: %zu, to_eval_size: %zu",
        n_past,
        // tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past).c_str(),
        (size_t)n_past,
        // tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
        embd.size() - n_past
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
        if (params.n_ctx <= params.n_keep + 1) {
             LOG_ERROR("Context size (%d) too small for keep (%d)", params.n_ctx, params.n_keep);
             has_next_token = false;
             return result;
        }

        const int n_left    = n_past - params.n_keep - 1;
        // Ensure n_left is positive before division
        const int n_discard = (n_left > 0) ? n_left/2 : 0;

        // Add bounds checking for sequence operations
        llama_seq_id seq_id = 0; // Assuming sequence ID 0
        int keep_start = 0;
        int keep_end = params.n_keep + 1; // Sequence is [start, end)
        int discard_start = keep_end;
        int discard_end = discard_start + n_discard;
        int add_start = discard_end;
        int add_end = n_past;
        int shift = -n_discard;

        // Ensure indices are valid before calling KV functions
        if (keep_start < keep_end && discard_start < discard_end && add_start <= add_end) {
            llama_kv_self_seq_rm (ctx, seq_id, discard_start, discard_end);
            llama_kv_self_seq_add(ctx, seq_id, add_start, add_end, shift);
        } else {
             LOG_WARNING("Invalid indices for KV cache shift operation.");
             // Potentially handle this error more gracefully
        }

        // Shift the embedding vector
        if ((size_t)n_discard < embd.size() && (size_t)(params.n_keep + 1 + n_discard) <= embd.size()) {
             std::vector<llama_token> temp_embd(embd.begin(), embd.begin() + params.n_keep + 1);
             temp_embd.insert(temp_embd.end(), embd.begin() + params.n_keep + 1 + n_discard, embd.end());
             embd = std::move(temp_embd);
        } else if ((size_t)n_discard >= embd.size()){ // If discarding more than available, keep only the prefix
            embd.resize(params.n_keep + 1);
        } // else: if start index is invalid, embd remains unchanged

        n_past -= n_discard;
        // n_past = std::max(0, n_past); // Ensure n_past doesn't become negative
        // Use explicit cast or ensure types match for std::max
        n_past = std::max<size_t>(0, n_past);

        // Restore log call
        LOG_VERBOSE("context shifted, n_ctx: %d, n_keep: %d, n_left: %d, n_discard: %d, n_past: %d",
            params.n_ctx,
            params.n_keep,
            n_left,
            n_discard,
            n_past
        );
    }

    bool tg = true;
    while ((size_t)n_past < embd.size()) // Use size_t for comparison
    {
        int n_eval = (int)embd.size() - n_past;
        tg = n_eval == 1;
        if (n_eval > params.n_batch)
        {
            n_eval = params.n_batch;
        }

        // Ensure we have tokens to evaluate
        if (n_eval <= 0) {
            LOG_WARNING("No tokens to evaluate (n_eval=%d)", n_eval);
            break; // Exit loop if nothing to evaluate
        }

        if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval)) != 0)
        {
            LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd_size: %zu",
                n_eval,
                n_past,
                params.cpuparams.n_threads,
                embd.size()
                // tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str() // Can be large
            );
            has_next_token = false;
            return result;
        }
        n_past += n_eval;

        if(is_interrupted) {
            // Restore log call
            LOG_INFO("Decoding Interrupted");
            embd.resize(n_past);
            has_next_token = false;
            return result;
        }
    }

    // Ensure model and vocab are valid
    if (!model) {
        LOG_ERROR("Model is null in nextToken");
        has_next_token = false;
        return result;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
         LOG_ERROR("Vocab is null in nextToken");
         has_next_token = false;
         return result;
    }

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
        const size_t vocab_size = llama_vocab_n_tokens(vocab);

        for (size_t i = 0; i < std::min((size_t)cur_p.size, (size_t)n_probs); ++i)
        {
            if (cur_p.data[i].id < vocab_size) { // Check bounds
                 result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
            }
        }

        common_sampler_accept(ctx_sampling, result.tok, true);
        if (tg) {
            num_tokens_predicted++;
        }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    if (n_remain > 0) {
        --n_remain;
    }

    if (!embd.empty() && embd.back() == llama_vocab_eos(vocab))
    {
        // stopping_word = llama_token_to_piece(ctx, embd.back());
        has_next_token = false;
        stopped_eos = true;
        // Restore log call
        LOG_VERBOSE("eos token found", "");
        return result;
    }

    has_next_token = params.n_predict == -1 || n_remain > 0;
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
        if (word.empty()) continue; // Skip empty stop words

        size_t pos;
        if (type == STOP_FULL)
        {
            // Ensure from_pos is not negative
            size_t from_pos = 0;
            size_t tmp_len = word.size() + last_token_size;
            if (text.size() > tmp_len) {
                from_pos = text.size() - tmp_len;
            }
            pos = text.find(word, from_pos);
        }
        else // STOP_PARTIAL
        {
            // Relying on find_partial_stop_string from utils
             pos = cactus::find_partial_stop_string(word, text);
        }

        if (pos != std::string::npos)
        {
            // If we found a stop string, update stop_pos if it's the earliest one found
            if (stop_pos == std::string::npos || pos < stop_pos)
            {
                stop_pos = pos;
                if (type == STOP_FULL)
                {
                    stopping_word = word;
                    stopped_word = true;
                    has_next_token = false;
                }
            }
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

    // Handle potential error from nextToken where tok is -1
    if (token_with_probs.tok == -1 && !has_next_token) {
        // If nextToken indicated an error or end of stream, propagate it
        return token_with_probs;
    }

    // Ensure context is valid before converting token to piece
    std::string token_text;
    if (ctx && token_with_probs.tok != -1) {
         token_text = common_token_to_piece(ctx, token_with_probs.tok);
    }
    generated_text += token_text;

    if (params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    // check if there is incomplete UTF-8 character at the end
    incomplete = false; // Assume complete unless found otherwise
    if (!generated_text.empty()) {
         unsigned char c = generated_text.back();
         int expected_continuation_bytes = 0;
         if ((c & 0xC0) == 0x80) { // Ends with continuation byte 10xxxxxx
             // Need to check previous bytes to see if it's incomplete
             int lookback = 1;
             while (lookback < 4 && lookback < generated_text.size()) {
                 unsigned char prev_c = generated_text[generated_text.size() - 1 - lookback];
                 if ((prev_c & 0xC0) == 0xC0) { // Found start byte 11xxxxxx
                     if      ((prev_c & 0xE0) == 0xC0) expected_continuation_bytes = 1;
                     else if ((prev_c & 0xF0) == 0xE0) expected_continuation_bytes = 2;
                     else if ((prev_c & 0xF8) == 0xF0) expected_continuation_bytes = 3;
                     incomplete = lookback < expected_continuation_bytes;
                     break;
                 } else if ((prev_c & 0x80) == 0x00) { // Found ASCII byte 0xxxxxxx
                     break; // Sequence broken by ASCII char
                 } // else: found another continuation byte, keep looking back
                 lookback++;
             }
         } else if ((c & 0xE0) == 0xC0) { // Starts with 110xxxxx
            incomplete = true; // Needs 1 more byte
         } else if ((c & 0xF0) == 0xE0) { // Starts with 1110xxxx
             incomplete = true; // Needs 2 more bytes
         } else if ((c & 0xF8) == 0xF0) { // Starts with 11110xxx
             incomplete = true; // Needs 3 more bytes
         }
         // else: starts with ASCII 0xxxxxxx or invalid byte, considered complete
    }

    if (incomplete && !has_next_token)
    {
        // Force generation of more tokens if the text ends mid-character
        has_next_token = true;
        if (params.n_predict != -1) { // Only increment n_remain if not infinite prediction
            n_remain++;
        }
    }

    if (!has_next_token && n_remain == 0 && params.n_predict != -1)
    {
        stopped_limit = true;
    }

    LOG_VERBOSE("next token, token_id: %d, token_text: %s, has_next_token: %d, n_remain: %d, incomplete: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
        token_with_probs.tok,
        tokens_to_output_formatted_string(ctx, token_with_probs.tok).c_str(),
        has_next_token,
        n_remain,
        incomplete,
        num_tokens_predicted,
        stopped_eos,
        stopped_word,
        stopped_limit,
        stopping_word.c_str()
    );
    return token_with_probs;
}


} // namespace cactus

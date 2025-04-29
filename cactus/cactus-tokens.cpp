#include "cactus.h"

/**
 * @file cactus-tokens.cpp
 * @brief Token handling utilities for the Cactus LLM interface
 * 
 * This file contains implementations for token manipulation, including
 * conversion between tokens and strings.
 */

namespace cactus {

/**
 * @brief Find the common prefix between two token sequences
 * 
 * @param a First token sequence
 * @param b Second token sequence
 * @return Length of the common prefix
 */
static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
}

/**
 * @brief Check if a string ends with a suffix
 * 
 * @param str String to check
 * @param suffix Suffix to look for
 * @return true if str ends with suffix, false otherwise
 */
static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

/**
 * @brief Find a partial stop string in text
 * 
 * Used for detecting if text is about to form a stopping string
 * 
 * @param stop The complete stop string to check for
 * @param text Text to check in
 * @return Position of the partial match if found, npos otherwise
 */
static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--)
        {
            if (stop[char_index] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

/**
 * @brief Formats incomplete UTF-8 multibyte characters for output
 * 
 * @param ctx The llama context
 * @param token The token to format
 * @return Formatted string representation of the token
 */
std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
}

/**
 * @brief Converts a range of tokens to a string
 * 
 * @param ctx The llama context
 * @param begin Iterator to the beginning of the token range
 * @param end Iterator to the end of the token range
 * @return String representation of the tokens
 */
std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end)
{
    std::string ret;
    for (auto it = begin; it != end; ++it)
    {
        ret += common_token_to_piece(ctx, *it);
    }
    return ret;
}

/**
 * @brief Clears a llama batch structure
 * 
 * @param batch The batch to clear
 */
static void llama_batch_clear(llama_batch *batch) {
    batch->n_tokens = 0;
}

/**
 * @brief Adds a token to a llama batch
 * 
 * @param batch The batch to add to
 * @param id Token ID
 * @param pos Token position
 * @param seq_ids Sequence IDs
 * @param logits Whether to compute logits for this token
 */
static void llama_batch_add(llama_batch *batch, llama_token id, llama_pos pos, std::vector<llama_seq_id> seq_ids, bool logits) {
    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits  [batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens += 1;
}

} // namespace cactus 
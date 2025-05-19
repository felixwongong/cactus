#include "cactus.h" 
#include "llama.h" 
#include "common.h"

#include <vector>
#include <string>
#include <stdarg.h> 
#include <stdio.h> 
#include <string.h> 
#include <sstream> 

namespace cactus {

bool cactus_verbose = false; 

void log(const char *level, const char *function, int line,
                const char *format, ...)
{
    va_list args;
    #if defined(__ANDROID__)
        // Restore Android logging path
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s:%d %s", function, line, format);
        va_start(args, format);
        android_LogPriority priority;
        if (strcmp(level, "ERROR") == 0) {
            priority = ANDROID_LOG_ERROR;
        } else if (strcmp(level, "WARNING") == 0) {
            priority = ANDROID_LOG_WARN;
        } else if (strcmp(level, "INFO") == 0) {
            priority = ANDROID_LOG_INFO;
        } else { // Treat VERBOSE or others as DEBUG
             if (!cactus_verbose && strcmp(level, "VERBOSE") == 0) {
                 va_end(args);
                 return; // Skip VERBOSE logs if not enabled
             }
            priority = ANDROID_LOG_DEBUG;
        }
        __android_log_vprint(priority, "Cactus", prefix, args);
        va_end(args);
    #else
        if (!cactus_verbose && strcmp(level, "VERBOSE") == 0) {
            return;
        }
        printf("[%s] %s:%d ", level, function, line);
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    #endif
}


/**
 * @brief Clears a llama batch structure
 * 
 * @param batch The batch to clear
 */
void llama_batch_clear(llama_batch *batch) {
    if (batch) { // Add null check
         batch->n_tokens = 0;
    }
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
void llama_batch_add(llama_batch *batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_ids, bool logits) {
     if (!batch) return; // Add null check
     // Check if the batch is full
     // Assuming batch has a capacity or max size defined somewhere, e.g., during llama_batch_init
     // if (batch->n_tokens >= batch_capacity) { /* Handle error or resize */ return; }

    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) { // Use ++i
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits  [batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens += 1;
}


/**
 * @brief Find the common prefix between two token sequences
 * 
 * @param a First token sequence
 * @param b Second token sequence
 * @return Length of the common prefix
 */
size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i = 0; // Initialize i
    size_t limit = std::min(a.size(), b.size());
    while (i < limit && a[i] == b[i]) {
        i++;
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
bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
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
size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        // Iterate backwards through the stop string
        for (int64_t i = stop.size() - 1; i >= 0; --i)
        {
            if (stop[i] == text_last_char)
            {
                // If the last char matches, check if the preceding part of stop string
                // matches the end of the text
                const std::string current_partial = stop.substr(0, i + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - current_partial.length();
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
    // Handle null context
    if (!ctx) return "<null_ctx>"; 
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (static_cast<unsigned int>(out[0]) & 0xff); // Cast to unsigned int for proper hex formatting
        std::string res(ss.str());
        // Pad with 0 if single hex digit
        if (res.length() == 1) {
            res = "0" + res;
        }
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
    // Handle null context
    if (!ctx) return "<null_ctx>"; 
    for (auto it = begin; it != end; ++it)
    {
        ret += common_token_to_piece(ctx, *it);
    }
    return ret;
}


} // namespace cactus
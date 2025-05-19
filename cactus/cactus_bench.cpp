#include "cactus.h"
#include "llama.h" 
#include <vector>
#include <string>
#include <cmath> 
#include <algorithm> 

namespace cactus {

/**
 * @brief Benchmarks the model performance
 * 
 * @param pp Prompt processing tokens
 * @param tg Text generation iterations
 * @param pl Parallel tokens to predict
 * @param nr Number of repetitions
 * @return JSON string with benchmark results
 */
std::string cactus_context::bench(int pp, int tg, int pl, int nr)
{
    if (is_predicting) {
        LOG_ERROR("cannot benchmark while predicting", "");
        return std::string("[]");
    }
    // Ensure context and model are valid
    if (!ctx || !model) {
        LOG_ERROR("Context or model not initialized for benchmarking.");
        return std::string("[]");
    }

    // Set the predicting flag to true to prevent concurrent operations
    is_predicting = true; 

    // Initialize average and standard deviation variables
    double pp_avg = 0.0;
    double tg_avg = 0.0;
    double pp_std = 0.0;
    double tg_std = 0.0;

    // Use the minimum of the prompt processing tokens and the batch size
    int batch_size = std::min(pp, (int)params.n_batch); 

    // Check if the batch size is valid
    if (batch_size <= 0) {
         LOG_ERROR("Invalid batch size for benchmark: %d (pp=%d, n_batch=%d)", batch_size, pp, params.n_batch);
         is_predicting = false;
         return std::string("[]");
    }

    llama_batch batch = llama_batch_init(
        batch_size,
        0, // No embeddings needed for benchmark
        pl // Number of sequences corresponds to parallel predictions
    );

    // Check if the batch was initialized successfully
    if (!batch.token) {
        LOG_ERROR("Failed to initialize llama_batch for benchmark.");
        is_predicting = false;
        return std::string("[]");
    }

    LOG_INFO("Starting benchmark: pp=%d, tg=%d, pl=%d, nr=%d, batch_size=%d", pp, tg, pl, nr, batch_size);

    for (int i = 0; i < nr; ++i)
    {
        if (is_interrupted) {
            LOG_INFO("Benchmark interrupted.");
            break;
        }

        // --- Prompt Processing Phase --- 

        llama_batch_clear(&batch);
        const int n_tokens_pp = pp;

        // Add tokens for prompt processing - ensure we don't exceed batch capacity
        for (int k = 0; k < n_tokens_pp; ++k) {
             if (batch.n_tokens >= batch_size) {
                 LOG_ERROR("Benchmark batch capacity (%d) exceeded during PP phase.", batch_size);
                 goto cleanup_and_exit;
             }
             // Use sequence ID 0 for the prompt
             llama_batch_add(&batch, 0, k, {0}, false); 
        }

        // Only need logits for the *last* token of the prompt to predict the next one
        if (batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = 1; 
        }

        llama_kv_self_clear(ctx); // Clear KV cache before prompt processing

        const int64_t t_pp_start = llama_time_us();
        if (llama_decode(ctx, batch) != 0)
        {
            LOG_ERROR("llama_decode() failed during prompt processing benchmark", "");
             continue;
        }
        const int64_t t_pp_end = llama_time_us();
        // Don't clear KV cache here, text generation needs it

        if (is_interrupted) {
             LOG_INFO("Benchmark interrupted after PP phase.");
             break;
        }

        // --- Text Generation Phase --- 

        const int64_t t_tg_start = llama_time_us();

        // KV cache position after prompt processing
        int n_past_tg = batch.n_tokens;

        // Text generation iterations
        for (int k = 0; k < tg; ++k) 
        {
            llama_batch_clear(&batch);

            // For each iteration, predict 'pl' tokens in parallel
            for (int j = 0; j < pl; ++j)
            {
                 if (batch.n_tokens >= batch_size) {
                      LOG_ERROR("Benchmark batch capacity (%d) exceeded during TG phase.", batch_size);
                      goto cleanup_and_exit; 
                 }
                 // Predict token at position n_past_tg + k for sequence j
                 llama_batch_add(&batch, 0, n_past_tg + k, {(llama_seq_id)j}, true); 
            }

            if (llama_decode(ctx, batch) != 0)
            {
                LOG_ERROR("llama_decode() failed during text generation benchmark", "");
                 break; 
            }
            if (is_interrupted) {
                 LOG_INFO("Benchmark interrupted during TG phase.");
                 goto cleanup_and_exit; // Use goto to ensure cleanup
            }
        }

        const int64_t t_tg_end = llama_time_us();

        // Calculate times and speeds for this repetition
        const double t_pp = (t_pp_end - t_pp_start) / 1000000.0;
        const double t_tg = (t_tg_end - t_tg_start) / 1000000.0;

        // Avoid division by zero
        const double speed_pp = (t_pp > 0) ? (double)n_tokens_pp / t_pp : 0.0;

        // Total tokens generated = pl * tg
        const double speed_tg = (t_tg > 0) ? (double)(pl * tg) / t_tg : 0.0; 

        // Accumulate stats
        pp_avg += speed_pp;
        tg_avg += speed_tg;
        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;
    }

cleanup_and_exit: // Label for cleanup
    llama_batch_free(batch); // Free the batch
    llama_kv_self_clear(ctx); // Clear KV cache finally
    is_predicting = false; // Reset prediction flag

    // Final calculations
    // Adjust count if interrupted (approximation)
    int valid_repetitions = is_interrupted ? 0 : nr; 
    if (valid_repetitions > 0) {
        pp_avg /= valid_repetitions;
        tg_avg /= valid_repetitions;

        if (valid_repetitions > 1) {
             // Prevent negative results from sqrt due to floating point inaccuracies
             double pp_var = pp_std / (valid_repetitions - 1) - pp_avg * pp_avg * valid_repetitions / (valid_repetitions - 1);
             double tg_var = tg_std / (valid_repetitions - 1) - tg_avg * tg_avg * valid_repetitions / (valid_repetitions - 1);
             pp_std = (pp_var > 0) ? sqrt(pp_var) : 0.0;
             tg_std = (tg_var > 0) ? sqrt(tg_var) : 0.0;
        } else {
            pp_std = 0.0;
            tg_std = 0.0;
        }
    } else { // If no valid repetitions completed
         pp_avg = 0.0; tg_avg = 0.0; pp_std = 0.0; tg_std = 0.0;
    }

    // Format result string
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    std::string result_str = "[\"" + std::string(model_desc) + "\"," +
                             std::to_string(llama_model_size(model)) + "," +
                             std::to_string(llama_model_n_params(model)) + "," +
                             std::to_string(pp_avg) + "," +
                             std::to_string(pp_std) + "," +
                             std::to_string(tg_avg) + "," +
                             std::to_string(tg_std) +
                             "]";
    LOG_INFO("Benchmark finished. Result: %s", result_str.c_str());
    return result_str;
}

} // namespace cactus 
#include "cactus.h"
#include <cmath> // for sqrt

/**
 * @file cactus-bench.cpp
 * @brief Benchmarking functionality for the Cactus LLM interface
 * 
 * This file contains the implementation for model performance benchmarking.
 */

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

    is_predicting = true;

    double pp_avg = 0;
    double tg_avg = 0;

    double pp_std = 0;
    double tg_std = 0;

    llama_batch batch = llama_batch_init(
        std::min(pp, params.n_ubatch), // max n_tokens is limited by n_ubatch
        0,                         // No embeddings
        1                          // Single sequence
    );

    for (int i = 0; i < nr; i++)
    {
        llama_batch_clear(&batch);

        const int n_tokens = pp;

        for (int i = 0; i < n_tokens; i++)
        {
            llama_batch_add(&batch, 0, i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = 1; // true

        llama_kv_self_clear(ctx);

        const int64_t t_pp_start = llama_time_us();
        if (llama_decode(ctx, batch) != 0)
        {
            LOG_ERROR("llama_decode() failed during prompt", "");
        }
        const int64_t t_pp_end = llama_time_us();
        llama_kv_self_clear(ctx);

        if (is_interrupted) break;

        const int64_t t_tg_start = llama_time_us();

        for (int i = 0; i < tg; i++)
        {
            llama_batch_clear(&batch);

            for (int j = 0; j < pl; j++)
            {
                llama_batch_add(&batch, 0, i, {j}, true);
            }

            if (llama_decode(ctx, batch) != 0)
            {
                LOG_ERROR("llama_decode() failed during text generation", "");
            }
            if (is_interrupted) break;
        }

        const int64_t t_tg_end = llama_time_us();

        llama_kv_self_clear(ctx);

        const double t_pp = (t_pp_end - t_pp_start) / 1000000.0;
        const double t_tg = (t_tg_end - t_tg_start) / 1000000.0;

        const double speed_pp = pp / t_pp;
        const double speed_tg = (pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;
    }

    pp_avg /= nr;
    tg_avg /= nr;

    if (nr > 1) {
        pp_std = sqrt(pp_std / (nr - 1) - pp_avg * pp_avg * nr / (nr - 1));
        tg_std = sqrt(tg_std / (nr - 1) - tg_avg * tg_avg * nr / (nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    if (is_interrupted) llama_kv_self_clear(ctx);
    is_predicting = false;

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    return std::string("[\"") + model_desc + std::string("\",") +
        std::to_string(llama_model_size(model)) + std::string(",") +
        std::to_string(llama_model_n_params(model)) + std::string(",") +
        std::to_string(pp_avg) + std::string(",") +
        std::to_string(pp_std) + std::string(",") +
        std::to_string(tg_avg) + std::string(",") +
        std::to_string(tg_std) +
        std::string("]");
}

} // namespace cactus 
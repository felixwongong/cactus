# Cactus Core C++ API Documentation

The Cactus Core C++ API provides a high-level interface for interacting with GGUF-formatted Large Language Models (LLMs) and related functionalities like Text-to-Speech (TTS), leveraging the power of `llama.cpp`. This document outlines the main components and usage of the API.

## Overview

The core of the API revolves around the `cactus::cactus_context` struct. This struct encapsulates all the necessary states and functionalities for:

*   Loading LLM and Vocoder (for TTS) models.
*   Performing text generation (completions).
*   Managing chat interactions with template support (including Jinja2).
*   Generating audio from text (Text-to-Speech).
*   Extracting embeddings from text.
*   Applying and managing LoRA adapters.
*   Benchmarking model performance.

It simplifies the complexities of `llama.cpp` by providing a more streamlined and user-friendly set of operations.

## Key Components

### `cactus::cactus_context`

This is the primary struct you will interact with. It manages the entire lifecycle of an LLM session.

**Key Member Variables:**

*   `llama_model *model`: Pointer to the loaded language model.
*   `llama_context *ctx`: The `llama.cpp` context for the language model.
*   `common_params params`: Holds various parameters for model loading, generation, sampling, etc. (defined in `common.h`).
*   `common_sampler *ctx_sampling`: Manages the token sampling process.
*   `common_chat_templates_ptr templates`: Handles chat message formatting based on predefined or custom templates.
*   `bool is_predicting`: Flag indicating if a prediction/generation is currently in progress.
*   `bool is_interrupted`: Flag to signal interruption of an ongoing generation.
*   `std::string generated_text`: Stores the accumulated text generated during a completion task.
*   `llama_model *vocoder_model`: Pointer to the loaded vocoder model (for TTS).
*   `llama_context *vocoder_ctx`: The `llama.cpp` context for the vocoder model.

**Core Functionalities (Member Functions):**

#### Model Management

*   `bool loadModel(common_params &params_)`:
    *   Loads an LLM based on the provided `common_params`.
    *   Parameters include model path, context size, GPU layers, etc.
    *   Returns `true` on success, `false` otherwise.
    *   Sets `loading_progress` (0-1) during loading.
    *   Can be interrupted by setting `is_load_interrupted` to `true`.

*   `bool loadVocoderModel(const common_params_vocoder &vocoder_params)`:
    *   Loads a vocoder model required for TTS.
    *   Takes vocoder-specific parameters.
    *   Returns `true` on success, `false` otherwise.

#### Text Completion

*   `void loadPrompt()`:
    *   Tokenizes and processes the prompt string (set in `params.prompt`) and prepares the context for generation.
    *   Handles truncation if the prompt is too long (`truncatePrompt`).

*   `void beginCompletion()`:
    *   Initializes the state for starting a new text generation sequence.
    *   Must be called after `loadPrompt()`.

*   `completion_token_output nextToken()`:
    *   Generates the next token in the sequence.
    *   Returns a `completion_token_output` struct containing the generated token and its probability distribution.

*   `completion_token_output doCompletion()`:
    *   A higher-level function that performs a single step of completion: generates the next token and appends it to `generated_text`.
    *   Handles stopping conditions (EOS, stop words, token limit).
    *   Updates internal state like `has_next_token`, `stopped_eos`, `stopped_word`, etc.

*   `void rewind()`:
    *   Resets the context's generation state (e.g., `n_past`, `generated_text`) to allow for a new generation with the same loaded model and initial prompt, or a new prompt.

#### Chat

*   `bool validateModelChatTemplate(bool use_jinja, const char *name) const`:
    *   Checks if a specified chat template is valid and available.

*   `common_chat_params getFormattedChatWithJinja(...) const`:
    *   Formats a series of chat messages (provided as a JSON string) using a Jinja2 template.
    *   Supports custom chat templates, JSON schema validation for outputs, and tool use definitions.
    *   The result (formatted prompt and other parameters) is returned in a `common_chat_params` struct. This formatted prompt can then be set in `params.prompt` before calling `loadPrompt()`.

*   `std::string getFormattedChat(...) const`:
    *   Formats chat messages using simpler, non-Jinja templating.
    *   Returns the formatted prompt string directly.

#### Text-to-Speech (TTS)

*   `bool synthesizeSpeech(const std::string& text, const std::string& output_wav_path, const std::string& speaker_id = "")`:
    *   Synthesizes speech from the input `text`.
    *   Saves the generated audio to the specified `output_wav_path`.
    *   Optionally takes a `speaker_id` for multi-speaker TTS models.
    *   Requires both the main TTS model (loaded via `loadModel`) and the vocoder model (loaded via `loadVocoderModel`) to be present.
    *   Returns `true` on success, `false` otherwise.

#### Embeddings

*   `std::vector<float> getEmbedding(common_params &embd_params)`:
    *   Generates embeddings for the prompt specified in `embd_params.prompt`.
    *   Returns a vector of floats representing the embedding.

#### LoRA (Low-Rank Adaptation)

*   `int applyLoraAdapters(std::vector<common_adapter_lora_info> lora)`:
    *   Applies one or more LoRA adapters to the loaded base model.
    *   `common_adapter_lora_info` contains the path to the LoRA file and a scaling factor.
    *   Returns `0` on success.

*   `void removeLoraAdapters()`:
    *   Removes all currently applied LoRA adapters from the model.

*   `std::vector<common_adapter_lora_info> getLoadedLoraAdapters()`:
    *   Returns information about the LoRA adapters currently applied to the model.

#### Utilities & Sampling

*   `bool initSampling()`:
    *   Initializes or re-initializes the token sampler (`ctx_sampling`) based on `params.sparams` (sampling parameters like temperature, top_k, top_p, etc.).
    *   Crucial for controlling the nature of the generated text.

*   `std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)`:
    *   Utility function to convert a single token to a string, handling UTF-8.

*   `std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end)`:
    *   Converts a sequence of tokens to a string.

*   `size_t findStoppingStrings(...)`:
    *   Helper to detect if any of the specified stop strings (from `params.sparams.stop`) are present at the end of the generated text.

*   `std::string bench(...)`:
    *   Performs benchmark tests on the loaded model for prompt processing and token generation speed.

### `common_params`

This struct (defined in `common.h`, which is part of `llama.cpp`) is extensively used to configure the behavior of `cactus_context`. It includes fields for:

*   Model path (`model`)
*   Prompt (`prompt`)
*   Context size (`n_ctx`)
*   Number of GPU layers (`n_gpu_layers`)
*   Seed (`seed`)
*   Batch size (`n_batch`)
*   Number of tokens to predict (`n_predict`)
*   Keep tokens from original prompt (`n_keep`)
*   Various LoRA parameters
*   Sampling parameters (`sparams` of type `llama_sampling_params`)
    *   Temperature, top_k, top_p, repetition penalty, stop strings, etc.
*   Embedding mode (`embedding`)

### `completion_token_output`

A struct returned by `nextToken()` and `doCompletion()`, containing:

*   `llama_token tok`: The ID of the generated token.
*   `std::vector<token_prob> probs`: A vector of `token_prob` (token ID and its probability) for the top-k most likely tokens, iflogit bias or full token probabilities are requested.

## Basic Usage Workflow

1.  **Initialization**:
    *   Create an instance of `cactus::cactus_context`.
    *   Populate a `common_params` struct with desired model path, generation settings, and sampling parameters.

2.  **Load Model**:
    *   Call `cctx.loadModel(params)`. Check the return value for success.

3.  **Set Prompt**:
    *   Set `params.prompt` to your input text.
    *   For chat, use `getFormattedChatWithJinja` or `getFormattedChat` to prepare the prompt and then assign it to `params.prompt`.

4.  **Load Prompt into Context**:
    *   Call `cctx.loadPrompt()`.

5.  **Initialize Sampling**:
    *   Call `cctx.initSampling()`.

6.  **Generate Text (Completion)**:
    *   Call `cctx.beginCompletion()`.
    *   Loop while `cctx.has_next_token` is true and `cctx.is_interrupted` is false:
        *   Call `completion_token_output token_out = cctx.doCompletion();`
        *   Process `token_out.tok` (e.g., convert to string using `tokens_to_output_formatted_string` and append to a result string).
        *   The full generated text is also available in `cctx.generated_text`.

7.  **Synthesize Speech (TTS) (Optional)**:
    *   If TTS is desired, load a vocoder model: `cctx.loadVocoderModel(vocoder_params)`.
    *   Call `cctx.synthesizeSpeech("Text to speak", "output.wav")`.

8.  **Generate Embeddings (Optional)**:
    *   Set `params.prompt` to the text you want embeddings for.
    *   Call `std::vector<float> embeddings = cctx.getEmbedding(params);`.

9.  **Cleanup**:
    *   The `cactus_context` destructor (`~cactus_context()`) handles the cleanup of allocated resources (model, context, sampler).

## Error Handling and Logging

*   Most functions in `cactus_context` return `bool` to indicate success or failure.
*   Progress and errors are often logged to `stderr` via `llama.cpp`'s internal logging.
*   The `cactus::log` function is available for custom logging within the Cactus layer.

This documentation provides a high-level overview. For detailed parameter options and advanced usage, refer to the comments within `cactus.h`, `common.h`, and the `llama.cpp` source code.

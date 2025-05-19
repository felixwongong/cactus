#define _USE_MATH_DEFINES // For M_PI on MSVC, ensure it's at the top if not already covered
#include "cactus.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "log.h" // Assuming log.h is used for LOG_ERROR, LOG_INFO etc.
#include "json.hpp" // For speaker file parsing

#include <fstream>
#include <vector>
#include <string>
#include <cmath> // For M_PI, cosf, etc.
#include <algorithm> // For std::clamp
#include <thread> // For std::thread if using threading from tts.cpp
#include <map>    // For text processing
#include <regex>  // For text processing
#include <iomanip> // For std::setprecision
#include <limits> // For std::numeric_limits

// Forward declare or include necessary headers for helper functions
// These would be moved/adapted from the original cactus/tts.cpp

namespace cactus {

// Namespace for internal TTS helper functions adapted from tts.cpp
namespace tts_internal {

    // --- Text processing maps (from tts.cpp) ---
    static const std::map<int, std::string> ones = {
        {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
        {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
        {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
        {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
    };

    static const std::map<int, std::string> tens = {
        {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
        {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
    };
    // --- End text processing maps ---

    // --- Copied/Adapted from cactus/tts.cpp (example) ---
    // Keep these static or in an anonymous namespace if they are file-local helpers
    // or make them part of tts_internal for use by cactus_context methods.

    enum outetts_version { // From tts.cpp
        OUTETTS_V0_2,
        OUTETTS_V0_3,
    };

    struct wav_header { // From tts.cpp
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t chunk_size;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_chunk_size = 16;
        uint16_t audio_format = 1; // PCM
        uint16_t num_channels = 1; // Mono
        uint32_t sample_rate; // To be filled
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample = 16;
        char data[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size;
    };

    static bool save_wav16(const std::string & fname, const std::vector<float> & audio_data, int sample_rate) {
        // Implementation from tts.cpp's save_wav16
        std::ofstream file(fname, std::ios::binary);
        if (!file) {
            LOG_ERROR("Failed to open file '%s' for writing.", fname.c_str());
            return false;
        }

        wav_header header;
        header.sample_rate = sample_rate;
        header.num_channels = 1; // Assuming mono
        header.bits_per_sample = 16;
        header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
        header.block_align = header.num_channels * (header.bits_per_sample / 8);
        header.data_size = audio_data.size() * (header.bits_per_sample / 8); // audio_data contains float samples
        header.chunk_size = 36 + header.data_size;

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const auto & sample : audio_data) {
            int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0f, -32768.0f, 32767.0f));
            file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
        }
        if (!file.good()) {
            LOG_ERROR("Error writing WAV data to '%s'.", fname.c_str());
            return false;
        }
        LOG_INFO("Audio saved to %s", fname.c_str());
        return true;
    }
    
    // --- DSP and Vocoder functions (adapted from tts.cpp) ---
    static void fill_hann_window(int length, bool periodic, float * output) {
        // Copied from tts.cpp
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5f * (1.0f - cosf((2.0f * M_PI * i) / (length + offset)));
        }
    }

    // very poor-man fft helper
    static void twiddle(float * real, float * imag, int k, int N) {
        // Copied from tts.cpp
        float angle = 2.0f * M_PI * k / N;
        *real = cosf(angle);
        *imag = sinf(angle);
    }

    static void irfft(int n, const float * inp_cplx, float * out_real) {
        // Copied from tts.cpp
        int N = n / 2 + 1;

        std::vector<float> real_input(N);
        std::vector<float> imag_input(N);
        for (int i = 0; i < N; ++i) {
            real_input[i] = inp_cplx[2 * i];
            imag_input[i] = inp_cplx[2 * i + 1];
        }

        std::vector<float> real_output(n);
        std::vector<float> imag_output(n);

        for (int k_loop = 0; k_loop < n; ++k_loop) { // Renamed k to k_loop to avoid conflict with outer scope if any
            real_output[k_loop] = 0.0f;
            imag_output[k_loop] = 0.0f;
            for (int m = 0; m < N; ++m) {
                float twiddle_real;
                float twiddle_imag;

                twiddle(&twiddle_real, &twiddle_imag, k_loop * m, n);

                real_output[k_loop] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
                imag_output[k_loop] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
            }
        }

        for (int i = 0; i < n; ++i) {
            out_real[i] = real_output[i] / N;
        }
    }

    static void fold(
        const std::vector<float> & data, // Input data (e.g., STFT frames after iFFT and windowing)
        int64_t n_out,                   // Expected total output samples (before final padding removal)
        int64_t n_win,                   // Window length used for STFT/iSTFT
        int64_t n_hop,                   // Hop length used for STFT/iSTFT
        int64_t n_pad,                   // Padding applied on each side of the frames (e.g. (n_win - n_hop)/2)
        std::vector<float> & output      // Output folded audio
    ) {
        // Copied from tts.cpp
        output.assign(n_out, 0.0f);
        int64_t current_data_ptr = 0;
        // Calculate num_frames based on data size and window length. This assumes data is concatenated frames.
        int num_frames = (n_win > 0) ? (data.size() / n_win) : 0;
        if (n_win > 0 && data.size() % n_win != 0) {
            LOG_WARNING("Fold: data size (%zu) is not a multiple of window length (%lld). Results might be incorrect.", data.size(), (long long)n_win);
        }

        for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            int64_t frame_output_start_pos = frame_idx * n_hop; // Start of this frame in the output array
            for (int i = 0; i < n_win; ++i) {
                int64_t target_idx = frame_output_start_pos + i;
                if (target_idx < n_out && current_data_ptr < (int64_t)data.size()) {
                     output[target_idx] += data[current_data_ptr];
                }
                current_data_ptr++;
            }
        }

        if (n_out > 2 * n_pad) {
             // Perform in-place trim or copy to new vector then assign
             // output.erase(output.begin() + (n_out - n_pad), output.end()); // Trim end first
             // output.erase(output.begin(), output.begin() + n_pad);       // Trim beginning
             // A safer way if n_out is the original intended *final* size before this function was called based on tts.cpp logic:
             // The tts.cpp logic `output.resize(n_out - 2 * n_pad);` implies n_out was the padded size.
             // Here, if `output` was already sized to `n_out` (final target size), then this step might be different.
             // Let's assume n_out passed to fold is the *padded* length, and we trim it.
             std::vector<float> temp_output(output.begin() + n_pad, output.begin() + (n_out - n_pad) );
             output = temp_output;
        } else if (n_out > 0) { // Only log if n_out was supposed to be positive
            LOG_WARNING("Fold: n_out (%lld) <= 2*n_pad (%lld), cannot trim padding. Output might be empty or incorrect.", (long long)n_out, (long long)(2*n_pad));
            // output.clear(); // Clearing might be too aggressive if some data is valid but not trimmable.
        }
    }
    // --- End DSP and Vocoder functions ---

    // --- embeddings_to_audio_samples IS NOW AFTER DSP HELPERS ---
    static std::vector<float> embeddings_to_audio_samples(
        const float * embeddings_ptr,       // Pointer to the flat embedding data
        int num_frames,                   // Number of frames (n_codes in tts.cpp)
        int frame_embedding_dim,          // Dimension of embedding per frame (n_embd in tts.cpp)
        llama_model * vocoder_model,      // Vocoder model (not directly used in original tts.cpp's embd_to_audio, but good to have if params are from it)
        llama_context * vocoder_ctx,      // Vocoder context (similarly, not directly used by original)
        int n_threads,
        int& out_sample_rate
    ) {
        LOG_INFO("embeddings_to_audio_samples: num_frames=%d, frame_embedding_dim=%d", num_frames, frame_embedding_dim);

        const int n_fft = 1280;
        const int n_hop = 320;
        const int n_win = n_fft;
        const int n_pad = (n_win - n_hop) / 2;
        const int64_t n_out_padded = (num_frames > 0) ? (static_cast<int64_t>(num_frames - 1) * n_hop + n_win) : 0;

        if (num_frames == 0 || frame_embedding_dim == 0) {
            LOG_WARNING("embeddings_to_audio_samples: No frames or zero embedding dimension. Returning empty audio.");
            out_sample_rate = 24000; 
            return {};
        }
        
        out_sample_rate = 24000; 

        std::vector<float> hann_window_coeffs(n_fft);
        fill_hann_window(n_fft, true, hann_window_coeffs.data()); // Now defined before this call

        int half_embedding_dim = frame_embedding_dim / 2;
        if (frame_embedding_dim % 2 != 0) {
            LOG_WARNING("Frame embedding dimension %d is not even. Vocoding might be incorrect.", frame_embedding_dim);
        }

        std::vector<float> complex_spectrum_ST(num_frames * frame_embedding_dim);

        for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            for (int k = 0; k < half_embedding_dim; ++k) {
                float mag_log = embeddings_ptr[frame_idx * frame_embedding_dim + k];
                float phi     = embeddings_ptr[frame_idx * frame_embedding_dim + k + half_embedding_dim];
                float mag = expf(mag_log);
                mag = std::min(mag, 100.0f);
                complex_spectrum_ST[frame_idx * frame_embedding_dim + 2 * k + 0] = mag * cosf(phi);
                complex_spectrum_ST[frame_idx * frame_embedding_dim + 2 * k + 1] = mag * sinf(phi);
            }
        }

        std::vector<float> all_ifft_frames_windowed(num_frames * n_fft);
        std::vector<float> hann_squared_frames(num_frames * n_fft);

        std::vector<std::thread> workers(n_threads);
        for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
            workers[thread_idx] = std::thread([&, thread_idx]() {
                for (int frame_idx = thread_idx; frame_idx < num_frames; frame_idx += n_threads) {
                    const float* current_frame_complex_spectrum = complex_spectrum_ST.data() + frame_idx * frame_embedding_dim;
                    float* current_frame_ifft_output = all_ifft_frames_windowed.data() + frame_idx * n_fft;
                    irfft(n_fft, current_frame_complex_spectrum, current_frame_ifft_output); // Now defined before this call
                    float* current_frame_hann_sq_output = hann_squared_frames.data() + frame_idx * n_fft;
                    for (int j = 0; j < n_fft; ++j) {
                        current_frame_ifft_output[j] *= hann_window_coeffs[j];
                        current_frame_hann_sq_output[j] = hann_window_coeffs[j] * hann_window_coeffs[j];
                    }
                }
            });
        }
        for (int i = 0; i < n_threads; ++i) {
            if(workers[i].joinable()) workers[i].join();
        }

        std::vector<float> audio_signal_folded;
        std::vector<float> window_energy_folded;

        fold(all_ifft_frames_windowed, n_out_padded, n_win, n_hop, n_pad, audio_signal_folded); // Now defined before this call
        fold(hann_squared_frames,      n_out_padded, n_win, n_hop, n_pad, window_energy_folded); // Now defined before this call

        if (audio_signal_folded.size() != window_energy_folded.size()) {
            LOG_ERROR("embeddings_to_audio_samples: Mismatch in folded signal size (%zu) and window energy size (%zu). Cannot normalize.", 
                      audio_signal_folded.size(), window_energy_folded.size());
            return {};
        }

        for (size_t i = 0; i < audio_signal_folded.size(); ++i) {
            if (window_energy_folded[i] > 1e-8f) { 
                audio_signal_folded[i] /= window_energy_folded[i];
            } else {
                audio_signal_folded[i] = 0.0f;
            }
        }
        LOG_INFO("Audio synthesized with %zu samples. Sample rate: %d Hz.", audio_signal_folded.size(), out_sample_rate);
        return audio_signal_folded;
    }

    // --- Text processing functions (adapted from tts.cpp) ---
    static std::string convert_less_than_thousand(int num) {
        std::string result;
        if (num == 0) { // Handle 0 explicitly if it's part of a larger number (e.g. 1000)
            return ""; // Or handle as per original logic if 0 itself should be "zero"
        }

        if (num >= 100) {
            result += ones.at(num / 100) + " hundred";
            num %= 100;
            if (num > 0) result += " "; // Add space if there are tens/ones following
        }

        if (num >= 20) {
            result += tens.at(num / 10);
            if (num % 10 > 0) {
                result += "-" + ones.at(num % 10);
            }
        } else if (num > 0) {
            result += ones.at(num);
        }
        return result;
    }

    static std::string number_to_words(const std::string & number_str) {
        try {
            size_t decimal_pos = number_str.find('.');
            std::string integer_part_str = number_str.substr(0, decimal_pos);
            if (integer_part_str.empty() && decimal_pos != std::string::npos) { // Case like ".5"
                integer_part_str = "0";
            }
            if (integer_part_str.empty() && decimal_pos == std::string::npos) { // Empty string
                return "";
            }

            long long ll_number = std::stoll(integer_part_str); // Use long long for larger numbers
            std::string result_words;

            if (ll_number == 0) {
                result_words = "zero";
            } else {
                std::string temp_res;
                if (ll_number < 0) { // Handle negative numbers
                    temp_res += "minus ";
                    ll_number = -ll_number;
                }

                if (ll_number >= 1000000000000LL) { // Trillions
                    long long trillions = ll_number / 1000000000000LL;
                    temp_res += convert_less_than_thousand(static_cast<int>(trillions)) + " trillion"; // Cast carefully
                    ll_number %= 1000000000000LL;
                    if (ll_number > 0) temp_res += " ";
                }
                if (ll_number >= 1000000000) { // Billions
                    int billions = static_cast<int>(ll_number / 1000000000);
                    temp_res += convert_less_than_thousand(billions) + " billion";
                    ll_number %= 1000000000;
                    if (ll_number > 0) temp_res += " ";
                }
                if (ll_number >= 1000000) { // Millions
                    int millions = static_cast<int>(ll_number / 1000000);
                    temp_res += convert_less_than_thousand(millions) + " million";
                    ll_number %= 1000000;
                    if (ll_number > 0) temp_res += " ";
                }
                if (ll_number >= 1000) { // Thousands
                    int thousands = static_cast<int>(ll_number / 1000);
                    temp_res += convert_less_than_thousand(thousands) + " thousand";
                    ll_number %= 1000;
                    if (ll_number > 0) temp_res += " ";
                }
                if (ll_number > 0) {
                    temp_res += convert_less_than_thousand(static_cast<int>(ll_number));
                }
                result_words = temp_res;
            }

            // Handle decimal part
            if (decimal_pos != std::string::npos) {
                result_words += " point";
                std::string decimal_part_str = number_str.substr(decimal_pos + 1);
                for (char digit : decimal_part_str) {
                    if (digit >= '0' && digit <= '9') {
                        result_words += " " + ones.at(digit - '0');
                    } else {
                        // Handle non-digit characters in decimal part if necessary, or ignore
                    }
                }
            }
            // Remove trailing space if any from parts
            if (!result_words.empty() && result_words.back() == ' ') {
                result_words.pop_back();
            }
            return result_words;
        } catch (const std::out_of_range& oor) {
            LOG_WARNING("Number out of range for stoll: %s", number_str.c_str());
            return number_str; // Return original number string if conversion fails
        } catch (const std::invalid_argument& ia) {
            LOG_WARNING("Invalid argument for stoll: %s", number_str.c_str());
            return number_str; // Return original number string if conversion fails
        }
    }

    static std::string replace_numbers_with_words(const std::string & input_text) {
        std::regex number_pattern(R"(([-+]?\d*\.\d+)|([-+]?\d+))"); // Improved regex for numbers
        std::string result;
        std::string temp_input = input_text;
        std::smatch match;

        while (std::regex_search(temp_input, match, number_pattern)) {
            result += match.prefix().str(); // Add text before the number
            result += number_to_words(match.str(0)); // Convert number and add
            temp_input = match.suffix().str(); // Continue with text after the number
        }
        result += temp_input; // Add any remaining text
        return result;
    }

    // Based on: https://github.com/edwko/OuteTTS/blob/a613e79c489d8256dd657ea9168d78de75895d82/outetts/version/v1/prompt_processor.py#L39
    std::string process_input_text(const std::string & text, outetts_version tts_version) {
        // For now I skipped text romanization as I am unsure how to handle
        // uroman and MeCab implementations in C++
        // maybe something like https://github.com/anyascii/anyascii/ could work.
        // currently only English would be supported in this function
        LOG_INFO("Original text for processing: '%s'", text.c_str());

        std::string processed_text = replace_numbers_with_words(text);
        LOG_INFO("Text after number replacement: '%s'", processed_text.c_str());

        std::transform(processed_text.begin(), processed_text.end(),
                    processed_text.begin(), ::tolower);
        LOG_INFO("Text after tolower: '%s'", processed_text.c_str());

        std::regex special_chars(R"([-_/,\.\\])"); // Original regex had double backslash for . and \ which is not needed in raw string literal for single chars.
        processed_text = std::regex_replace(processed_text, special_chars, " ");
        LOG_INFO("Text after special_chars replacement: '%s'", processed_text.c_str());

        std::regex non_alpha(R"([^a-z\s])");
        processed_text = std::regex_replace(processed_text, non_alpha, "");
        LOG_INFO("Text after non_alpha replacement: '%s'", processed_text.c_str());

        std::regex multiple_spaces(R"(\s+)");
        processed_text = std::regex_replace(processed_text, multiple_spaces, " ");
        LOG_INFO("Text after multiple_spaces collapse: '%s'", processed_text.c_str());

        // Trim leading and trailing whitespace
        processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");
        LOG_INFO("Text after trim: '%s'", processed_text.c_str());

        std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);
        LOG_INFO("Final processed text: '%s' with separator '%s'", processed_text.c_str(), separator.c_str());

        return processed_text;
    }
    // --- End text processing functions ---
    
    // --- Speaker, Version, and Prompt helpers (adapted from tts.cpp) ---
    static nlohmann::ordered_json load_speaker_embedding_json(const std::string & speaker_file_path) {
        // Adapted from tts.cpp's speaker_from_file
        LOG_INFO("Attempting to load speaker file: %s", speaker_file_path.c_str());
        std::ifstream file(speaker_file_path);
        if (!file) {
            LOG_ERROR("Failed to open speaker file: %s", speaker_file_path.c_str());
            return nlohmann::ordered_json(); // Return empty/null json on failure
        }
        try {
            return nlohmann::ordered_json::parse(file);
        } catch (const nlohmann::json::parse_error& e) {
            LOG_ERROR("Failed to parse speaker JSON from %s: %s", speaker_file_path.c_str(), e.what());
            return nlohmann::ordered_json();
        }
    }

    static outetts_version determine_tts_version(llama_model* model, const nlohmann::ordered_json& speaker_json) {
        // Adapted from tts.cpp's get_tts_version
        if (!speaker_json.is_null() && speaker_json.contains("version")) {
            std::string version = speaker_json["version"].get<std::string>();
            if (version == "0.2") {
                return OUTETTS_V0_2;
            } else if (version == "0.3") {
                return OUTETTS_V0_3;
            } else {
                LOG_WARNING("Unsupported speaker version '%s' in JSON. Checking model template.", version.c_str());
            }
        }

        if (model) {
            const char *chat_template = llama_model_chat_template(model, nullptr);
            if (chat_template && std::string(chat_template) == "outetts-0.3") {
                LOG_INFO("Determined TTS version OUTETTS_V0_3 from model chat template.");
                return OUTETTS_V0_3;
            }
        }
        LOG_INFO("Defaulting TTS version to OUTETTS_V0_2.");
        return OUTETTS_V0_2; // Default if not found or model is null
    }

    static std::string get_speaker_audio_text(const nlohmann::ordered_json& speaker_json, outetts_version tts_version) {
        // Adapted from tts.cpp's audio_text_from_speaker
        if (speaker_json.is_null() || !speaker_json.contains("words")) {
            LOG_WARNING("Speaker JSON is null or does not contain 'words' field.");
            return "<|text_start|>"; // Minimal default
        }
        std::string audio_text = "<|text_start|>";
        std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        try {
            for (const auto &word_item : speaker_json["words"]) {
                if (word_item.contains("word")) {
                    audio_text += word_item["word"].get<std::string>() + separator;
                } else {
                    LOG_WARNING("Speaker JSON 'words' item missing 'word' field.");
                }
            }
        } catch (const nlohmann::json::exception& e) {
            LOG_ERROR("Error processing speaker JSON for audio_text: %s", e.what());
            return "<|text_start|>"; // Fallback
        }
        // Remove last separator if added
        if (!audio_text.empty() && audio_text.length() > separator.length() && audio_text.rfind(separator) == audio_text.length() - separator.length()) {
             audio_text = audio_text.substr(0, audio_text.length() - separator.length());
        }
        return audio_text;
    }

    static std::string get_speaker_audio_data(const nlohmann::ordered_json& speaker_json, outetts_version tts_version) {
        // Adapted from tts.cpp's audio_data_from_speaker
        if (speaker_json.is_null() || !speaker_json.contains("words")) {
            LOG_WARNING("Speaker JSON is null or does not contain 'words' field for audio_data.");
            return "<|audio_start|>\n"; // Minimal default
        }
        std::string audio_data = "<|audio_start|>\n";
        std::string code_start_token = (tts_version == OUTETTS_V0_3) ? "" : "<|code_start|>";
        std::string code_end_token = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|code_end|>";
        try {
            for (const auto &word_item : speaker_json["words"]) {
                if (word_item.contains("word") && word_item.contains("duration") && word_item.contains("codes")) {
                    std::string word_text = word_item["word"].get<std::string>();
                    double duration = word_item["duration"].get<double>();
                    std::vector<int> codes = word_item["codes"].get<std::vector<int>>();

                    std::ostringstream word_entry;
                    word_entry << word_text << "<|t_" << std::fixed << std::setprecision(2)
                            << duration << "|>" << code_start_token;
                    for (const auto &code_val : codes) {
                        word_entry << "<|" << code_val << "|>";
                    }
                    word_entry << code_end_token << "\n";
                    audio_data += word_entry.str();
                } else {
                     LOG_WARNING("Speaker JSON 'words' item missing required fields (word, duration, codes).");
                }
            }
        } catch (const nlohmann::json::exception& e) {
            LOG_ERROR("Error processing speaker JSON for audio_data: %s", e.what());
            return "<|audio_start|>\n"; // Fallback
        }
        return audio_data;
    }

    static void prompt_add_token(std::vector<llama_token>& prompt, llama_token token) {
        prompt.push_back(token);
    }

    static void prompt_add_tokens(std::vector<llama_token>& prompt, const std::vector<llama_token>& tokens) {
        prompt.insert(prompt.end(), tokens.begin(), tokens.end());
    }

    // Note: common_tokenize is defined in common.h/common.cpp and used by cactus generally.
    // We assume it's available. It typically takes (llama_vocab*, string, bool add_bos, bool special)
    static void prompt_add_string(std::vector<llama_token>& prompt, const llama_vocab * vocab, const std::string & txt, bool add_bos, bool special_tokens) {
        if (!vocab) {
            LOG_ERROR("Cannot add string to prompt: vocab is null.");
            return;
        }
        std::vector<llama_token> tmp = common_tokenize(vocab, txt, add_bos, special_tokens);
        prompt_add_tokens(prompt, tmp);
    }

    static void prompt_initialize(std::vector<llama_token>& prompt, const llama_vocab * vocab) {
        // Adapted from tts.cpp's prompt_init
        prompt.clear();
        // The initial prompt token <|im_start|>\n seems specific to certain chat formats or the OuteTTS example setup.
        // This might need to be configurable or conditional based on the TTS model.
        // For OuteTTS, this seems to be the required start.
        prompt_add_string(prompt, vocab, "<|im_start|>\n", true, true);
    }

    static std::vector<llama_token> prepare_guide_tokens(const llama_vocab * vocab, const std::string & str, outetts_version tts_version) {
        // Adapted from tts.cpp
        if (!vocab) {
            LOG_ERROR("Cannot prepare guide tokens: vocab is null.");
            return {};
        }
        const std::string& delimiter = (tts_version == OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");
        std::vector<llama_token> result_tokens;
        size_t start_pos = 0;
        size_t end_pos = str.find(delimiter);

        // First token is always a newline, as it was not previously added (as per tts.cpp example)
        std::vector<llama_token> newline_token_vec = common_tokenize(vocab, "\n", false, true);
        if (!newline_token_vec.empty()) {
            result_tokens.push_back(newline_token_vec[0]);
        }

        while (end_pos != std::string::npos) {
            std::string current_word = str.substr(start_pos, end_pos - start_pos);
            if (!current_word.empty()) {
                std::vector<llama_token> tmp_tokens = common_tokenize(vocab, current_word, false, true);
                if (!tmp_tokens.empty()) {
                     // OuteTTS example seems to take only the first token of a word for guide tokens.
                     // This might be specific to how OuteTTS uses guide tokens.
                    result_tokens.push_back(tmp_tokens[0]);
                }
            }
            start_pos = end_pos + delimiter.length();
            end_pos = str.find(delimiter, start_pos);
        }

        // Add the last part
        std::string last_word = str.substr(start_pos);
        if (!last_word.empty()) {
            std::vector<llama_token> tmp_tokens = common_tokenize(vocab, last_word, false, true);
            if (!tmp_tokens.empty()) {
                result_tokens.push_back(tmp_tokens[0]);
            }
        }
        return result_tokens;
    }
    // --- End Speaker, Version, and Prompt helpers ---

} // namespace tts_internal


bool cactus_context::loadVocoderModel(const common_params_vocoder &vocoder_params) {
    if (this->vocoder_model != nullptr) {
        LOG_INFO("Vocoder model already loaded. Freeing existing model.");
        if (this->vocoder_ctx) llama_free(this->vocoder_ctx);
        llama_model_free(this->vocoder_model);
        this->vocoder_model = nullptr;
        this->vocoder_ctx = nullptr;
    }

    LOG_INFO("Loading vocoder model from: %s", vocoder_params.model.path.c_str());
    if (vocoder_params.model.path.empty()) {
        LOG_ERROR("Vocoder model path is empty.");
        return false;
    }

    auto mparams = llama_model_default_params();
    // Vocoders might not need extensive GPU offloading or specific settings like main LLMs.
    // These params might need to be adjusted based on the vocoder model's requirements.
    // For now, use defaults and allow common_params to override if necessary via main params.
    mparams.n_gpu_layers = params.n_gpu_layers; // Use main model's GPU layer count for now
    mparams.main_gpu     = params.main_gpu;
    mparams.split_mode   = params.split_mode;
    // Copy other relevant model params from `this->params` if needed.

    this->vocoder_model = llama_model_load_from_file(vocoder_params.model.path.c_str(), mparams);
    if (this->vocoder_model == nullptr) {
        LOG_ERROR("Failed to load vocoder model from '%s'", vocoder_params.model.path.c_str());
        return false;
    }

    auto cparams = llama_context_default_params();
    // Context size for vocoders is usually small or fixed.
    // This might need to be determined from model metadata or tts.cpp example.
    // Max codes generated by primary TTS model is params.n_predict.
    // Vocoder context and batch size should accommodate this.
    int max_codes = params.n_predict > 0 ? params.n_predict : 768; // Default from main.cpp if not set
    cparams.n_ctx   = max_codes > 0 ? (uint32_t)max_codes : 1024; // Ensure context can hold max codes, min 1024
    cparams.n_batch = max_codes > 0 ? (uint32_t)max_codes : 1024; // Batch size should be able to process all codes at once
    cparams.n_ubatch = max_codes > 0 ? (uint32_t)max_codes : 1024; // Physical batch size also needs to be sufficient
    cparams.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; // Use this to specify non-causal attention for vocoder
    cparams.embeddings = true; // Ensure vocoder context is set to output embeddings
                                            // Seed is typically handled by llama_sampling_params or model_params for specific RNG needs.
    // Copy other relevant context params from `this->params` if needed.
    
    // Check if a specific number of threads is set for the vocoder, otherwise use main params.
    // int vocoder_threads = params.vocoder.n_threads > 0 ? params.vocoder.n_threads : params.cpuparams.n_threads;
    // cparams.n_threads = vocoder_threads;
    // cparams.n_threads_batch = vocoder_threads;
    // Currently common_params_vocoder does not have n_threads, so we use main params.cpuparams
    cparams.n_threads = params.cpuparams.n_threads;
    cparams.n_threads_batch = params.cpuparams.n_threads;


    this->vocoder_ctx = llama_init_from_model(this->vocoder_model, cparams);
    if (this->vocoder_ctx == nullptr) {
        LOG_ERROR("Failed to create context for vocoder model '%s'", vocoder_params.model.path.c_str());
        llama_model_free(this->vocoder_model);
        this->vocoder_model = nullptr;
        return false;
    }

    LOG_INFO("Vocoder model '%s' loaded successfully.", vocoder_params.model.path.c_str());
    return true;
}

bool cactus_context::synthesizeSpeech(const std::string& text, const std::string& output_wav_path, const std::string& speaker_id_or_path) {
    if (!this->ctx || !this->model) {
        LOG_ERROR("Primary TTS model or context not loaded. Cannot synthesize speech.");
        return false;
    }
    if (!this->vocoder_model || !this->vocoder_ctx) {
         LOG_ERROR("Vocoder model and context must be loaded via loadVocoderModel() first.");
        return false;
    }

    nlohmann::ordered_json speaker_json;
    std::string actual_speaker_file_path = speaker_id_or_path;
    if (actual_speaker_file_path.empty()) actual_speaker_file_path = params.vocoder.speaker_file;
    if (!actual_speaker_file_path.empty()) {
        speaker_json = tts_internal::load_speaker_embedding_json(actual_speaker_file_path);
        if (speaker_json == nullptr || speaker_json.is_null()) {
            LOG_ERROR("Failed to load speaker data from: %s", actual_speaker_file_path.c_str());
        }
    }
    
    tts_internal::outetts_version tts_version = tts_internal::determine_tts_version(this->model, speaker_json);
    std::string processed_text = tts_internal::process_input_text(text, tts_version);
    const llama_vocab * vocab = llama_model_get_vocab(this->model);
    if (!vocab) { LOG_ERROR("Failed to get vocabulary from primary TTS model."); return false; }
    
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    std::vector<llama_token> prompt_tokens;
    tts_internal::prompt_initialize(prompt_tokens, vocab);
    if (!speaker_json.is_null() && speaker_json.contains("words")) {
        std::string speaker_audio_text_str = tts_internal::get_speaker_audio_text(speaker_json, tts_version);
        if (!speaker_audio_text_str.empty()) {
            tts_internal::prompt_add_string(prompt_tokens, vocab, speaker_audio_text_str, true, true);
        }
    }
    tts_internal::prompt_add_string(prompt_tokens, vocab, processed_text, true, true);
    std::vector<llama_token> guide_tokens;
    if (params.vocoder.use_guide_tokens) {
        guide_tokens = tts_internal::prepare_guide_tokens(vocab, processed_text, tts_version);
    }
    tts_internal::prompt_add_string(prompt_tokens, vocab, "\n<|audio_start|>\n", true, true);

    if (prompt_tokens.empty()) { LOG_ERROR("Failed to tokenize prompt."); llama_batch_free(batch); return false; }
    LOG_INFO("Prompt tokenized into %zu tokens.", prompt_tokens.size());
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        llama_batch_add(&batch, prompt_tokens[i], i, {0}, false);
    }
    // Ensure logits are requested for the last token of the initial prompt
    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = true;
    }
    
    llama_kv_self_clear(this->ctx);
    if (llama_decode(this->ctx, batch) != 0) {
        LOG_ERROR("llama_decode failed for initial prompt processing.");
        llama_batch_free(batch);
        return false; 
    }
    
    std::vector<float> tts_model_output_embeddings; 
    std::vector<llama_token> generated_codes;
    int n_max_codes = params.n_predict > 0 ? params.n_predict : 768;
    int eos_token = llama_vocab_eos(vocab); // Correct non-deprecated version

    if (!this->ctx_sampling) { LOG_ERROR("Sampling context not initialized."); llama_batch_free(batch); return false; }
    
    common_sampler_reset(this->ctx_sampling);

    // Logic from original tts.cpp for guide token application
    bool next_token_uses_guide_token = true; 
    // IMPORTANT: Confirm token 198 is the correct word separator for the model/tokenizer.
    // It was used in the llama.cpp example. If OuteTTS uses <|space|> and that tokenizes to something else,
    // this ID needs to be adjusted. For now, we use 198 as per the example.
    llama_token word_separator_token_id = -1; // Default to an invalid token ID
    const std::string separator_str = (tts_version == tts_internal::OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    std::vector<llama_token> sep_tokens = common_tokenize(vocab, separator_str, false, true);
    if (!sep_tokens.empty()) {
        word_separator_token_id = sep_tokens[0];
        LOG_INFO("Using token ID %d for word separator '%s'", word_separator_token_id, separator_str.c_str());
    } else {
        LOG_WARNING("Could not tokenize word separator '%s'. Guide token logic might be impaired.", separator_str.c_str());
    }


    for (int i = 0; i < n_max_codes; ++i) {
        llama_token id = common_sampler_sample(this->ctx_sampling, this->ctx, batch.n_tokens - 1);

        if (params.vocoder.use_guide_tokens && !guide_tokens.empty() && next_token_uses_guide_token &&
            !llama_vocab_is_control(vocab, id) && !llama_vocab_is_eog(vocab, id)) {
            id = guide_tokens[0];
            guide_tokens.erase(guide_tokens.begin());
        }
        
        next_token_uses_guide_token = (id == word_separator_token_id);

        common_sampler_accept(this->ctx_sampling, id, true);

        if (id == eos_token) { LOG_INFO("EOS token encountered during code generation."); break; }
        generated_codes.push_back(id);

        batch.n_tokens = 0; 
        llama_batch_add(&batch, id, prompt_tokens.size() + i, {0}, true);
        
        if (llama_decode(this->ctx, batch) != 0) {
            LOG_ERROR("llama_decode failed during code generation loop.");
            llama_batch_free(batch);
            return false; 
        }
    }

    if (!generated_codes.empty()) {
        LOG_INFO("Generated %zu raw tokens before filtering.", generated_codes.size());
        
        // Filter codes to keep only actual audio codes from the OuteTTS range
        const llama_token audio_code_min = 151672;
        const llama_token audio_code_max = 155772; // As per original tts.cpp

        std::vector<llama_token> filtered_codes;
        for (llama_token code : generated_codes) {
            if (code >= audio_code_min && code <= audio_code_max) {
                filtered_codes.push_back(code - audio_code_min); // Offset immediately
            }
        }
        generated_codes = filtered_codes; // Replace with filtered and offset codes

        LOG_INFO("Filtered and offset to %zu audio codes.", generated_codes.size());
    }

    if (!generated_codes.empty()) {
        LOG_INFO("Processing %zu audio codes with vocoder model.", generated_codes.size());
        llama_batch vocoder_batch = llama_batch_init(generated_codes.size(), 0, 1);
        for(size_t i=0; i < generated_codes.size(); ++i) {
            llama_batch_add(&vocoder_batch, generated_codes[i], i, {0}, true);
        }

        if (!this->vocoder_model || !this->vocoder_ctx) {
             LOG_ERROR("Vocoder model not loaded.");
             llama_batch_free(batch);
             llama_batch_free(vocoder_batch);
             return false;
        }
        
        llama_kv_self_clear(this->vocoder_ctx);
        if (llama_decode(this->vocoder_ctx, vocoder_batch) != 0) {
             LOG_ERROR("llama_decode failed for vocoder model processing codes.");
             llama_batch_free(batch);
             llama_batch_free(vocoder_batch);
             return false; 
        }

        const float * vocoder_embeddings_output = llama_get_embeddings(this->vocoder_ctx);
        if (!vocoder_embeddings_output) {
             LOG_ERROR("Failed to get embeddings from vocoder model.");
             llama_batch_free(batch);
             llama_batch_free(vocoder_batch);
             return false;
        }
        
        int n_embd_vocoder = llama_model_n_embd(this->vocoder_model);
        tts_model_output_embeddings.assign(vocoder_embeddings_output, vocoder_embeddings_output + generated_codes.size() * n_embd_vocoder);
        LOG_INFO("Successfully extracted %zu embedding values.", tts_model_output_embeddings.size());
        llama_batch_free(vocoder_batch);
    } else { LOG_WARNING("No codes generated or all codes were filtered out."); }

    if (tts_model_output_embeddings.empty() && !generated_codes.empty()) {
        LOG_ERROR("No embeddings generated from vocoder model, though there were codes.");
        llama_batch_free(batch);
        return false;
    }

    int vocoder_sample_rate = 24000;
    std::vector<float> audio_samples = tts_internal::embeddings_to_audio_samples(
        tts_model_output_embeddings.data(),
        generated_codes.size(), // This is now the size of filtered_codes
        llama_model_n_embd(this->vocoder_model),
        this->vocoder_model, this->vocoder_ctx,
        params.cpuparams.n_threads, vocoder_sample_rate
    );

    if (audio_samples.empty()) {
        LOG_ERROR("Failed to generate audio samples from embeddings.");
        llama_batch_free(batch);
        return false;
    }
    if (!tts_internal::save_wav16(output_wav_path, audio_samples, vocoder_sample_rate)) {
        LOG_ERROR("Failed to save audio samples to file.");
        llama_batch_free(batch);
        return false;
    }

    LOG_INFO("Speech synthesized successfully to '%s'.", output_wav_path.c_str());
    llama_batch_free(batch);
    return true;
}

} // namespace cactus 
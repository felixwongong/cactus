---
title: "Ridiculously Fast On-Device Transcription: Reviewing Parakeet CTC 1.1B with Cactus"
description: "Review of NVIDIA's Parakeet-CTC-1.1B model running locally on Mac with Cactus. Architecture breakdown, benchmarks, and transcription use cases."
keywords: ["Parakeet", "mixture of experts", "MoE", "on-device coding", "Mac inference", "NVIDIA", "Apple Silicon"]
author: "Satyajit Kumar and Henry Ndubuaku"
date: 2026-02-26
tags: ["Parakeet", "ASR", "speech-to-text", "Apple Silicon", "Transcription"]
---

# Ridiculously Fast On-Device Transcription: Reviewing Parakeet CTC 1.1B with Cactus

*By Satyajit Kumar (and Henry Ndubuaku)*

Placeholder for video

Parakeet CTC 1.1B is NVIDIA’s large-scale, non-autoregressive English speech-to-text model built on FastConformer. It uses Limited Context Attention in the encoder and a lightweight CTC projection head instead of an autoregressive decoder, which makes the decoding stage extremely efficient. Using Cactus we achieve up to **6 million tokens/second** on decode speed. This makes Parakeet CTC 1.1B a strong choice for low-latency, high-throughput transcription.

## Architecture Details

Parakeet CTC 1.1B is built on NVIDIA's FastConformer encoder and optimized for non-autoregressive ASR. At a high level:

1. **Audio front-end (mel + subsampling):** Input audio is converted to log-mel features, then an 8x depthwise-separable convolutional subsampler reduces sequence length before the encoder stack.
2. **FastConformer encoder blocks:** The encoder combines Conformer layers with **Limited Context Attention (LCA)** for local efficiency and periodic **Global Tokens (GT)** so long-range context is still preserved.
3. **CTC projection head:** Instead of an autoregressive decoder, Parakeet projects encoder states directly to token logits and uses **CTC** decoding (blank/repeat collapse), making inference highly parallel and low latency.

This architecture is why Parakeet works well for both real-time and batch transcription: most compute is in the encoder pass, and decoding stays lightweight.

## Model Architecture Diagram

```text
                                      ┌───────────────────────┐
                                      │     CTC Collapse      │
                                      │ remove blanks / merge │
                                      │ repeated labels       │
                                      └───────────┬───────────┘
                                                  ▲
                                      ┌───────────┴───────────┐
                                      │   CTC Projection Head │
                                      │  Conv1D / Linear → V  │
                                      └───────────┬───────────┘
                                                  ▲
                                      ┌───────────┴───────────┐
                                      │         Norm          │
                                      └───────────┬───────────┘
                                                  ▲
                         ┌────────────────────────⊕───────────────────────┐
                         │                        │                       │
                         │          FastConformer Encoder Stack           │
                         │                  × Num Layers                  │
                         │                                                │
                         │   ┌────────────────────────────────────────┐   │
                         │   │           FastConformer Block          │   │
                         │   │                                        │   │
                         │   │             ┌──────────────┐           │   │
                         │   │             │     FFN      │           │   │
                         │   │             │ Linear       │           │   │
                         │   │             │ SwiGLU/Act   │           │   │
                         │   │             │ Linear       │           │   │
                         │   │             └──────┬───────┘           │   │
                         │   │                    │                   │   │
                         │   │                    ⊕                   │   │
                         │   │                    │                   │   │
                         │   │             ┌──────┴───────┐           │   │
                         │   │             │  Conv Module │           │   │
                         │   │             │ Pointwise    │           │   │
                         │   │             │ Depthwise    │           │   │
                         │   │             │ Pointwise    │           │   │
                         │   │             └──────┬───────┘           │   │
                         │   │                    │                   │   │
                         │   │                    ⊕                   │   │
                         │   │                    │                   │   │
                         │   │    ┌───────────────┴──────────────┐    │   │
                         │   │    │   Limited Context Attention  │    │   │
                         │   │    │     local / sliding window   │    │   │
                         │   │    │                              │    │   │
                         │   │    │      Q        K        V     │    │   │
                         │   │    │      ↑        ↑        ↑     │    │   │
                         │   │    │ ┌────┴────────┴────────┴───┐ │    │   │
                         │   │    │ │           Linear         │ │    │   │
                         │   │    │ └─────────────┬────────────┘ │    │   │
                         │   │    └───────────────┼──────────────┘    │   │
                         │   │                    │                   │   │
                         │   │                    ⊕                   │   │
                         │   │                    │                   │   │
                         │   │    ┌───────────────┴──────────────┐    │   │
                         │   │    │            FFN               │    │   │
                         │   │    │    Linear → Act → Linear     │    │   │
                         │   │    └──────────────────────────────┘    │   │
                         │   │                                        │   │
                         │   └────────────────────────────────────────┘   │
                         └────────────────────────┬───────────────────────┘
                                                  ▲
                                      ┌───────────┴───────────┐
                                      │  Conv Subsampling /   │
                                      │ Sequence Reduction    │
                                      │  (time downsample)    │
                                      └───────────┬───────────┘
                                                  ▲
                                      ┌───────────┴───────────┐
                                      │   Mel-Spectrogram /   │
                                      │   Acoustic Features   │
                                      └───────────┬───────────┘
                                                  ▲
                                      ┌───────────┴───────────┐
                                      │     16 kHz Audio      │
                                      │      Waveform In      │
                                      └───────────────────────┘
```

*Diagram based on NVIDIA's Parakeet / FastConformer architecture description:*  
https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/

## Getting Started with Parakeet-CTC-1.1B on Cactus

### Prerequisites

- macOS with Apple Silicon and 16GB+ RAM (M1 or later recommended)
- Python 3.10+
- CMake (`brew install cmake`)
- Git

### 1. Clone and Build

```bash
git clone https://github.com/cactus-compute/cactus.git
cd cactus

# Build the Cactus engine (shared library for Python FFI)
cactus build --python
```

### 2. Download the Model

Cactus handles downloading and converting HuggingFace models to its optimized binary format with INT4/INT8 quantization, all in one command:

```bash
cactus download nvidia/parakeet-ctc-1.1b
```

### 3. Live Transcription

The fastest way to start transcribing with parakeet is by running the CLI command. It also provides an input for your cloud handoff key, for transcription models that support it (Parakeet currently does not support cloud handoff, but it will be added):

```bash
cactus transcribe nvidia/parakeet-ctc-1.1b
```

This command downloads and converts the model if needed, then starts a live transcription session from your computer's microphone (or an external mic). To transcribe a specific WAV file, pass `--file`:

```bash
cactus transcribe nvidia/parakeet-ctc-1.1b --file /path/to/your/file.wav
```

### 4. Use the Python API

For integrating Parakeet into your own applications, use the Python FFI bindings directly:

```python
import json
from cactus import cactus_init, cactus_transcribe, cactus_destroy

def on_token(token, token_id, user_data):
    print(token.decode(), end="", flush=True)

model = cactus_init("weights/parakeet-ctc-1.1b")

result = json.loads(
    cactus_transcribe(model, "/path/to/audio.wav", callback=on_token)
)

if not result["success"]:
    raise RuntimeError(result["error"])

print("\n\nFinal transcript:")
print(result["response"])
print(f"Decode speed: {result['decode_tps']:.1f} tokens/sec")

    cactus_destroy(model)
```

You can reuse the same initialized model to transcribe multiple files in a batch job and track `total_time_ms` / `decode_tps` from each response for throughput monitoring.

## Conclusion

## See Also

- [Cactus Engine API Reference](/docs/cactus_engine.md) — Full C API docs for completion, tool calling, and cloud handoff
- [Python SDK](/python/) — Python bindings used in the examples above
- [Fine-tuning Guide](/docs/finetuning.md) — Deploy your own LoRA fine-tunes to mobile
- [Hybrid Transcription](/blog/hybrid_transcription.md) — On-device/cloud hybrid speech transcription with Cactus
- [LFM2-24B-A2B](/blog/lfm2_24b_a2b.md) - Reviewing LFM2 24B MoE A2B with Cactus
- [Runtime Compatibility](/docs/compatibility.md) — Weight versioning across Cactus releases

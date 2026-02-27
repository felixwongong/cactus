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

The architecture has a lot of innovations compared to a base transformer. Firstly, the mel spectrogram features pass through a conformer, which is a transformer where every layer has a few convoolutional layers at the beginning. Another piece of innovation is the limited context attention, which intuitively makes sense because only the most immediate audio would affect the current decoding. then there's also the decoding head, which is a pointwise 1d conv on the output. It was trained with Connectionist Temporal Classification (CTC) loss, which is perfect for this decoding since it acts on empty space as well.

## Model Architecture Diagram

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

This builds, downloads (if needed), and launches an session that you can live transcribe from using your computers microphone or an external mic. to run transcribe on a specific WAV file, simply pass in the --file argument to the command.

```bash
cactus transcribe nvidia/parakeet-ctc-1.1b --file /path/to/your/file.wav
```

### 4. Use the Python API

## Conclusion

## See Also

- [Cactus Engine API Reference](/docs/cactus_engine.md) — Full C API docs for completion, tool calling, and cloud handoff
- [Python SDK](/python/) — Python bindings used in the examples above
- [Fine-tuning Guide](/docs/finetuning.md) — Deploy your own LoRA fine-tunes to mobile
- [Hybrid Transcription](/blog/hybrid_transcription.md) — On-device/cloud hybrid speech transcription with Cactus
- [LFM2-24B-A2B](/blog/lfm2_24b_a2b.md) - Reviewing LFM2 24B MoE A2B with Cactus
- [Runtime Compatibility](/docs/compatibility.md) — Weight versioning across Cactus releases
---
title: "The Sweet Spot for Mac Code Use: Reviewing LFM2 24B MoE A2B with Cactus"
description: "LFM2-24B-A2B features 24B total parameters but only activates 2B during inference. We break down the MoE architecture, GQA, gated convolutions, and show how to run it locally with Cactus."
date: "2026-02-24"
readTime: "10 min read"
authors:
  - name: "Noah Cylich"
    role: "Co-founder"
  - name: "Henry Ndubuaku"
    role: "Co-founder & CTO"
tags: ["Models", "Applications"]
youtubeId: "-duR4gh_O10"
featured: true
---

[![Watch the video](https://img.youtube.com/vi/-duR4gh_O10/maxresdefault.jpg)](https://youtu.be/-duR4gh_O10)

LFM2-24B-A2B is a really great next step to see over the LFM2-8B-A1B model. The model features 24B total parameters, but only activates a sparse subset of 2B during inference. This allows it to be competitive in inference speed to 2B dense models, while delivering far greater performance.

> "LFM2-24B-A1 excels at coding, keen to see on-device coding agents built with these."
> — Henry Ndubuaku, Cactus Co-founder & CTO

## Architecture Breakdown

Going into more depth about the model, there's really a lot to appreciate with all the architectural work LFM has accomplished, here's the breakdown:

1. **GQA**: Grouped-query attention is the industry standard choice for efficient LLMs, their choice of a group size of 4 means that the KV cache is 4x smaller than standard, baseline attention.
2. **Gated Convolution**: This is the signature design choice of the Liquid series of models and efficiently adds parameters and expressiveness without much compute cost.
3. **Efficient Vocab**: The small vocab size of 65k is actually a strength for these models, as the final matmul vocab projection is the slowest static part of every model and is extremely parameter efficient. Gemma3 270m for instance dedicated 170m params just to its vocab projection since it has a vocab of 250k tokens.
4. **MoE**: Mixture of experts is the most important choice for this model that really separates it from Liquid's prior work, it scales up parameters without sacrificing speed.

Together with Cactus, these choices enable lightning fast inference at low energy. Ultimately, despite being 24B params, only 200mb of running memory is used, while generating 25 TPS with our m4 pro chips with 48gb of ram.


## Model Architecture Diagram

```
                                    ┌───────────────┐
                                    │    Linear     │
                                    │Tied w/ Embed. │
                                    └───────┬───────┘
                                            │
                                      ┌─────┴─────┐                                 ┌───────────────────────┐
                                      │   Norm    │                                 │ Gated Short           │
                                      └─────┬─────┘                                 │ Convolution Block     │
                                            │                                       │                       │
 ┌──────────────────┐             ┌─────────⊕─────────────┐                         │             ↑         │
 │ SwiGLU Expert    │             │         │             │                         │         ┌───┴───┐     │
 │                  │             │ ┌───────┴───────────┐ │                         │         │Linear │     │
 │        ↑         │             │ │     MoE Block     │ │                         │         └───┬───┘     │
 │    ┌───┴───┐     │             │ │         ↑         │ │                         │             │         │
 │    │Linear │     │             │ │         ⊕         │ │                         │     ┌──────►⊗         │
 │    └───┬───┘     │             │ │  ↗    ↗   ↖  ↖    │ │                         │     │       ↑         │
 │        │         │             │ │ ⊗    ⊗     ⊗    ⊗ │ │                         │ ┌───┴────┐  │         │
 │        ⊗ ◄───┐   │◄------------┤ │ ╎    ╎     ╎    ╎ │ │                 ┌------►│ │ Conv1D │  │         │
 │        ↑     │   │             │ │ ↑    ↑     ↑    ↑ │ │                 ╎       │ └───┬────┘  │         │
 │        │  ┌──┴──┐│             │ │ E1...E4...E9...E64│ │                 ╎       │     │       │         │
 │        │  │SiLU ││             │ │ ↑    ↑     ↑    ↑ │ │                 ╎       │     ⊗ ◄───┐ │         │
 │        │  └──┬──┘│             │ ├─┴────┴─────┴────┴─┤ │                 ╎       │     ↑     │ │         │
 │    ┌───┴───┐ │   │             │ │   ┌───────────┐   │ │                 ╎       │     B     X C         │
 │    │Linear ├─┘   │             │ │   │  Router   │   │ │                 ╎       │     ↑     ↑ ↑         │
 │    └───┬───┘     │             │ │   └─────┬─────┘   │ │                 ╎       │  ┌──┴─────┴─┴──────┐  │
 │        ↑         │             │ │         │         │ │                 ╎       │  │     Linear      │  │
 └────────│─────────┘             │ └─────────┴─────────┘ │                 ╎       │  └────────┬────────┘  │
                                  │           │           │                 ╎       │           ↑           |
                                  └───────────│───────────┘ × Num of Layers ╎       └───────────│───────────┘
                                              │                             ╎
                                        ┌─────┴─────┐                       ╎
                                        │   Norm    │                       ╎       ┌───────────────────────┐
                                        └─────┬─────┘                       ╎       │ GQA Block             │
                                              │                             ╎       │                       │
                                    ┌─────────⊕─────────┐                   ╎       │           ↑           │
                                    │         │         │                   ╎       │       ┌───┴───┐       │
                                    │ ┌───────┴───────┐ │                   │       │       │Linear │       │
                                    │ │Sequence Block │ │                   │       │       └───┬───┘       │
                                    │ └───────┬───────┘ ├-------------------┤ OR    │           │           │
                                    │         │         │                   │       │ ┌─────────┴─────────┐ │
                                    └─────────│─────────┘                   │       │ │  Grouped Query    │ │
                                              │                             └------►│ │    Attention      │ │
                                        ┌─────┴─────┐                               │ └─┬───────┬───────┬─┘ │
                                        │   Norm    │                               │   Q       K       V   │
                                        └─────┬─────┘                               │   ↑       ↑       ↑   │
                                              │                                     │ ┌─┴──┐  ┌─┴──┐    │   │
                                        ┌─────┴─────┐                               │ │Norm│  │Norm│    │   │
                                        │ Embedding │                               │ └─┬──┘  └─┬──┘    │   │
                                        └─────┬─────┘                               │   ↑       ↑       │   │
                                              │                                     │ ┌─┴───────┴───────┴─┐ │
                                            Input                                   │ │      Linear       │ │
                                                                                    │ └─────────┬─────────┘ │
                                                                                    │           ↑           │
                                                                                    └───────────│───────────┘
```

## Getting Started with LFM2-24B on Cactus

Ready to run LFM2-24B locally on your Mac? Here's how to get up and running.

### Prerequisites

- macOS with Apple Silicon (M1 or later recommended; M4 Pro with 48GB RAM for best results)
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
cactus download LiquidAI/LFM2-24B-A2B
```

### 3. Chat Interactively

The fastest way to start chatting with the model:

```bash
cactus run LiquidAI/LFM2-24B-A2B
```

This builds, downloads (if needed), and launches an interactive chat session.

### 4. Use the Python API

For building your own applications and agents, use the Python FFI bindings directly:

```python
import json
from cactus import cactus_init, cactus_complete, cactus_reset, cactus_destroy

# Load the model
model = cactus_init("weights/lfm2-24b-a2b")

# Simple chat completion
messages = [{"role": "user", "content": "Write a Python function to sort a list"}]
response = json.loads(cactus_complete(model, messages))

print(response["response"])       # Generated text
print(f"{response['decode_tps']:.1f} tokens/sec")

# Streaming with a callback
def on_token(token, token_id, user_data):
    print(token.decode(), end="", flush=True)

cactus_complete(model, messages, callback=on_token)

# Clean up
cactus_reset(model)
cactus_destroy(model)
```

### 5. Function Calling for Agents

Cactus supports tool use out of the box, a key building block for on-device coding agents:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    }
]

messages = [{"role": "user", "content": "Calculate the factorial of 10"}]
response = json.loads(cactus_complete(model, messages, tools=tools))

if response["function_calls"]:
    print(response["function_calls"])  # Model's tool invocation
```

### Cloud Handoff

Cactus measures model confidence during generation. When the model isn't confident enough for a query, the response signals `cloud_handoff: true`, letting your agent route complex requests to a cloud API while keeping simple ones fast and local:

```python
response = json.loads(cactus_complete(model, messages, confidence_threshold=0.7))

if response["cloud_handoff"]:
    # Route to cloud API for this query
    pass
else:
    print(response["response"])
```

This hybrid local-cloud pattern is what makes on-device coding agents practical: fast local inference for the majority of tasks, with automatic escalation when needed.

## Conclusion

LFM2-24B-A2B represents a compelling sweet spot for on-device coding. The MoE architecture activates just 2B of its 24B parameters per token, delivering quality that punches well above its compute class while keeping inference fast and memory-lean at ~200MB of running RAM. Paired with Cactus's INT4 quantization, SIMD-optimized kernels, and built-in function calling, this is a model you can actually build local coding agents on top of today.

The pieces are coming together: models that are smart enough to be useful, efficient enough to run on a laptop, and runtimes that make it all accessible through a few lines of Python. Whether you're building a code assistant that works offline, a privacy-first dev tool, or just experimenting with what's possible without a cloud API, LFM2-24B with Cactus is a great place to start.

Give it a try, build something, and let us know what you think.

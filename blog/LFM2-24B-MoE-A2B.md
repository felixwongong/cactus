# The Sweet Spot for Mac Code Use: Reviewing LFM2 24B MoE A2B with Cactus

*By Noah Cylich and Henry Ndubuaku*

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

## Real-World Performance

For our use cases, this model is the sweet spot for mac use. The 8B model was too big for mobile use cases, but not quite hefty enough for mac use. This model filled that niche: it's fast enough to still be usable, while also being big enough to have some intelligence. This model has finally made edge coding truly usable (I know I'll be running this on my next plane flight). It provides real world value to the Cactus team. Here is a video of me demoing the model with Cactus in int4!

[![Watch the demo](https://img.youtube.com/vi/WG3nJW7vZLE/maxresdefault.jpg)](https://youtu.be/WG3nJW7vZLE?si=x_9x3CVgd_qtsYug)

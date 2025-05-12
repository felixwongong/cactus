henry@Henrys-MacBook-Air CactusKit % tree
.
├── Frameworks
│   └── cactus.xcframework
│       ├── info.plist
│       ├── ios-arm64
│       │   └── cactus.framework
│       │       ├── Headers
│       │       │   ├── cactus.h
│       │       │   ├── chat.h
│       │       │   ├── common.h
│       │       │   ├── ggml-alloc.h
│       │       │   ├── ggml-backend-impl.h
│       │       │   ├── ggml-backend.h
│       │       │   ├── ggml-common.h
│       │       │   ├── ggml-cpp.h
│       │       │   ├── ggml-cpu-aarch64.h
│       │       │   ├── ggml-cpu-impl.h
│       │       │   ├── ggml-cpu-quants.h
│       │       │   ├── ggml-cpu-traits.h
│       │       │   ├── ggml-cpu.h
│       │       │   ├── ggml-impl.h
│       │       │   ├── ggml-metal-impl.h
│       │       │   ├── ggml-metal.h
│       │       │   ├── ggml-opt.h
│       │       │   ├── ggml-quants.h
│       │       │   ├── ggml-threading.h
│       │       │   ├── ggml.h
│       │       │   ├── gguf.h
│       │       │   ├── json-schema-to-grammar.h
│       │       │   ├── json.hpp
│       │       │   ├── llama-adapter.h
│       │       │   ├── llama-arch.h
│       │       │   ├── llama-batch.h
│       │       │   ├── llama-chat.h
│       │       │   ├── llama-context.h
│       │       │   ├── llama-cparams.h
│       │       │   ├── llama-cpp.h
│       │       │   ├── llama-grammar.h
│       │       │   ├── llama-graph.h
│       │       │   ├── llama-hparams.h
│       │       │   ├── llama-impl.h
│       │       │   ├── llama-io.h
│       │       │   ├── llama-kv-cache.h
│       │       │   ├── llama-memory.h
│       │       │   ├── llama-mmap.h
│       │       │   ├── llama-model-loader.h
│       │       │   ├── llama-model.h
│       │       │   ├── llama-sampling.h
│       │       │   ├── llama-vocab.h
│       │       │   ├── llama.h
│       │       │   ├── log.h
│       │       │   ├── minja
│       │       │   │   ├── chat-template.hpp
│       │       │   │   └── minja.hpp
│       │       │   ├── sampling.h
│       │       │   ├── sgemm.h
│       │       │   ├── unicode-data.h
│       │       │   └── unicode.h
│       │       ├── Info.plist
│       │       ├── cactus
│       │       └── ggml-llama.metallib
│       ├── ios-arm64_x86_64-simulator
│       │   └── cactus.framework
│       │       ├── Headers
│       │       │   ├── cactus.h
│       │       │   ├── chat.h
│       │       │   ├── common.h
│       │       │   ├── ggml-alloc.h
│       │       │   ├── ggml-backend-impl.h
│       │       │   ├── ggml-backend.h
│       │       │   ├── ggml-common.h
│       │       │   ├── ggml-cpp.h
│       │       │   ├── ggml-cpu-aarch64.h
│       │       │   ├── ggml-cpu-impl.h
│       │       │   ├── ggml-cpu-quants.h
│       │       │   ├── ggml-cpu-traits.h
│       │       │   ├── ggml-cpu.h
│       │       │   ├── ggml-impl.h
│       │       │   ├── ggml-metal-impl.h
│       │       │   ├── ggml-metal.h
│       │       │   ├── ggml-opt.h
│       │       │   ├── ggml-quants.h
│       │       │   ├── ggml-threading.h
│       │       │   ├── ggml.h
│       │       │   ├── gguf.h
│       │       │   ├── json-schema-to-grammar.h
│       │       │   ├── json.hpp
│       │       │   ├── llama-adapter.h
│       │       │   ├── llama-arch.h
│       │       │   ├── llama-batch.h
│       │       │   ├── llama-chat.h
│       │       │   ├── llama-context.h
│       │       │   ├── llama-cparams.h
│       │       │   ├── llama-cpp.h
│       │       │   ├── llama-grammar.h
│       │       │   ├── llama-graph.h
│       │       │   ├── llama-hparams.h
│       │       │   ├── llama-impl.h
│       │       │   ├── llama-io.h
│       │       │   ├── llama-kv-cache.h
│       │       │   ├── llama-memory.h
│       │       │   ├── llama-mmap.h
│       │       │   ├── llama-model-loader.h
│       │       │   ├── llama-model.h
│       │       │   ├── llama-sampling.h
│       │       │   ├── llama-vocab.h
│       │       │   ├── llama.h
│       │       │   ├── log.h
│       │       │   ├── minja
│       │       │   │   ├── chat-template.hpp
│       │       │   │   └── minja.hpp
│       │       │   ├── sampling.h
│       │       │   ├── sgemm.h
│       │       │   ├── unicode-data.h
│       │       │   └── unicode.h
│       │       ├── Info.plist
│       │       ├── _CodeSignature
│       │       │   └── CodeResources
│       │       ├── cactus
│       │       └── ggml-llama-sim.metallib
│       ├── tvos-arm64
│       │   └── cactus.framework
│       │       ├── Headers
│       │       │   ├── cactus.h
│       │       │   ├── chat.h
│       │       │   ├── common.h
│       │       │   ├── ggml-alloc.h
│       │       │   ├── ggml-backend-impl.h
│       │       │   ├── ggml-backend.h
│       │       │   ├── ggml-common.h
│       │       │   ├── ggml-cpp.h
│       │       │   ├── ggml-cpu-aarch64.h
│       │       │   ├── ggml-cpu-impl.h
│       │       │   ├── ggml-cpu-quants.h
│       │       │   ├── ggml-cpu-traits.h
│       │       │   ├── ggml-cpu.h
│       │       │   ├── ggml-impl.h
│       │       │   ├── ggml-metal-impl.h
│       │       │   ├── ggml-metal.h
│       │       │   ├── ggml-opt.h
│       │       │   ├── ggml-quants.h
│       │       │   ├── ggml-threading.h
│       │       │   ├── ggml.h
│       │       │   ├── gguf.h
│       │       │   ├── json-schema-to-grammar.h
│       │       │   ├── json.hpp
│       │       │   ├── llama-adapter.h
│       │       │   ├── llama-arch.h
│       │       │   ├── llama-batch.h
│       │       │   ├── llama-chat.h
│       │       │   ├── llama-context.h
│       │       │   ├── llama-cparams.h
│       │       │   ├── llama-cpp.h
│       │       │   ├── llama-grammar.h
│       │       │   ├── llama-graph.h
│       │       │   ├── llama-hparams.h
│       │       │   ├── llama-impl.h
│       │       │   ├── llama-io.h
│       │       │   ├── llama-kv-cache.h
│       │       │   ├── llama-memory.h
│       │       │   ├── llama-mmap.h
│       │       │   ├── llama-model-loader.h
│       │       │   ├── llama-model.h
│       │       │   ├── llama-sampling.h
│       │       │   ├── llama-vocab.h
│       │       │   ├── llama.h
│       │       │   ├── log.h
│       │       │   ├── minja
│       │       │   │   ├── chat-template.hpp
│       │       │   │   └── minja.hpp
│       │       │   ├── sampling.h
│       │       │   ├── sgemm.h
│       │       │   ├── unicode-data.h
│       │       │   └── unicode.h
│       │       ├── Info.plist
│       │       ├── cactus
│       │       └── ggml-llama.metallib
│       └── tvos-arm64_x86_64-simulator
│           └── cactus.framework
│               ├── Headers
│               │   ├── cactus.h
│               │   ├── chat.h
│               │   ├── common.h
│               │   ├── ggml-alloc.h
│               │   ├── ggml-backend-impl.h
│               │   ├── ggml-backend.h
│               │   ├── ggml-common.h
│               │   ├── ggml-cpp.h
│               │   ├── ggml-cpu-aarch64.h
│               │   ├── ggml-cpu-impl.h
│               │   ├── ggml-cpu-quants.h
│               │   ├── ggml-cpu-traits.h
│               │   ├── ggml-cpu.h
│               │   ├── ggml-impl.h
│               │   ├── ggml-metal-impl.h
│               │   ├── ggml-metal.h
│               │   ├── ggml-opt.h
│               │   ├── ggml-quants.h
│               │   ├── ggml-threading.h
│               │   ├── ggml.h
│               │   ├── gguf.h
│               │   ├── json-schema-to-grammar.h
│               │   ├── json.hpp
│               │   ├── llama-adapter.h
│               │   ├── llama-arch.h
│               │   ├── llama-batch.h
│               │   ├── llama-chat.h
│               │   ├── llama-context.h
│               │   ├── llama-cparams.h
│               │   ├── llama-cpp.h
│               │   ├── llama-grammar.h
│               │   ├── llama-graph.h
│               │   ├── llama-hparams.h
│               │   ├── llama-impl.h
│               │   ├── llama-io.h
│               │   ├── llama-kv-cache.h
│               │   ├── llama-memory.h
│               │   ├── llama-mmap.h
│               │   ├── llama-model-loader.h
│               │   ├── llama-model.h
│               │   ├── llama-sampling.h
│               │   ├── llama-vocab.h
│               │   ├── llama.h
│               │   ├── log.h
│               │   ├── minja
│               │   │   ├── chat-template.hpp
│               │   │   └── minja.hpp
│               │   ├── sampling.h
│               │   ├── sgemm.h
│               │   ├── unicode-data.h
│               │   └── unicode.h
│               ├── Info.plist
│               ├── _CodeSignature
│               │   └── CodeResources
│               ├── cactus
│               └── ggml-llama-sim.metallib
├── Package.swift
├── README.md
└── Sources
    └── CactusKit
        ├── CactusError.swift
        ├── CactusModels.swift
        ├── CactusParams.swift
        └── CactusSession.swift

23 directories, 221 files
henry@Henrys-MacBook-Air CactusKit % 
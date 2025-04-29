#include "cactus.h"

/**
 * @file cactus.cpp
 * @brief Main file for the Cactus LLM interface
 * 
 * This file simply includes all the module implementations.
 */

// Include all module implementations
// Include logging first so log function is defined before it's used
#include "cactus-log.cpp"
#include "cactus-core.cpp"
#include "cactus-tokens.cpp"
#include "cactus-generation.cpp"
#include "cactus-chat.cpp"
#include "cactus-embedding.cpp"
#include "cactus-lora.cpp"
#include "cactus-bench.cpp" 
#include "cactus.h"

/**
 * @file cactus-log.cpp
 * @brief Logging functionality for the Cactus LLM interface
 * 
 * This file contains the implementation of logging functions for Cactus.
 */

namespace cactus {

// Flag controlling verbosity
bool cactus_verbose = false;

/**
 * @brief Log function for different verbosity levels
 * 
 * @param level Log level (ERROR, WARNING, INFO, VERBOSE)
 * @param function Function name where log was called
 * @param line Line number in source file
 * @param format Printf-style format string
 * @param ... Format arguments
 */
void log(const char *level, const char *function, int line,
                const char *format, ...)
{
    va_list args;
    #if defined(__ANDROID__)
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s:%d %s", function, line, format);

        va_start(args, format);
        android_LogPriority priority;
        if (strcmp(level, "ERROR") == 0) {
            priority = ANDROID_LOG_ERROR;
        } else if (strcmp(level, "WARNING") == 0) {
            priority = ANDROID_LOG_WARN;
        } else if (strcmp(level, "INFO") == 0) {
            priority = ANDROID_LOG_INFO;
        } else {
            priority = ANDROID_LOG_DEBUG;
        }
        __android_log_vprint(priority, "Cactus", prefix, args);
        va_end(args);
    #else
        printf("[%s] %s:%d ", level, function, line);
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    #endif
}

} // namespace cactus 
#include "test_utils.h"
#include "../cactus/ffi/cactus_cloud.h"
#include "../cactus/telemetry/telemetry.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

std::string make_temp_dir(const char* prefix) {
    char pattern[256] = {0};
    std::snprintf(pattern, sizeof(pattern), "/tmp/%s_XXXXXX", prefix);
    return std::string(mkdtemp(pattern));
}

void cleanup_cache_dir(const std::string& cache_dir) {
    if (cache_dir.empty()) return;
    std::remove((cache_dir + "/cloud_api_key").c_str());
    rmdir(cache_dir.c_str());
}

std::string read_first_line(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return {};
    std::string line;
    if (std::getline(in, line)) return line;
    return {};
}

bool test_cloud_key_cache_roundtrip() {
    const std::string cache_dir = make_temp_dir("cactus_cloud_key_cache");
    const std::string key_file = cache_dir + "/cloud_api_key";
    const std::string expected_key = "env-test-key-123";

    cactus::telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    std::remove(key_file.c_str());
    unsetenv("CACTUS_CLOUD_KEY");

    bool ok = true;
    ok = ok && cactus::ffi::resolve_cloud_api_key(nullptr).empty();

    setenv("CACTUS_CLOUD_KEY", expected_key.c_str(), 1);
    ok = ok && (cactus::ffi::resolve_cloud_api_key(nullptr) == expected_key);
    ok = ok && (read_first_line(key_file) == expected_key);

    unsetenv("CACTUS_CLOUD_KEY");
    ok = ok && (cactus::ffi::resolve_cloud_api_key(nullptr) == expected_key);

    cleanup_cache_dir(cache_dir);
    return ok;
}

enum class CloudEndpointTestResult {
    Passed,
    Failed,
    Skipped,
};

CloudEndpointTestResult test_cloud_handoff_key_resolution_unified() {
    const std::string cache_dir = make_temp_dir("cactus_cloud_handoff_key");
    const std::string key_file = cache_dir + "/cloud_api_key";
    const std::string expected_key = "env-handoff-key-456";

    cactus::telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    std::remove(key_file.c_str());
    unsetenv("CACTUS_CLOUD_KEY");

    cactus::ffi::CloudCompletionRequest request;
    cactus::engine::ChatMessage msg;
    msg.role = "user";
    msg.content = "ping";
    request.messages.push_back(msg);

    cactus::ffi::CloudCompletionResult without_key = cactus::ffi::cloud_complete_request(request, 1L);
    if (without_key.error == "curl_not_enabled") {
        cleanup_cache_dir(cache_dir);
        return CloudEndpointTestResult::Skipped;
    }
    if (without_key.error != "missing_api_key") {
        cleanup_cache_dir(cache_dir);
        return CloudEndpointTestResult::Failed;
    }

    setenv("CACTUS_CLOUD_KEY", expected_key.c_str(), 1);
    cactus::ffi::CloudCompletionResult with_env = cactus::ffi::cloud_complete_request(request, 1L);
    unsetenv("CACTUS_CLOUD_KEY");

    bool env_path_used = with_env.error != "missing_api_key";
    bool cache_written = read_first_line(key_file) == expected_key;

    cactus::ffi::CloudCompletionResult with_cache = cactus::ffi::cloud_complete_request(request, 1L);
    bool cache_path_used = with_cache.error != "missing_api_key";

    cleanup_cache_dir(cache_dir);

    if (env_path_used && cache_written && cache_path_used) {
        return CloudEndpointTestResult::Passed;
    }
    return CloudEndpointTestResult::Failed;
}

} // namespace

int main() {
    TestUtils::TestRunner runner("Cloud Tests");
    runner.run_test("Cloud key cache roundtrip", test_cloud_key_cache_roundtrip());

    CloudEndpointTestResult endpoint_result = test_cloud_handoff_key_resolution_unified();
    if (endpoint_result == CloudEndpointTestResult::Skipped) {
        runner.log_skip("Cloud handoff key source", "libcactus built without curl support");
    } else {
        runner.run_test("Cloud handoff key source", endpoint_result == CloudEndpointTestResult::Passed);
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "aclrtlaunch_direct_matmul_fp16_k0.h"
#include "aclrtlaunch_direct_matmul_fp16_k1.h"
#include "aclrtlaunch_direct_matmul_fp16_k20.h"
#include "aclrtlaunch_direct_matmul_fp16_k21.h"
#include "aclrtlaunch_direct_matmul_fp16_k30.h"
#include "aclrtlaunch_direct_matmul_fp16_k31.h"
#include "aclrtlaunch_direct_matmul_fp16_k201.h"
#include "aclrtlaunch_direct_matmul_fp16_k10201.h"
#include "aclrtlaunch_direct_matmul_bf16_k1.h"
#include "aclrtlaunch_direct_matmul_bf16_k0.h"
#include "aclrtlaunch_direct_matmul_bf16_k20.h"
#include "aclrtlaunch_direct_matmul_bf16_k21.h"
#include "aclrtlaunch_direct_matmul_bf16_k30.h"
#include "aclrtlaunch_direct_matmul_bf16_k31.h"
#include "aclrtlaunch_direct_matmul_bf16_k201.h"
#include "aclrtlaunch_direct_matmul_bf16_k10201.h"
#include "aclrtlaunch_direct_matmul_fp32_k1.h"
#include "aclrtlaunch_direct_matmul_fp32_k21.h"
#include "aclrtlaunch_direct_matmul_fp32_k31.h"
#include "aclrtlaunch_direct_matmul_fp32_k101.h"
#include "aclrtlaunch_direct_matmul_fp32_k201.h"
#include "aclrtlaunch_direct_matmul_fp32_k10201.h"
#include "aclrtlaunch_direct_matmul_fp32_k20201.h"
#include "mat_mul_v3_tiling_data.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string manifest;
    int device = 0;
    int warmup = 1;
    int repeat = 1;
    int samples = 3;
};

struct Candidate {
    std::string workloadId;
    std::string rank;
    std::string role;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    std::string dtype;
    bool transA = false;
    bool transB = false;
    uint32_t usedCores = 0;
    uint32_t suffix = 0;
    size_t workspaceBytes = 0;
    std::string tilingPath;
    std::string tilingSha256;
    std::string tilingFnv1a64;
    std::string scheduleSha256;
    bool reserve = false;
    uint32_t requiredSuccessfulTilings = 0;
};

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { Allocate(count); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { Release(); }

    void Allocate(size_t count)
    {
        Release();
        if (count == 0) return;
        const aclError rc = aclrtMalloc(&ptr, count, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS) {
            throw std::runtime_error(
                "aclrtMalloc failed bytes=" + std::to_string(count) +
                " rc=" + std::to_string(rc));
        }
        bytes = count;
    }

    void Release()
    {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
};

void Check(aclError rc, const std::string &operation)
{
    if (rc != ACL_SUCCESS) {
        throw std::runtime_error(
            operation + " failed rc=" + std::to_string(rc));
    }
}

double ElapsedMs(const Clock::time_point &started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::string JsonEscape(const std::string &value)
{
    std::ostringstream output;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(ch)
                           << std::dec;
                } else {
                    output << ch;
                }
        }
    }
    return output.str();
}

std::vector<std::string> SplitCsv(const std::string &line)
{
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (size_t index = 0; index < line.size(); ++index) {
        const char ch = line[index];
        if (ch == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                current.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    fields.push_back(current);
    return fields;
}

bool Truthy(const std::string &value)
{
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string Field(
    const std::vector<std::string> &fields,
    const std::unordered_map<std::string, size_t> &columns,
    const std::string &name)
{
    const auto found = columns.find(name);
    if (found == columns.end() || found->second >= fields.size()) {
        throw std::runtime_error("manifest missing field " + name);
    }
    return fields[found->second];
}

std::vector<Candidate> LoadManifest(const std::string &path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open direct manifest " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("direct manifest is empty");
    std::unordered_map<std::string, size_t> columns;
    const auto header = SplitCsv(line);
    for (size_t index = 0; index < header.size(); ++index) columns[header[index]] = index;
    std::vector<Candidate> result;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = SplitCsv(line);
        Candidate row;
        row.workloadId = Field(fields, columns, "workload_id");
        row.rank = Field(fields, columns, "rank");
        row.role = Field(fields, columns, "candidate_role");
        row.m = std::stoll(Field(fields, columns, "m"));
        row.n = std::stoll(Field(fields, columns, "n"));
        row.k = std::stoll(Field(fields, columns, "k"));
        row.dtype = Field(fields, columns, "dtype");
        row.transA = Truthy(Field(fields, columns, "trans_a"));
        row.transB = Truthy(Field(fields, columns, "trans_b"));
        row.usedCores = static_cast<uint32_t>(
            std::stoul(Field(fields, columns, "used_core_num")));
        row.suffix = static_cast<uint32_t>(
            std::stoul(Field(fields, columns, "kernel_suffix")));
        row.workspaceBytes = static_cast<size_t>(
            std::stoull(Field(fields, columns, "workspace_bytes")));
        row.tilingPath = Field(fields, columns, "tiling_path");
        row.tilingSha256 = Field(fields, columns, "tiling_sha256");
        row.tilingFnv1a64 = Field(fields, columns, "tiling_fnv1a64");
        row.scheduleSha256 = Field(fields, columns, "model_schedule_sha256");
        row.reserve = Truthy(Field(fields, columns, "is_reserve"));
        row.requiredSuccessfulTilings = static_cast<uint32_t>(
            std::stoul(Field(fields, columns, "required_successful_tilings")));
        if (row.workloadId.empty() || row.rank.empty() || row.role != "searched" ||
            row.m <= 0 || row.n <= 0 || row.k <= 0 || row.usedCores == 0 ||
            row.workspaceBytes < 20U * 1024U * 1024U ||
            row.requiredSuccessfulTilings == 0) {
            throw std::runtime_error("direct manifest contains an invalid candidate");
        }
        result.push_back(std::move(row));
    }
    if (result.empty()) throw std::runtime_error("direct manifest has no candidates");
    return result;
}

size_t ElementBytes(const std::string &dtype)
{
    if (dtype == "fp16" || dtype == "bf16") return 2;
    if (dtype == "fp32") return 4;
    throw std::runtime_error("unsupported dtype " + dtype);
}

size_t MatrixBytes(int64_t first, int64_t second, size_t width)
{
    const auto elements = static_cast<uint64_t>(first) * static_cast<uint64_t>(second);
    if (first <= 0 || second <= 0 ||
        elements > std::numeric_limits<size_t>::max() / width) {
        throw std::runtime_error("matrix allocation overflow");
    }
    return static_cast<size_t>(elements) * width;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + bias) >> 16U);
}

void StoreScalar(uint8_t *destination, const std::string &dtype, float value)
{
    if (dtype == "fp16") {
        const aclFloat16 encoded = aclFloatToFloat16(value);
        std::memcpy(destination, &encoded, sizeof(encoded));
    } else if (dtype == "bf16") {
        const uint16_t encoded = FloatToBfloat16(value);
        std::memcpy(destination, &encoded, sizeof(encoded));
    } else if (dtype == "fp32") {
        std::memcpy(destination, &value, sizeof(value));
    } else {
        throw std::runtime_error("unsupported dtype " + dtype);
    }
}

int PatternSign(int64_t index, int axis)
{
    static constexpr std::array<int64_t, 3> moduli{251, 257, 263};
    static constexpr std::array<int64_t, 3> multipliers{73, 97, 101};
    static constexpr std::array<int64_t, 3> offsets{19, 31, 47};
    const int64_t mixed =
        (index * multipliers.at(static_cast<size_t>(axis)) +
         offsets.at(static_cast<size_t>(axis))) %
        moduli.at(static_cast<size_t>(axis));
    return mixed < moduli.at(static_cast<size_t>(axis)) / 2 ? -1 : 1;
}

std::vector<uint8_t> PatternVector(
    int64_t count, int axis, int multiplier, float magnitude,
    const std::string &dtype)
{
    const size_t width = ElementBytes(dtype);
    std::vector<uint8_t> output(static_cast<size_t>(count) * width);
    for (int64_t index = 0; index < count; ++index) {
        StoreScalar(
            output.data() + static_cast<size_t>(index) * width,
            dtype,
            static_cast<float>(multiplier * PatternSign(index, axis)) * magnitude);
    }
    return output;
}

std::vector<uint8_t> ConstantVector(
    int64_t count, float value, const std::string &dtype)
{
    const size_t width = ElementBytes(dtype);
    std::vector<uint8_t> output(static_cast<size_t>(count) * width);
    for (int64_t index = 0; index < count; ++index) {
        StoreScalar(
            output.data() + static_cast<size_t>(index) * width,
            dtype, value);
    }
    return output;
}

int64_t KChecksum(int64_t k)
{
    // Keep the accumulated result small and exactly representable in all
    // output dtypes.  This avoids fp16 overflow/rounding hiding a missing K
    // tile while retaining a non-zero deterministic checksum for every
    // workload in the fixed campaign.
    int64_t result = 0;
    for (int64_t index = 0; index < k; ++index) {
        result += index == 0 ? 7 : PatternSign(index, 2);
    }
    if (result == 0) {
        throw std::runtime_error("structured K checksum is zero");
    }
    return result;
}

void CopyRows(
    std::vector<uint8_t> &destination,
    int64_t rows,
    const std::vector<uint8_t> &positive,
    const std::vector<uint8_t> &negative,
    int signAxis)
{
    const size_t rowBytes = positive.size();
    if (negative.size() != rowBytes ||
        destination.size() != static_cast<size_t>(rows) * rowBytes) {
        throw std::runtime_error("structured input size mismatch");
    }
    for (int64_t row = 0; row < rows; ++row) {
        const auto &source = PatternSign(row, signAxis) > 0 ? positive : negative;
        std::memcpy(
            destination.data() + static_cast<size_t>(row) * rowBytes,
            source.data(), rowBytes);
    }
}

void StructuredInputs(
    std::vector<uint8_t> &a,
    std::vector<uint8_t> &b,
    const Candidate &workload)
{
    const auto kPositive = PatternVector(workload.k, 2, 1, 1.0F, workload.dtype);
    const auto kNegative = PatternVector(workload.k, 2, -1, 1.0F, workload.dtype);
    StoreScalar(kPositive.data(), workload.dtype, 7.0F);
    StoreScalar(kNegative.data(), workload.dtype, -7.0F);
    const auto kOnes = ConstantVector(workload.k, 1.0F, workload.dtype);
    const auto kMinusOnes = ConstantVector(workload.k, -1.0F, workload.dtype);
    const auto mPositive = PatternVector(workload.m, 0, 1, 1.0F, workload.dtype);
    const auto mNegative = PatternVector(workload.m, 0, -1, 1.0F, workload.dtype);
    const auto nPositive = PatternVector(workload.n, 1, 1, 1.0F, workload.dtype);
    const auto nNegative = PatternVector(workload.n, 1, -1, 1.0F, workload.dtype);
    const auto nSeven = PatternVector(workload.n, 1, 1, 7.0F, workload.dtype);
    if (workload.transA) {
        CopyRows(a, workload.k, mPositive, mPositive, 2);
    } else {
        CopyRows(a, workload.m, kOnes, kMinusOnes, 0);
    }
    if (workload.transB) {
        CopyRows(b, workload.n, kPositive, kNegative, 1);
    } else {
        CopyRows(b, workload.k, nPositive, nNegative, 2);
        // KChecksum assigns coefficient +7 to k=0.  Non-transposed B is
        // stored as [K,N], so its first row must use that same coefficient.
        std::memcpy(b.data(), nSeven.data(), nSeven.size());
    }
}

std::vector<uint8_t> StructuredExpected(const Candidate &workload)
{
    std::vector<uint8_t> output(
        MatrixBytes(workload.m, workload.n, ElementBytes(workload.dtype)));
    const float checksum = static_cast<float>(KChecksum(workload.k));
    const auto positive = PatternVector(
        workload.n, 1, 1, checksum, workload.dtype);
    const auto negative = PatternVector(
        workload.n, 1, -1, checksum, workload.dtype);
    CopyRows(output, workload.m, positive, negative, 0);
    return output;
}

std::vector<uint8_t> ReadTiling(const Candidate &candidate)
{
    std::ifstream input(candidate.tilingPath, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read tiling file " + candidate.tilingPath);
    std::vector<uint8_t> result(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (result.size() != sizeof(MatmulTilingData)) {
        throw std::runtime_error(
            "tiling ABI size mismatch expected=" +
            std::to_string(sizeof(MatmulTilingData)) + " actual=" +
            std::to_string(result.size()));
    }
    uint64_t fnv = 0xCBF29CE484222325ULL;
    for (uint8_t value : result) fnv = (fnv ^ value) * 0x100000001B3ULL;
    std::ostringstream digest;
    digest << std::hex << std::setw(16) << std::setfill('0') << fnv;
    if (digest.str() != candidate.tilingFnv1a64) {
        throw std::runtime_error("tiling content digest mismatch");
    }
    MatmulTilingData decoded{};
    std::memcpy(&decoded, result.data(), sizeof(decoded));
    if (decoded.matmulTiling.usedCoreNum != static_cast<int32_t>(candidate.usedCores) ||
        decoded.matmulTiling.M != candidate.m ||
        decoded.matmulTiling.N != candidate.n ||
        decoded.matmulTiling.Ka != candidate.k ||
        decoded.matmulTiling.Kb != candidate.k) {
        throw std::runtime_error("tiling content does not match manifest shape/core count");
    }
    return result;
}

aclError Launch(
    const Candidate &candidate,
    aclrtStream stream,
    void *a,
    void *b,
    void *c,
    void *workspace,
    void *tiling)
{
#define DIRECT_LAUNCH(dtype_name, suffix_value) \
    return ACLRT_LAUNCH_KERNEL(direct_matmul_##dtype_name##_k##suffix_value)( \
        candidate.usedCores, stream, a, b, nullptr, nullptr, c, workspace, tiling)
    if (candidate.dtype == "fp16") {
        switch (candidate.suffix) {
            case 0: DIRECT_LAUNCH(fp16, 0);
            case 1: DIRECT_LAUNCH(fp16, 1);
            case 20: DIRECT_LAUNCH(fp16, 20);
            case 21: DIRECT_LAUNCH(fp16, 21);
            case 30: DIRECT_LAUNCH(fp16, 30);
            case 31: DIRECT_LAUNCH(fp16, 31);
            case 201: DIRECT_LAUNCH(fp16, 201);
            case 10201: DIRECT_LAUNCH(fp16, 10201);
        }
    } else if (candidate.dtype == "bf16") {
        switch (candidate.suffix) {
            case 0: DIRECT_LAUNCH(bf16, 0);
            case 1: DIRECT_LAUNCH(bf16, 1);
            case 20: DIRECT_LAUNCH(bf16, 20);
            case 21: DIRECT_LAUNCH(bf16, 21);
            case 30: DIRECT_LAUNCH(bf16, 30);
            case 31: DIRECT_LAUNCH(bf16, 31);
            case 201: DIRECT_LAUNCH(bf16, 201);
            case 10201: DIRECT_LAUNCH(bf16, 10201);
        }
    } else if (candidate.dtype == "fp32") {
        switch (candidate.suffix) {
            case 1: DIRECT_LAUNCH(fp32, 1);
            case 21: DIRECT_LAUNCH(fp32, 21);
            case 31: DIRECT_LAUNCH(fp32, 31);
            case 101: DIRECT_LAUNCH(fp32, 101);
            case 201: DIRECT_LAUNCH(fp32, 201);
            case 10201: DIRECT_LAUNCH(fp32, 10201);
            case 20201: DIRECT_LAUNCH(fp32, 20201);
        }
    }
#undef DIRECT_LAUNCH
    return ACL_ERROR_INVALID_PARAM;
}

struct Stats {
    double minimum = 0;
    double mean = 0;
    double median = 0;
    double stddev = 0;
    double p95 = 0;
    double maximum = 0;
};

Stats ComputeStats(std::vector<double> values)
{
    if (values.empty()) throw std::runtime_error("no device-event samples");
    std::sort(values.begin(), values.end());
    Stats result;
    result.minimum = values.front();
    result.maximum = values.back();
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    result.median = values.size() % 2 == 0
        ? (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2
        : values[values.size() / 2];
    double variance = 0;
    for (double value : values) variance += (value - result.mean) * (value - result.mean);
    result.stddev = std::sqrt(variance / values.size());
    const size_t p95 = std::min(
        values.size() - 1,
        static_cast<size_t>(std::ceil(0.95 * values.size()) - 1));
    result.p95 = values[p95];
    return result;
}

void EmitFailure(const Candidate &candidate, const std::string &error)
{
    std::cout
        << "DIRECT_MATMUL_RESULT {\"status\":\"failed\","
        << "\"workload_id\":\"" << JsonEscape(candidate.workloadId) << "\","
        << "\"rank\":\"" << JsonEscape(candidate.rank) << "\","
        << "\"candidate_role\":\"searched\","
        << "\"model_schedule_sha256\":\"" << candidate.scheduleSha256 << "\","
        << "\"tiling_sha256\":\"" << candidate.tilingSha256 << "\","
        << "\"error\":\"" << JsonEscape(error) << "\"}" << std::endl;
}

void EmitSuccess(
    const Candidate &candidate,
    const std::vector<double> &samples,
    const Stats &stats,
    double devicePrepareMs,
    double warmupWallMs,
    double measurementWallMs,
    double validationMs,
    double totalMs)
{
    const double operations =
        2.0 * static_cast<double>(candidate.m) * candidate.n * candidate.k;
    std::cout << std::setprecision(12)
        << "DIRECT_MATMUL_RESULT {\"status\":\"success\","
        << "\"measurement_source\":\"direct_tiling_buffer\","
        << "\"tiling_applied\":1,\"full_output_validated\":1,"
        << "\"workload_id\":\"" << JsonEscape(candidate.workloadId) << "\","
        << "\"rank\":\"" << JsonEscape(candidate.rank) << "\","
        << "\"candidate_role\":\"searched\","
        << "\"model_schedule_sha256\":\"" << candidate.scheduleSha256 << "\","
        << "\"actual_tiling_sha256\":\"" << candidate.tilingSha256 << "\","
        << "\"actual_tiling_fnv1a64\":\"" << candidate.tilingFnv1a64 << "\","
        << "\"actual_kernel_suffix\":" << candidate.suffix << ','
        << "\"actual_block_dim\":" << candidate.usedCores << ','
        << "\"workspace_bytes\":" << candidate.workspaceBytes << ','
        << "\"is_reserve\":" << (candidate.reserve ? 1 : 0) << ','
        << "\"min_ms\":" << stats.minimum << ','
        << "\"mean_ms\":" << stats.mean << ','
        << "\"median_ms\":" << stats.median << ','
        << "\"stddev_ms\":" << stats.stddev << ','
        << "\"p95_ms\":" << stats.p95 << ','
        << "\"max_ms\":" << stats.maximum << ','
        << "\"tflops\":" << operations / (stats.median * 1.0e9) << ','
        << "\"device_prepare_ms\":" << devicePrepareMs << ','
        << "\"warmup_wall_ms\":" << warmupWallMs << ','
        << "\"measurement_wall_ms\":" << measurementWallMs << ','
        << "\"numeric_preflight_ms\":" << validationMs << ','
        << "\"runner_total_ms\":" << totalMs << ','
        << "\"samples_ms\":[";
    for (size_t index = 0; index < samples.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << samples[index];
    }
    std::cout << "]}" << std::endl;
}

void RunCandidate(
    const Candidate &candidate,
    aclrtStream stream,
    DeviceBuffer &a,
    DeviceBuffer &b,
    DeviceBuffer &c,
    DeviceBuffer &workspace,
    DeviceBuffer &tiling,
    const std::vector<uint8_t> &expected,
    const Options &options,
    double devicePrepareMs)
{
    const auto totalStarted = Clock::now();
    const auto tilingHost = ReadTiling(candidate);
    Check(aclrtMemcpy(
        tiling.ptr, tiling.bytes, tilingHost.data(), tilingHost.size(),
        ACL_MEMCPY_HOST_TO_DEVICE), "copy exact tiling buffer");

    auto reset = [&]() {
        Check(aclrtMemset(c.ptr, c.bytes, 0x5a, c.bytes), "poison output");
        Check(aclrtMemset(
            workspace.ptr, workspace.bytes, 0, workspace.bytes),
            "clear workspace");
    };
    auto launch = [&]() {
        Check(Launch(
            candidate, stream, a.ptr, b.ptr, c.ptr, workspace.ptr, tiling.ptr),
            "direct CANN 8.1 MatMulV3 kernel launch");
    };

    const auto warmupStarted = Clock::now();
    for (int index = 0; index < options.warmup; ++index) {
        reset();
        launch();
        Check(aclrtSynchronizeStream(stream), "direct warmup synchronize");
    }
    const double warmupWallMs = ElapsedMs(warmupStarted);

    const auto measurementStarted = Clock::now();
    std::vector<double> samples;
    aclrtEvent begin = nullptr;
    aclrtEvent end = nullptr;
    Check(aclrtCreateEvent(&begin), "create begin event");
    try {
        Check(aclrtCreateEvent(&end), "create end event");
        for (int sample = 0; sample < options.samples; ++sample) {
            reset();
            Check(aclrtSynchronizeStream(stream), "pre-measurement reset synchronize");
            Check(aclrtRecordEvent(begin, stream), "record begin event");
            launch();
            Check(aclrtRecordEvent(end, stream), "record end event");
            Check(aclrtSynchronizeEvent(end), "synchronize end event");
            float elapsed = 0;
            Check(aclrtEventElapsedTime(&elapsed, begin, end), "event elapsed time");
            samples.push_back(elapsed);
        }
        (void)aclrtDestroyEvent(end);
        (void)aclrtDestroyEvent(begin);
    } catch (...) {
        if (end != nullptr) (void)aclrtDestroyEvent(end);
        if (begin != nullptr) (void)aclrtDestroyEvent(begin);
        throw;
    }
    const double measurementWallMs = ElapsedMs(measurementStarted);

    const auto validationStarted = Clock::now();
    std::vector<uint8_t> observed(c.bytes);
    Check(aclrtMemcpy(
        observed.data(), observed.size(), c.ptr, c.bytes,
        ACL_MEMCPY_DEVICE_TO_HOST), "copy full direct output");
    if (observed != expected) {
        const size_t width = ElementBytes(candidate.dtype);
        size_t first = 0;
        while (first * width < observed.size() &&
               std::memcmp(observed.data() + first * width,
                           expected.data() + first * width, width) == 0) {
            ++first;
        }
        throw std::runtime_error(
            "full numeric validation mismatch at C index=" +
            std::to_string(first));
    }
    const double validationMs = ElapsedMs(validationStarted);
    const Stats stats = ComputeStats(samples);
    EmitSuccess(
        candidate, samples, stats, devicePrepareMs, warmupWallMs,
        measurementWallMs, validationMs, ElapsedMs(totalStarted));
}

std::unordered_map<std::string, std::string> ParseArgs(int argc, char **argv)
{
    std::unordered_map<std::string, std::string> result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help" || key == "--validate-input") {
            result[key] = "1";
            continue;
        }
        if (key.rfind("--", 0) != 0 || index + 1 >= argc) {
            throw std::runtime_error("invalid argument " + key);
        }
        result[key] = argv[++index];
    }
    return result;
}

std::string Get(
    const std::unordered_map<std::string, std::string> &args,
    const std::string &name,
    const std::string &fallback = "")
{
    const auto found = args.find(name);
    return found == args.end() ? fallback : found->second;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const auto args = ParseArgs(argc, argv);
        if (args.count("--help")) {
            std::cout << "direct_matmul_runner --manifest FILE --device N "
                         "--warmup N --repeat N --samples N\n";
            return 0;
        }
        Options options;
        options.manifest = Get(args, "--manifest");
        options.device = std::stoi(Get(args, "--device", "0"));
        options.warmup = std::stoi(Get(args, "--warmup", "1"));
        options.repeat = std::stoi(Get(args, "--repeat", "1"));
        options.samples = std::stoi(Get(args, "--samples", "3"));
        if (options.manifest.empty() || options.warmup < 0 ||
            options.repeat != 1 || options.samples <= 0) {
            throw std::runtime_error("invalid direct runner options");
        }
        const auto candidates = LoadManifest(options.manifest);
        for (const auto &candidate : candidates) (void)ReadTiling(candidate);
        if (args.count("--validate-input")) {
            std::cout << "DIRECT_MATMUL_INPUT status=passed candidates="
                      << candidates.size() << " tiling_bytes="
                      << sizeof(MatmulTilingData) << '\n';
            return 0;
        }

        bool initialized = false;
        bool deviceSet = false;
        aclrtContext context = nullptr;
        aclrtStream stream = nullptr;
        try {
            Check(aclInit(nullptr), "aclInit");
            initialized = true;
            Check(aclrtSetDevice(options.device), "aclrtSetDevice");
            deviceSet = true;
            Check(aclrtCreateContext(&context, options.device), "aclrtCreateContext");
            Check(aclrtCreateStream(&stream), "aclrtCreateStream");

            size_t index = 0;
            while (index < candidates.size()) {
                const Candidate &workload = candidates[index];
                size_t end = index;
                size_t maxWorkspace = 0;
                while (end < candidates.size() &&
                       candidates[end].workloadId == workload.workloadId) {
                    const Candidate &row = candidates[end];
                    if (row.m != workload.m || row.n != workload.n ||
                        row.k != workload.k || row.dtype != workload.dtype ||
                        row.transA != workload.transA || row.transB != workload.transB) {
                        throw std::runtime_error("one workload ID has inconsistent shapes");
                    }
                    if (row.requiredSuccessfulTilings !=
                        workload.requiredSuccessfulTilings) {
                        throw std::runtime_error(
                            "one workload ID has inconsistent success targets");
                    }
                    maxWorkspace = std::max(maxWorkspace, row.workspaceBytes);
                    ++end;
                }
                const auto prepareStarted = Clock::now();
                const size_t width = ElementBytes(workload.dtype);
                const size_t aBytes = MatrixBytes(workload.m, workload.k, width);
                const size_t bBytes = MatrixBytes(workload.k, workload.n, width);
                const size_t cBytes = MatrixBytes(workload.m, workload.n, width);
                std::vector<uint8_t> aHost(aBytes);
                std::vector<uint8_t> bHost(bBytes);
                StructuredInputs(aHost, bHost, workload);
                const auto expected = StructuredExpected(workload);
                DeviceBuffer a(aBytes), b(bBytes), c(cBytes), workspace(maxWorkspace);
                DeviceBuffer tiling(sizeof(MatmulTilingData));
                Check(aclrtMemcpy(
                    a.ptr, a.bytes, aHost.data(), aHost.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE), "copy structured A");
                Check(aclrtMemcpy(
                    b.ptr, b.bytes, bHost.data(), bHost.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE), "copy structured B");
                const double devicePrepareMs = ElapsedMs(prepareStarted);
                uint32_t successful = 0;
                for (size_t candidateIndex = index; candidateIndex < end; ++candidateIndex) {
                    if (successful >= workload.requiredSuccessfulTilings) break;
                    const Candidate &candidate = candidates[candidateIndex];
                    std::cout << "DIRECT_MATMUL_START workload_id="
                              << candidate.workloadId << " rank=" << candidate.rank
                              << std::endl;
                    try {
                        RunCandidate(
                            candidate, stream, a, b, c, workspace, tiling,
                            expected, options, devicePrepareMs);
                        ++successful;
                    } catch (const std::exception &error) {
                        EmitFailure(candidate, error.what());
                        if (std::string(error.what()).rfind(
                                "full numeric validation mismatch", 0) != 0) {
                            throw;
                        }
                    }
                }
                if (successful != workload.requiredSuccessfulTilings) {
                    throw std::runtime_error(
                        "workload exhausted legal reserves before reaching success target");
                }
                index = end;
            }
            (void)aclrtDestroyStream(stream);
            stream = nullptr;
            (void)aclrtDestroyContext(context);
            context = nullptr;
            (void)aclrtResetDevice(options.device);
            deviceSet = false;
            (void)aclFinalize();
            initialized = false;
        } catch (...) {
            if (stream != nullptr) (void)aclrtDestroyStream(stream);
            if (context != nullptr) (void)aclrtDestroyContext(context);
            if (deviceSet) (void)aclrtResetDevice(options.device);
            if (initialized) (void)aclFinalize();
            throw;
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "DIRECT_MATMUL_FATAL " << error.what() << '\n';
        return 1;
    }
}

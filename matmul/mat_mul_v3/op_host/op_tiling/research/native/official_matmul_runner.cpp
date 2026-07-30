#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_matmul.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string candidatesCsv = "results/candidates.csv";
    std::string outputCsv = "results/official_matmul_profile.csv";
    std::string samplesCsv = "results/official_matmul_samples.csv";
    std::string onlyWorkload;
    int32_t workloadLimit = -1;
    int32_t deviceId = 0;
    int32_t warmup = 10;
    int32_t repeat = 50;
    int32_t samples = 15;
    int32_t numericPreflightMaxMiB = 4;
};

struct Workload {
    std::string id;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    std::string dtype;
    bool transA = false;
    bool transB = false;
};

struct ProfileSummary {
    bool supported = true;
    bool success = false;
    bool preflightPassed = false;
    std::string preflightMode;
    std::string error;
    std::vector<double> valuesMs;
    double minimum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double standardDeviation = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
    double tflops = 0.0;
};

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t requested) { Allocate(requested); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { Release(); }

    void Allocate(size_t requested)
    {
        Release();
        if (requested == 0) return;
        bytes = requested;
        const aclError rc = aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS) {
            ptr = nullptr;
            bytes = 0;
            throw std::runtime_error(
                "aclrtMalloc failed, bytes=" + std::to_string(requested) +
                ", rc=" + std::to_string(rc));
        }
    }

    void Release()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
};

struct TensorHandle {
    aclTensor *ptr = nullptr;

    TensorHandle() = default;
    TensorHandle(const TensorHandle &) = delete;
    TensorHandle &operator=(const TensorHandle &) = delete;
    TensorHandle(TensorHandle &&other) noexcept : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }
    TensorHandle &operator=(TensorHandle &&other) noexcept
    {
        if (this != &other) {
            if (ptr != nullptr) aclDestroyTensor(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~TensorHandle()
    {
        if (ptr != nullptr) aclDestroyTensor(ptr);
    }
};

struct ExecutorHandle {
    aclOpExecutor *ptr = nullptr;

    ExecutorHandle() = default;
    ExecutorHandle(const ExecutorHandle &) = delete;
    ExecutorHandle &operator=(const ExecutorHandle &) = delete;
    ExecutorHandle(ExecutorHandle &&other) noexcept : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }
    ExecutorHandle &operator=(ExecutorHandle &&other) noexcept
    {
        if (this != &other) {
            if (ptr != nullptr) aclDestroyAclOpExecutor(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~ExecutorHandle()
    {
        if (ptr != nullptr) aclDestroyAclOpExecutor(ptr);
    }
};

void CheckAcl(aclError rc, const std::string &operation)
{
    if (rc != ACL_SUCCESS) {
        throw std::runtime_error(operation + " failed, rc=" + std::to_string(rc));
    }
}

void CheckAclnn(aclnnStatus rc, const std::string &operation)
{
    if (rc != 0) {
        throw std::runtime_error(operation + " failed, rc=" + std::to_string(rc));
    }
}

int RunAclOnly(int32_t deviceId)
{
    std::cout << "official runner ACL check\n"
              << "device=" << deviceId << '\n';
    aclError rc = aclInit(nullptr);
    std::cout << "aclInit rc=" << rc << '\n';
    if (rc != ACL_SUCCESS) return 1;
    rc = aclrtSetDevice(deviceId);
    std::cout << "aclrtSetDevice rc=" << rc << '\n';
    if (rc != ACL_SUCCESS) {
        aclFinalize();
        return 1;
    }
    const char *soc = aclrtGetSocName();
    std::cout << "aclrtGetSocName=" << (soc == nullptr ? "<null>" : soc) << '\n';
    const aclError resetRc = aclrtResetDevice(deviceId);
    std::cout << "aclrtResetDevice rc=" << resetRc << '\n';
    const aclError finalizeRc = aclFinalize();
    std::cout << "aclFinalize rc=" << finalizeRc << '\n';
    return soc != nullptr && resetRc == ACL_SUCCESS && finalizeRc == ACL_SUCCESS ? 0 : 1;
}

int RunSocOnly(int32_t deviceId)
{
    std::cout << "soc_probe_stage=aclInit" << std::endl;
    const aclError initRc = aclInit(nullptr);
    if (initRc != ACL_SUCCESS) {
        std::cerr << "fatal: aclInit failed, rc=" << initRc << std::endl;
        return 1;
    }
    std::cout << "soc_probe_stage=aclrtSetDevice device=" << deviceId << std::endl;
    const aclError setRc = aclrtSetDevice(deviceId);
    if (setRc != ACL_SUCCESS) {
        std::cerr << "fatal: aclrtSetDevice failed, rc=" << setRc << std::endl;
        return 1;
    }
    std::cout << "soc_probe_stage=aclrtGetSocName" << std::endl;
    const char *soc = aclrtGetSocName();
    if (soc == nullptr || soc[0] == '\0') {
        std::cerr << "fatal: aclrtGetSocName returned an empty value" << std::endl;
        return 1;
    }
    std::cout << "aclrtGetSocName=" << soc << std::endl;
    std::cout.flush();
    std::_Exit(0);
}

std::vector<std::string> SplitCsv(const std::string &line)
{
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
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

std::string EscapeCsv(const std::string &value)
{
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string out = "\"";
    for (char ch : value) out += ch == '"' ? "\"\"" : std::string(1, ch);
    return out + '"';
}

template <typename T>
std::string ToText(const T &value)
{
    std::ostringstream output;
    output << std::setprecision(12) << value;
    return output.str();
}

const std::vector<std::string> &ProfileColumns()
{
    static const std::vector<std::string> columns{
        "workload_id", "rank", "source", "candidate_role",
        "m", "n", "k", "dtype", "trans_a", "trans_b", "execution_mode",
        "used_core_num", "hint_single_core_m", "hint_single_core_n",
        "hint_single_core_k", "hint_base_m", "hint_base_n", "hint_base_k",
        "official_base_m", "official_base_n", "official_base_k",
        "official_core_num", "official_m_dim", "official_n_dim",
        "proxy_total", "success", "preflight_passed", "preflight_mode", "error",
        "min_ms", "mean_ms", "median_ms", "stddev_ms", "p95_ms", "max_ms",
        "tflops", "warmup", "repeat", "samples", "tiling_signature", "tiling_bin",
    };
    return columns;
}

void WriteCsvRecord(std::ostream &output, const std::vector<std::string> &fields)
{
    if (fields.size() != ProfileColumns().size()) {
        throw std::runtime_error(
            "official profile CSV schema mismatch: expected " +
            std::to_string(ProfileColumns().size()) + " fields, got " +
            std::to_string(fields.size()));
    }
    for (size_t index = 0; index < fields.size(); ++index) {
        if (index != 0) output << ',';
        output << EscapeCsv(fields[index]);
    }
    output << '\n';
}

bool ParseBool(const std::string &value)
{
    return value == "1" || value == "true" || value == "TRUE" || value == "yes";
}

std::string GetField(
    const std::vector<std::string> &fields,
    const std::unordered_map<std::string, size_t> &columns,
    const std::string &name)
{
    const auto it = columns.find(name);
    return it == columns.end() || it->second >= fields.size() ? "" : fields[it->second];
}

bool SameWorkload(const Workload &a, const Workload &b)
{
    return a.m == b.m && a.n == b.n && a.k == b.k && a.dtype == b.dtype &&
           a.transA == b.transA && a.transB == b.transB;
}

std::vector<Workload> LoadWorkloads(const std::string &path, const Options &options)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open candidate CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("candidate CSV is empty");

    const auto header = SplitCsv(line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;
    const std::string idColumn =
        columns.count("workload_id") != 0 ? "workload_id" : "id";
    if (columns.count(idColumn) == 0) {
        throw std::runtime_error("workload CSV missing column: id or workload_id");
    }
    for (const char *required : {"m", "n", "k", "dtype", "trans_a", "trans_b"}) {
        if (columns.count(required) == 0) {
            throw std::runtime_error(std::string("workload CSV missing column: ") + required);
        }
    }

    std::vector<Workload> workloads;
    std::unordered_map<std::string, size_t> indices;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = SplitCsv(line);
        Workload workload;
        workload.id = GetField(fields, columns, idColumn);
        workload.m = std::stoll(GetField(fields, columns, "m"));
        workload.n = std::stoll(GetField(fields, columns, "n"));
        workload.k = std::stoll(GetField(fields, columns, "k"));
        workload.dtype = GetField(fields, columns, "dtype");
        workload.transA = ParseBool(GetField(fields, columns, "trans_a"));
        workload.transB = ParseBool(GetField(fields, columns, "trans_b"));
        if (workload.id.empty() || workload.m <= 0 || workload.n <= 0 || workload.k <= 0) {
            throw std::runtime_error("candidate CSV contains invalid workload shape");
        }
        if (!options.onlyWorkload.empty() && workload.id != options.onlyWorkload) continue;

        const auto found = indices.find(workload.id);
        if (found != indices.end()) {
            if (!SameWorkload(workloads[found->second], workload)) {
                throw std::runtime_error(
                    "workload ID maps to inconsistent shapes: " + workload.id);
            }
            continue;
        }
        if (options.workloadLimit > 0 &&
            static_cast<int32_t>(workloads.size()) >= options.workloadLimit) {
            continue;
        }
        indices[workload.id] = workloads.size();
        workloads.push_back(std::move(workload));
    }
    if (workloads.empty()) throw std::runtime_error("no workloads matched official baseline filters");
    return workloads;
}

size_t CheckedBytes(int64_t x, int64_t y, size_t elementBytes, const char *name)
{
    if (x <= 0 || y <= 0) throw std::runtime_error(std::string(name) + " shape is invalid");
    const auto product = static_cast<unsigned long long>(x) * static_cast<unsigned long long>(y);
    if (product > std::numeric_limits<size_t>::max() / elementBytes) {
        throw std::runtime_error(std::string(name) + " byte size overflow");
    }
    return static_cast<size_t>(product) * elementBytes;
}

size_t ElementBytes(const std::string &dtype)
{
    if (dtype == "fp16" || dtype == "bf16") return 2;
    if (dtype == "fp32") return 4;
    throw std::runtime_error("aclnnMatmul unsupported dtype: " + dtype);
}

aclDataType AclDType(const std::string &dtype)
{
    if (dtype == "fp16") return ACL_FLOAT16;
    if (dtype == "bf16") return ACL_BF16;
    if (dtype == "fp32") return ACL_FLOAT;
    throw std::runtime_error("aclnnMatmul unsupported dtype: " + dtype);
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float output = 0.0F;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

void FillOnes(std::vector<uint8_t> &buffer, const std::string &dtype)
{
    if (dtype == "fp16") {
        const aclFloat16 one = aclFloatToFloat16(1.0F);
        auto *values = reinterpret_cast<aclFloat16 *>(buffer.data());
        std::fill(values, values + buffer.size() / sizeof(*values), one);
    } else if (dtype == "bf16") {
        const uint16_t one = FloatToBfloat16(1.0F);
        auto *values = reinterpret_cast<uint16_t *>(buffer.data());
        std::fill(values, values + buffer.size() / sizeof(*values), one);
    } else if (dtype == "fp32") {
        auto *values = reinterpret_cast<float *>(buffer.data());
        std::fill(values, values + buffer.size() / sizeof(*values), 1.0F);
    } else {
        throw std::runtime_error("cannot initialize unsupported dtype: " + dtype);
    }
}

double DecodeOutput(uint32_t observed, const std::string &dtype)
{
    if (dtype == "fp16") {
        aclFloat16 value = 0;
        std::memcpy(&value, &observed, sizeof(value));
        return aclFloat16ToFloat(value);
    }
    if (dtype == "bf16") {
        uint16_t value = 0;
        std::memcpy(&value, &observed, sizeof(value));
        return Bfloat16ToFloat(value);
    }
    if (dtype == "fp32") {
        float value = 0.0F;
        std::memcpy(&value, &observed, sizeof(value));
        return value;
    }
    throw std::runtime_error("cannot decode unsupported dtype: " + dtype);
}

double ExpectedOnesOutput(int64_t k, const std::string &dtype)
{
    const float value = static_cast<float>(k);
    if (dtype == "fp16") return aclFloat16ToFloat(aclFloatToFloat16(value));
    if (dtype == "bf16") return Bfloat16ToFloat(FloatToBfloat16(value));
    return static_cast<double>(k);
}

TensorHandle CreateTensor(
    void *data,
    aclDataType dtype,
    const std::array<int64_t, 2> &view,
    const std::array<int64_t, 2> &strides,
    const std::array<int64_t, 2> &storage)
{
    TensorHandle tensor;
    tensor.ptr = aclCreateTensor(
        view.data(), view.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
        storage.data(), storage.size(), data);
    if (tensor.ptr == nullptr) throw std::runtime_error("aclCreateTensor returned null");
    return tensor;
}

void ComputeStats(ProfileSummary &summary, const Workload &workload)
{
    std::sort(summary.valuesMs.begin(), summary.valuesMs.end());
    summary.minimum = summary.valuesMs.front();
    summary.maximum = summary.valuesMs.back();
    summary.mean = std::accumulate(
        summary.valuesMs.begin(), summary.valuesMs.end(), 0.0) /
        static_cast<double>(summary.valuesMs.size());
    const size_t count = summary.valuesMs.size();
    summary.median = count % 2 == 0
        ? 0.5 * (summary.valuesMs[count / 2 - 1] + summary.valuesMs[count / 2])
        : summary.valuesMs[count / 2];
    double variance = 0.0;
    for (double value : summary.valuesMs) {
        const double delta = value - summary.mean;
        variance += delta * delta;
    }
    summary.standardDeviation = std::sqrt(variance / static_cast<double>(count));
    const size_t p95Index =
        std::min(count - 1, static_cast<size_t>(std::ceil(0.95 * count) - 1));
    summary.p95 = summary.valuesMs[p95Index];
    const double operations =
        2.0 * static_cast<double>(workload.m) * workload.n * workload.k;
    summary.tflops = operations / (summary.median * 1.0e9);
    summary.success = true;
}

void LogStage(const Workload &workload, const std::string &stage)
{
    std::cout << "official_stage " << workload.id << " stage=" << stage << std::endl;
}

ProfileSummary ProfileOfficial(
    const Workload &workload,
    aclrtStream stream,
    const Options &options,
    std::ofstream &samplesOutput)
{
    ProfileSummary summary;
    if (workload.dtype == "int8") {
        summary.supported = false;
        summary.error = "unsupported_by_aclnnMatmul:int8";
        return summary;
    }

    try {
        const size_t elementBytes = ElementBytes(workload.dtype);
        const size_t aBytes = CheckedBytes(workload.m, workload.k, elementBytes, "A");
        const size_t bBytes = CheckedBytes(workload.k, workload.n, elementBytes, "B");
        const size_t cBytes = CheckedBytes(workload.m, workload.n, elementBytes, "C");
        LogStage(workload, "device_malloc");
        DeviceBuffer a(aBytes);
        DeviceBuffer b(bBytes);
        DeviceBuffer c(cBytes);

        const size_t numericLimit =
            static_cast<size_t>(std::max(0, options.numericPreflightMaxMiB)) * 1024 * 1024;
        const bool numericPreflight =
            numericLimit > 0 && aBytes <= numericLimit && bBytes <= numericLimit - aBytes &&
            workload.k <= 60000;
        if (numericPreflight) {
            std::vector<uint8_t> aHost(aBytes);
            std::vector<uint8_t> bHost(bBytes);
            FillOnes(aHost, workload.dtype);
            FillOnes(bHost, workload.dtype);
            CheckAcl(aclrtMemcpy(
                a.ptr, a.bytes, aHost.data(), aHost.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy official numeric A");
            CheckAcl(aclrtMemcpy(
                b.ptr, b.bytes, bHost.data(), bHost.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy official numeric B");
            summary.preflightMode = "numeric_ones_grid9_v1";
        } else {
            CheckAcl(aclrtMemset(a.ptr, a.bytes, 0, a.bytes), "aclrtMemset official A");
            CheckAcl(aclrtMemset(b.ptr, b.bytes, 0, b.bytes), "aclrtMemset official B");
            summary.preflightMode = "zero_coverage_grid9_v1";
        }
        CheckAcl(aclrtMemset(c.ptr, c.bytes, 0x5a, c.bytes), "aclrtMemset official C poison");

        const std::array<int64_t, 2> aView{workload.m, workload.k};
        const std::array<int64_t, 2> bView{workload.k, workload.n};
        const std::array<int64_t, 2> cView{workload.m, workload.n};
        const std::array<int64_t, 2> aStorage =
            workload.transA ? std::array<int64_t, 2>{workload.k, workload.m} : aView;
        const std::array<int64_t, 2> bStorage =
            workload.transB ? std::array<int64_t, 2>{workload.n, workload.k} : bView;
        const std::array<int64_t, 2> aStrides = workload.transA
            ? std::array<int64_t, 2>{1, workload.m}
            : std::array<int64_t, 2>{workload.k, 1};
        const std::array<int64_t, 2> bStrides = workload.transB
            ? std::array<int64_t, 2>{1, workload.k}
            : std::array<int64_t, 2>{workload.n, 1};
        const std::array<int64_t, 2> cStrides{workload.n, 1};
        const aclDataType dtype = AclDType(workload.dtype);
        TensorHandle aTensor = CreateTensor(a.ptr, dtype, aView, aStrides, aStorage);
        TensorHandle bTensor = CreateTensor(b.ptr, dtype, bView, bStrides, bStorage);
        TensorHandle cTensor = CreateTensor(c.ptr, dtype, cView, cStrides, cView);

        uint64_t workspaceBytes = 0;
        ExecutorHandle executor;
        LogStage(workload, "get_workspace");
        CheckAclnn(aclnnMatmulGetWorkspaceSize(
            aTensor.ptr, bTensor.ptr, cTensor.ptr, 0, &workspaceBytes, &executor.ptr),
            "aclnnMatmulGetWorkspaceSize");
        if (executor.ptr == nullptr) {
            throw std::runtime_error("aclnnMatmulGetWorkspaceSize returned null executor");
        }
        CheckAclnn(
            aclSetAclOpExecutorRepeatable(executor.ptr),
            "aclSetAclOpExecutorRepeatable");
        DeviceBuffer workspace(static_cast<size_t>(workspaceBytes));

        auto launch = [&]() {
            CheckAclnn(
                aclnnMatmul(workspace.ptr, workspaceBytes, executor.ptr, stream),
                "aclnnMatmul");
        };

        LogStage(workload, "preflight_launch");
        launch();
        CheckAcl(aclrtSynchronizeStream(stream), "official preflight synchronize");
        const size_t outputBytes = ElementBytes(workload.dtype);
        constexpr int64_t coverageGrid = 9;
        std::set<int64_t> sampleIndices;
        for (int64_t rowProbe = 0; rowProbe < coverageGrid; ++rowProbe) {
            const int64_t row =
                (workload.m - 1) * rowProbe / (coverageGrid - 1);
            for (int64_t columnProbe = 0; columnProbe < coverageGrid;
                 ++columnProbe) {
                const int64_t column =
                    (workload.n - 1) * columnProbe / (coverageGrid - 1);
                sampleIndices.insert(row * workload.n + column);
            }
        }
        for (int64_t index : sampleIndices) {
            uint32_t observed = 0;
            auto *source =
                static_cast<uint8_t *>(c.ptr) + static_cast<size_t>(index) * outputBytes;
            CheckAcl(aclrtMemcpy(
                &observed, outputBytes, source, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy official preflight sample");
            if (numericPreflight) {
                const double actual = DecodeOutput(observed, workload.dtype);
                const double expected = ExpectedOnesOutput(workload.k, workload.dtype);
                const double tolerance = std::max(0.03, std::abs(expected) * 0.01);
                if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
                    throw std::runtime_error(
                        "official numeric preflight failed at C index=" +
                        std::to_string(index) + ", actual=" + std::to_string(actual) +
                        ", expected=" + std::to_string(expected));
                }
            } else {
                const auto *bytes = reinterpret_cast<const uint8_t *>(&observed);
                for (size_t byte = 0; byte < outputBytes; ++byte) {
                    if (bytes[byte] != 0) {
                        std::ostringstream detail;
                        detail << "official output coverage failed at C index="
                               << index << ", observed=0x"
                               << std::hex << std::setfill('0')
                               << std::setw(static_cast<int>(outputBytes * 2))
                               << observed;
                        throw std::runtime_error(
                            detail.str());
                    }
                }
            }
        }
        summary.preflightPassed = true;

        LogStage(workload, "warmup");
        for (int32_t i = 0; i < options.warmup; ++i) launch();
        CheckAcl(aclrtSynchronizeStream(stream), "official warmup synchronize");

        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        CheckAcl(aclrtCreateEvent(&start), "aclrtCreateEvent official start");
        try {
            CheckAcl(aclrtCreateEvent(&end), "aclrtCreateEvent official end");
            for (int32_t sample = 0; sample < options.samples; ++sample) {
                LogStage(workload, "sample_" + std::to_string(sample));
                CheckAcl(aclrtRecordEvent(start, stream), "aclrtRecordEvent official start");
                for (int32_t repeat = 0; repeat < options.repeat; ++repeat) launch();
                CheckAcl(aclrtRecordEvent(end, stream), "aclrtRecordEvent official end");
                CheckAcl(aclrtSynchronizeEvent(end), "aclrtSynchronizeEvent official end");
                float batchMs = 0.0F;
                CheckAcl(
                    aclrtEventElapsedTime(&batchMs, start, end),
                    "aclrtEventElapsedTime official");
                const double latencyMs =
                    static_cast<double>(batchMs) / std::max(1, options.repeat);
                summary.valuesMs.push_back(latencyMs);
                samplesOutput << EscapeCsv(workload.id)
                              << ",-1,official_operator_baseline," << sample << ','
                              << std::setprecision(12) << latencyMs << '\n';
                samplesOutput.flush();
            }
            aclrtDestroyEvent(end);
            aclrtDestroyEvent(start);
        } catch (...) {
            if (end != nullptr) aclrtDestroyEvent(end);
            if (start != nullptr) aclrtDestroyEvent(start);
            throw;
        }
        ComputeStats(summary, workload);
    } catch (const std::exception &exception) {
        summary.error = exception.what();
    }
    return summary;
}

void WriteProfileHeader(std::ostream &output)
{
    WriteCsvRecord(output, ProfileColumns());
}

void WriteProfileRow(
    std::ostream &output,
    const Workload &workload,
    const ProfileSummary &summary,
    const Options &options)
{
    WriteCsvRecord(output, {
        workload.id,
        "-1",
        "installed_aclnn_matmul",
        "official_operator_baseline",
        ToText(workload.m),
        ToText(workload.n),
        ToText(workload.k),
        workload.dtype,
        ToText(workload.transA),
        ToText(workload.transB),
        "official_matmul_v3",
        "0",
        "0", "0", "0",
        "0", "0", "0",
        "0", "0", "0",
        "0", "0", "0",
        "0",
        ToText(summary.success),
        ToText(summary.preflightPassed),
        summary.preflightMode,
        summary.error,
        ToText(summary.minimum),
        ToText(summary.mean),
        ToText(summary.median),
        ToText(summary.standardDeviation),
        ToText(summary.p95),
        ToText(summary.maximum),
        ToText(summary.tflops),
        ToText(options.warmup),
        ToText(options.repeat),
        ToText(options.samples),
        "installed_cann_aclnn_matmul",
        "",
    });
    output.flush();
}

void ValidateProfileCsvContract(const Workload &workload, const Options &options)
{
    ProfileSummary summary;
    std::ostringstream output;
    WriteProfileHeader(output);
    WriteProfileRow(output, workload, summary, options);
    std::istringstream input(output.str());
    std::string header;
    std::string row;
    if (!std::getline(input, header) || !std::getline(input, row) ||
        SplitCsv(header).size() != ProfileColumns().size() ||
        SplitCsv(row).size() != ProfileColumns().size()) {
        throw std::runtime_error("official profile CSV contract validation failed");
    }
}

std::unordered_map<std::string, std::string> ParseArgs(int argc, char **argv)
{
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h" || key == "--validate-input" ||
            key == "--acl-only" || key == "--soc-only") {
            args[key] = "1";
            continue;
        }
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("invalid argument: " + key);
        }
        args[key] = argv[++i];
    }
    return args;
}

std::string Get(
    const std::unordered_map<std::string, std::string> &args,
    const std::string &name,
    const std::string &fallback)
{
    const auto it = args.find(name);
    return it == args.end() ? fallback : it->second;
}

int32_t GetInt(
    const std::unordered_map<std::string, std::string> &args,
    const std::string &name,
    int32_t fallback)
{
    const auto it = args.find(name);
    return it == args.end() ? fallback : std::stoi(it->second);
}

void PrintUsage()
{
    std::cout
        << "official_matmul_runner [options]\n"
        << "  --candidates FILE\n"
        << "  --output FILE\n"
        << "  --samples-output FILE\n"
        << "  --device N\n"
        << "  --warmup N\n"
        << "  --repeat N\n"
        << "  --samples N\n"
        << "  --only-workload ID\n"
        << "  --workload-limit N\n"
        << "  --numeric-preflight-max-mib N\n"
        << "  --acl-only            initialize the linked ACL runtime without profiling\n"
        << "  --soc-only            print the exact device SoC and exit without teardown\n"
        << "  --validate-input       validate input and CSV schema without ACL/NPU\n";
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const auto args = ParseArgs(argc, argv);
        if (args.count("--help") || args.count("-h")) {
            PrintUsage();
            return 0;
        }
        Options options;
        options.candidatesCsv = Get(args, "--candidates", options.candidatesCsv);
        options.outputCsv = Get(args, "--output", options.outputCsv);
        options.samplesCsv = Get(args, "--samples-output", options.samplesCsv);
        options.onlyWorkload = Get(args, "--only-workload", options.onlyWorkload);
        options.workloadLimit = GetInt(args, "--workload-limit", options.workloadLimit);
        options.deviceId = GetInt(args, "--device", options.deviceId);
        options.warmup = GetInt(args, "--warmup", options.warmup);
        options.repeat = GetInt(args, "--repeat", options.repeat);
        options.samples = GetInt(args, "--samples", options.samples);
        options.numericPreflightMaxMiB =
            GetInt(args, "--numeric-preflight-max-mib", options.numericPreflightMaxMiB);
        if (options.warmup < 0 || options.repeat <= 0 || options.samples <= 0) {
            throw std::runtime_error("warmup/repeat/samples values are invalid");
        }
        if (args.count("--acl-only")) {
            return RunAclOnly(options.deviceId);
        }
        if (args.count("--soc-only")) {
            return RunSocOnly(options.deviceId);
        }

        const auto workloads = LoadWorkloads(options.candidatesCsv, options);
        ValidateProfileCsvContract(workloads.front(), options);
        if (args.count("--validate-input")) {
            std::cout << "official baseline input passed: workloads=" << workloads.size()
                      << " profile_columns=" << ProfileColumns().size() << '\n';
            return 0;
        }
        const auto outputParent = std::filesystem::path(options.outputCsv).parent_path();
        const auto samplesParent = std::filesystem::path(options.samplesCsv).parent_path();
        if (!outputParent.empty()) std::filesystem::create_directories(outputParent);
        if (!samplesParent.empty()) std::filesystem::create_directories(samplesParent);
        std::ofstream output(options.outputCsv);
        std::ofstream samplesOutput(options.samplesCsv);
        if (!output || !samplesOutput) {
            throw std::runtime_error("cannot create official baseline output CSV");
        }
        WriteProfileHeader(output);
        samplesOutput << "workload_id,rank,candidate_role,sample,latency_ms\n";

        bool aclInitialized = false;
        bool deviceSet = false;
        aclrtContext context = nullptr;
        aclrtStream stream = nullptr;
        int failures = 0;
        try {
            CheckAcl(aclInit(nullptr), "aclInit");
            aclInitialized = true;
            CheckAcl(aclrtSetDevice(options.deviceId), "aclrtSetDevice");
            deviceSet = true;
            CheckAcl(aclrtCreateContext(&context, options.deviceId), "aclrtCreateContext");
            CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");

            for (size_t index = 0; index < workloads.size(); ++index) {
                const Workload &workload = workloads[index];
                std::cout << "official_progress: [" << index + 1 << '/' << workloads.size()
                          << "] " << workload.id << " M=" << workload.m << " N=" << workload.n
                          << " K=" << workload.k << " dtype=" << workload.dtype << std::endl;
                const ProfileSummary summary =
                    ProfileOfficial(workload, stream, options, samplesOutput);
                WriteProfileRow(output, workload, summary, options);
                std::cout << "official_done " << workload.id
                          << " supported=" << summary.supported
                          << " success=" << summary.success
                          << " median_ms=" << std::setprecision(12) << summary.median;
                if (!summary.error.empty()) std::cout << " reason=" << summary.error;
                std::cout << std::endl;
                if (summary.supported && !summary.success) {
                    ++failures;
                    break;
                }
            }

            if (stream != nullptr) {
                aclrtDestroyStream(stream);
                stream = nullptr;
            }
            if (context != nullptr) {
                aclrtDestroyContext(context);
                context = nullptr;
            }
            aclrtResetDevice(options.deviceId);
            deviceSet = false;
            aclFinalize();
            aclInitialized = false;
        } catch (...) {
            if (stream != nullptr) aclrtDestroyStream(stream);
            if (context != nullptr) aclrtDestroyContext(context);
            if (deviceSet) aclrtResetDevice(options.deviceId);
            if (aclInitialized) aclFinalize();
            throw;
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception &exception) {
        std::cerr << "fatal: " << exception.what() << '\n';
        return 1;
    }
}

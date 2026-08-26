#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#ifndef SCATTER_ELEMENTS_ONLY
#include <aclnnop/aclnn_flash_attention_score_grad.h>
#include <aclnnop/aclnn_fused_infer_attention_score.h>
#include <aclnnop/aclnn_gather.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_permute.h>
#endif
#ifndef REMAINING_OPERATOR_ONLY
#include <aclnnop/aclnn_scatter.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>

namespace {

std::string JsonEscape(const std::string &value);
void CheckAcl(aclError rc, const std::string &what);
std::string gWorkloadId = "unknown";
std::string gBackend = "aclnn_real_npu";
#ifndef SCATTER_ELEMENTS_ONLY
constexpr const char *kGatherElementsSourceOperatorType = "GatherElementsV2";

using GatherElementsGetWorkspace = aclnnStatus (*)(const aclTensor *, const aclTensor *, int64_t,
                                                    aclTensor *, uint64_t *, aclOpExecutor **);
using GatherElementsLaunch = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

struct GatherElementsPrivateApi {
    void *library = nullptr;
    GatherElementsGetWorkspace getWorkspace = nullptr;
    GatherElementsLaunch launch = nullptr;
    std::string path;
};
#endif

#ifndef REMAINING_OPERATOR_ONLY
using ScatterElementsGetWorkspace = aclnnStatus (*)(const aclTensor *, int64_t, const aclTensor *,
                                                     const aclTensor *, int64_t, aclTensor *,
                                                     uint64_t *, aclOpExecutor **);
using ScatterElementsLaunch = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

struct ScatterElementsPrivateApi {
    void *library = nullptr;
    ScatterElementsGetWorkspace getWorkspace = nullptr;
    ScatterElementsLaunch launch = nullptr;
    std::string path;
};
#endif

void Stage(const std::string &stage, const std::string &detail = "")
{
    std::cout << "MULTIOP_NPU_STAGE {\"workload_id\":\"" << JsonEscape(gWorkloadId)
              << "\",\"stage\":\"" << JsonEscape(stage) << "\"";
    if (!detail.empty()) std::cout << ",\"detail\":\"" << JsonEscape(detail) << "\"";
    std::cout << "}" << std::endl;
}

#ifndef SCATTER_ELEMENTS_ONLY
void *LoadRemainingPrivateLibrary(const char *path, const std::string &label)
{
    if (path == nullptr || *path == '\0') throw std::runtime_error(label + " path is absent");
    Stage("private_host_tiler_load_begin", label);
    // Host-tiler registration is performed by library constructors and must
    // be globally visible to the installed CANN 8.1 OpAPI invoked below.
    void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *error = dlerror();
        throw std::runtime_error("cannot load " + label + ": " +
                                 (error == nullptr ? "unknown dlopen error" : std::string(error)));
    }
    Stage("private_host_tiler_load_done", label);
    return handle;  // Isolated worker owns this host tiler until process exit.
}

#endif

#ifndef SCATTER_ELEMENTS_ONLY
const GatherElementsPrivateApi &LoadGatherElementsPrivateApi(const char *libraryPath)
{
    // Match CANN 8.1's official custom-OpAPI loader: the library handle and
    // resolved functions live for the whole worker process.  dlclose is
    // intentionally absent because Nnopbase retains process-local state
    // created by GetWorkspaceSize; unloading the code behind that state is
    // unsafe.  The isolated worker exits immediately after one operation.
    static GatherElementsPrivateApi *api = nullptr;
    if (api != nullptr) {
        if (api->path != libraryPath) {
            throw std::runtime_error("private GatherElementsV2 OpAPI path changed within one worker");
        }
        return *api;
    }

    Stage("private_opapi_load_begin", libraryPath);
    auto *loaded = new GatherElementsPrivateApi();
    loaded->path = libraryPath;
    loaded->library = dlopen(libraryPath, RTLD_LAZY | RTLD_LOCAL);
    if (loaded->library == nullptr) {
        const char *error = dlerror();
        const std::string message = error == nullptr ? "unknown dlopen error" : error;
        delete loaded;
        throw std::runtime_error("cannot load private GatherElementsV2 C++ OpAPI: " + message);
    }
    Stage("private_opapi_load_done");

    dlerror();
    loaded->getWorkspace = reinterpret_cast<GatherElementsGetWorkspace>(
        dlsym(loaded->library, "aclnnPrivateGatherElementsV2GetWorkspaceSize"));
    const char *workspaceSymbolError = dlerror();
    if (loaded->getWorkspace == nullptr || workspaceSymbolError != nullptr) {
        throw std::runtime_error("private GatherElementsV2 CANN-8.1 OpAPI GetWorkspaceSize symbol is absent");
    }
    Stage("private_opapi_get_workspace_symbol_done");

    dlerror();
    loaded->launch = reinterpret_cast<GatherElementsLaunch>(
        dlsym(loaded->library, "aclnnPrivateGatherElementsV2"));
    const char *launchSymbolError = dlerror();
    if (loaded->launch == nullptr || launchSymbolError != nullptr) {
        throw std::runtime_error("private GatherElementsV2 CANN-8.1 OpAPI launch symbol is absent");
    }
    Stage("private_opapi_launch_symbol_done");
    api = loaded;
    return *api;
}
#endif

#ifndef REMAINING_OPERATOR_ONLY
const ScatterElementsPrivateApi &LoadScatterElementsPrivateApi(const char *libraryPath)
{
    // CANN's generated OpAPI keeps process-local executor state.  Match the
    // official loader lifetime and let the isolated worker own the handle
    // until process exit; unloading it after GetWorkspaceSize is unsafe.
    static ScatterElementsPrivateApi *api = nullptr;
    if (api != nullptr) {
        if (api->path != libraryPath) {
            throw std::runtime_error("private ScatterElementsV2 OpAPI path changed within one worker");
        }
        return *api;
    }
    Stage("private_scatter_opapi_load_begin", libraryPath);
    auto *loaded = new ScatterElementsPrivateApi();
    loaded->path = libraryPath;
    loaded->library = dlopen(libraryPath, RTLD_LAZY | RTLD_LOCAL);
    if (loaded->library == nullptr) {
        const char *error = dlerror();
        const std::string message = error == nullptr ? "unknown dlopen error" : error;
        delete loaded;
        throw std::runtime_error("cannot load private ScatterElementsV2 C++ OpAPI: " + message);
    }
    dlerror();
    loaded->getWorkspace = reinterpret_cast<ScatterElementsGetWorkspace>(
        dlsym(loaded->library, "aclnnScatterEleGetWorkspaceSize"));
    const char *workspaceError = dlerror();
    if (loaded->getWorkspace == nullptr || workspaceError != nullptr) {
        throw std::runtime_error("private ScatterElementsV2 GetWorkspaceSize symbol is absent");
    }
    dlerror();
    loaded->launch = reinterpret_cast<ScatterElementsLaunch>(
        dlsym(loaded->library, "aclnnScatterEle"));
    const char *launchError = dlerror();
    if (loaded->launch == nullptr || launchError != nullptr) {
        throw std::runtime_error("private ScatterElementsV2 launch symbol is absent");
    }
    Stage("private_scatter_opapi_load_done");
    api = loaded;
    return *api;
}
#endif

void DeviceException(aclrtExceptionInfo *info)
{
    std::cout << "MULTIOP_NPU_EXCEPTION {\"workload_id\":\"" << JsonEscape(gWorkloadId)
              << "\",\"device_id\":" << aclrtGetDeviceIdFromExceptionInfo(info)
              << ",\"stream_id\":" << aclrtGetStreamIdFromExceptionInfo(info)
              << ",\"task_id\":" << aclrtGetTaskIdFromExceptionInfo(info)
              << ",\"thread_id\":" << aclrtGetThreadIdFromExceptionInfo(info)
              << ",\"error_code\":" << aclrtGetErrorCodeFromExceptionInfo(info) << "}" << std::endl;
}

void ReportPackage(aclCANNPackageName name, const std::string &label, bool required)
{
    aclCANNPackageVersion version{};
    const aclError rc = aclsysGetCANNVersion(name, &version);
    if (rc != ACL_SUCCESS) {
        Stage("cann_package_unavailable", label + ",rc=" + std::to_string(rc));
        if (required) CheckAcl(rc, "aclsysGetCANNVersion " + label);
        return;
    }
    Stage("cann_package_version", label + "=" + std::string(version.version));
}

void CheckAcl(aclError rc, const std::string &what)
{
    if (rc != ACL_SUCCESS) {
        const char *recent = aclGetRecentErrMsg();
        throw std::runtime_error(
            what + " failed, rc=" + std::to_string(rc) +
            (recent == nullptr ? "" : ", recent=" + std::string(recent)));
    }
}

void CheckAclnn(aclnnStatus rc, const std::string &what)
{
    if (rc != 0) {
        throw std::runtime_error(what + " failed, rc=" + std::to_string(rc));
    }
}

std::string JsonEscape(const std::string &value)
{
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

class Arguments {
public:
    Arguments(int argc, char **argv)
    {
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0) {
                throw std::runtime_error("arguments must be --name value pairs");
            }
            values_[std::string(argv[i]).substr(2)] = argv[i + 1];
        }
    }

    std::string Get(const std::string &name) const
    {
        auto it = values_.find(name);
        if (it == values_.end()) throw std::runtime_error("missing --" + name);
        return it->second;
    }

    bool Has(const std::string &name) const
    {
        return values_.find(name) != values_.end();
    }

    int64_t Int(const std::string &name) const
    {
        return std::stoll(Get(name));
    }

    std::vector<int64_t> IntList(const std::string &name, bool requirePositive) const
    {
        const std::string text = Get(name);
        std::vector<int64_t> result;
        std::stringstream input(text);
        std::string token;
        while (std::getline(input, token, ',')) {
            if (token.empty()) throw std::runtime_error("empty dimension in --" + name);
            const int64_t value = std::stoll(token);
            if (requirePositive && value <= 0) throw std::runtime_error("non-positive dimension in --" + name);
            result.push_back(value);
        }
        if (result.empty()) throw std::runtime_error("empty shape in --" + name);
        return result;
    }

    std::vector<int64_t> Shape(const std::string &name) const
    {
        return IntList(name, true);
    }

private:
    std::map<std::string, std::string> values_;
};

size_t ElementBytes(aclDataType dtype)
{
    switch (dtype) {
        case ACL_INT8:
        case ACL_UINT8: return 1;
        case ACL_FLOAT16:
        case ACL_BF16: return 2;
        case ACL_FLOAT:
        case ACL_INT32: return 4;
        case ACL_INT64: return 8;
        default: throw std::runtime_error("unsupported acl dtype");
    }
}

aclDataType DType(const std::string &name)
{
    if (name == "fp16") return ACL_FLOAT16;
    if (name == "bf16") return ACL_BF16;
    if (name == "fp32") return ACL_FLOAT;
    if (name == "int8") return ACL_INT8;
    if (name == "int32") return ACL_INT32;
    if (name == "int64") return ACL_INT64;
    throw std::runtime_error("unsupported dtype: " + name);
}

uint64_t Elements(const std::vector<int64_t> &shape)
{
    uint64_t result = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || result > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            throw std::runtime_error("tensor element count overflow");
        }
        result *= static_cast<uint64_t>(dim);
    }
    return result;
}

std::vector<int64_t> Strides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;) strides[i - 1] = strides[i] * shape[i];
    return strides;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(uint64_t elements, aclDataType dtype)
    {
        const uint64_t bytes = elements * ElementBytes(dtype);
        if (elements != 0 && bytes / elements != ElementBytes(dtype)) {
            throw std::runtime_error("device allocation overflow");
        }
        bytes_ = static_cast<size_t>(bytes);
        if (bytes_ > 0) {
            CheckAcl(aclrtMalloc(&ptr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
            CheckAcl(aclrtMemset(ptr_, bytes_, 0, bytes_), "aclrtMemset");
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) aclrtFree(ptr_);
    }
    void *Data() const { return ptr_; }
    size_t Bytes() const { return bytes_; }

    template <class T>
    void CopyFrom(const std::vector<T> &host)
    {
        const size_t bytes = host.size() * sizeof(T);
        if (bytes != bytes_) throw std::runtime_error("host/device tensor byte mismatch");
        CheckAcl(aclrtMemcpy(ptr_, bytes_, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
    }

    template <class T>
    std::vector<T> CopyTo() const
    {
        if (bytes_ % sizeof(T) != 0) throw std::runtime_error("device/host tensor byte mismatch");
        std::vector<T> host(bytes_ / sizeof(T));
        CheckAcl(aclrtMemcpy(host.data(), bytes_, ptr_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
        return host;
    }

private:
    void *ptr_ = nullptr;
    size_t bytes_ = 0;
};

class Tensor {
public:
    Tensor(DeviceBuffer &buffer, aclDataType dtype, const std::vector<int64_t> &shape,
           const std::vector<int64_t> &strides = {}, const std::vector<int64_t> &storage = {})
    {
        const auto actualStrides = strides.empty() ? Strides(shape) : strides;
        const auto actualStorage = storage.empty() ? shape : storage;
        ptr_ = aclCreateTensor(
            shape.data(), shape.size(), dtype, actualStrides.data(), 0, ACL_FORMAT_ND,
            actualStorage.data(), actualStorage.size(), buffer.Data());
        if (ptr_ == nullptr) throw std::runtime_error("aclCreateTensor returned null");
    }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    ~Tensor()
    {
        if (ptr_ != nullptr) aclDestroyTensor(ptr_);
    }
    aclTensor *Get() const { return ptr_; }

private:
    aclTensor *ptr_ = nullptr;
};

class IntArray {
public:
    explicit IntArray(const std::vector<int64_t> &values)
    {
        ptr_ = aclCreateIntArray(values.data(), values.size());
        if (ptr_ == nullptr) throw std::runtime_error("aclCreateIntArray returned null");
    }
    ~IntArray()
    {
        if (ptr_ != nullptr) aclDestroyIntArray(ptr_);
    }
    const aclIntArray *Get() const { return ptr_; }

private:
    aclIntArray *ptr_ = nullptr;
};

class TensorList {
public:
    explicit TensorList(const std::vector<const aclTensor *> &tensors)
    {
        ptr_ = aclCreateTensorList(tensors.data(), tensors.size());
        if (ptr_ == nullptr) throw std::runtime_error("aclCreateTensorList returned null");
    }
    ~TensorList()
    {
        if (ptr_ != nullptr) aclDestroyTensorList(ptr_);
    }
    const aclTensorList *Get() const { return ptr_; }

private:
    aclTensorList *ptr_ = nullptr;
};

class Executor {
public:
    ~Executor()
    {
        // For a normal (non-repeatable) ACLNN executor, CANN consumes and
        // releases it in the second-stage aclnnXxx call.  It is ours to
        // destroy only when GetWorkspaceSize succeeded but no one-shot
        // launch consumed the executor (for example, a failed launch).
        if (ptr != nullptr) aclDestroyAclOpExecutor(ptr);
    }
    void MarkConsumedByOneShotLaunch() { ptr = nullptr; }
    void ReleaseToIsolatedProcessExit() { ptr = nullptr; }
    aclOpExecutor *ptr = nullptr;
};

struct Measurement {
    uint64_t workspaceBytes = 0;
    std::vector<double> samplesMs;
    uint64_t probeBytes = 0;
    uint64_t probeNonzeroBytes = 0;
    uint64_t outputBytes = 0;
    uint64_t outputDigest = 0;
    bool outputReferenceChecked = false;
    bool outputReferenceEqual = false;
};

uint64_t Fnv1a64(const std::vector<uint8_t> &data)
{
    uint64_t value = 1469598103934665603ULL;
    for (uint8_t byte : data) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

void FillAttentionPattern(DeviceBuffer &buffer, aclDataType dtype, uint64_t count, uint32_t pattern)
{
    if (dtype == ACL_FLOAT) {
        std::vector<float> host(count);
        for (uint64_t index = 0; index < count; ++index) {
            host[index] = (static_cast<float>((index + pattern) % 7U) + 1.0F) / 16.0F;
        }
        buffer.CopyFrom(host);
        return;
    }
    if (dtype == ACL_FLOAT16 || dtype == ACL_BF16) {
        // These are finite, positive, normal values in the two IEEE-like
        // encodings. The input pattern is for correctness comparison only; it
        // is never included in a timing interval.
        const uint16_t fp16[] = {0x3000U, 0x3400U, 0x3800U, 0x3A00U, 0x3C00U, 0x3D00U, 0x3E00U};
        const uint16_t bf16[] = {0x3E00U, 0x3E40U, 0x3E80U, 0x3EA0U, 0x3EC0U, 0x3F00U, 0x3F20U};
        const uint16_t *table = dtype == ACL_FLOAT16 ? fp16 : bf16;
        std::vector<uint16_t> host(count);
        for (uint64_t index = 0; index < count; ++index) host[index] = table[(index + pattern) % 7U];
        buffer.CopyFrom(host);
        return;
    }
    throw std::runtime_error("attention pattern requires fp16, bf16, or fp32");
}

void FillGeneralPattern(DeviceBuffer &buffer, aclDataType dtype, uint64_t count, uint32_t pattern)
{
    if (dtype == ACL_FLOAT || dtype == ACL_FLOAT16 || dtype == ACL_BF16) {
        FillAttentionPattern(buffer, dtype, count, pattern);
        return;
    }
    if (dtype == ACL_INT32) {
        std::vector<int32_t> values(count);
        for (uint64_t index = 0; index < count; ++index) values[index] = static_cast<int32_t>((index + pattern) % 251U);
        buffer.CopyFrom(values);
        return;
    }
    if (dtype == ACL_INT64) {
        std::vector<int64_t> values(count);
        for (uint64_t index = 0; index < count; ++index) values[index] = static_cast<int64_t>((index + pattern) % 251U);
        buffer.CopyFrom(values);
        return;
    }
    if (dtype == ACL_INT8) {
        std::vector<int8_t> values(count);
        for (uint64_t index = 0; index < count; ++index) values[index] = static_cast<int8_t>((index + pattern) % 97U);
        buffer.CopyFrom(values);
        return;
    }
    throw std::runtime_error("unsupported deterministic tensor pattern dtype");
}

std::vector<uint8_t> SnapshotOutputs(const std::vector<const DeviceBuffer *> &outputs)
{
    std::vector<uint8_t> all;
    for (const DeviceBuffer *output : outputs) {
        const std::vector<uint8_t> part = output->CopyTo<uint8_t>();
        const uint64_t size = static_cast<uint64_t>(part.size());
        for (int byte = 0; byte < 8; ++byte) all.push_back(static_cast<uint8_t>((size >> (byte * 8)) & 0xFFU));
        all.insert(all.end(), part.begin(), part.end());
    }
    return all;
}

void WriteReference(const std::string &path, const std::vector<uint8_t> &data)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create output reference: " + path);
    stream.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!stream) throw std::runtime_error("cannot write output reference: " + path);
}

std::vector<uint8_t> ReadReference(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot read output reference: " + path);
    const std::streamsize size = stream.tellg();
    if (size < 0) throw std::runtime_error("cannot determine output reference size: " + path);
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size > 0 && !stream.read(reinterpret_cast<char *>(data.data()), size)) {
        throw std::runtime_error("cannot read complete output reference: " + path);
    }
    return data;
}

template <class GetWorkspace, class Launch>
Measurement Measure(aclrtStream stream, int warmup, int samples, DeviceBuffer &output,
                    GetWorkspace getWorkspace, Launch launch)
{
    Measurement result;
    if (samples == 0) {
        Executor executor;
        Stage("verification_get_workspace_begin");
        CheckAclnn(getWorkspace(&result.workspaceBytes, &executor.ptr), "GetWorkspaceSize verification");
        if (executor.ptr == nullptr) throw std::runtime_error("verification GetWorkspaceSize returned null executor");
        DeviceBuffer workspace(result.workspaceBytes, ACL_UINT8);
        Stage("verification_get_workspace_done", "workspace_bytes=" + std::to_string(result.workspaceBytes));
        Stage("verification_launch_begin");
        CheckAclnn(launch(workspace.Data(), result.workspaceBytes, executor.ptr, stream), "operator verification launch");
        executor.MarkConsumedByOneShotLaunch();
        Stage("verification_launch_returned");
        Stage("verification_sync_begin");
        // This is an actual viability launch, not a host watchdog probe.
        // Do not inject a timeout that turns a live device operation into a
        // host-forced failure; process isolation is the containment boundary.
        CheckAcl(aclrtSynchronizeStream(stream), "verification stream synchronize");
        Stage("verification_sync_done");
        return result;
    }
    for (int i = 0; i < warmup; ++i) {
        Executor executor;
        uint64_t workspaceBytes = 0;
        Stage("warmup_get_workspace_begin", "index=" + std::to_string(i));
        CheckAclnn(getWorkspace(&workspaceBytes, &executor.ptr), "GetWorkspaceSize warmup");
        if (executor.ptr == nullptr) throw std::runtime_error("warmup GetWorkspaceSize returned null executor");
        result.workspaceBytes = std::max(result.workspaceBytes, workspaceBytes);
        DeviceBuffer workspace(workspaceBytes, ACL_UINT8);
        Stage("warmup_get_workspace_done", "workspace_bytes=" + std::to_string(workspaceBytes));
        Stage("warmup_launch_begin", "index=" + std::to_string(i));
        CheckAclnn(launch(workspace.Data(), workspaceBytes, executor.ptr, stream), "operator warmup launch");
        executor.MarkConsumedByOneShotLaunch();
        Stage("warmup_launch_returned", "index=" + std::to_string(i));
        Stage("warmup_sync_begin", "index=" + std::to_string(i));
        CheckAcl(aclrtSynchronizeStream(stream), "warmup synchronize");
        Stage("warmup_sync_done", "index=" + std::to_string(i));
    }

    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    Stage("event_create_begin");
    CheckAcl(aclrtCreateEvent(&start), "aclrtCreateEvent start");
    try {
        CheckAcl(aclrtCreateEvent(&end), "aclrtCreateEvent end");
        Stage("event_create_done");
        for (int i = 0; i < samples; ++i) {
            Executor executor;
            uint64_t workspaceBytes = 0;
            Stage("sample_get_workspace_begin", "index=" + std::to_string(i));
            CheckAclnn(getWorkspace(&workspaceBytes, &executor.ptr), "GetWorkspaceSize sample");
            if (executor.ptr == nullptr) throw std::runtime_error("sample GetWorkspaceSize returned null executor");
            result.workspaceBytes = std::max(result.workspaceBytes, workspaceBytes);
            DeviceBuffer workspace(workspaceBytes, ACL_UINT8);
            Stage("sample_get_workspace_done", "workspace_bytes=" + std::to_string(workspaceBytes));
            Stage("sample_begin", "index=" + std::to_string(i));
            CheckAcl(aclrtRecordEvent(start, stream), "aclrtRecordEvent start");
            CheckAclnn(launch(workspace.Data(), workspaceBytes, executor.ptr, stream), "operator sample launch");
            executor.MarkConsumedByOneShotLaunch();
            Stage("sample_launch_returned", "index=" + std::to_string(i));
            CheckAcl(aclrtRecordEvent(end, stream), "aclrtRecordEvent end");
            Stage("sample_stream_sync_begin", "index=" + std::to_string(i));
            CheckAcl(aclrtSynchronizeStream(stream), "sample stream synchronize");
            Stage("sample_stream_sync_done", "index=" + std::to_string(i));
            float elapsedMs = 0.0F;
            CheckAcl(aclrtEventElapsedTime(&elapsedMs, start, end), "aclrtEventElapsedTime");
            if (!std::isfinite(elapsedMs) || elapsedMs < 0.0F) {
                throw std::runtime_error("invalid device event time");
            }
            result.samplesMs.push_back(elapsedMs);
        }
        aclrtDestroyEvent(end);
        aclrtDestroyEvent(start);
    } catch (...) {
        if (end != nullptr) aclrtDestroyEvent(end);
        if (start != nullptr) aclrtDestroyEvent(start);
        throw;
    }
    const size_t probeBytes = std::min<size_t>(output.Bytes(), 4096);
    std::vector<uint8_t> probe(probeBytes);
    if (probeBytes > 0) {
        Stage("output_probe_begin");
        CheckAcl(aclrtMemcpy(probe.data(), probe.size(), output.Data(), probeBytes,
                             ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy output probe");
        Stage("output_probe_done");
    }
    result.probeBytes = probeBytes;
    result.probeNonzeroBytes = std::count_if(probe.begin(), probe.end(), [](uint8_t value) { return value != 0; });
    return result;
}

int64_t NormalizeAxis(int64_t axis, size_t rank)
{
    if (axis < 0) axis += static_cast<int64_t>(rank);
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) throw std::runtime_error("axis is out of range");
    return axis;
}

void FillIndices(DeviceBuffer &buffer, aclDataType dtype, uint64_t count, int64_t axisExtent)
{
    if (axisExtent <= 0) throw std::runtime_error("invalid index axis extent");
    if (dtype == ACL_INT32) {
        std::vector<int32_t> data(count);
        for (uint64_t i = 0; i < count; ++i) data[i] = static_cast<int32_t>(i % axisExtent);
        buffer.CopyFrom(data);
    } else if (dtype == ACL_INT64) {
        std::vector<int64_t> data(count);
        for (uint64_t i = 0; i < count; ++i) data[i] = static_cast<int64_t>(i % axisExtent);
        buffer.CopyFrom(data);
    } else {
        throw std::runtime_error("index dtype must be int32 or int64");
    }
}

std::vector<uint8_t> GatherElementsHostReference(DeviceBuffer &input, DeviceBuffer &index,
                                                 aclDataType dtype, aclDataType indexDtype,
                                                 const std::vector<int64_t> &shape,
                                                 const std::vector<int64_t> &indexShape,
                                                 int64_t axis)
{
    const std::vector<uint8_t> inputBytes = input.CopyTo<uint8_t>();
    const uint64_t outputElements = Elements(indexShape);
    std::vector<int64_t> indices(outputElements);
    if (indexDtype == ACL_INT32) {
        const std::vector<int32_t> values = index.CopyTo<int32_t>();
        std::copy(values.begin(), values.end(), indices.begin());
    } else if (indexDtype == ACL_INT64) {
        indices = index.CopyTo<int64_t>();
    } else {
        throw std::runtime_error("GatherElements reference index dtype must be int32 or int64");
    }
    const std::vector<int64_t> inputStrides = Strides(shape);
    const std::vector<int64_t> outputStrides = Strides(indexShape);
    const size_t elementBytes = ElementBytes(dtype);
    std::vector<uint8_t> outputBytes(static_cast<size_t>(outputElements) * elementBytes);
    for (uint64_t linear = 0; linear < outputElements; ++linear) {
        uint64_t inputOffset = 0;
        for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
            int64_t coordinate = static_cast<int64_t>(
                (linear / static_cast<uint64_t>(outputStrides[dimension])) %
                static_cast<uint64_t>(indexShape[dimension]));
            if (static_cast<int64_t>(dimension) == axis) coordinate = indices[linear];
            if (coordinate < 0 || coordinate >= shape[dimension]) {
                throw std::runtime_error("GatherElements deterministic reference index is out of range");
            }
            inputOffset += static_cast<uint64_t>(coordinate * inputStrides[dimension]);
        }
        std::memcpy(outputBytes.data() + linear * elementBytes,
                    inputBytes.data() + inputOffset * elementBytes, elementBytes);
    }
    std::vector<uint8_t> snapshot;
    const uint64_t byteCount = static_cast<uint64_t>(outputBytes.size());
    snapshot.reserve(8 + outputBytes.size());
    for (int byte = 0; byte < 8; ++byte) {
        snapshot.push_back(static_cast<uint8_t>((byteCount >> (byte * 8)) & 0xFFU));
    }
    snapshot.insert(snapshot.end(), outputBytes.begin(), outputBytes.end());
    return snapshot;
}

#ifndef SCATTER_ELEMENTS_ONLY
Measurement RunMatmul(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const int64_t m = args.Int("m"), n = args.Int("n"), k = args.Int("k");
    const bool transA = args.Int("trans-a") != 0, transB = args.Int("trans-b") != 0;
    const aclDataType dtype = DType(args.Get("dtype"));
    DeviceBuffer a(static_cast<uint64_t>(m) * k, dtype);
    DeviceBuffer b(static_cast<uint64_t>(k) * n, dtype);
    DeviceBuffer c(static_cast<uint64_t>(m) * n, dtype);
    const std::vector<int64_t> aView{m, k}, bView{k, n}, cView{m, n};
    const std::vector<int64_t> aStrides = transA ? std::vector<int64_t>{1, m} : std::vector<int64_t>{k, 1};
    const std::vector<int64_t> bStrides = transB ? std::vector<int64_t>{1, k} : std::vector<int64_t>{n, 1};
    const std::vector<int64_t> aStorage = transA ? std::vector<int64_t>{k, m} : aView;
    const std::vector<int64_t> bStorage = transB ? std::vector<int64_t>{n, k} : bView;
    Tensor aTensor(a, dtype, aView, aStrides, aStorage);
    Tensor bTensor(b, dtype, bView, bStrides, bStorage);
    Tensor cTensor(c, dtype, cView);
    return Measure(stream, warmup, samples, c,
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnMatmulGetWorkspaceSize(aTensor.Get(), bTensor.Get(), cTensor.Get(), 0, workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnMatmul(workspace, bytes, executor, launchStream);
        });
}

Measurement RunTranspose(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const auto shape = args.Shape("shape");
    const auto perm = args.IntList("perm", false);
    if (shape.size() != perm.size()) throw std::runtime_error("shape/perm rank mismatch");
    std::vector<int64_t> outputShape(shape.size());
    std::vector<bool> seen(shape.size(), false);
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(shape.size()) || seen[perm[i]]) {
            throw std::runtime_error("invalid permutation");
        }
        seen[perm[i]] = true;
        outputShape[i] = shape[perm[i]];
    }
    const aclDataType dtype = DType(args.Get("dtype"));
    DeviceBuffer input(Elements(shape), dtype), output(Elements(outputShape), dtype);
    const bool officialVerification = samples == 0 && dtype == ACL_FLOAT &&
        shape == std::vector<int64_t>({4, 2}) && perm == std::vector<int64_t>({1, 0});
    if (officialVerification) {
        input.CopyFrom(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    }
    Tensor inputTensor(input, dtype, shape), outputTensor(output, dtype, outputShape);
    IntArray dims(perm);
    Measurement result = Measure(stream, warmup, samples, output,
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnPermuteGetWorkspaceSize(inputTensor.Get(), dims.Get(), outputTensor.Get(), workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnPermute(workspace, bytes, executor, launchStream);
        });
    if (officialVerification) {
        const std::vector<float> expected{1, 3, 5, 7, 2, 4, 6, 8};
        if (output.CopyTo<float>() != expected) throw std::runtime_error("official transpose output mismatch");
        Stage("verification_output_exact", "official_4x2_fp32=pass");
    }
    return result;
}

Measurement RunGatherV2(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const auto shape = args.Shape("shape"), indexShape = args.Shape("index-shape");
    const int64_t axis = NormalizeAxis(args.Int("axis"), shape.size());
    const aclDataType dtype = DType(args.Get("dtype")), indexDtype = DType(args.Get("index-dtype"));
    std::vector<int64_t> outputShape;
    outputShape.insert(outputShape.end(), shape.begin(), shape.begin() + axis);
    outputShape.insert(outputShape.end(), indexShape.begin(), indexShape.end());
    outputShape.insert(outputShape.end(), shape.begin() + axis + 1, shape.end());
    DeviceBuffer input(Elements(shape), dtype), index(Elements(indexShape), indexDtype), output(Elements(outputShape), dtype);
    FillIndices(index, indexDtype, Elements(indexShape), shape[axis]);
    Tensor inputTensor(input, dtype, shape), indexTensor(index, indexDtype, indexShape), outputTensor(output, dtype, outputShape);
    return Measure(stream, warmup, samples, output,
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnGatherV2GetWorkspaceSize(inputTensor.Get(), axis, indexTensor.Get(), outputTensor.Get(), workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnGatherV2(workspace, bytes, executor, launchStream);
        });
}

Measurement RunGatherElements(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const auto shape = args.Shape("shape"), indexShape = args.Shape("index-shape");
    if (shape.size() != indexShape.size()) throw std::runtime_error("gather-elements rank mismatch");
    const int64_t axis = NormalizeAxis(args.Int("axis"), shape.size());
    const aclDataType dtype = DType(args.Get("dtype")), indexDtype = DType(args.Get("index-dtype"));
    DeviceBuffer input(Elements(shape), dtype), index(Elements(indexShape), indexDtype), output(Elements(indexShape), dtype);
    FillGeneralPattern(input, dtype, Elements(shape), 0);
    FillIndices(index, indexDtype, Elements(indexShape), shape[axis]);
    const bool needsReference = args.Has("write-reference") || args.Has("compare-reference");
    const std::vector<uint8_t> exactReference = needsReference
        ? GatherElementsHostReference(input, index, dtype, indexDtype, shape, indexShape, axis)
        : std::vector<uint8_t>{};
    // A private source candidate explicitly opts into generic OPP dispatch.
    // The reference has neither variable and therefore cannot accidentally
    // take this path. An audit path without the dispatch contract is an error,
    // never a silent ACLNN fallback.
    const char *dispatch = std::getenv("GATHER_ELEMENTS_SOURCE_DISPATCH");
    const bool auditRequested = std::getenv("GATHER_ELEMENTS_TILING_AUDIT_PATH") != nullptr;
    const bool useCustomSource = dispatch != nullptr;
    if (useCustomSource && std::string(dispatch) != "cann81_prebuilt_aclnn") {
        throw std::runtime_error("invalid GATHER_ELEMENTS_SOURCE_DISPATCH");
    }
    if (auditRequested != useCustomSource) {
        throw std::runtime_error("GatherElements source audit/dispatch environment mismatch");
    }
    if (useCustomSource) {
        const char *operatorType = std::getenv("GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE");
        if (operatorType == nullptr || std::string(operatorType) != kGatherElementsSourceOperatorType) {
            throw std::runtime_error("private GatherElementsV2 package operator-type environment mismatch");
        }
        const char *libraryPath = std::getenv("GATHER_ELEMENTS_SOURCE_OPAPI_LIBRARY");
        if (libraryPath == nullptr || *libraryPath == '\0') {
            throw std::runtime_error("private GatherElementsV2 C++ OpAPI path is absent");
        }
        const GatherElementsPrivateApi &api = LoadGatherElementsPrivateApi(libraryPath);
        gBackend = "private_cann81_prebuilt_ascendc_aclnn_real_npu";
        if (args.Has("source-load-only")) {
            Stage("private_opapi_load_probe_done");
            return Measurement{};
        }
        Tensor inputTensor(input, dtype, shape), indexTensor(index, indexDtype, indexShape), outputTensor(output, dtype, indexShape);
        const auto getWorkspace = [&](uint64_t *workspace, aclOpExecutor **executor) {
            return api.getWorkspace(inputTensor.Get(), indexTensor.Get(), axis, outputTensor.Get(), workspace, executor);
        };
        const auto launch = [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return api.launch(workspace, bytes, executor, launchStream);
        };
        if (args.Has("source-host-tiling-only") || args.Has("source-executor-planning-only")) {
            Measurement result;
            Executor executor;
            Stage("private_host_tiling_get_workspace_begin");
            CheckAclnn(getWorkspace(&result.workspaceBytes, &executor.ptr),
                       "private GatherElementsV2 GetWorkspaceSize probe");
            if (executor.ptr == nullptr) {
                throw std::runtime_error("private GatherElementsV2 GetWorkspaceSize returned null executor");
            }
            Stage("private_host_tiling_get_workspace_done",
                  "workspace_bytes=" + std::to_string(result.workspaceBytes));
            // This probe deliberately stops between ACLNN's two stages.  Its
            // isolated process exits immediately, so leave Nnopbase's
            // executor space process-owned instead of invoking a cleanup path
            // that a normal GetWorkspace+launch operation never takes.
            executor.ReleaseToIsolatedProcessExit();
            Stage("private_host_tiling_probe_done");
            return result;
        }
        Measurement result;
        result = Measure(stream, args.Has("source-tiling-only") ? 0 : warmup,
                         args.Has("source-tiling-only") ? 0 : samples, output,
                         getWorkspace, launch);
        Stage("private_precompiled_kernel_execution_done");
        const std::vector<uint8_t> snapshot = SnapshotOutputs({&output});
        result.outputBytes = snapshot.size();
        result.outputDigest = Fnv1a64(snapshot);
        if (args.Has("write-reference")) WriteReference(args.Get("write-reference"), snapshot);
        if (args.Has("compare-reference")) {
            result.outputReferenceChecked = true;
            result.outputReferenceEqual = snapshot == ReadReference(args.Get("compare-reference"));
            Stage("private_output_compare_done", result.outputReferenceEqual ? "equal=true" : "equal=false");
        }
        return result;
    }

    // CANN 8.1 exposes aclnnGather but not a public aclnnGatherElements API.
    // Rank-one Gather and GatherElements are equivalent, so preflight still
    // proves the installed NPU route on the first workload.  Higher-rank
    // reference files are generated from the exact deterministic tensor
    // coordinates; this host work is never inside a device-event interval.
    if (args.Has("write-reference") && shape.size() != 1) {
        WriteReference(args.Get("write-reference"), exactReference);
        Measurement result;
        result.outputBytes = exactReference.size();
        result.outputDigest = Fnv1a64(exactReference);
        gBackend = "host_exact_gather_elements_reference";
        Stage("gather_elements_host_reference_done");
        return result;
    }

    Tensor inputTensor(input, dtype, shape), indexTensor(index, indexDtype, indexShape), outputTensor(output, dtype, indexShape);
    const auto execute = [&](const auto &getWorkspace, const auto &launch) {
        if (args.Has("source-tiling-only")) {
            // A native-source audit counts only when the same official ACLNN
            // call launches and synchronizes once through the private OPP
            // overlay.  The one-shot launch consumes its executor.
            return Measure(stream, 0, 0, output, getWorkspace, launch);
        }
        Measurement result = Measure(stream, warmup, samples, output, getWorkspace, launch);
        const std::vector<uint8_t> snapshot = SnapshotOutputs({&output});
        result.outputBytes = snapshot.size();
        result.outputDigest = Fnv1a64(snapshot);
        if (args.Has("write-reference")) {
            if (snapshot != exactReference) throw std::runtime_error("installed rank-one Gather reference mismatch");
            WriteReference(args.Get("write-reference"), exactReference);
        }
        if (args.Has("compare-reference")) {
            result.outputReferenceChecked = true;
            result.outputReferenceEqual = snapshot == ReadReference(args.Get("compare-reference"));
        }
        return result;
    };

    // ``aclnnGather`` is intentionally used only for the installed reference.
    // It is an OpAPI entry point, not the dynamic custom-OPP dispatch path.
    return execute(
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnGatherGetWorkspaceSize(inputTensor.Get(), axis, indexTensor.Get(), outputTensor.Get(), workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnGather(workspace, bytes, executor, launchStream);
        });
}
#endif

#ifndef REMAINING_OPERATOR_ONLY
Measurement RunScatterElements(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const auto shape = args.Shape("shape"), indexShape = args.Shape("index-shape");
    if (shape.size() != indexShape.size()) throw std::runtime_error("scatter-elements rank mismatch");
    const int64_t axis = NormalizeAxis(args.Int("axis"), shape.size());
    const int64_t reduce = args.Int("reduce");
    const aclDataType dtype = DType(args.Get("dtype")), indexDtype = DType(args.Get("index-dtype"));
    DeviceBuffer input(Elements(shape), dtype), index(Elements(indexShape), indexDtype);
    DeviceBuffer source(Elements(indexShape), dtype), output(Elements(shape), dtype);
    FillGeneralPattern(input, dtype, Elements(shape), 0);
    FillGeneralPattern(source, dtype, Elements(indexShape), 17);
    FillIndices(index, indexDtype, Elements(indexShape), shape[axis]);
    Tensor inputTensor(input, dtype, shape), indexTensor(index, indexDtype, indexShape);
    Tensor sourceTensor(source, dtype, indexShape), outputTensor(output, dtype, shape);
    const char *dispatch = std::getenv("SCATTER_ELEMENTS_SOURCE_DISPATCH");
    const bool auditRequested = std::getenv("SCATTER_ELEMENTS_TILING_AUDIT_PATH") != nullptr;
    const bool usePrivateSource = dispatch != nullptr;
    if (usePrivateSource && std::string(dispatch) != "cann81_native_aclnn") {
        throw std::runtime_error("invalid SCATTER_ELEMENTS_SOURCE_DISPATCH");
    }
    if (auditRequested != usePrivateSource) {
        throw std::runtime_error("ScatterElements source audit/dispatch environment mismatch");
    }

    ScatterElementsGetWorkspace privateGetWorkspace = nullptr;
    ScatterElementsLaunch privateLaunch = nullptr;
    if (usePrivateSource) {
        const char *libraryPath = std::getenv("SCATTER_ELEMENTS_SOURCE_OPAPI_LIBRARY");
        if (libraryPath == nullptr || *libraryPath == '\0') {
            throw std::runtime_error("private ScatterElementsV2 C++ OpAPI path is absent");
        }
        const ScatterElementsPrivateApi &api = LoadScatterElementsPrivateApi(libraryPath);
        privateGetWorkspace = api.getWorkspace;
        privateLaunch = api.launch;
        gBackend = "private_cann81_native_scatter_aclnn_real_npu";
        if (args.Has("source-load-only")) {
            Stage("private_scatter_opapi_load_probe_done");
            return Measurement{};
        }
    }

    const auto getWorkspace = [&](uint64_t *workspace, aclOpExecutor **executor) {
        if (usePrivateSource) {
            return privateGetWorkspace(inputTensor.Get(), axis, indexTensor.Get(), sourceTensor.Get(), reduce,
                                       outputTensor.Get(), workspace, executor);
        }
        return aclnnScatterGetWorkspaceSize(inputTensor.Get(), axis, indexTensor.Get(), sourceTensor.Get(), reduce,
                                            outputTensor.Get(), workspace, executor);
    };
    const auto launch = [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
        return usePrivateSource ? privateLaunch(workspace, bytes, executor, launchStream)
                                : aclnnScatter(workspace, bytes, executor, launchStream);
    };
    if (args.Has("source-executor-planning-only")) {
        Measurement result;
        Executor executor;
        Stage("private_scatter_executor_planning_begin");
        CheckAclnn(getWorkspace(&result.workspaceBytes, &executor.ptr),
                   "private ScatterElementsV2 GetWorkspaceSize probe");
        if (executor.ptr == nullptr) {
            throw std::runtime_error("private ScatterElementsV2 GetWorkspaceSize returned null executor");
        }
        executor.ReleaseToIsolatedProcessExit();
        Stage("private_scatter_executor_planning_done", "workspace_bytes=" + std::to_string(result.workspaceBytes));
        return result;
    }
    Measurement result = Measure(stream, args.Has("source-tiling-only") ? 0 : warmup,
                                 args.Has("source-tiling-only") ? 0 : samples,
                                 output, getWorkspace, launch);
    const std::vector<uint8_t> snapshot = SnapshotOutputs({&output});
    result.outputBytes = snapshot.size();
    result.outputDigest = Fnv1a64(snapshot);
    if (args.Has("write-reference")) WriteReference(args.Get("write-reference"), snapshot);
    if (args.Has("compare-reference")) {
        result.outputReferenceChecked = true;
        result.outputReferenceEqual = snapshot == ReadReference(args.Get("compare-reference"));
    }
    return result;
}
#endif

#ifndef SCATTER_ELEMENTS_ONLY
std::vector<int64_t> AttentionShape(const std::string &layout, int64_t batch, int64_t heads,
                                    int64_t sequence, int64_t headDim)
{
    if (layout == "BNSD") return {batch, heads, sequence, headDim};
    if (layout == "BSND") return {batch, sequence, heads, headDim};
    if (layout == "BSH") return {batch, sequence, heads * headDim};
    if (layout == "SBH") return {sequence, batch, heads * headDim};
    throw std::runtime_error("unsupported attention layout: " + layout);
}

Measurement RunFlashAttentionScoreGrad(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const int64_t batch = args.Int("batch"), qHeads = args.Int("q-heads"), kvHeads = args.Int("kv-heads");
    const int64_t qSeq = args.Int("q-seq"), kvSeq = args.Int("kv-seq"), headDim = args.Int("head-dim");
    const std::string layout = args.Get("layout");
    const aclDataType dtype = DType(args.Get("dtype"));
    const auto qShape = AttentionShape(layout, batch, qHeads, qSeq, headDim);
    const auto kvShape = AttentionShape(layout, batch, kvHeads, kvSeq, headDim);
    const std::vector<int64_t> softmaxShape{batch, qHeads, qSeq, 8};
    DeviceBuffer q(Elements(qShape), dtype), k(Elements(kvShape), dtype), v(Elements(kvShape), dtype);
    DeviceBuffer dy(Elements(qShape), dtype), attention(Elements(qShape), dtype);
    DeviceBuffer softmaxMax(Elements(softmaxShape), ACL_FLOAT), softmaxSum(Elements(softmaxShape), ACL_FLOAT);
    DeviceBuffer dq(Elements(qShape), dtype), dk(Elements(kvShape), dtype), dv(Elements(kvShape), dtype);
    // A non-zero deterministic input makes a cross-overlay comparison
    // meaningful. It is populated before warmup/measurement and is never
    // included in the device-event interval.
    FillAttentionPattern(q, dtype, Elements(qShape), 0);
    FillAttentionPattern(k, dtype, Elements(kvShape), 1);
    FillAttentionPattern(v, dtype, Elements(kvShape), 2);
    FillAttentionPattern(dy, dtype, Elements(qShape), 3);
    FillAttentionPattern(attention, dtype, Elements(qShape), 4);
    std::vector<float> maxValues(Elements(softmaxShape));
    for (uint64_t index = 0; index < maxValues.size(); ++index) {
        maxValues[index] = (static_cast<float>(index % 5U) + 1.0F) / 8.0F;
    }
    softmaxMax.CopyFrom(maxValues);
    std::vector<float> sumValues(Elements(softmaxShape), 1.0F);
    softmaxSum.CopyFrom(sumValues);
    Tensor qTensor(q, dtype, qShape), kTensor(k, dtype, kvShape), vTensor(v, dtype, kvShape);
    Tensor dyTensor(dy, dtype, qShape), attentionTensor(attention, dtype, qShape);
    Tensor maxTensor(softmaxMax, ACL_FLOAT, softmaxShape), sumTensor(softmaxSum, ACL_FLOAT, softmaxShape);
    Tensor dqTensor(dq, dtype, qShape), dkTensor(dk, dtype, kvShape), dvTensor(dv, dtype, kvShape);
    std::vector<char> layoutText(layout.begin(), layout.end());
    layoutText.push_back('\0');
    const double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    const char *dispatch = std::getenv("FASG_SOURCE_DISPATCH");
    const bool privateSource = dispatch != nullptr;
    const bool auditRequested = std::getenv("FASG_TILING_AUDIT_PATH") != nullptr;
    if (privateSource && std::string(dispatch) != "cann81_native_aclnn") throw std::runtime_error("invalid FASG_SOURCE_DISPATCH");
    if (privateSource != auditRequested) throw std::runtime_error("FASG source audit/dispatch mismatch");
    if (privateSource) {
        LoadRemainingPrivateLibrary(std::getenv("FASG_SOURCE_TILING_LIBRARY"),
                                    "FlashAttentionScoreGrad CANN-8.1 host tiler");
        gBackend = "private_cann81_fasg_host_tiler_installed_kernel_real_npu";
        if (args.Has("source-load-only")) return Measurement{};
    }
    const auto getWorkspace = [&](uint64_t *workspace, aclOpExecutor **executor) {
        return aclnnFlashAttentionScoreGradGetWorkspaceSize(
            qTensor.Get(), kTensor.Get(), vTensor.Get(), dyTensor.Get(),
            nullptr, nullptr, nullptr, nullptr, maxTensor.Get(), sumTensor.Get(), nullptr, attentionTensor.Get(),
            nullptr, scale, 1.0, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(),
            qHeads, layoutText.data(), 0, 0, dqTensor.Get(), dkTensor.Get(), dvTensor.Get(), nullptr,
            workspace, executor);
    };
    const auto launch = [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
        return aclnnFlashAttentionScoreGrad(workspace, bytes, executor, launchStream);
    };
    if (args.Has("source-executor-planning-only")) {
        Measurement result;
        Executor executor;
        CheckAclnn(getWorkspace(&result.workspaceBytes, &executor.ptr), "private FASG GetWorkspaceSize probe");
        if (executor.ptr == nullptr) throw std::runtime_error("private FASG returned null executor");
        executor.ReleaseToIsolatedProcessExit();
        return result;
    }
    // Source discovery performs one real installed-kernel launch/sync after
    // registering the selected instrumented CANN 8.1 host tiler.
    Measurement result = Measure(stream, args.Has("source-tiling-only") ? 0 : warmup,
                                 args.Has("source-tiling-only") ? 0 : samples, dq, getWorkspace, launch);
    const std::vector<uint8_t> snapshot = SnapshotOutputs({&dq, &dk, &dv});
    result.outputBytes = snapshot.size();
    result.outputDigest = Fnv1a64(snapshot);
    if (args.Has("write-reference")) WriteReference(args.Get("write-reference"), snapshot);
    if (args.Has("compare-reference")) {
        result.outputReferenceChecked = true;
        result.outputReferenceEqual = snapshot == ReadReference(args.Get("compare-reference"));
    }
    return result;
}

Measurement RunFusedInferAttentionScore(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const int64_t batch = args.Int("batch"), qHeads = args.Int("q-heads"), kvHeads = args.Int("kv-heads");
    const int64_t qSeq = args.Int("q-seq"), kvSeq = args.Int("kv-seq"), headDim = args.Int("head-dim");
    const std::string layout = args.Get("layout");
    const aclDataType dtype = DType(args.Get("dtype"));
    if (layout != "BNSD" && layout != "BSND" && layout != "BSH") {
        throw std::runtime_error("unsupported fused-attention layout: " + layout);
    }
    const auto qShape = AttentionShape(layout, batch, qHeads, qSeq, headDim);
    const auto kvShape = AttentionShape(layout, batch, kvHeads, kvSeq, headDim);
    DeviceBuffer q(Elements(qShape), dtype), k(Elements(kvShape), dtype), v(Elements(kvShape), dtype);
    DeviceBuffer output(Elements(qShape), dtype);
    FillAttentionPattern(q, dtype, Elements(qShape), 0);
    FillAttentionPattern(k, dtype, Elements(kvShape), 1);
    FillAttentionPattern(v, dtype, Elements(kvShape), 2);
    Tensor qTensor(q, dtype, qShape), kTensor(k, dtype, kvShape), vTensor(v, dtype, kvShape);
    Tensor outputTensor(output, dtype, qShape);
    TensorList keyList({kTensor.Get()}), valueList({vTensor.Get()});
    std::vector<char> layoutText(layout.begin(), layout.end());
    layoutText.push_back('\0');
    const double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    const char *dispatch = std::getenv("FIAS_SOURCE_DISPATCH");
    const bool privateSource = dispatch != nullptr;
    const bool auditRequested = std::getenv("FIAS_TILING_AUDIT_PATH") != nullptr;
    if (privateSource && std::string(dispatch) != "cann81_native_aclnn") throw std::runtime_error("invalid FIAS_SOURCE_DISPATCH");
    if (privateSource != auditRequested) throw std::runtime_error("FIAS source audit/dispatch mismatch");
    if (privateSource) {
        LoadRemainingPrivateLibrary(std::getenv("FIAS_SOURCE_TILING_LIBRARY"),
                                    "FusedInferAttentionScore CANN-8.1 host tiler");
        gBackend = "private_cann81_fias_host_tiler_installed_kernel_real_npu";
        if (args.Has("source-load-only")) return Measurement{};
    }
    const auto getWorkspace = [&](uint64_t *workspace, aclOpExecutor **executor) {
        return aclnnFusedInferAttentionScoreGetWorkspaceSize(
            qTensor.Get(), keyList.Get(), valueList.Get(), nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            qHeads, scale, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(),
            layoutText.data(), kvHeads, 0, 0, 0, 0, false, outputTensor.Get(), nullptr, workspace, executor);
    };
    const auto launch = [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
        return aclnnFusedInferAttentionScore(workspace, bytes, executor, launchStream);
    };
    if (args.Has("source-executor-planning-only")) {
        Measurement result;
        Executor executor;
        CheckAclnn(getWorkspace(&result.workspaceBytes, &executor.ptr), "private FIAS GetWorkspaceSize probe");
        if (executor.ptr == nullptr) throw std::runtime_error("private FIAS returned null executor");
        executor.ReleaseToIsolatedProcessExit();
        return result;
    }
    Measurement result = Measure(stream, args.Has("source-tiling-only") ? 0 : warmup,
                                 args.Has("source-tiling-only") ? 0 : samples, output, getWorkspace, launch);
    const std::vector<uint8_t> snapshot = SnapshotOutputs({&output});
    result.outputBytes = snapshot.size();
    result.outputDigest = Fnv1a64(snapshot);
    if (args.Has("write-reference")) WriteReference(args.Get("write-reference"), snapshot);
    if (args.Has("compare-reference")) {
        result.outputReferenceChecked = true;
        result.outputReferenceEqual = snapshot == ReadReference(args.Get("compare-reference"));
    }
    return result;
}
#endif

Measurement RunOperation(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const std::string op = args.Get("op");
#ifdef SCATTER_ELEMENTS_ONLY
    if (op == "scatter_elements") return RunScatterElements(args, stream, warmup, samples);
#elif defined(GATHER_ELEMENTS_ONLY)
    if (op == "gather_elements") return RunGatherElements(args, stream, warmup, samples);
#elif defined(FASG_ONLY)
    if (op == "flash_attention_score_grad") return RunFlashAttentionScoreGrad(args, stream, warmup, samples);
#elif defined(FIAS_ONLY)
    if (op == "fused_infer_attention_score") return RunFusedInferAttentionScore(args, stream, warmup, samples);
#else
    if (op == "matmul") return RunMatmul(args, stream, warmup, samples);
    if (op == "transpose") return RunTranspose(args, stream, warmup, samples);
    if (op == "gather_v2") return RunGatherV2(args, stream, warmup, samples);
    if (op == "gather_elements") return RunGatherElements(args, stream, warmup, samples);
    if (op == "scatter_elements") return RunScatterElements(args, stream, warmup, samples);
    if (op == "flash_attention_score_grad") return RunFlashAttentionScoreGrad(args, stream, warmup, samples);
    if (op == "fused_infer_attention_score") return RunFusedInferAttentionScore(args, stream, warmup, samples);
#endif
    throw std::runtime_error("unknown op: " + op);
}

double Median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

}  // namespace

int main(int argc, char **argv)
{
    std::string workloadId = "unknown";
    std::string op = "unknown";
    std::string socName = "unknown";
    aclrtStream stream = nullptr;
    bool normalCleanup = false;
    try {
        Arguments args(argc, argv);
        normalCleanup = args.Has("normal-cleanup");
        workloadId = args.Get("workload-id");
        gWorkloadId = workloadId;
        op = args.Get("op");
        const int device = static_cast<int>(args.Int("device"));
        const int warmup = static_cast<int>(args.Int("warmup"));
        const int samples = static_cast<int>(args.Int("samples"));
        if (warmup < 0 || samples < 0) throw std::runtime_error("invalid warmup/sample count");
        Stage("process_begin", "op=" + op);
        Stage("acl_init_begin");
        CheckAcl(aclInit(nullptr), "aclInit");
        Stage("acl_init_done");
        ReportPackage(ACL_PKG_NAME_TOOLKIT, "toolkit", false);
        ReportPackage(ACL_PKG_NAME_OPP, "opp", true);
        ReportPackage(ACL_PKG_NAME_OPP_KERNEL, "opp_kernel", true);
        Stage("set_device_begin", "device=" + std::to_string(device));
        CheckAcl(aclrtSetDevice(device), "aclrtSetDevice");
        Stage("set_device_done");
        CheckAcl(aclrtSetExceptionInfoCallback(DeviceException), "aclrtSetExceptionInfoCallback");
        const char *detectedSoc = aclrtGetSocName();
        if (detectedSoc == nullptr) throw std::runtime_error("aclrtGetSocName returned null");
        socName = detectedSoc;
        Stage("soc_detected", socName);
        if (socName != args.Get("expected-soc")) {
            throw std::runtime_error("unexpected SoC: " + socName + ", expected " + args.Get("expected-soc"));
        }
        Stage("stream_create_begin");
        CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
        Stage("stream_create_done");
        Stage("operation_prepare_begin");
        Measurement result = RunOperation(args, stream, warmup, samples);
        Stage("operation_measurement_done");
        CheckAcl(aclrtSynchronizeStream(stream), "final synchronize");
        Stage("final_sync_done");
        if (normalCleanup) {
            // Generic Python-source compilation owns host-side TBE workers.
            // Do normal process-local ACL cleanup so those workers can finish
            // and emit their audit before the runner returns. This destroys
            // only this process's stream; it does not reset the NPU.
            Stage("normal_cleanup_stream_destroy_begin");
            CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
            stream = nullptr;
            Stage("normal_cleanup_stream_destroy_done");
            Stage("normal_cleanup_acl_finalize_begin");
            CheckAcl(aclFinalize(), "aclFinalize");
            Stage("normal_cleanup_acl_finalize_done");
        }
        const bool verificationOnly = result.samplesMs.empty();
        const double median = verificationOnly ? 0.0 : Median(result.samplesMs);
        std::cout << "MULTIOP_NPU_RESULT {\"schema\":\"multi_op_real_npu_v2\","
                  << "\"status\":\"success\",\"backend\":\"" << JsonEscape(gBackend) << "\","
                  << "\"measurement_kind\":\"" << (verificationOnly ? "viability_only" : "device_event_latency") << "\","
                  << "\"soc\":\"" << JsonEscape(socName) << "\","
                  << "\"workload_id\":\"" << JsonEscape(workloadId) << "\","
                  << "\"op\":\"" << JsonEscape(op) << "\","
                  << "\"workspace_bytes\":" << result.workspaceBytes << ','
                  << "\"median_ms\":";
        if (verificationOnly) std::cout << "null,";
        else std::cout << std::setprecision(12) << median << ',';
        std::cout
                  << "\"samples_ms\":[";
        for (size_t i = 0; i < result.samplesMs.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << std::setprecision(12) << result.samplesMs[i];
        }
        std::cout << "],\"output_probe_bytes\":" << result.probeBytes
                  << ",\"output_probe_nonzero_bytes\":" << result.probeNonzeroBytes
                  << ",\"output_bytes\":" << result.outputBytes
                  << ",\"output_digest_fnv1a64\":" << result.outputDigest
                  << ",\"output_reference_checked\":" << (result.outputReferenceChecked ? "true" : "false")
                  << ",\"output_reference_equal\":" << (result.outputReferenceEqual ? "true" : "false")
                  << "}" << std::endl;
        if (normalCleanup) {
            Stage("process_exit", "normal_process_cleanup_without_global_device_reset");
            return 0;
        }
        Stage("process_exit", "isolated_worker_exit_without_global_device_reset");
        std::_Exit(0);
    } catch (const std::exception &error) {
        if (normalCleanup) {
            if (stream != nullptr) {
                (void)aclrtDestroyStream(stream);
                stream = nullptr;
            }
            (void)aclFinalize();
        }
        Stage("failure", error.what());
        std::cout << "MULTIOP_NPU_RESULT {\"schema\":\"multi_op_real_npu_v2\","
                  << "\"status\":\"failed\",\"backend\":\"" << JsonEscape(gBackend) << "\","
                  << "\"soc\":\"" << JsonEscape(socName) << "\","
                  << "\"workload_id\":\"" << JsonEscape(workloadId) << "\","
                  << "\"op\":\"" << JsonEscape(op) << "\","
                  << "\"error\":\"" << JsonEscape(error.what()) << "\"}" << std::endl;
        if (normalCleanup) {
            Stage("process_exit", "normal_failed_process_cleanup_without_global_device_reset");
            return 1;
        }
        Stage("process_exit", "isolated_failed_worker_exit_without_global_device_reset");
        std::_Exit(1);
    }
}

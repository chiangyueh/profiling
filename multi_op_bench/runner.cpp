#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_flash_attention_score_grad.h>
#include <aclnnop/aclnn_fused_infer_attention_score.h>
#include <aclnnop/aclnn_gather.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_scatter.h>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string JsonEscape(const std::string &value);
void CheckAcl(aclError rc, const std::string &what);
std::string gWorkloadId = "unknown";

void Stage(const std::string &stage, const std::string &detail = "")
{
    std::cout << "MULTIOP_NPU_STAGE {\"workload_id\":\"" << JsonEscape(gWorkloadId)
              << "\",\"stage\":\"" << JsonEscape(stage) << "\"";
    if (!detail.empty()) std::cout << ",\"detail\":\"" << JsonEscape(detail) << "\"";
    std::cout << "}" << std::endl;
}

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
        if (ptr != nullptr) aclDestroyAclOpExecutor(ptr);
    }
    aclOpExecutor *ptr = nullptr;
};

struct Measurement {
    uint64_t workspaceBytes = 0;
    std::vector<double> samplesMs;
    uint64_t probeBytes = 0;
    uint64_t probeNonzeroBytes = 0;
};

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
    FillIndices(index, indexDtype, Elements(indexShape), shape[axis]);
    Tensor inputTensor(input, dtype, shape), indexTensor(index, indexDtype, indexShape), outputTensor(output, dtype, indexShape);
    return Measure(stream, warmup, samples, output,
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnGatherGetWorkspaceSize(inputTensor.Get(), axis, indexTensor.Get(), outputTensor.Get(), workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnGather(workspace, bytes, executor, launchStream);
        });
}

Measurement RunScatterElements(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const auto shape = args.Shape("shape"), indexShape = args.Shape("index-shape");
    if (shape.size() != indexShape.size()) throw std::runtime_error("scatter-elements rank mismatch");
    const int64_t axis = NormalizeAxis(args.Int("axis"), shape.size());
    const int64_t reduce = args.Int("reduce");
    const aclDataType dtype = DType(args.Get("dtype")), indexDtype = DType(args.Get("index-dtype"));
    DeviceBuffer input(Elements(shape), dtype), index(Elements(indexShape), indexDtype);
    DeviceBuffer source(Elements(indexShape), dtype), output(Elements(shape), dtype);
    FillIndices(index, indexDtype, Elements(indexShape), shape[axis]);
    Tensor inputTensor(input, dtype, shape), indexTensor(index, indexDtype, indexShape);
    Tensor sourceTensor(source, dtype, indexShape), outputTensor(output, dtype, shape);
    return Measure(stream, warmup, samples, output,
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnScatterGetWorkspaceSize(inputTensor.Get(), axis, indexTensor.Get(), sourceTensor.Get(), reduce,
                                                outputTensor.Get(), workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnScatter(workspace, bytes, executor, launchStream);
        });
}

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
    std::vector<float> sumValues(Elements(softmaxShape), 1.0F);
    softmaxSum.CopyFrom(sumValues);
    Tensor qTensor(q, dtype, qShape), kTensor(k, dtype, kvShape), vTensor(v, dtype, kvShape);
    Tensor dyTensor(dy, dtype, qShape), attentionTensor(attention, dtype, qShape);
    Tensor maxTensor(softmaxMax, ACL_FLOAT, softmaxShape), sumTensor(softmaxSum, ACL_FLOAT, softmaxShape);
    Tensor dqTensor(dq, dtype, qShape), dkTensor(dk, dtype, kvShape), dvTensor(dv, dtype, kvShape);
    std::vector<char> layoutText(layout.begin(), layout.end());
    layoutText.push_back('\0');
    const double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    return Measure(stream, warmup, samples, dq,
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnFlashAttentionScoreGradGetWorkspaceSize(
                qTensor.Get(), kTensor.Get(), vTensor.Get(), dyTensor.Get(),
                nullptr, nullptr, nullptr, nullptr, maxTensor.Get(), sumTensor.Get(), nullptr, attentionTensor.Get(),
                nullptr, scale, 1.0, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(),
                qHeads, layoutText.data(), 0, 0, dqTensor.Get(), dkTensor.Get(), dvTensor.Get(), nullptr,
                workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnFlashAttentionScoreGrad(workspace, bytes, executor, launchStream);
        });
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
    Tensor qTensor(q, dtype, qShape), kTensor(k, dtype, kvShape), vTensor(v, dtype, kvShape);
    Tensor outputTensor(output, dtype, qShape);
    TensorList keyList({kTensor.Get()}), valueList({vTensor.Get()});
    std::vector<char> layoutText(layout.begin(), layout.end());
    layoutText.push_back('\0');
    const double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    return Measure(stream, warmup, samples, output,
        [&](uint64_t *workspace, aclOpExecutor **executor) {
            return aclnnFusedInferAttentionScoreGetWorkspaceSize(
                qTensor.Get(), keyList.Get(), valueList.Get(), nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                qHeads, scale, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(),
                layoutText.data(), kvHeads, 0, 0, 0, 0, false, outputTensor.Get(), nullptr,
                workspace, executor);
        },
        [&](void *workspace, uint64_t bytes, aclOpExecutor *executor, aclrtStream launchStream) {
            return aclnnFusedInferAttentionScore(workspace, bytes, executor, launchStream);
        });
}

Measurement RunOperation(const Arguments &args, aclrtStream stream, int warmup, int samples)
{
    const std::string op = args.Get("op");
    if (op == "matmul") return RunMatmul(args, stream, warmup, samples);
    if (op == "transpose") return RunTranspose(args, stream, warmup, samples);
    if (op == "gather_v2") return RunGatherV2(args, stream, warmup, samples);
    if (op == "gather_elements") return RunGatherElements(args, stream, warmup, samples);
    if (op == "scatter_elements") return RunScatterElements(args, stream, warmup, samples);
    if (op == "flash_attention_score_grad") return RunFlashAttentionScoreGrad(args, stream, warmup, samples);
    if (op == "fused_infer_attention_score") return RunFusedInferAttentionScore(args, stream, warmup, samples);
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
    try {
        Arguments args(argc, argv);
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
        const bool verificationOnly = result.samplesMs.empty();
        const double median = verificationOnly ? 0.0 : Median(result.samplesMs);
        std::cout << "MULTIOP_NPU_RESULT {\"schema\":\"multi_op_real_npu_v2\","
                  << "\"status\":\"success\",\"backend\":\"aclnn_real_npu\","
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
                  << ",\"output_probe_nonzero_bytes\":" << result.probeNonzeroBytes << "}" << std::endl;
        Stage("process_exit", "isolated_worker_exit_without_global_device_reset");
        std::_Exit(0);
    } catch (const std::exception &error) {
        Stage("failure", error.what());
        std::cout << "MULTIOP_NPU_RESULT {\"schema\":\"multi_op_real_npu_v2\","
                  << "\"status\":\"failed\",\"backend\":\"aclnn_real_npu\","
                  << "\"soc\":\"" << JsonEscape(socName) << "\","
                  << "\"workload_id\":\"" << JsonEscape(workloadId) << "\","
                  << "\"op\":\"" << JsonEscape(op) << "\","
                  << "\"error\":\"" << JsonEscape(error.what()) << "\"}" << std::endl;
        Stage("process_exit", "isolated_failed_worker_exit_without_global_device_reset");
        std::_Exit(1);
    }
}

#include "search_types.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace matmul_search {

std::string Candidate::Key() const
{
    std::ostringstream os;
    os << singleM << ':' << singleN << ':' << singleK << ':'
       << baseM << ':' << baseN << ':' << baseK << ':' << traverse << ':'
       << static_cast<int>(dbA) << ':' << static_cast<int>(dbB) << ':' << static_cast<int>(splitK);
    return os.str();
}

bool Candidate::operator==(const Candidate &other) const
{
    return singleM == other.singleM && singleN == other.singleN && singleK == other.singleK &&
           baseM == other.baseM && baseN == other.baseN && baseK == other.baseK &&
           traverse == other.traverse && dbA == other.dbA && dbB == other.dbB && splitK == other.splitK;
}

std::string DTypeToString(DType dtype)
{
    switch (dtype) {
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
        case DType::FP32: return "fp32";
        case DType::INT8: return "int8";
    }
    return "unknown";
}

std::string ExecutionModeToString(ExecutionMode mode)
{
    switch (mode) {
        case ExecutionMode::BASE_ITERATE_ALL: return "base_iterate_all";
        case ExecutionMode::NONE: return "none";
    }
    return "none";
}

bool ParseDType(const std::string &text, DType &dtype)
{
    std::string value = text;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "fp16" || value == "float16" || value == "half") {
        dtype = DType::FP16;
        return true;
    }
    if (value == "bf16" || value == "bfloat16") {
        dtype = DType::BF16;
        return true;
    }
    if (value == "fp32" || value == "float32" || value == "float") {
        dtype = DType::FP32;
        return true;
    }
    if (value == "int8" || value == "i8") {
        dtype = DType::INT8;
        return true;
    }
    return false;
}

int32_t DTypeBits(DType dtype)
{
    switch (dtype) {
        case DType::FP16:
        case DType::BF16: return 16;
        case DType::FP32: return 32;
        case DType::INT8: return 8;
    }
    return 16;
}

int32_t DTypeBytes(DType dtype)
{
    return std::max(1, DTypeBits(dtype) / 8);
}

int32_t OutputBytes(DType dtype)
{
    return dtype == DType::INT8 ? 4 : DTypeBytes(dtype);
}

int32_t AccumulatorBytes(DType dtype)
{
    return dtype == DType::INT8 ? 4 : 4;
}

std::string TilingSignature(const TCubeTiling &t)
{
    std::ostringstream os;
    os << t.usedCoreNum << ':' << t.singleCoreM << ':' << t.singleCoreN << ':' << t.singleCoreK << ':'
       << t.baseM << ':' << t.baseN << ':' << t.baseK << ':' << t.depthA1 << ':' << t.depthB1 << ':'
       << t.stepM << ':' << t.stepN << ':' << t.stepKa << ':' << t.stepKb << ':'
       << t.iterateOrder << ':' << t.dbL0A << ':' << t.dbL0B << ':' << t.dbL0C;
    return os.str();
}

}  // namespace matmul_search

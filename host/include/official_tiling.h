#pragma once

#include <string>
#include "search_types.h"
#include "lib/matmul/bmm_tiling.h"

namespace matmul_search {

std::string ValidateTilingContract(const Workload &workload, const TCubeTiling &tiling,
                                   const PlatformCaps &caps, ExecutionMode &mode);

class OfficialTilingEngine {
public:
    explicit OfficialTilingEngine(const std::string &socVersion = "Ascend910B");

    bool IsReady() const;
    const std::string &Error() const;
    const PlatformCaps &Caps() const;
    Evaluation Evaluate(const Workload &workload, const Candidate &candidate) const;

private:
    platform_ascendc::PlatformAscendC *platform_ = nullptr;
    PlatformCaps caps_{};
    std::string error_;
};

}  // namespace matmul_search

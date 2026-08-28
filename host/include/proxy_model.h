#pragma once

#include "search_types.h"

namespace matmul_search {

struct MatmulPathParameters {
    // These counts describe the MatMul control program lowered into the
    // generic hardware IR. Hardware rates live only in HardwareProfile.
    double cubeIssueCycles = 1.0;
    double scalarCoreSetupCycles = 24.0;
    double scalarPerMmadCycles = 2.0;
    double scalarPerOutputTileCycles = 6.0;
    double kernelFixedCycles = 64.0;
    double splitKBaseCycles = 180.0;
    double splitKPerCoreCycles = 12.0;
};

class ProxyModel {
public:
    ProxyModel(PlatformCaps caps, MatmulPathParameters parameters = {});
    ProxyBreakdown Score(const Workload &workload, const Candidate &candidate, const TCubeTiling &tiling) const;

private:
    PlatformCaps caps_;
    MatmulPathParameters parameters_;
};

}  // namespace matmul_search

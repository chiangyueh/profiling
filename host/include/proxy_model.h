#pragma once

#include "search_types.h"

namespace matmul_search {

struct ProxyWeights {
    // HBM comes from PlatformAscendC when available. The fallback and local
    // pipeline rates are explicit calibration priors in byte/cycle.
    double fallbackHbmBytesPerCycle = 32.0;
    double mte2BytesPerCyclePerCore = 128.0;
    double mte1BytesPerCyclePerCore = 256.0;
    double fixpipeBytesPerCyclePerCore = 128.0;
    double mte2IssueCycles = 4.0;
    double mte1IssueCycles = 2.0;
    double cubeIssueCycles = 1.0;
    double fixpipeIssueCycles = 4.0;
    double scalarCoreSetupCycles = 24.0;
    double scalarPerMmadCycles = 2.0;
    double scalarPerOutputTileCycles = 6.0;
    double kernelFixedCycles = 64.0;
    double splitKBaseCycles = 180.0;
    double splitKPerCoreCycles = 12.0;
};

class ProxyModel {
public:
    ProxyModel(PlatformCaps caps, ProxyWeights weights = {});
    ProxyBreakdown Score(const Workload &workload, const Candidate &candidate, const TCubeTiling &tiling) const;

private:
    PlatformCaps caps_;
    ProxyWeights weights_;
};

}  // namespace matmul_search

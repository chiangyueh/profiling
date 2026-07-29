#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "proxy_model.h"

namespace {

TCubeTiling MakeTiling(int m, int n, int k, int baseM, int baseN, int baseK)
{
    TCubeTiling t{};
    t.usedCoreNum = 1;
    t.M = m;
    t.N = n;
    t.Ka = k;
    t.Kb = k;
    t.singleCoreM = m;
    t.singleCoreN = n;
    t.singleCoreK = k;
    t.baseM = baseM;
    t.baseN = baseN;
    t.baseK = baseK;
    t.stepM = 1;
    t.stepN = 1;
    t.stepKa = 1;
    t.stepKb = 1;
    t.depthA1 = 1;
    t.depthB1 = 1;
    t.iterateOrder = 0;
    t.dbL0A = 1;
    t.dbL0B = 1;
    t.dbL0C = 1;
    return t;
}

bool Near(double a, double b)
{
    return std::abs(a - b) <= 1e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}

}  // namespace

int main()
{
    using namespace matmul_search;

    PlatformCaps caps;
    caps.coreNum = 24;
    caps.hbmBytesPerCycle = 32;
    ProxyModel model(caps);
    Candidate candidate;

    Workload square{"square", 512, 512, 512, DType::FP16, false, false, 24};
    TCubeTiling small = MakeTiling(512, 512, 512, 32, 32, 32);
    small.stepM = 2;
    small.stepN = 2;
    small.stepKa = 16;
    small.stepKb = 16;
    small.depthA1 = 64;
    small.depthB1 = 64;
    small.dbL0A = 2;
    small.dbL0B = 2;

    TCubeTiling large = MakeTiling(512, 512, 512, 128, 128, 64);
    large.stepM = 2;
    large.stepN = 2;
    large.stepKa = 8;
    large.stepKb = 8;
    large.depthA1 = 32;
    large.depthB1 = 32;
    large.dbL0A = 2;
    large.dbL0B = 2;

    const ProxyBreakdown smallScore = model.Score(square, candidate, small);
    const ProxyBreakdown largeScore = model.Score(square, candidate, large);
    assert(std::isfinite(smallScore.total));
    assert(largeScore.total < smallScore.total);
    assert(largeScore.estimatedMmadCount < smallScore.estimatedMmadCount);

    TCubeTiling noDb = large;
    noDb.dbL0A = 1;
    noDb.dbL0B = 1;
    noDb.depthA1 /= 2;
    noDb.depthB1 /= 2;
    const ProxyBreakdown noDbScore = model.Score(square, candidate, noDb);
    assert(largeScore.criticalCoreCycles <= noDbScore.criticalCoreCycles);

    Workload rectangular{"rectangular", 64, 1024, 512, DType::FP16, false, false, 24};
    TCubeTiling orderM = MakeTiling(64, 1024, 512, 64, 64, 64);
    orderM.stepM = 1;
    orderM.stepN = 4;
    orderM.stepKa = 8;
    orderM.stepKb = 8;
    orderM.depthA1 = 8;
    orderM.depthB1 = 32;
    TCubeTiling orderN = orderM;
    orderN.iterateOrder = 1;
    const ProxyBreakdown orderMScore = model.Score(rectangular, candidate, orderM);
    const ProxyBreakdown orderNScore = model.Score(rectangular, candidate, orderN);
    assert(orderNScore.estimatedAGmBytes < orderMScore.estimatedAGmBytes);
    assert(orderNScore.total < orderMScore.total);

    Workload tail{"tail", 257, 1009, 4097, DType::FP16, false, false, 24};
    TCubeTiling tailTiling = MakeTiling(257, 1009, 4097, 192, 256, 64);
    tailTiling.stepM = 2;
    tailTiling.stepN = 4;
    tailTiling.stepKa = 3;
    tailTiling.stepKb = 3;
    tailTiling.depthA1 = 6;
    tailTiling.depthB1 = 12;
    const ProxyBreakdown tailScore = model.Score(tail, candidate, tailTiling);
    assert(tailScore.tailEfficiency > 0.0 && tailScore.tailEfficiency < 1.0);
    assert(tailScore.tailPenalty > 0.0);

    Workload imbalanced{"imbalanced", 257, 256, 256, DType::FP16, false, false, 3};
    TCubeTiling multi = MakeTiling(257, 256, 256, 128, 256, 64);
    multi.usedCoreNum = 3;
    multi.singleCoreM = 128;
    multi.singleCoreN = 256;
    multi.singleCoreK = 256;
    multi.stepKa = 4;
    multi.stepKb = 4;
    multi.depthA1 = 4;
    multi.depthB1 = 4;
    const ProxyBreakdown multiScore = model.Score(imbalanced, candidate, multi);
    assert(multiScore.criticalCoreCycles > multiScore.averageCoreCycles);
    assert(multiScore.balancePenalty > 0.0);

    const ProxyBreakdown repeat = model.Score(square, candidate, large);
    assert(Near(repeat.total, largeScore.total));
    assert(Near(repeat.estimatedGmBytes, largeScore.estimatedGmBytes));
    assert(repeat.criticalCoreId == largeScore.criticalCoreId);

    std::cout << "proxy_model_test passed\n";
    return 0;
}

#include <cassert>
#include <cmath>
#include <iostream>

#include "hardware_cost_model.h"
#include "hardware_profiles.h"
#include "indexed_update_path.h"

int main()
{
    using namespace hardware_cost;
    IndexedUpdateProblem problem;
    problem.valueBytes = 2;
    problem.indexBytes = 4;
    problem.availableVectorCores = 40;

    IndexedUpdateSchedule oneCore;
    oneCore.usedCoreNum = 1;
    oneCore.inputOneTime = 17;
    oneCore.indicesOneTime = 1;
    oneCore.updatesOneTime = 1;
    oneCore.indicesEach = 1024;
    oneCore.indicesLast = 1024;
    oneCore.indicesLoop = 1;
    oneCore.lastIndicesEach = 1024;
    oneCore.lastIndicesLast = 1024;
    oneCore.lastIndicesLoop = 1;
    oneCore.mode = 1;

    IndexedUpdateSchedule manyCores = oneCore;
    manyCores.usedCoreNum = 16;
    manyCores.indicesEach = 64;
    manyCores.indicesLast = 64;
    manyCores.lastIndicesEach = 64;
    manyCores.lastIndicesLast = 64;

    const HardwareCostModel model(Ascend910B3VectorProfile());
    const KernelCost serial = model.Evaluate(LowerIndexedUpdatePath(problem, oneCore));
    const KernelCost parallel = model.Evaluate(LowerIndexedUpdatePath(problem, manyCores));
    assert(serial.valid && parallel.valid);
    assert(std::isfinite(serial.totalCycles) && std::isfinite(parallel.totalCycles));
    assert(parallel.totalCycles < serial.totalCycles);
    assert(parallel.sharedResourceCycles > 0.0);

    problem.indexBytes = 8;
    problem.accumulate = true;
    problem.promoteAccumulatorToFp32 = true;
    const KernelCost converted = model.Evaluate(LowerIndexedUpdatePath(problem, manyCores));
    assert(converted.valid);
    assert(converted.criticalResourceCycles[
        static_cast<std::size_t>(Resource::VECTOR)] > 0.0);
    assert(converted.criticalResourceCycles[
        static_cast<std::size_t>(Resource::SYNC)] > 0.0);

    // Exercise the other source-kernel control-flow branch: input is split
    // into UB-sized pieces and every index tile is inspected for each piece.
    IndexedUpdateSchedule partitioned;
    partitioned.usedCoreNum = 4;
    partitioned.eachNum = 1;
    partitioned.eachPiece = 1;
    partitioned.inputOnePiece = 4096;
    partitioned.inputOneTime = 4096;
    partitioned.indicesOneTime = 1024;
    partitioned.updatesOneTime = 1024;
    partitioned.inputEach = 1024;
    partitioned.inputLast = 1024;
    partitioned.inputLoop = 4;
    partitioned.indicesEach = 256;
    partitioned.indicesLast = 256;
    partitioned.indicesLoop = 4;
    partitioned.inputAlign = 1024;
    partitioned.indicesAlign = 256;
    partitioned.updatesAlign = 256;
    partitioned.mode = 0;
    const KernelProgram partitionedProgram = LowerIndexedUpdatePath(problem, partitioned);
    const KernelCost partitionedCost = model.Evaluate(partitionedProgram);
    assert(partitionedCost.valid);
    assert(partitionedCost.criticalResourceCycles[
        static_cast<std::size_t>(Resource::MTE2)] > 0.0);
    assert(partitionedCost.criticalResourceCycles[
        static_cast<std::size_t>(Resource::SCALAR)] > 0.0);
    assert(partitionedProgram.peakMemoryBytes[
        static_cast<std::size_t>(MemorySpace::UB)] > 0);

    IndexedUpdateSchedule fewerInputTiles = partitioned;
    fewerInputTiles.inputEach = 2048;
    fewerInputTiles.inputLast = 2048;
    fewerInputTiles.inputLoop = 2;
    fewerInputTiles.inputAlign = 2048;
    const KernelCost fewerTileCost = model.Evaluate(
        LowerIndexedUpdatePath(problem, fewerInputTiles));
    assert(fewerTileCost.valid);
    assert(fewerTileCost.totalCycles < partitionedCost.totalCycles);

    std::cout << "indexed_update_path_test passed\n";
    return 0;
}

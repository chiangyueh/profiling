#include <cassert>
#include <cmath>
#include <iostream>

#include "hardware_cost_model.h"
#include "hardware_profiles.h"
#include "indexed_read_path.h"

int main()
{
    using namespace hardware_cost;
    IndexedReadProblem problem;
    problem.sourceShape = {128, 4096};
    problem.indexShape = {128, 1024};
    problem.axis = 1;
    problem.valueBytes = 2;
    problem.indexBytes = 4;

    const HardwareCostModel model(Ascend910B3VectorProfile());
    const auto oneCoreSchedule = PlanIndexedReadSchedule(problem, 1, 1);
    const auto manyCoreSchedule = PlanIndexedReadSchedule(problem, 20, 1);
    const KernelCost oneCore = model.Evaluate(
        LowerIndexedReadPath(problem, oneCoreSchedule));
    const KernelCost manyCore = model.Evaluate(
        LowerIndexedReadPath(problem, manyCoreSchedule));
    assert(oneCore.valid && manyCore.valid);
    assert(std::isfinite(oneCore.totalCycles));
    assert(manyCore.totalCycles < oneCore.totalCycles);
    assert(manyCoreSchedule.route == IndexedReadRoute::LAST_DIMENSION);
    assert(manyCore.criticalResourceCycles[
        static_cast<std::size_t>(Resource::MTE2)] > 0.0);
    assert(manyCore.criticalResourceCycles[
        static_cast<std::size_t>(Resource::VECTOR)] > 0.0);

    const auto smallUbSchedule = PlanIndexedReadSchedule(problem, 20, 8);
    const KernelCost smallUb = model.Evaluate(
        LowerIndexedReadPath(problem, smallUbSchedule));
    assert(smallUb.valid);
    assert(smallUb.totalCycles > manyCore.totalCycles);

    IndexedReadProblem transpose = problem;
    transpose.sourceShape = {8, 64, 128};
    transpose.indexShape = {8, 17, 128};
    transpose.axis = 1;
    const auto transposeSchedule = PlanIndexedReadSchedule(transpose, 20, 1);
    const KernelCost transposeCost = model.Evaluate(
        LowerIndexedReadPath(transpose, transposeSchedule));
    assert(transposeSchedule.route == IndexedReadRoute::TRANSPOSE);
    assert(transposeCost.valid);
    assert(transposeCost.readBytes[
        static_cast<std::size_t>(MemorySpace::GM)] > 0.0);
    assert(transposeCost.writeBytes[
        static_cast<std::size_t>(MemorySpace::GM)] > 0.0);

    // A large source-to-output ratio selects the scalar random-read path;
    // it must expose both scalar address work and physical GM transactions.
    IndexedReadProblem irregular = problem;
    irregular.sourceShape = {64, 65536};
    irregular.indexShape = {64, 17};
    const auto irregularSchedule = PlanIndexedReadSchedule(irregular, 20, 8);
    const KernelCost irregularCost = model.Evaluate(
        LowerIndexedReadPath(irregular, irregularSchedule));
    assert(irregularSchedule.scalarMode);
    assert(irregularCost.valid);
    assert(irregularCost.criticalResourceCycles[
        static_cast<std::size_t>(Resource::SCALAR)] > 0.0);

    std::cout << "indexed_read_path_test passed\n";
    return 0;
}

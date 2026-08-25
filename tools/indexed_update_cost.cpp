#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include "hardware_cost_model.h"
#include "hardware_profiles.h"
#include "indexed_update_path.h"

int main()
{
    using namespace hardware_cost;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout << std::setprecision(17);

    std::string id;
    while (std::cin >> id) {
        IndexedUpdateProblem problem;
        IndexedUpdateSchedule schedule;
        int accumulate = 0;
        int promote = 0;
        if (!(std::cin >> problem.valueBytes >> problem.indexBytes >> accumulate >> promote
              >> problem.availableVectorCores
              >> schedule.usedCoreNum >> schedule.eachNum >> schedule.extraTaskCore
              >> schedule.eachPiece >> schedule.inputOnePiece >> schedule.inputOneTime
              >> schedule.indicesOneTime >> schedule.updatesOneTime >> schedule.inputEach
              >> schedule.indicesEach >> schedule.inputLast >> schedule.indicesLast
              >> schedule.inputLoop >> schedule.indicesLoop >> schedule.inputAlign
              >> schedule.indicesAlign >> schedule.updatesAlign >> schedule.lastIndicesLoop
              >> schedule.lastIndicesEach >> schedule.lastIndicesLast >> schedule.oneTime
              >> schedule.mode)) {
            std::cerr << "invalid indexed-update hardware-path row after id=" << id << '\n';
            return 2;
        }
        problem.accumulate = accumulate != 0;
        problem.promoteAccumulatorToFp32 = promote != 0;
        const HardwareCostModel model(Ascend910B3VectorProfile());
        const KernelCost cost = model.Evaluate(LowerIndexedUpdatePath(problem, schedule));
        if (!cost.valid) {
            std::cout << id << " invalid " << cost.error << '\n';
            continue;
        }
        const auto index = [](Resource value) { return static_cast<std::size_t>(value); };
        std::cout << id << " valid " << cost.totalCycles << ' '
                  << cost.criticalCoreCycles << ' ' << cost.hbmCycles << ' '
                  << cost.l2Cycles << ' ' << cost.sharedResourceCycles << ' '
                  << cost.criticalResourceCycles[index(Resource::MTE2)] << ' '
                  << cost.criticalResourceCycles[index(Resource::VECTOR)] << ' '
                  << cost.criticalResourceCycles[index(Resource::SCALAR)] << ' '
                  << cost.criticalResourceCycles[index(Resource::MTE3)] << ' '
                  << cost.coreUtilization << '\n';
    }
    return 0;
}

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include "hardware_cost_model.h"
#include "hardware_profiles.h"
#include "indexed_read_path.h"

int main()
{
    using namespace hardware_cost;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout << std::setprecision(17);

    std::string id;
    while (std::cin >> id) {
        IndexedReadProblem problem;
        uint64_t coreCap = 0;
        uint64_t ubDivisor = 0;
        uint64_t rank = 0;
        if (!(std::cin >> problem.valueBytes >> problem.indexBytes
              >> problem.availableVectorCores >> problem.physicalUbBytes
              >> problem.axis >> coreCap >> ubDivisor >> rank)) {
            std::cerr << "invalid indexed-read hardware-path row after id=" << id << '\n';
            return 2;
        }
        problem.sourceShape.resize(rank);
        problem.indexShape.resize(rank);
        for (uint64_t &dimension : problem.sourceShape) {
            if (!(std::cin >> dimension)) return 2;
        }
        for (uint64_t &dimension : problem.indexShape) {
            if (!(std::cin >> dimension)) return 2;
        }

        try {
            const IndexedReadSchedule schedule = PlanIndexedReadSchedule(
                problem, coreCap, ubDivisor);
            const HardwareCostModel model(Ascend910B3VectorProfile());
            const KernelCost cost = model.Evaluate(
                LowerIndexedReadPath(problem, schedule));
            if (!cost.valid) {
                std::cout << id << " invalid " << cost.error << '\n';
                continue;
            }
            const auto index = [](Resource value) {
                return static_cast<std::size_t>(value);
            };
            std::cout << id << " valid " << cost.totalCycles << ' '
                      << cost.criticalCoreCycles << ' ' << cost.hbmCycles << ' '
                      << cost.l2Cycles << ' ' << cost.sharedResourceCycles << ' '
                      << cost.criticalResourceCycles[index(Resource::MTE2)] << ' '
                      << cost.criticalResourceCycles[index(Resource::VECTOR)] << ' '
                      << cost.criticalResourceCycles[index(Resource::SCALAR)] << ' '
                      << cost.criticalResourceCycles[index(Resource::MTE3)] << ' '
                      << cost.coreUtilization << '\n';
        } catch (const std::exception &error) {
            std::cout << id << " invalid " << error.what() << '\n';
        }
    }
    return 0;
}

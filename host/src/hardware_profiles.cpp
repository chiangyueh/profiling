#include "hardware_profiles.h"

namespace hardware_cost {

HardwareProfile Ascend910B3VectorProfile()
{
    const auto resource = [](Resource value) { return static_cast<std::size_t>(value); };
    const auto memory = [](MemorySpace value) { return static_cast<std::size_t>(value); };
    HardwareProfile profile;

    // Ascend 910B3 hardware limits and CCE primitive service breakpoints.
    // They are intentionally compiled into the executable: runtime scoring
    // reads neither CCE measurements nor historical operator latency.
    profile.rates[resource(Resource::MTE2)] = {16.0, 0.0, 10.0, 337.0};
    profile.rates[resource(Resource::MTE1)] = {256.0, 0.0, 2.0, 0.0};
    profile.rates[resource(Resource::MTE3)] = {64.0, 0.0, 10.0, 25.0};
    profile.rates[resource(Resource::VECTOR)] = {0.0, 128.0, 1.0, 0.0};
    profile.rates[resource(Resource::SCALAR)] = {0.0, 1.0, 1.0, 0.0};
    profile.rates[resource(Resource::FIXPIPE)] = {64.0, 0.0, 31.0, 0.0};
    // The CCE PIPE_ALL probe measured 545 cycles for 64 barriers.  Event
    // set/wait instructions remain one issue cycle each and are represented
    // through ResourceWork::issues rather than ResourceWork::operations.
    profile.rates[resource(Resource::SYNC)] = {0.0, 64.0 / 545.0, 1.0, 0.0};

    profile.capacityBytes[memory(MemorySpace::L0A)] = 64ULL * 1024;
    profile.capacityBytes[memory(MemorySpace::L0B)] = 64ULL * 1024;
    profile.capacityBytes[memory(MemorySpace::L0C)] = 128ULL * 1024;
    profile.capacityBytes[memory(MemorySpace::L1)] = 524032;
    profile.capacityBytes[memory(MemorySpace::UB)] = 196352;
    profile.capacityBytes[memory(MemorySpace::L2)] = 192ULL * 1024 * 1024;

    profile.aggregateHbmBytesPerCycle = 20.0 * 32.0;
    profile.aggregateL2BytesPerCycle = 20.0 * 110.0;
    profile.kernelLaunchCycles = 96.0;
    profile.parallelUnits[resource(Resource::MTE2)] = 16.0;
    profile.parallelUnits[resource(Resource::MTE3)] = 16.0;
    profile.parallelUnits[resource(Resource::VECTOR)] = 40.0;
    profile.parallelUnits[resource(Resource::SCALAR)] = 40.0;
    return profile;
}

}  // namespace hardware_cost

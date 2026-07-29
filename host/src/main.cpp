#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "beam_lns_search.h"
#include "csv_io.h"

namespace {

void PrintUsage()
{
    std::cout
        << "matmul_tiling_search [options]\n"
        << "  --workloads FILE            workload CSV\n"
        << "  --m M --n N --k K           single workload\n"
        << "  --dtype fp16|bf16|fp32|int8\n"
        << "  --trans-a 0|1 --trans-b 0|1\n"
        << "  --cores N                   maximum AI Core count\n"
        << "  --beam-width N              beam width\n"
        << "  --tabu-iters N              tabu/LNS iterations\n"
        << "  --lns-rounds N              large-neighborhood rounds\n"
        << "  --top-k N                   candidates saved per workload\n"
        << "  --max-core-rounds N         optional M/N rounds cap; 0 disables the cap\n"
        << "  --max-base-m N --max-base-n N --max-base-k N\n"
        << "  --platform-only             print capacities and emit empty CSV headers\n"
        << "  --soc VERSION               default Ascend910B\n"
        << "  --output FILE               top-candidate CSV\n"
        << "  --all-output FILE           all evaluated candidates CSV\n"
        << "  --tiling-dir DIR            raw TCubeTiling files\n";
}

bool ToBool(const std::string &value)
{
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::unordered_map<std::string, std::string> ParseArgs(int argc, char **argv)
{
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h" || key == "--platform-only") {
            args[key] = "1";
            continue;
        }
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("invalid argument: " + key);
        }
        args[key] = argv[++i];
    }
    return args;
}

std::string Get(const std::unordered_map<std::string, std::string> &args,
                const std::string &key, const std::string &fallback)
{
    const auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
}

int32_t GetInt(const std::unordered_map<std::string, std::string> &args,
               const std::string &key, int32_t fallback)
{
    const auto it = args.find(key);
    return it == args.end() ? fallback : std::stoi(it->second);
}

}  // namespace

int main(int argc, char **argv)
{
    using namespace matmul_search;
    try {
        const auto args = ParseArgs(argc, argv);
        if (args.count("--help") || args.count("-h")) {
            PrintUsage();
            return 0;
        }

        const std::string output = Get(args, "--output", "results/candidates.csv");
        const std::string allOutput = Get(args, "--all-output", "results/all_evaluated.csv");
        const std::string tilingDir = Get(args, "--tiling-dir", "results/tilings");
        const std::string soc = Get(args, "--soc", "Ascend910B");

        OfficialTilingEngine engine(soc);
        if (!engine.IsReady()) {
            std::cerr << "CANN platform initialization failed: " << engine.Error() << '\n';
            return 2;
        }
        const auto &caps = engine.Caps();
        std::cout << "CANN platform=" << soc << " cores=" << caps.coreNum
                  << " L0A=" << caps.l0aBytes << " L0B=" << caps.l0bBytes
                  << " L0C=" << caps.l0cBytes << " L1=" << caps.l1Bytes
                  << " UB=" << caps.ubBytes << " L2=" << caps.l2Bytes
                  << " L2_Bpc_per_core=" << caps.l2BytesPerCycle
                  << " HBM_Bpc_per_core=" << caps.hbmBytesPerCycle << '\n';

        if (args.count("--platform-only")) {
            std::vector<Evaluation> empty;
            std::string error;
            if (!WriteEvaluationsCsv(output, empty, true, tilingDir, error) ||
                !WriteEvaluationsCsv(allOutput, empty, false, "", error)) {
                std::cerr << error << '\n';
                return 4;
            }
            return 0;
        }

        std::vector<Workload> workloads;
        std::string error;
        const auto workloadIt = args.find("--workloads");
        if (workloadIt != args.end()) {
            if (!LoadWorkloadsCsv(workloadIt->second, workloads, error)) {
                std::cerr << error << '\n';
                return 3;
            }
        } else {
            if (args.count("--m") == 0 || args.count("--n") == 0 || args.count("--k") == 0) {
                PrintUsage();
                return 3;
            }
            Workload w;
            w.id = Get(args, "--id", "single");
            w.m = GetInt(args, "--m", 1);
            w.n = GetInt(args, "--n", 1);
            w.k = GetInt(args, "--k", 1);
            if (!ParseDType(Get(args, "--dtype", "fp16"), w.dtype)) {
                std::cerr << "unsupported dtype\n";
                return 3;
            }
            w.transA = ToBool(Get(args, "--trans-a", "0"));
            w.transB = ToBool(Get(args, "--trans-b", "0"));
            w.maxCores = GetInt(args, "--cores", caps.coreNum);
            if (w.m <= 0 || w.n <= 0 || w.k <= 0 || w.maxCores <= 0) {
                std::cerr << "shape and --cores must be positive\n";
                return 3;
            }
            workloads.push_back(w);
        }

        SearchOptions options;
        options.beamWidth = GetInt(args, "--beam-width", options.beamWidth);
        options.tabuIterations = GetInt(args, "--tabu-iters", options.tabuIterations);
        options.lnsRounds = GetInt(args, "--lns-rounds", options.lnsRounds);
        options.topK = GetInt(args, "--top-k", options.topK);
        options.seedCount = GetInt(args, "--seed-count", options.seedCount);
        options.maxCoreRounds = GetInt(args, "--max-core-rounds", options.maxCoreRounds);
        options.maxBaseM = GetInt(args, "--max-base-m", options.maxBaseM);
        options.maxBaseN = GetInt(args, "--max-base-n", options.maxBaseN);
        options.maxBaseK = GetInt(args, "--max-base-k", options.maxBaseK);
        if (options.beamWidth <= 0 || options.topK <= 0 || options.seedCount <= 0 ||
            options.maxCoreRounds < 0 || options.maxBaseM <= 0 ||
            options.maxBaseN <= 0 || options.maxBaseK <= 0 ||
            options.tabuIterations < 0 || options.lnsRounds < 0) {
            std::cerr << "search widths, limits, and rounds are invalid\n";
            return 3;
        }

        std::vector<Evaluation> allEvaluations;
        std::vector<Evaluation> allTop;
        for (const auto &workload : workloads) {
            std::cout << "search " << workload.id << " M=" << workload.m << " N=" << workload.n
                      << " K=" << workload.k << " dtype=" << DTypeToString(workload.dtype) << std::endl;
            BeamLnsSearcher searcher(engine, options);
            SearchResult result = searcher.Search(workload);
            allEvaluations.insert(allEvaluations.end(), result.evaluated.begin(), result.evaluated.end());
            allTop.insert(allTop.end(), result.top.begin(), result.top.end());
            const auto bestIt = std::find_if(result.top.begin(), result.top.end(), [](const Evaluation &evaluation) {
                return evaluation.source != "official_default";
            });
            if (bestIt != result.top.end()) {
                const auto &best = *bestIt;
                std::cout << "  evaluated=" << result.evaluated.size()
                          << " baseline=" << std::count_if(
                                 result.top.begin(), result.top.end(), [](const Evaluation &evaluation) {
                                     return evaluation.source == "official_default";
                                 })
                          << " searched_top=" << std::count_if(
                                 result.top.begin(), result.top.end(), [](const Evaluation &evaluation) {
                                     return evaluation.source != "official_default";
                                 })
                          << " single_hint=" << best.candidate.singleM << 'x'
                          << best.candidate.singleN << 'x' << best.candidate.singleK
                          << " base_hint=" << best.candidate.baseM << 'x' << best.candidate.baseN << 'x'
                          << best.candidate.baseK
                          << " official=" << best.tiling.baseM << 'x' << best.tiling.baseN << 'x'
                          << best.tiling.baseK << " cores=" << best.tiling.usedCoreNum
                          << " score=" << best.proxy.total << '\n';
            } else {
                std::cout << "  searched_top=0 optimization_status=no_candidate reason=";
                if (workload.dtype == DType::INT8) {
                    std::cout << "MatMulV3_dtype_unsupported\n";
                } else if (options.maxCoreRounds > 0) {
                    std::cout << "no_distinct_candidate_within_"
                              << options.maxCoreRounds << "_core_rounds\n";
                } else {
                    std::cout << "no_distinct_legal_candidate\n";
                }
            }
        }

        if (!WriteEvaluationsCsv(output, allTop, true, tilingDir, error)) {
            std::cerr << error << '\n';
            return 4;
        }
        if (!WriteEvaluationsCsv(allOutput, allEvaluations, false, "", error)) {
            std::cerr << error << '\n';
            return 4;
        }
        std::cout << "top CSV: " << output << '\n'
                  << "all CSV: " << allOutput << '\n'
                  << "tiling binaries: " << tilingDir << '\n';
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "fatal: " << ex.what() << '\n';
        return 1;
    }
}

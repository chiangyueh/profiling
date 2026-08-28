#include "csv_io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace matmul_search {
namespace {

std::vector<std::string> Split(const std::string &line)
{
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    fields.push_back(current);
    return fields;
}

std::string Escape(const std::string &value)
{
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') out += "\"\"";
        else out += ch;
    }
    out += '"';
    return out;
}

bool ParseBool(const std::string &value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower == "1" || lower == "true" || lower == "yes" || lower == "y";
}

std::string SafeName(std::string value)
{
    for (char &ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_')) ch = '_';
    }
    return value;
}

int64_t CeilDiv(int64_t x, int64_t y)
{
    return y <= 0 ? 0 : (x + y - 1) / y;
}

}  // namespace

bool LoadWorkloadsCsv(const std::string &path, std::vector<Workload> &workloads, std::string &error)
{
    std::ifstream input(path);
    if (!input) {
        error = "cannot open workload CSV: " + path;
        return false;
    }
    std::string line;
    if (!std::getline(input, line)) {
        error = "empty workload CSV";
        return false;
    }
    const auto headers = Split(line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t i = 0; i < headers.size(); ++i) columns[headers[i]] = i;
    for (const char *required : {"id", "m", "n", "k", "dtype"}) {
        if (columns.count(required) == 0) {
            error = std::string("missing CSV column: ") + required;
            return false;
        }
    }

    size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty() || line[0] == '#') continue;
        const auto fields = Split(line);
        auto get = [&](const std::string &name, const std::string &fallback = "") {
            const auto it = columns.find(name);
            if (it == columns.end() || it->second >= fields.size()) return fallback;
            return fields[it->second];
        };
        try {
            Workload w;
            w.id = get("id");
            w.m = std::stoi(get("m"));
            w.n = std::stoi(get("n"));
            w.k = std::stoi(get("k"));
            if (!ParseDType(get("dtype"), w.dtype)) {
                error = "unsupported dtype at line " + std::to_string(lineNumber);
                return false;
            }
            w.transA = ParseBool(get("trans_a", "0"));
            w.transB = ParseBool(get("trans_b", "0"));
            w.maxCores = std::stoi(get("max_cores", "24"));
            if (w.id.empty() || w.m <= 0 || w.n <= 0 || w.k <= 0 || w.maxCores <= 0) {
                error = "workload id, shape, and max_cores must be positive at line " +
                    std::to_string(lineNumber);
                return false;
            }
            workloads.push_back(w);
        } catch (const std::exception &ex) {
            error = "invalid workload at line " + std::to_string(lineNumber) + ":" + ex.what();
            return false;
        }
    }
    if (workloads.empty()) {
        error = "workload CSV contains no rows";
        return false;
    }
    return true;
}

bool WriteTilingBinary(const std::string &path, const TCubeTiling &tiling, std::string &error)
{
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "cannot write tiling binary: " + path;
        return false;
    }
    output.write(reinterpret_cast<const char *>(&tiling), sizeof(tiling));
    if (!output) {
        error = "failed while writing tiling binary: " + path;
        return false;
    }
    return true;
}

bool WriteEvaluationsCsv(const std::string &path, const std::vector<Evaluation> &evaluations,
                         bool includeRank, const std::string &tilingDirectory, std::string &error)
{
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    if (!tilingDirectory.empty()) std::filesystem::create_directories(tilingDirectory);
    std::ofstream output(path);
    if (!output) {
        error = "cannot write CSV: " + path;
        return false;
    }

    if (includeRank) output << "rank,";
    output << "workload_id,m,n,k,dtype,trans_a,trans_b,max_cores,source,candidate_role,source_iteration,valid,error,execution_mode,"
              "candidate_single_core_m,candidate_single_core_n,candidate_single_core_k,"
              "candidate_base_m,candidate_base_n,candidate_base_k,candidate_traverse,candidate_db_a,candidate_db_b,candidate_split_k,"
              "used_core_num,official_core_num,official_m_dim,official_n_dim,"
              "m_core_parts,n_core_parts,k_core_parts,single_core_m,single_core_n,single_core_k,"
              "base_m,base_n,base_k,depth_a1,depth_b1,"
              "step_m,step_n,step_ka,step_kb,iterate_order,db_l0a,db_l0b,db_l0c,"
              "share_l1_size,share_l0c_size,share_ub_size,"
              "proxy_total,critical_core_id,critical_core_cycles,average_core_cycles,pipeline_cycles,"
              "proxy_cube_cycles,proxy_gm_cycles,proxy_l1_cycles,proxy_mte2_cycles,proxy_mte1_cycles,"
              "proxy_fixpipe_cycles,proxy_scalar_cycles,fill_drain_cycles,proxy_launch_cycles,"
              "tail_penalty,balance_penalty,split_k_penalty,tail_efficiency,core_utilization,arithmetic_intensity,estimated_gm_bytes,"
              "estimated_a_gm_bytes,estimated_b_gm_bytes,estimated_c_gm_bytes,estimated_mte1_bytes,"
              "l1_cache_hit_rate,estimated_mmad_count,estimated_output_tile_count,"
              "official_return,tiling_signature,tiling_bin\n";

    output << std::setprecision(12);
    std::unordered_map<std::string, size_t> workloadRanks;
    for (size_t i = 0; i < evaluations.size(); ++i) {
        const auto &e = evaluations[i];
        const bool isBaseline = e.source == "official_default";
        const size_t workloadRank = isBaseline ? 0 : ++workloadRanks[e.workload.id];
        std::string binPath;
        if (e.valid && !tilingDirectory.empty()) {
            binPath = (std::filesystem::path(tilingDirectory) /
                (SafeName(e.workload.id) +
                 (isBaseline ? "_baseline.bin" : "_rank" + std::to_string(workloadRank) + ".bin"))).string();
            if (!WriteTilingBinary(binPath, e.tiling, error)) return false;
        }
        if (includeRank) output << workloadRank << ',';
        output << Escape(e.workload.id) << ',' << e.workload.m << ',' << e.workload.n << ',' << e.workload.k << ','
               << DTypeToString(e.workload.dtype) << ',' << e.workload.transA << ',' << e.workload.transB << ','
               << e.workload.maxCores << ',' << Escape(e.source) << ','
               << (isBaseline ? "api_auto_baseline" : "searched") << ','
               << e.sourceIteration << ',' << e.valid << ','
               << Escape(e.error) << ',' << ExecutionModeToString(e.executionMode) << ','
               << e.candidate.singleM << ',' << e.candidate.singleN << ',' << e.candidate.singleK << ','
               << e.candidate.baseM << ',' << e.candidate.baseN << ',' << e.candidate.baseK << ','
               << e.candidate.traverse << ',' << e.candidate.dbA << ',' << e.candidate.dbB << ',' << e.candidate.splitK << ','
               << e.tiling.usedCoreNum << ',' << e.officialCoreNum << ','
               << e.officialMDim << ',' << e.officialNDim << ','
               << CeilDiv(e.workload.m, e.tiling.singleCoreM) << ','
               << CeilDiv(e.workload.n, e.tiling.singleCoreN) << ','
               << CeilDiv(e.workload.k, e.tiling.singleCoreK) << ','
               << e.tiling.singleCoreM << ',' << e.tiling.singleCoreN << ','
               << e.tiling.singleCoreK << ',' << e.tiling.baseM << ',' << e.tiling.baseN << ',' << e.tiling.baseK << ','
               << e.tiling.depthA1 << ',' << e.tiling.depthB1 << ',' << e.tiling.stepM << ',' << e.tiling.stepN << ','
               << e.tiling.stepKa << ',' << e.tiling.stepKb << ',' << e.tiling.iterateOrder << ',' << e.tiling.dbL0A << ','
               << e.tiling.dbL0B << ',' << e.tiling.dbL0C << ','
               << e.tiling.shareL1Size << ',' << e.tiling.shareL0CSize << ',' << e.tiling.shareUbSize << ','
               << e.proxy.total << ',' << e.proxy.criticalCoreId << ',' << e.proxy.criticalCoreCycles << ','
               << e.proxy.averageCoreCycles << ',' << e.proxy.pipelineCycles << ',' << e.proxy.cubeCycles << ','
               << e.proxy.gmCycles << ',' << e.proxy.l1Cycles << ',' << e.proxy.mte2Cycles << ','
               << e.proxy.mte1Cycles << ',' << e.proxy.fixpipeCycles << ',' << e.proxy.scalarCycles << ','
               << e.proxy.fillDrainCycles << ',' << e.proxy.launchCycles << ',' << e.proxy.tailPenalty << ','
               << e.proxy.balancePenalty << ',' << e.proxy.splitKPenalty << ',' << e.proxy.tailEfficiency << ','
               << e.proxy.coreUtilization << ',' << e.proxy.arithmeticIntensity << ',' << e.proxy.estimatedGmBytes << ','
               << e.proxy.estimatedAGmBytes << ',' << e.proxy.estimatedBGmBytes << ',' << e.proxy.estimatedCGmBytes << ','
               << e.proxy.estimatedMte1Bytes << ',' << e.proxy.l1CacheHitRate << ',' << e.proxy.estimatedMmadCount << ','
               << e.proxy.estimatedOutputTileCount << ','
               << e.officialReturn << ',' << Escape(e.tilingSignature) << ',' << Escape(binPath) << '\n';
    }
    return true;
}

}  // namespace matmul_search

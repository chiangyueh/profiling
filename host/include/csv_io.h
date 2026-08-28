#pragma once

#include <string>
#include <vector>
#include "search_types.h"

namespace matmul_search {

bool LoadWorkloadsCsv(const std::string &path, std::vector<Workload> &workloads, std::string &error);
bool WriteEvaluationsCsv(const std::string &path, const std::vector<Evaluation> &evaluations,
                         bool includeRank, const std::string &tilingDirectory, std::string &error);
bool WriteTilingBinary(const std::string &path, const TCubeTiling &tiling, std::string &error);

}  // namespace matmul_search

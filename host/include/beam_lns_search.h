#pragma once

#include <unordered_map>
#include <vector>
#include "official_tiling.h"
#include "proxy_model.h"

namespace matmul_search {

struct SearchResult {
    std::vector<Evaluation> evaluated;
    std::vector<Evaluation> top;
};

class BeamLnsSearcher {
public:
    BeamLnsSearcher(const OfficialTilingEngine &engine, SearchOptions options = {}, ProxyWeights weights = {});
    SearchResult Search(const Workload &workload);

private:
    struct Partial {
        Candidate candidate;
        double score = 0.0;
    };

    const OfficialTilingEngine &engine_;
    SearchOptions options_;
    ProxyModel proxy_;
    std::vector<Candidate> coreSplits_;
    std::vector<int32_t> mValues_;
    std::vector<int32_t> nValues_;
    std::vector<int32_t> kValues_;
    std::unordered_map<std::string, Evaluation> cache_;
    std::vector<std::string> evaluationOrder_;

    std::vector<int32_t> BuildAxisValues(int32_t shape, int32_t alignment, int32_t maxValue) const;
    std::vector<Candidate> BuildCoreSplits(const Workload &workload) const;
    bool HardLegal(const Workload &workload, const Candidate &candidate, bool requireAll) const;
    double RoughScore(const Workload &workload, const Candidate &candidate, int stage) const;
    Evaluation Evaluate(const Workload &workload, const Candidate &candidate,
                        const std::string &source, int32_t iteration);
    std::vector<Evaluation> RunBeam(const Workload &workload);
    void RunTabuLns(const Workload &workload, const std::vector<Evaluation> &beamBest);
    std::vector<Candidate> Neighbors(const Workload &workload, const Candidate &candidate) const;
    std::vector<Candidate> LargeNeighborhood(const Workload &workload, const Candidate &center, int32_t round) const;
    std::vector<Evaluation> CollectTop(int32_t topK) const;
};

}  // namespace matmul_search

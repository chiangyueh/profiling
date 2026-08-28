#include "beam_lns_search.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <random>
#include <set>
#include <unordered_set>

namespace matmul_search {
namespace {

template <typename T>
T CeilDiv(T x, T y)
{
    return y <= 0 ? 0 : (x + y - 1) / y;
}

int32_t RoundUp(int32_t x, int32_t alignment)
{
    return static_cast<int32_t>(CeilDiv<int64_t>(x, alignment) * alignment);
}

int FindIndex(const std::vector<int32_t> &values, int32_t value)
{
    const auto it = std::lower_bound(values.begin(), values.end(), value);
    if (it == values.end()) {
        return static_cast<int>(values.size()) - 1;
    }
    if (it == values.begin()) {
        return 0;
    }
    const auto hi = static_cast<int>(it - values.begin());
    const auto lo = hi - 1;
    return std::abs(values[lo] - value) <= std::abs(values[hi] - value) ? lo : hi;
}

int32_t BaseKAlignment(const Workload &w)
{
    return w.dtype == DType::FP32 && !w.transA && w.transB ? 8 : 16;
}

}  // namespace

BeamLnsSearcher::BeamLnsSearcher(
    const OfficialTilingEngine &engine, SearchOptions options, MatmulPathParameters parameters)
    : engine_(engine), options_(options), proxy_(engine.Caps(), parameters)
{
}

std::vector<int32_t> BeamLnsSearcher::BuildAxisValues(
    int32_t shape, int32_t alignment, int32_t maxValue) const
{
    const int32_t cap = std::max(alignment, std::min(maxValue, std::max(alignment, RoundUp(shape, alignment))));
    std::set<int32_t> values;
    const int32_t denseCap = std::min(cap, 256);
    for (int32_t v = alignment; v <= denseCap; v += alignment) {
        values.insert(v);
    }
    for (int32_t v : {288, 320, 384, 448, 512, 640, 768, 896, 1024}) {
        const int32_t aligned = RoundUp(v, alignment);
        if (aligned <= cap) {
            values.insert(aligned);
        }
    }
    values.insert(std::min(cap, RoundUp(shape, alignment)));
    return {values.begin(), values.end()};
}

std::vector<Candidate> BeamLnsSearcher::BuildCoreSplits(const Workload &w) const
{
    (void)w;
    // This first stage asks the official MultiCoreMatmulTiling API for legal
    // BASE tile seeds only. MatMulV3-specific single-core partitioning,
    // Split-K and full-load families are generated from their exact callback
    // contracts in refine_matmul_v3_candidates.py.
    Candidate base;
    base.dbA = false;
    base.dbB = false;
    return {base};
}

bool BeamLnsSearcher::HardLegal(
    const Workload &w, const Candidate &c, bool requireAll) const
{
    const int32_t k0 = BaseKAlignment(w);
    const int32_t coreLimit = std::max(1, std::min(w.maxCores, engine_.Caps().coreNum));
    if (c.splitK) {
        return false;
    }
    if (c.traverse < 0 || c.traverse > 2) {
        return false;
    }
    const bool hasSingleM = c.singleM > 0;
    const bool hasSingleN = c.singleN > 0;
    if (hasSingleM != hasSingleN) {
        return false;
    }
    if (hasSingleM) {
        if (c.singleM % 16 != 0 || c.singleN % 16 != 0 ||
            c.singleM > RoundUp(w.m, 16) || c.singleN > RoundUp(w.n, 16)) {
            return false;
        }
        if (c.singleK > 0 &&
            (c.singleK > RoundUp(w.k, k0) || (c.splitK && c.singleK % k0 != 0))) {
            return false;
        }
        const int64_t mParts = CeilDiv<int64_t>(w.m, c.singleM);
        const int64_t nParts = CeilDiv<int64_t>(w.n, c.singleN);
        const int64_t kParts = c.singleK > 0 ? CeilDiv<int64_t>(w.k, c.singleK) : 1;
        const int64_t logicalTiles = mParts * nParts * kParts;
        const int64_t tileLimit =
            static_cast<int64_t>(coreLimit) * options_.maxCoreRounds;
        if (options_.maxCoreRounds > 0 && logicalTiles > tileLimit) {
            return false;
        }
        if (kParts > 1) {
            return false;
        }
    } else if (c.singleK > 0) {
        return false;
    }
    if (c.baseM > 0 && (c.baseM % 16 != 0 || c.baseM > options_.maxBaseM)) {
        return false;
    }
    if (c.baseN > 0 && (c.baseN % 16 != 0 || c.baseN > options_.maxBaseN)) {
        return false;
    }
    if (c.baseK > 0 && (c.baseK % k0 != 0 || c.baseK > options_.maxBaseK)) {
        return false;
    }
    if (requireAll && (c.baseM <= 0 || c.baseN <= 0 || c.baseK <= 0)) {
        return false;
    }
    if (requireAll && !hasSingleM) {
        // This generic seed stage intentionally asks MultiCoreMatmulTiling for
        // SetSingleShape(baseM, baseN, -1). The later MatMulV3-specific stage
        // independently searches larger singleCoreM/N partitions.
        const int64_t mParts = CeilDiv<int64_t>(w.m, c.baseM);
        const int64_t nParts = CeilDiv<int64_t>(w.n, c.baseN);
        const int64_t logicalTiles = mParts * nParts;
        const int64_t tileLimit =
            static_cast<int64_t>(coreLimit) * options_.maxCoreRounds;
        if (options_.maxCoreRounds > 0 && logicalTiles > tileLimit) {
            return false;
        }
    }
    if (c.singleM > 0 &&
        ((c.baseM > 0 && c.baseM > RoundUp(c.singleM, 16)) ||
         (c.baseN > 0 && c.baseN > RoundUp(c.singleN, 16)) ||
         (c.singleK > 0 && c.baseK > 0 && c.baseK > RoundUp(c.singleK, k0)))) {
        return false;
    }

    const auto &caps = engine_.Caps();
    const uint64_t inputBytes = static_cast<uint64_t>(DTypeBytes(w.dtype));
    if (c.baseM > 0 && c.baseN > 0) {
        const uint64_t l0c = static_cast<uint64_t>(c.baseM) * c.baseN * AccumulatorBytes(w.dtype);
        if (l0c > caps.l0cBytes) {
            return false;
        }
    }
    if (c.baseM > 0 && c.baseK > 0) {
        const uint64_t l0a = static_cast<uint64_t>(c.baseM) * c.baseK * inputBytes;
        if (l0a > caps.l0aBytes) {
            return false;
        }
    }
    if (c.baseN > 0 && c.baseK > 0) {
        const uint64_t l0b = static_cast<uint64_t>(c.baseN) * c.baseK * inputBytes;
        if (l0b > caps.l0bBytes) {
            return false;
        }
    }
    return true;
}

double BeamLnsSearcher::RoughScore(
    const Workload &w, const Candidate &c, int stage) const
{
    (void)stage;
    double score = 0.0;
    if (c.singleM > 0 && c.singleN > 0) {
        const int64_t kSingle = c.singleK > 0 ? c.singleK : w.k;
        const int64_t mParts = CeilDiv<int64_t>(w.m, c.singleM);
        const int64_t nParts = CeilDiv<int64_t>(w.n, c.singleN);
        const int64_t kParts = c.singleK > 0 ? CeilDiv<int64_t>(w.k, c.singleK) : 1;
        const int64_t used = mParts * nParts * kParts;
        const int64_t available = std::max(1, std::min(w.maxCores, engine_.Caps().coreNum));
        const double padded = static_cast<double>(c.singleM) * c.singleN * kSingle * used;
        const double actual = static_cast<double>(w.m) * w.n * w.k;
        const double tailEfficiency = std::min(1.0, actual / std::max(1.0, padded));
        const double utilization = std::min(1.0, static_cast<double>(used) / available);
        const double rounds = std::ceil(static_cast<double>(used) / available);
        const double aTraffic = static_cast<double>(w.m) * w.k * nParts * kParts;
        const double bTraffic = static_cast<double>(w.k) * w.n * mParts * kParts;
        const double logicalInput = std::max(1.0,
            static_cast<double>(w.m) * w.k + static_cast<double>(w.k) * w.n);
        score += 5.0 * (1.0 - utilization) +
            3.0 * (1.0 - tailEfficiency) +
            0.08 * std::max(0.0, rounds - 1.0) +
            0.02 * (aTraffic + bTraffic) / logicalInput;
    }
    if (c.baseK > 0) {
        const double paddedK = static_cast<double>(CeilDiv<int64_t>(w.k, c.baseK) * c.baseK);
        const double kEfficiency = static_cast<double>(w.k) / paddedK;
        score += 3.0 * (1.0 - kEfficiency) + 16.0 / c.baseK;
    }
    if (c.baseM > 0) {
        const double paddedM = static_cast<double>(CeilDiv<int64_t>(w.m, c.baseM) * c.baseM);
        const double mEfficiency = static_cast<double>(w.m) / paddedM;
        score += 4.0 * (1.0 - mEfficiency) + 32.0 / c.baseM;
    }
    if (c.baseM > 0 && c.baseN > 0) {
        const double paddedN = static_cast<double>(CeilDiv<int64_t>(w.n, c.baseN) * c.baseN);
        const double nEfficiency = static_cast<double>(w.n) / paddedN;
        const double intensity = 2.0 * c.baseM * c.baseN /
            std::max(1.0, static_cast<double>(DTypeBytes(w.dtype)) * (c.baseM + c.baseN));
        const double l0cUse = static_cast<double>(c.baseM) * c.baseN * AccumulatorBytes(w.dtype) /
            std::max<uint64_t>(1, engine_.Caps().l0cBytes);
        score += 4.0 * (1.0 - nEfficiency) - 0.02 * intensity + 0.15 * std::abs(0.65 - l0cUse);
    }
    if (c.baseM > 0 && c.baseN > 0 && c.baseK > 0) {
        const double l0aUse = static_cast<double>(c.baseM) * c.baseK * DTypeBytes(w.dtype) * (c.dbA ? 2 : 1) /
            std::max<uint64_t>(1, engine_.Caps().l0aBytes);
        const double l0bUse = static_cast<double>(c.baseN) * c.baseK * DTypeBytes(w.dtype) * (c.dbB ? 2 : 1) /
            std::max<uint64_t>(1, engine_.Caps().l0bBytes);
        const double loops = static_cast<double>(CeilDiv<int64_t>(w.m, c.baseM)) *
            CeilDiv<int64_t>(w.n, c.baseN) * CeilDiv<int64_t>(w.k, c.baseK);
        score += 0.000001 * loops +
            0.10 * std::abs(0.70 - l0aUse) + 0.10 * std::abs(0.70 - l0bUse);
    }
    return score;
}

Evaluation BeamLnsSearcher::Evaluate(
    const Workload &w, const Candidate &candidate, const std::string &source, int32_t iteration)
{
    const std::string key = candidate.Key();
    const auto found = cache_.find(key);
    if (found != cache_.end()) {
        return found->second;
    }

    Evaluation evaluation = engine_.Evaluate(w, candidate);
    evaluation.source = source;
    evaluation.sourceIteration = iteration;
    if (evaluation.valid) {
        evaluation.proxy = proxy_.Score(w, candidate, evaluation.tiling);
    } else {
        evaluation.proxy.total = std::numeric_limits<double>::infinity();
    }
    cache_[key] = evaluation;
    evaluationOrder_.push_back(key);
    return evaluation;
}

std::vector<Evaluation> BeamLnsSearcher::RunBeam(const Workload &w)
{
    const int32_t k0 = BaseKAlignment(w);
    coreSplits_ = BuildCoreSplits(w);
    mValues_ = BuildAxisValues(w.m, 16, options_.maxBaseM);
    nValues_ = BuildAxisValues(w.n, 16, options_.maxBaseN);
    kValues_ = BuildAxisValues(w.k, k0, options_.maxBaseK);

    std::vector<Partial> beam;
    for (const Candidate &c : coreSplits_) {
        if (HardLegal(w, c, false)) {
            beam.push_back({c, RoughScore(w, c, 0)});
        }
    }
    auto trim = [this](std::vector<Partial> &states) {
        std::sort(states.begin(), states.end(), [](const Partial &a, const Partial &b) {
            if (a.score != b.score) return a.score < b.score;
            return a.candidate.Key() < b.candidate.Key();
        });
        const size_t target = static_cast<size_t>(options_.beamWidth);
        if (states.size() <= target) {
            return;
        }

        states.resize(target);
    };
    trim(beam);

    std::vector<Partial> expanded;
    // NPUMeter's guided search first grows K to use L0A/L0B, then trades K
    // for M/N to improve L0C reuse. Keep that dependency order in the beam.
    for (const auto &state : beam) {
        for (int32_t k : kValues_) {
            Candidate c = state.candidate;
            c.baseK = k;
            if (HardLegal(w, c, false)) {
                expanded.push_back({c, RoughScore(w, c, 1)});
            }
        }
    }
    trim(expanded);
    beam.swap(expanded);
    expanded.clear();

    for (const auto &state : beam) {
        for (int32_t m : mValues_) {
            Candidate c = state.candidate;
            c.baseM = m;
            if (HardLegal(w, c, false)) {
                expanded.push_back({c, RoughScore(w, c, 2)});
            }
        }
    }
    trim(expanded);
    beam.swap(expanded);
    expanded.clear();

    for (const auto &state : beam) {
        for (int32_t n : nValues_) {
            Candidate c = state.candidate;
            c.baseN = n;
            if (HardLegal(w, c, true)) {
                expanded.push_back({c, RoughScore(w, c, 3)});
            }
        }
    }
    trim(expanded);
    beam.swap(expanded);

    std::vector<Evaluation> valid;
    int32_t sequence = 0;
    for (const auto &state : beam) {
        for (int32_t traverse : {0, 1, 2}) {
            for (int32_t dbMask = 0; dbMask < 4; ++dbMask) {
                Candidate c = state.candidate;
                c.traverse = traverse;
                c.dbA = (dbMask & 1) != 0;
                c.dbB = (dbMask & 2) != 0;
                if (!HardLegal(w, c, true)) {
                    continue;
                }
                Evaluation e = Evaluate(w, c, "beam", sequence++);
                if (e.valid) {
                    valid.push_back(e);
                }
            }
        }
    }

    std::sort(valid.begin(), valid.end(), [](const Evaluation &a, const Evaluation &b) {
        if (a.proxy.total != b.proxy.total) return a.proxy.total < b.proxy.total;
        return a.candidate.Key() < b.candidate.Key();
    });
    return valid;
}

std::vector<Candidate> BeamLnsSearcher::Neighbors(const Workload &w, const Candidate &center) const
{
    std::vector<Candidate> out;
    std::unordered_set<std::string> seen;
    auto add = [&](Candidate c) {
        if (seen.insert(c.Key()).second) {
            out.push_back(c);
        }
    };

    const int mi = FindIndex(mValues_, center.baseM);
    const int ni = FindIndex(nValues_, center.baseN);
    const int ki = FindIndex(kValues_, center.baseK);
    int splitIndex = 0;
    int64_t splitDistance = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < coreSplits_.size(); ++i) {
        const Candidate &split = coreSplits_[i];
        if (split.singleM == center.singleM && split.singleN == center.singleN &&
            split.singleK == center.singleK) {
            splitIndex = static_cast<int>(i);
            splitDistance = 0;
            break;
        }
        const int64_t distance =
            std::abs(static_cast<int64_t>(split.singleM) - center.singleM) +
            std::abs(static_cast<int64_t>(split.singleN) - center.singleN) +
            std::abs(static_cast<int64_t>(split.singleK) - center.singleK);
        if (distance < splitDistance) {
            splitDistance = distance;
            splitIndex = static_cast<int>(i);
        }
    }
    for (int delta : {-2, -1, 1, 2}) {
        if (splitIndex + delta >= 0 &&
            splitIndex + delta < static_cast<int>(coreSplits_.size())) {
            const Candidate &split = coreSplits_[splitIndex + delta];
            Candidate c = center;
            c.singleM = split.singleM;
            c.singleN = split.singleN;
            c.singleK = split.singleK;
            c.splitK = false;
            add(c);
        }
        if (mi + delta >= 0 && mi + delta < static_cast<int>(mValues_.size())) {
            Candidate c = center;
            c.baseM = mValues_[mi + delta];
            add(c);
        }
        if (ni + delta >= 0 && ni + delta < static_cast<int>(nValues_.size())) {
            Candidate c = center;
            c.baseN = nValues_[ni + delta];
            add(c);
        }
        if (ki + delta >= 0 && ki + delta < static_cast<int>(kValues_.size())) {
            Candidate c = center;
            c.baseK = kValues_[ki + delta];
            add(c);
        }
    }
    for (int traverse : {0, 1, 2}) {
        if (traverse != center.traverse) {
            Candidate c = center;
            c.traverse = traverse;
            add(c);
        }
    }
    Candidate c = center;
    c.dbA = !c.dbA;
    add(c);
    c = center;
    c.dbB = !c.dbB;
    add(c);
    (void)w;
    return out;
}

std::vector<Candidate> BeamLnsSearcher::LargeNeighborhood(
    const Workload &w, const Candidate &center, int32_t round) const
{
    std::vector<Partial> pool;
    const int mode = round % 4;
    if (mode == 0) {
        for (const Candidate &split : coreSplits_) {
            Candidate c = center;
            c.singleM = split.singleM;
            c.singleN = split.singleN;
            c.singleK = split.singleK;
            c.splitK = false;
            if (HardLegal(w, c, true)) pool.push_back({c, RoughScore(w, c, 3)});
        }
    } else if (mode == 1) {
        for (int32_t m : mValues_) {
            for (int32_t n : nValues_) {
                Candidate c = center;
                c.baseM = m;
                c.baseN = n;
                if (HardLegal(w, c, true)) pool.push_back({c, RoughScore(w, c, 3)});
            }
        }
    } else if (mode == 2) {
        for (int32_t k : kValues_) {
            for (int dbMask = 0; dbMask < 4; ++dbMask) {
                Candidate c = center;
                c.baseK = k;
                c.dbA = (dbMask & 1) != 0;
                c.dbB = (dbMask & 2) != 0;
                if (HardLegal(w, c, true)) pool.push_back({c, RoughScore(w, c, 3)});
            }
        }
    } else {
        for (int32_t traverse : {0, 1, 2}) {
            for (int deltaM : {-2, -1, 0, 1, 2}) {
                for (int deltaN : {-2, -1, 0, 1, 2}) {
                    Candidate c = center;
                    const int mi = std::clamp(FindIndex(mValues_, center.baseM) + deltaM,
                                              0, static_cast<int>(mValues_.size()) - 1);
                    const int ni = std::clamp(FindIndex(nValues_, center.baseN) + deltaN,
                                              0, static_cast<int>(nValues_.size()) - 1);
                    c.baseM = mValues_[mi];
                    c.baseN = nValues_[ni];
                    c.traverse = traverse;
                    if (HardLegal(w, c, true)) pool.push_back({c, RoughScore(w, c, 3)});
                }
            }
        }
    }
    std::sort(pool.begin(), pool.end(), [](const Partial &a, const Partial &b) {
        return a.score < b.score;
    });
    const size_t limit = std::min<size_t>(48, pool.size());
    std::vector<Candidate> result;
    std::unordered_set<std::string> seen;
    for (size_t i = 0; i < limit; ++i) {
        if (seen.insert(pool[i].candidate.Key()).second) result.push_back(pool[i].candidate);
    }
    return result;
}

void BeamLnsSearcher::RunTabuLns(
    const Workload &w, const std::vector<Evaluation> &beamBest)
{
    if (beamBest.empty()) return;
    const int32_t seedCount = std::min<int32_t>(options_.seedCount, beamBest.size());
    const int32_t iterationsPerSeed = std::max(1, options_.tabuIterations / seedCount);
    int32_t globalIteration = 0;

    for (int32_t seed = 0; seed < seedCount; ++seed) {
        Evaluation current = beamBest[seed];
        Evaluation best = current;
        std::deque<std::string> tabuQueue;
        std::unordered_set<std::string> tabu;
        auto markTabu = [&](const std::string &key) {
            if (tabu.insert(key).second) tabuQueue.push_back(key);
            while (tabuQueue.size() > 32) {
                tabu.erase(tabuQueue.front());
                tabuQueue.pop_front();
            }
        };
        markTabu(current.candidate.Key());

        for (int32_t iter = 0; iter < iterationsPerSeed; ++iter, ++globalIteration) {
            std::vector<Candidate> candidates = Neighbors(w, current.candidate);
            if (options_.lnsRounds > 0 && iter % std::max(1, iterationsPerSeed / options_.lnsRounds) == 0) {
                auto lns = LargeNeighborhood(w, current.candidate, iter);
                candidates.insert(candidates.end(), lns.begin(), lns.end());
            }

            Evaluation next;
            next.proxy.total = std::numeric_limits<double>::infinity();
            bool found = false;
            for (const Candidate &candidate : candidates) {
                if (!HardLegal(w, candidate, true)) continue;
                const std::string key = candidate.Key();
                Evaluation e = Evaluate(w, candidate, iter % 2 == 0 ? "tabu" : "lns", globalIteration);
                if (!e.valid) continue;
                const bool aspiration = e.proxy.total < best.proxy.total;
                if (tabu.count(key) != 0 && !aspiration) continue;
                if (!found || e.proxy.total < next.proxy.total) {
                    next = e;
                    found = true;
                }
            }
            if (!found) break;
            current = next;
            markTabu(current.candidate.Key());
            if (current.proxy.total < best.proxy.total) best = current;
        }
    }
}

std::vector<Evaluation> BeamLnsSearcher::CollectTop(int32_t topK) const
{
    std::vector<Evaluation> valid;
    std::unordered_set<std::string> signatures;
    for (const auto &key : evaluationOrder_) {
        const auto it = cache_.find(key);
        if (it == cache_.end() || !it->second.valid) continue;
        if (it->second.source == "official_default") {
            signatures.insert(it->second.tilingSignature);
            continue;
        }
        valid.push_back(it->second);
    }
    std::sort(valid.begin(), valid.end(), [](const Evaluation &a, const Evaluation &b) {
        if (a.proxy.total != b.proxy.total) return a.proxy.total < b.proxy.total;
        return a.candidate.Key() < b.candidate.Key();
    });

    std::vector<Evaluation> unique;
    for (const auto &e : valid) {
        if (signatures.insert(e.tilingSignature).second) {
            unique.push_back(e);
            if (unique.size() >= static_cast<size_t>(topK)) break;
        }
    }
    return unique;
}

SearchResult BeamLnsSearcher::Search(const Workload &workload)
{
    cache_.clear();
    evaluationOrder_.clear();

    Candidate baseline;
    Evaluation baselineEval = engine_.Evaluate(workload, baseline);
    baselineEval.source = "official_default";
    baselineEval.sourceIteration = 0;
    if (baselineEval.valid) {
        baselineEval.proxy = proxy_.Score(workload, baseline, baselineEval.tiling);
    } else {
        baselineEval.proxy.total = std::numeric_limits<double>::infinity();
    }
    cache_[baseline.Key()] = baselineEval;
    evaluationOrder_.push_back(baseline.Key());

    auto beamBest = RunBeam(workload);

    // The unconstrained official result is a proven legal tiling, but its
    // Candidate uses -1 hints and therefore has no meaningful neighborhood.
    // Materialize its final baseMNK as a search center. Even if SetFixSplit
    // cannot reproduce that exact result, Tabu/LNS can still evaluate the
    // adjacent legal hint values instead of stopping when Beam found no valid
    // fixed hint.
    if (baselineEval.valid) {
        Candidate officialCenter;
        const bool baselineSplitK =
            CeilDiv<int64_t>(workload.k, baselineEval.tiling.singleCoreK) > 1;
        officialCenter.baseM = baselineEval.tiling.baseM;
        officialCenter.baseN = baselineEval.tiling.baseN;
        officialCenter.baseK = baselineEval.tiling.baseK;
        officialCenter.traverse =
            baselineEval.tiling.iterateOrder == 0 ? 1 : 2;
        officialCenter.dbA = baselineEval.tiling.dbL0A == 2;
        officialCenter.dbB = baselineEval.tiling.dbL0B == 2;
        officialCenter.splitK = false;

        if (!baselineSplitK && HardLegal(workload, officialCenter, true)) {
            Evaluation concrete = Evaluate(workload, officialCenter, "official_seed", 0);
            if (concrete.valid) {
                beamBest.push_back(concrete);
            } else {
                Evaluation neighborhoodCenter = baselineEval;
                neighborhoodCenter.candidate = officialCenter;
                neighborhoodCenter.source = "official_neighborhood_seed";
                beamBest.push_back(neighborhoodCenter);
            }
        }
        std::sort(beamBest.begin(), beamBest.end(), [](const Evaluation &a, const Evaluation &b) {
            if (a.proxy.total != b.proxy.total) return a.proxy.total < b.proxy.total;
            return a.candidate.Key() < b.candidate.Key();
        });
    }
    RunTabuLns(workload, beamBest);

    SearchResult result;
    for (const auto &key : evaluationOrder_) {
        result.evaluated.push_back(cache_.at(key));
    }
    if (baselineEval.valid) {
        result.top.push_back(baselineEval);
    }
    auto searchedTop = CollectTop(options_.topK);
    result.top.insert(result.top.end(), searchedTop.begin(), searchedTop.end());
    return result;
}

}  // namespace matmul_search

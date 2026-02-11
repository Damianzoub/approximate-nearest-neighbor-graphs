// HNSWDarth.cpp
#include "HNSWDarth.h"
#include <limits>
#include <utility>

namespace {

struct Cand {
    float d;
    int id;
};
struct CandMinCmp {
    bool operator()(const Cand& a, const Cand& b) const {
        return a.d > b.d; // min-heap by distance
    }
};

struct Res {
    float d;
    int id;
};
struct ResMaxCmp {
    bool operator()(const Res& a, const Res& b) const {
        return a.d < b.d; // max-heap by distance (worst on top)
    }
};

inline bool try_insert_result(std::priority_queue<Res, std::vector<Res>, ResMaxCmp>& rs,
                              float d, int id, int k, int& ninserts) {
    if ((int)rs.size() < k) {
        rs.push({d, id});
        ninserts++;
        return true;
    }
    float worst = rs.top().d;
    if (d < worst) {
        rs.pop();
        rs.push({d, id});
        ninserts++;
        return true;
    }
    return false;
}

inline float worst_result_dist(const std::priority_queue<Res, std::vector<Res>, ResMaxCmp>& rs, int k) {
    if ((int)rs.size() < k) return std::numeric_limits<float>::infinity();
    return rs.top().d;
}

inline void dump_sorted(const std::priority_queue<Res, std::vector<Res>, ResMaxCmp>& rs,
                        std::vector<std::pair<float,int>>& out_sorted) {
    out_sorted.clear();
    out_sorted.reserve(rs.size());
    auto tmp = rs;
    while (!tmp.empty()) {
        out_sorted.push_back({tmp.top().d, tmp.top().id});
        tmp.pop();
    }
    std::sort(out_sorted.begin(), out_sorted.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
}

} // namespace

std::vector<int> DarthSearcher::search_layer0_darth(
    const float* q,
    int ep_id,
    const DarthParams& params,
    const IRecallPredictor& model,
    NeighFn neigh,
    DistFn dist,
    void* ctx
) {
    if (!q) return {};
    if (params.k <= 0) return {};
    if (params.efSearch <= 0) return {};
    if (params.ipi <= 0) return {};
    if (params.mpi <= 0) return {};

    std::priority_queue<Cand, std::vector<Cand>, CandMinCmp> C;
    std::priority_queue<Res,  std::vector<Res>,  ResMaxCmp>  R;

    std::unordered_set<int> visited;
    visited.reserve((size_t)params.efSearch * 8u);

    // counters/features
    int ndis = 0;      // total distance computations
    int nstep = 0;     // number of pops from candidate queue
    int ninserts = 0;  // number of result-set updates
    int idis = 0;      // distance computations since last prediction

    float d0 = dist(q, ep_id, ctx); ndis++; idis++;
    float firstNN = d0;

    visited.insert(ep_id);
    C.push({d0, ep_id});
    (void)try_insert_result(R, d0, ep_id, params.k, ninserts);

    int pi = std::max(1, params.ipi);

    while (!C.empty()) {
        Cand cur = C.top(); C.pop();
        nstep++;

        float worst = worst_result_dist(R, params.k);
        if (cur.d > worst) break;

        const std::vector<int>& nbs = neigh(cur.id, ctx);
        for (int nb : nbs) {
            if (visited.find(nb) != visited.end()) continue;

            float d = dist(q, nb, ctx); ndis++; idis++;

            bool pushed = false;

            // keep exploration alive under a soft cap
            worst = worst_result_dist(R, params.k);
            if ((int)C.size() < params.efSearch || (int)R.size() < params.k || d < worst) {
                C.push({d, nb});
                pushed = true;
            }

            bool inserted = try_insert_result(R, d, nb, params.k, ninserts);

            // IMPORTANT: mark visited only if node enters frontier or result
            if (pushed || inserted) visited.insert(nb);

            // Predictor call every pi distance computations
            if (idis >= pi) {
                idis = 0;

                std::vector<std::pair<float,int>> sorted;
                dump_sorted(R, sorted);

                std::vector<float> dists_sorted;
                dists_sorted.reserve(sorted.size());
                for (auto& p : sorted) dists_sorted.push_back(p.first);

                float closestNN = dists_sorted.empty() ? std::numeric_limits<float>::infinity() : dists_sorted.front();
                float furthestNN = dists_sorted.empty() ? std::numeric_limits<float>::infinity() : dists_sorted.back();

                float avg, var, med, p25, p75;
                nn_stats(dists_sorted, avg, var, med, p25, p75);

                float feat[11] = {
                    (float)nstep,
                    (float)ndis,
                    (float)ninserts,
                    (float)firstNN,
                    (float)closestNN,
                    (float)furthestNN,
                    avg, var, med, p25, p75
                };

                float Rp = model.predict_one(feat);

                if (Rp >= params.Rt) {
                    std::vector<int> out;
                    out.reserve(std::min((int)sorted.size(), params.k));
                    for (int i = 0; i < (int)sorted.size() && (int)out.size() < params.k; ++i) {
                        out.push_back(sorted[i].second);
                    }
                    return out;
                }

                pi = darth_update_pi(params.ipi, params.mpi, params.Rt, Rp);
            }
        }
    }

    // natural termination
    std::vector<std::pair<float,int>> sorted;
    dump_sorted(R, sorted);

    std::vector<int> out;
    out.reserve(std::min((int)sorted.size(), params.k));
    for (int i = 0; i < (int)sorted.size() && (int)out.size() < params.k; ++i) {
        out.push_back(sorted[i].second);
    }
    return out;
}

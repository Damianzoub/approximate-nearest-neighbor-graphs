// hnsw_darth.cpp
#include "hnswDarth.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

HNSW_DARTH::HNSW_DARTH(int dim, int M, int efConstruction, Metric metric, uint64_t seed)
    : dim_(dim),
      M_(M),
      M0_(2 * M),
      efConstruction_(efConstruction),
      metric_(metric),
      maxlevel_(-1),
      entry_id_(-1),
      mL_(1.0 / std::log((double)M)),
      rng_(seed),
      unif_(0.0, 1.0) {
    if (dim_ <= 0) throw std::runtime_error("dim must be >0");
    if (M_ < 2) throw std::runtime_error("M must be >=2");
}

float HNSW_DARTH::dist(const std::vector<float>& a, const std::vector<float>& b) const {
    if (metric_ == Metric::L2) {
        float s = 0.0f;
        for (int i = 0; i < dim_; ++i) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    } else { // Cosine distance = 1 - cos
        double dot=0, na=0, nb=0;
        for (int i = 0; i < dim_; ++i) {
            dot += (double)a[i] * (double)b[i];
            na  += (double)a[i] * (double)a[i];
            nb  += (double)b[i] * (double)b[i];
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom == 0.0) return 1.0f;
        double cosv = dot / denom;
        return (float)(1.0 - cosv);
    }
}

int HNSW_DARTH::sample_level() const {
    double r = -std::log(unif_(rng_)) * mL_;
    return (int)r;
}

static inline void erase_swap(std::vector<int>& v, int x) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == x) {
            v[i] = v.back();
            v.pop_back();
            return;
        }
    }
}

int HNSW_DARTH::insert(const std::vector<float>& vec, int node_id) {
    check_dim(vec);

    if (node_id < 0) {
        node_id = (int)vectors_.size();
        while (vectors_.find(node_id) != vectors_.end()) ++node_id;
    }
    if (vectors_.find(node_id) != vectors_.end()) {
        throw std::runtime_error("Node id already exists");
    }

    vectors_[node_id] = vec;

    int l = sample_level();

    if (entry_id_ < 0) {
        maxlevel_ = l;
        entry_id_ = node_id;
        layers_.resize(maxlevel_ + 1);
        for (int lc = 0; lc <= maxlevel_; ++lc) {
            layers_[lc][node_id] = {};
        }
        return node_id;
    }

    if (l > maxlevel_) {
        layers_.resize(l + 1);
        maxlevel_ = l;
    }

    for (int lc = 0; lc <= l; ++lc) {
        layers_[lc][node_id] = {};
    }

    int ep = entry_id_;

    for (int lc = maxlevel_; lc > l; --lc) {
        ep = search_layer_greedy(vec, ep, lc);
    }

    for (int lc = std::min(l, maxlevel_); lc >= 0; --lc) {
        auto candidates = search_layer(vec, ep, lc, efConstruction_);

        int Mmax = (lc == 0) ? M0_ : M_;

        std::vector<int> selected;
        if (use_heuristic_) {
            selected = select_neighbors_heuristic_paper(vec, candidates, lc, Mmax, true, true);
        } else {
            selected = candidates;
            if ((int)selected.size() > Mmax) selected.resize(Mmax);
        }

        layers_[lc][node_id] = selected;

        for (int nb : selected) {
            auto& nbrs = layers_[lc][nb];
            nbrs.push_back(node_id);
            if ((int)nbrs.size() > Mmax) prune_connections(nb, lc, Mmax);
        }

        if (!selected.empty()) ep = selected[0];
    }

    if (l > maxlevel_) entry_id_ = node_id;
    return node_id;
}

int HNSW_DARTH::search_layer_greedy(const std::vector<float>& q, int ep, int layer) const {
    int cur = ep;
    float curDist = dist(q, vectors_.at(cur));
    bool changed = true;

    while (changed) {
        changed = false;
        const auto it = layers_[layer].find(cur);
        if (it == layers_[layer].end()) break;

        for (int nb : it->second) {
            float d = dist(q, vectors_.at(nb));
            if (d < curDist) {
                curDist = d;
                cur = nb;
                changed = true;
            }
        }
    }
    return cur;
}

std::vector<int> HNSW_DARTH::search_layer(const std::vector<float>& q, int ep_id, int lc, int ef) const {
    using Pair = std::pair<float,int>;

    struct MinCmp { bool operator()(const Pair& a, const Pair& b) const { return a.first > b.first; } };
    struct MaxCmp { bool operator()(const Pair& a, const Pair& b) const { return a.first < b.first; } };

    std::priority_queue<Pair, std::vector<Pair>, MinCmp> cand;
    std::priority_queue<Pair, std::vector<Pair>, MaxCmp> best;

    std::unordered_set<int> visited;
    visited.reserve((size_t)ef * 2);

    float d0 = dist(q, vectors_.at(ep_id));
    cand.push({d0, ep_id});
    best.push({d0, ep_id});
    visited.insert(ep_id);

    while (!cand.empty()) {
        Pair c = cand.top(); cand.pop();

        float worst = best.top().first;
        if (c.first > worst) break;

        const auto it = layers_[lc].find(c.second);
        if (it == layers_[lc].end()) continue;

        for (int nb : it->second) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_.at(nb));
            if ((int)best.size() < ef || d < best.top().first) {
                cand.push({d, nb});
                best.push({d, nb});
                if ((int)best.size() > ef) best.pop();
            }
        }
    }

    std::vector<Pair> tmp;
    tmp.reserve(best.size());
    while (!best.empty()) { tmp.push_back(best.top()); best.pop(); }
    std::sort(tmp.begin(), tmp.end(), [](const Pair& a, const Pair& b){ return a.first < b.first; });

    std::vector<int> ids;
    ids.reserve(tmp.size());
    for (auto& p : tmp) ids.push_back(p.second);
    return ids;
}

std::vector<int> HNSW_DARTH::select_neighbors_heuristic_paper(const std::vector<float>& q,
                                                              const std::vector<int>& candidates,
                                                              int lc,
                                                              int M,
                                                              bool extend_candidates,
                                                              bool keep_pruned_connections) const {
    std::vector<int> cand = candidates;

    if (extend_candidates) {
        std::unordered_set<int> extra;
        for (int c : candidates) {
            const auto it = layers_[lc].find(c);
            if (it == layers_[lc].end()) continue;
            for (int nb : it->second) extra.insert(nb);
        }
        for (int nb : extra) cand.push_back(nb);
    }

    std::vector<std::pair<float,int>> dlist;
    dlist.reserve(cand.size());
    for (int c : cand) dlist.push_back({dist(q, vectors_.at(c)), c});
    std::sort(dlist.begin(), dlist.end(), [](auto& a, auto& b){ return a.first < b.first; });

    std::vector<int> result;
    result.reserve(M);

    std::vector<int> discarded;
    discarded.reserve(dlist.size());

    for (auto& di : dlist) {
        int c = di.second;
        bool good = true;
        for (int r : result) {
            float dcr = dist(vectors_.at(c), vectors_.at(r));
            if (dcr < di.first) { good = false; break; }
        }
        if (good) {
            result.push_back(c);
            if ((int)result.size() == M) break;
        } else if (keep_pruned_connections) {
            discarded.push_back(c);
        }
    }

    if (keep_pruned_connections && (int)result.size() < M) {
        for (int c : discarded) {
            result.push_back(c);
            if ((int)result.size() == M) break;
        }
    }

    return result;
}

void HNSW_DARTH::prune_connections(int node_id, int lc, int Mmax) {
    auto& nbrs = layers_[lc][node_id];

    std::vector<std::pair<float,int>> dlist;
    dlist.reserve(nbrs.size());
    for (int nb : nbrs) dlist.push_back({dist(vectors_.at(node_id), vectors_.at(nb)), nb});
    std::sort(dlist.begin(), dlist.end(), [](auto& a, auto& b){ return a.first < b.first; });

    nbrs.clear();
    for (int i = 0; i < (int)dlist.size() && i < Mmax; ++i) nbrs.push_back(dlist[i].second);
}

HNSW_DARTH::DarthFeatures HNSW_DARTH::darth_extract_features(
    const std::vector<std::pair<float,int>>& result_maxheap,
    int ndis, int nstep, float firstNN, int ninserts
) {
    DarthFeatures f;
    f.ndis = ndis;
    f.nstep = nstep;
    f.ninserts = ninserts;
    f.firstNN = firstNN;

    if (result_maxheap.empty()) return f;

    std::vector<float> ds;
    ds.reserve(result_maxheap.size());
    float minv = std::numeric_limits<float>::infinity();
    float maxv = 0.0f;

    for (auto& p : result_maxheap) {
        float d = p.first;
        ds.push_back(d);
        minv = std::min(minv, d);
        maxv = std::max(maxv, d);
    }

    std::sort(ds.begin(), ds.end());
    f.closestNN = minv;
    f.furthestNN = maxv;

    double mean = 0.0;
    for (float x : ds) mean += x;
    mean /= (double)ds.size();
    f.meanNN = (float)mean;

    double var = 0.0;
    for (float x : ds) {
        double t = (double)x - mean;
        var += t * t;
    }
    var /= (double)ds.size();
    f.varNN = (float)var;

    auto pct = [&](double p) -> float {
        if (ds.empty()) return std::numeric_limits<float>::infinity();
        double idx = p * (ds.size() - 1);
        size_t i0 = (size_t)std::floor(idx);
        size_t i1 = std::min(i0 + 1, ds.size() - 1);
        double a = idx - (double)i0;
        return (float)((1.0 - a) * ds[i0] + a * ds[i1]);
    };

    f.p25NN = pct(0.25);
    f.p50NN = pct(0.50);
    f.p75NN = pct(0.75);

    return f;
}

std::vector<int> HNSW_DARTH::query(const std::vector<float>& q, int k, int efSearch) const {
    check_dim(q);
    if (entry_id_ < 0) return {};

    int ep = entry_id_;

    // greedy descent from top layer down to 1
    for (int lc = maxlevel_; lc > 0; --lc) {
        ep = search_layer_greedy(q, ep, lc);
    }

    // layer 0 best-first search with efSearch
    auto cand = search_layer(q, ep, 0, efSearch);

    // sort defensively by actual distance
    std::vector<std::pair<float,int>> tmp;
    tmp.reserve(cand.size());
    for (int id : cand) tmp.push_back({dist(q, vectors_.at(id)), id});
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first; });

    const int m = std::min((int)tmp.size(), k);
    std::vector<int> out;
    out.reserve(m);
    for (int i = 0; i < m; ++i) out.push_back(tmp[i].second);
    return out;
}
std::vector<int> HNSW_DARTH::search_layer_darth(
    const std::vector<float>& q,
    int ep_id,
    int lc,
    int efSearch,
    int k,
    float Rt,
    const IPredictor& predictor,
    int ipi,
    int mpi
) const
{
    using Pair = std::pair<float,int>;

    struct MinCmp { bool operator()(const Pair& a, const Pair& b) const { return a.first > b.first; } };
    struct MaxCmp { bool operator()(const Pair& a, const Pair& b) const { return a.first < b.first; } };

    std::priority_queue<Pair, std::vector<Pair>, MinCmp> cand;
    std::priority_queue<Pair, std::vector<Pair>, MaxCmp> best;

    std::unordered_set<int> visited;
    visited.reserve((size_t)efSearch * 2);

    float d0 = dist(q, vectors_.at(ep_id));
    cand.push({d0, ep_id});
    best.push({d0, ep_id});
    visited.insert(ep_id);

    int ndis = 1;
    int nstep = 0;
    int ninserts = 1;
    float firstNN = d0;

    while (!cand.empty()) {
        Pair c = cand.top(); cand.pop();
        nstep++;

        float worst = best.top().first;
        if (c.first > worst) break;

        const auto it = layers_[lc].find(c.second);
        if (it == layers_[lc].end()) continue;

        for (int nb : it->second) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_.at(nb));
            ndis++;

            if ((int)best.size() < efSearch || d < best.top().first) {
                cand.push({d, nb});
                best.push({d, nb});
                ninserts++;

                if ((int)best.size() > efSearch)
                    best.pop();
            }
        }

        // ---- DARTH EARLY TERMINATION ----
        if (nstep % ipi == 0 && (int)best.size() >= k) {
            std::vector<Pair> heapCopy;
            auto tmp = best;
            while (!tmp.empty()) {
                heapCopy.push_back(tmp.top());
                tmp.pop();
            }

            auto feats = darth_extract_features(heapCopy, ndis, nstep, firstNN, ninserts);
            float Rp = predictor.predict(feats);

            if (Rp >= Rt)
                break;
        }
    }

    std::vector<Pair> tmp;
    tmp.reserve(best.size());
    while (!best.empty()) {
        tmp.push_back(best.top());
        best.pop();
    }

    std::sort(tmp.begin(), tmp.end(),
              [](const Pair& a, const Pair& b){ return a.first < b.first; });

    const int m = std::min((int)tmp.size(), k);
    std::vector<int> out;
    out.reserve(m);

    for (int i = 0; i < m; ++i)
        out.push_back(tmp[i].second);

    return out;
}

std::vector<int> HNSW_DARTH::query_darth(const std::vector<float>& q,
                                        int k,
                                        int efSearch,
                                        float Rt,
                                        const IPredictor& predictor,
                                        int ipi,
                                        int mpi) const {
    check_dim(q);
    if (entry_id_ < 0) return {};

    int ep = entry_id_;
    for (int lc = maxlevel_; lc > 0; --lc) {
        ep = search_layer_greedy(q, ep, lc);
    }

    // DARTH on layer 0
    return search_layer_darth(q, ep, 0, efSearch, k, Rt, predictor, ipi, mpi);
}

void HNSW_DARTH::search(const std::vector<std::vector<float>>& Xq,
                        int k,
                        int efSearch,
                        std::vector<std::vector<float>>& D,
                        std::vector<std::vector<int>>& I) const {
    const int nq = (int)Xq.size();
    D.assign(nq, std::vector<float>(k, std::numeric_limits<float>::infinity()));
    I.assign(nq, std::vector<int>(k, -1));

    for (int i = 0; i < nq; ++i) {
        auto ids = query(Xq[i], k, efSearch);   // classic HNSW query
        const int m = std::min((int)ids.size(), k);

        for (int j = 0; j < m; ++j) {
            I[i][j] = ids[j];
            D[i][j] = dist(Xq[i], vectors_.at(ids[j]));
        }
    }
}

void HNSW_DARTH::search_darth(const std::vector<std::vector<float>>& Xq,
                              int k,
                              int efSearch,
                              float Rt,
                              const IPredictor& predictor,
                              int ipi,
                              int mpi,
                              std::vector<std::vector<float>>& D,
                              std::vector<std::vector<int>>& I) const {
    const int nq = (int)Xq.size();
    D.assign(nq, std::vector<float>(k, std::numeric_limits<float>::infinity()));
    I.assign(nq, std::vector<int>(k, -1));

    for (int i = 0; i < nq; ++i) {
        auto ids = query_darth(Xq[i], k, efSearch, Rt, predictor, ipi, mpi);
        const int m = std::min((int)ids.size(), k);

        for (int j = 0; j < m; ++j) {
            I[i][j] = ids[j];
            D[i][j] = dist(Xq[i], vectors_.at(ids[j]));
        }
    }
}



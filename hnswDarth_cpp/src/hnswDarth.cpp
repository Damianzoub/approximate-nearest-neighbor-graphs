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
    if (dim_ <= 0) throw std::runtime_error("dim must be > 0");
    if (M_ < 2) throw std::runtime_error("M must be >= 2");
    if (efConstruction_ < 2) efConstruction_ = 2;
}

float HNSW_DARTH::dist(const std::vector<float>& a, const std::vector<float>& b) const {
    if ((int)a.size() != dim_ || (int)b.size() != dim_)
        throw std::runtime_error("Vector dimension mismatch");
    if (metric_ == Metric::L2) {
        double s = 0.0;
        for (int i = 0; i < dim_; ++i) {
            double d = (double)a[i] - (double)b[i];
            s += d * d;
        }
        return (float)s;
    } else { // COSINE (returns 1-cos)
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < dim_; ++i) {
            dot += (double)a[i] * (double)b[i];
            na += (double)a[i] * (double)a[i];
            nb += (double)b[i] * (double)b[i];
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom <= 0.0) return 1.0f;
        double c = dot / denom;
        return (float)(1.0 - c);
    }
}

int HNSW_DARTH::sample_level() const {
    double U = std::max(unif_(rng_), 1e-12);
    return (int)(-std::log(U) * mL_);
}

int HNSW_DARTH::search_layer_greedy(const std::vector<float>& q, int ep_id, int lc) const {
    int best = ep_id;
    float best_dist = dist(q, vectors_.at(best));

    while (true) {
        bool improved = false;
        auto it = layers_[lc].find(best);
        if (it == layers_[lc].end()) break;

        for (int nb : it->second) {
            float d = dist(q, vectors_.at(nb));
            if (d < best_dist) {
                best_dist = d;
                best = nb;
                improved = true;
            }
        }
        if (!improved) break;
    }
    return best;
}

// HNSW search_layer with visited-once (paper-style)
std::vector<int> HNSW_DARTH::search_layer(const std::vector<float>& q, int ep_id, int lc, int ef) const {
    if (lc < 0 || lc >= (int)layers_.size()) return {};
    if (layers_[lc].empty()) return {};
    if (!vectors_.count(ep_id)) return {};

    // C: min-heap (dist, id)
    using MinItem = std::pair<float,int>;
    std::priority_queue<MinItem, std::vector<MinItem>, std::greater<MinItem>> C;

    // W: max-heap via (-dist, id) stored as pair(neg_dist, id)
    // We'll keep W as max-heap by using neg distances and default comparator.
    std::priority_queue<std::pair<float,int>> W;

    std::unordered_set<int> visited;
    visited.reserve((size_t)ef * 2);

    float dep = dist(q, vectors_.at(ep_id));
    visited.insert(ep_id);
    C.push({dep, ep_id});
    W.push({-dep, ep_id});

    while (!C.empty()) {
        auto [dc, cid] = C.top(); C.pop();
        float worst = -W.top().first;
        if (dc > worst) break;

        auto it = layers_[lc].find(cid);
        if (it == layers_[lc].end()) continue;

        for (int nb : it->second) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_.at(nb));

            if ((int)W.size() < ef) {
                C.push({d, nb});
                W.push({-d, nb});
            } else {
                worst = -W.top().first;
                if (d < worst) {
                    C.push({d, nb});
                    W.pop();
                    W.push({-d, nb});
                }
            }
        }
    }

    std::vector<std::pair<float,int>> tmp;
    tmp.reserve(W.size());
    while (!W.empty()) {
        tmp.push_back({-W.top().first, W.top().second});
        W.pop();
    }
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first; });

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
    std::unordered_set<int> Wset;
    Wset.reserve(candidates.size() * 2 + 16);
    for (int x : candidates) Wset.insert(x);

    if (extend_candidates) {
        std::vector<int> base;
        base.reserve(Wset.size());
        for (int x : Wset) base.push_back(x);

        for (int e : base) {
            auto it = layers_[lc].find(e);
            if (it == layers_[lc].end()) continue;
            for (int adj : it->second) Wset.insert(adj);
        }
    }

    std::vector<std::pair<float,int>> cand;
    cand.reserve(Wset.size());
    for (int e : Wset) {
        cand.push_back({dist(q, vectors_.at(e)), e});
    }
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b){ return a.first < b.first; });

    std::vector<int> R;
    R.reserve(M);
    std::vector<std::pair<float,int>> discarded;

    for (auto& [de, e] : cand) {
        bool good = true;
        for (int r : R) {
            // paper check: dist(e,r) < dist(q,e) => e is redundant
            if (dist(vectors_.at(e), vectors_.at(r)) < de) {
                good = false;
                break;
            }
        }
        if (good) {
            R.push_back(e);
            if ((int)R.size() == M) break;
        } else {
            discarded.push_back({de, e});
        }
    }

    if (keep_pruned_connections && (int)R.size() < M) {
        std::sort(discarded.begin(), discarded.end(), [](auto& a, auto& b){ return a.first < b.first; });
        for (auto& [_, e] : discarded) {
            bool exists = false;
            for (int r : R) if (r == e) { exists = true; break; }
            if (!exists) {
                R.push_back(e);
                if ((int)R.size() == M) break;
            }
        }
    }
    return R;
}

void HNSW_DARTH::prune_connections(int node_id, int lc, int Mmax) {
    auto it = layers_[lc].find(node_id);
    if (it == layers_[lc].end()) return;
    if ((int)it->second.size() <= Mmax) return;

    std::vector<int> neigh = it->second;
    auto keep = select_neighbors_heuristic_paper(vectors_.at(node_id), neigh, lc, Mmax,
                                                 /*extend_candidates=*/true,
                                                 /*keep_pruned_connections=*/false);
    std::unordered_set<int> keepSet(keep.begin(), keep.end());

    // Remove edges (bidirectional)
    for (int nb : neigh) {
        if (keepSet.find(nb) != keepSet.end()) continue;
        auto itnb = layers_[lc].find(nb);
        if (itnb != layers_[lc].end()) {
            auto& adj = itnb->second;
            adj.erase(std::remove(adj.begin(), adj.end(), node_id), adj.end());
        }
    }

    it->second = keep;
}

int HNSW_DARTH::insert(const std::vector<float>& vec, int node_id) {
    if ((int)vec.size() != dim_) throw std::runtime_error("insert: dim mismatch");

    if (node_id < 0) {
        // simple id assignment: next integer not used
        int cand = (int)vectors_.size();
        while (vectors_.count(cand)) ++cand;
        node_id = cand;
    }
    if (vectors_.count(node_id)) throw std::runtime_error("insert: node id already exists");

    vectors_[node_id] = vec;
    int l = sample_level();

    // first node
    if (entry_id_ < 0) {
        layers_.resize(l + 1);
        for (int lc = 0; lc <= l; ++lc) layers_[lc][node_id] = {};
        entry_id_ = node_id;
        maxlevel_ = l;
        return node_id;
    }

    int old_top = maxlevel_;
    if (l > maxlevel_) {
        layers_.resize(l + 1);
        maxlevel_ = l;
    }

    int ep = entry_id_;
    int L = old_top;

    // greedy in upper layers (L down to l+1)
    for (int lc = L; lc > l; --lc) {
        ep = search_layer_greedy(vec, ep, lc);
    }

    // connect down to layer 0
    for (int lc = std::min(L, l); lc >= 0; --lc) {
        // ensure node exists in this layer
        if (!layers_[lc].count(node_id)) layers_[lc][node_id] = {};

        // find candidates
        auto W = search_layer(vec, ep, lc, efConstruction_);
        int Mmax = (lc == 0) ? M0_ : M_;

        auto neighbors = select_neighbors_heuristic_paper(vec, W, lc, Mmax,
                                                          /*extend_candidates=*/true,
                                                          /*keep_pruned_connections=*/false);

        // link bidirectionally
        auto& adj_v = layers_[lc][node_id];
        for (int nb : neighbors) {
            if (!layers_[lc].count(nb)) layers_[lc][nb] = {};
            adj_v.push_back(nb);
            layers_[lc][nb].push_back(node_id);

            // prune neighbor if too big
            if ((int)layers_[lc][nb].size() > Mmax) prune_connections(nb, lc, Mmax);
        }

        // update ep for next lower layer
        if (!W.empty()) ep = W.front();
    }

    if (l > old_top) entry_id_ = node_id;
    return node_id;
}

HNSW_DARTH::DarthFeatures
HNSW_DARTH::darth_extract_features(const std::vector<std::pair<float,int>>& result_maxheap,
                                  int ndis, int nstep, float firstNN, int ninserts) {
    // result_maxheap stores (-dist, id) pairs
    std::vector<float> dists;
    dists.reserve(result_maxheap.size());
    for (auto& p : result_maxheap) dists.push_back(-p.first);

    DarthFeatures f;
    f.ndis = ndis;
    f.nstep = nstep;
    f.ninserts = ninserts;
    f.firstNN = firstNN;

    if (dists.empty()) return f;

    std::sort(dists.begin(), dists.end());
    f.closestNN = dists.front();
    f.furthestNN = dists.back();

    // mean/var
    double sum = 0.0;
    for (float x : dists) sum += x;
    double mean = sum / (double)dists.size();
    f.meanNN = (float)mean;

    double var = 0.0;
    for (float x : dists) {
        double t = (double)x - mean;
        var += t * t;
    }
    var /= (double)dists.size();
    f.varNN = (float)var;

    auto pct = [&](double p)->float {
        if (dists.size() == 1) return dists[0];
        double pos = p * (double)(dists.size() - 1);
        size_t i = (size_t)std::floor(pos);
        size_t j = std::min(i + 1, dists.size() - 1);
        double w = pos - (double)i;
        return (float)((1.0 - w) * dists[i] + w * dists[j]);
    };
    f.p25NN = pct(0.25);
    f.p50NN = pct(0.50);
    f.p75NN = pct(0.75);
    return f;
}

std::vector<int> HNSW_DARTH::search_layer_darth(const std::vector<float>& q,
                                               int ep_id,
                                               int lc,
                                               int efSearch,
                                               int k,
                                               float Rt,
                                               const IPredictor& predictor,
                                               int ipi,
                                               int mpi) const {
    if (lc < 0 || lc >= (int)layers_.size()) return {};
    if (layers_[lc].empty()) return {};
    if (!vectors_.count(ep_id)) return {};

    std::unordered_set<int> visited;
    visited.reserve((size_t)efSearch * 4 + 64);

    // C: min-heap
    using MinItem = std::pair<float,int>;
    std::priority_queue<MinItem, std::vector<MinItem>, std::greater<MinItem>> C;

    // W: max-heap but we store (-dist, id) and use default comparator
    std::priority_queue<std::pair<float,int>> W;

    int ndis = 0, idis = 0, nstep = 0, ninserts = 0;

    float dep = dist(q, vectors_.at(ep_id)); ndis++; idis++;
    float firstNN = dep;

    visited.insert(ep_id);
    C.push({dep, ep_id});
    W.push({-dep, ep_id}); ninserts++;

    int pi = std::max(ipi, mpi);

    while (!C.empty()) {
        auto [dc, cid] = C.top(); C.pop();
        nstep++;

        float maxDist = (W.size() < (size_t)k) ? std::numeric_limits<float>::infinity()
                                              : (-W.top().first);
        if (dc > maxDist) break;

        bool should_predict = false;

        auto it = layers_[lc].find(cid);
        if (it != layers_[lc].end()) {
            for (int nb : it->second) {
                if (visited.find(nb) != visited.end()) continue;
                visited.insert(nb);

                float d = dist(q, vectors_.at(nb)); ndis++; idis++;

                // update result set (size k)
                if ((int)W.size() < k) {
                    W.push({-d, nb}); ninserts++;
                } else if (d < -W.top().first) {
                    W.pop();
                    W.push({-d, nb}); ninserts++;
                }

                maxDist = (W.size() < (size_t)k) ? std::numeric_limits<float>::infinity()
                                                : (-W.top().first);

                // candidate queue condition (paper-like)
                if (d < maxDist || (int)C.size() < efSearch) {
                    C.push({d, nb});
                }

                if (idis % pi == 0) should_predict = true;
            }
        }

        if (should_predict) {
            // snapshot W into a vector to compute features
            std::vector<std::pair<float,int>> wvec;
            wvec.reserve(W.size());
            {
                auto Wcopy = W;
                while (!Wcopy.empty()) {
                    wvec.push_back(Wcopy.top());
                    Wcopy.pop();
                }
            }

            auto feats = darth_extract_features(wvec, ndis, nstep, firstNN, ninserts);
            float Rp = predictor.predict(feats);

            if (Rp >= Rt) break;

            pi = (int)(mpi + (ipi - mpi) * (Rt - Rp));
            if (pi < mpi) pi = mpi;
            idis = 0;
        }
    }

    // return sorted top-k from W
    std::vector<std::pair<float,int>> tmp;
    tmp.reserve(W.size());
    while (!W.empty()) {
        tmp.push_back({-W.top().first, W.top().second});
        W.pop();
    }
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first; });

    std::vector<int> ids;
    ids.reserve(std::min((int)tmp.size(), k));
    for (int i = 0; i < (int)tmp.size() && (int)ids.size() < k; ++i) ids.push_back(tmp[i].second);
    return ids;
}

std::vector<int> HNSW_DARTH::query_darth(const std::vector<float>& q,
                                        int k,
                                        int efSearch,
                                        float Rt,
                                        const IPredictor& predictor,
                                        int ipi,
                                        int mpi) const {
    if (entry_id_ < 0) return {};
    if ((int)q.size() != dim_) throw std::runtime_error("query_darth: dim mismatch");

    int ep = entry_id_;

    // greedy descent upper layers
    for (int lc = maxlevel_; lc > 0; --lc) {
        ep = search_layer_greedy(q, ep, lc);
    }

    // DARTH on layer 0
    return search_layer_darth(q, ep, 0, efSearch, k, Rt, predictor, ipi, mpi);
}
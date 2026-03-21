#include "hnsw_pip.h"

HNSWPiPIndex::HNSWPiPIndex(int dim,
                           int M,
                           int efConstruction,
                           const std::string& metric,
                           uint32_t seed,
                           double pip_gamma,
                           int pip_delta)
    : dim_(dim),
      M_(M),
      M0_(2 * M),
      efConstruction_(efConstruction),
      metric_(metric),
      maxlevel_(-1),
      entry_id_(-1),
      mL_(1.0 / std::log((double)M)),
      use_heuristic_(true),
      pip_gamma_(pip_gamma),
      pip_delta_(pip_delta),
      rng_(seed),
      uni_(0.0, 1.0) {
    if (dim_ <= 0) throw std::invalid_argument("dim must be positive");
    if (M_ <= 1) throw std::invalid_argument("M must be > 1");
    if (metric_ != "l2" && metric_ != "cosine") {
        throw std::invalid_argument("metric must be 'l2' or 'cosine'");
    }
}

float HNSWPiPIndex::dist(const std::vector<float>& a, const std::vector<float>& b) const {
    if ((int)a.size() != dim_ || (int)b.size() != dim_) {
        throw std::runtime_error("dimension mismatch in dist()");
    }

    if (metric_ == "l2") {
        float s = 0.0f;
        for (int i = 0; i < dim_; ++i) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < dim_; ++i) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    if (denom == 0.0) return 1.0f;
    return (float)(1.0 - dot / denom);
}

int HNSWPiPIndex::sample_level() {
    double U = std::max(uni_(rng_), 1e-12);
    return (int)(-std::log(U) * mL_);
}

void HNSWPiPIndex::add(const std::vector<std::vector<float>>& X) {
    for (int i = 0; i < (int)X.size(); ++i) {
        add_point(X[i], (int)vectors_.size());
    }
}

int HNSWPiPIndex::add_point(const std::vector<float>& vec, int node_id) {
    if ((int)vec.size() != dim_) {
        throw std::invalid_argument("vector dimension mismatch");
    }

    if (node_id < 0) node_id = (int)vectors_.size();
    if (node_id != (int)vectors_.size()) {
        throw std::invalid_argument("this standalone version expects sequential node ids");
    }

    vectors_.push_back(vec);
    int l = sample_level();

    if (entry_id_ == -1) {
        layers_.resize(l + 1);
        for (int lc = 0; lc <= l; ++lc) {
            layers_[lc].resize(node_id + 1);
            layers_[lc][node_id] = {};
        }
        entry_id_ = node_id;
        maxlevel_ = l;
        return node_id;
    }

    int old_top = maxlevel_;
    if (l > maxlevel_) {
        layers_.resize(l + 1);
        maxlevel_ = l;
    }

    for (auto& layer : layers_) {
        if ((int)layer.size() <= node_id) {
            layer.resize(node_id + 1);
        }
    }

    int ep = entry_id_;
    int L = old_top;

    for (int lc = L; lc > l; --lc) {
        ep = search_layer_greedy(vec, ep, lc);
    }

    for (int lc = std::min(L, l); lc >= 0; --lc) {
        std::vector<int> W = search_layer_standard(vec, ep, lc, efConstruction_);

        std::vector<int> neighbors = select_neighbors_heuristic(
            vec, W, lc, (lc > 0 ? M_ : M0_), true, false
        );

        for (int nb : neighbors) {
            layers_[lc][node_id].insert(nb);
            layers_[lc][nb].insert(node_id);

            int Mmax = (lc == 0 ? M0_ : M_);
            if ((int)layers_[lc][nb].size() > Mmax) {
                layers_[lc][nb] = prune_connections(nb, lc, Mmax);
            }
        }

        if (!W.empty()) ep = W[0];
    }

    if (l > old_top) {
        entry_id_ = node_id;
    }

    return node_id;
}

int HNSWPiPIndex::search_layer_greedy(const std::vector<float>& q, int ep, int lc) const {
    int best = ep;
    float best_dist = dist(q, vectors_[best]);

    while (true) {
        bool improved = false;
        for (int nb : layers_[lc][best]) {
            float d = dist(q, vectors_[nb]);
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

std::vector<int> HNSWPiPIndex::search_layer_standard(const std::vector<float>& q,
                                                     int ep,
                                                     int layer,
                                                     int ef) const {
    if (layer < 0 || layer >= (int)layers_.size()) return {};
    if (ep < 0 || ep >= (int)vectors_.size()) return {};
    if (ef <= 0) return {};

    std::unordered_set<int> visited;
    std::priority_queue<CandidateMin, std::vector<CandidateMin>, std::greater<CandidateMin>> C;
    std::priority_queue<CandidateMax> W;

    float dist_ep = dist(q, vectors_[ep]);
    visited.insert(ep);
    C.push({dist_ep, ep});
    W.push({dist_ep, ep});

    while (!C.empty()) {
        CandidateMin cur = C.top();
        C.pop();

        float worst_dist = W.top().dist;
        if (cur.dist > worst_dist) break;

        for (int nb : layers_[layer][cur.id]) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_[nb]);

            if ((int)W.size() < ef) {
                C.push({d, nb});
                W.push({d, nb});
            } else {
                worst_dist = W.top().dist;
                if (d < worst_dist) {
                    C.push({d, nb});
                    W.pop();
                    W.push({d, nb});
                }
            }
        }
    }

    std::vector<std::pair<float, int>> result;
    while (!W.empty()) {
        result.push_back({W.top().dist, W.top().id});
        W.pop();
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> ids;
    ids.reserve(result.size());
    for (const auto& p : result) ids.push_back(p.second);
    return ids;
}

std::vector<int> HNSWPiPIndex::heap_topk_ids(const std::priority_queue<CandidateMax>& heap, int k) {
    auto copy = heap;
    std::vector<std::pair<float, int>> items;
    while (!copy.empty()) {
        items.push_back({copy.top().dist, copy.top().id});
        copy.pop();
    }
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> ids;
    for (int i = 0; i < std::min(k, (int)items.size()); ++i) {
        ids.push_back(items[i].second);
    }
    return ids;
}

std::vector<int> HNSWPiPIndex::search_layer_pip(const std::vector<float>& q,
                                                int ep,
                                                int layer,
                                                int ef,
                                                int k) const {
    if (layer < 0 || layer >= (int)layers_.size()) return {};
    if (ep < 0 || ep >= (int)vectors_.size()) return {};
    if (ef <= 0 || k <= 0) return {};

    std::unordered_set<int> visited;
    std::priority_queue<CandidateMin, std::vector<CandidateMin>, std::greater<CandidateMin>> C;
    std::priority_queue<CandidateMax> W;
    std::priority_queue<CandidateMax> Kheap;

    auto threshold = [&]() -> float {
        return W.top().dist;
    };

    auto try_add_topk = [&](float d, int id) {
        if ((int)Kheap.size() < k) {
            Kheap.push({d, id});
        } else if (d < Kheap.top().dist) {
            Kheap.pop();
            Kheap.push({d, id});
        }
    };

    float dist_ep = dist(q, vectors_[ep]);
    visited.insert(ep);
    C.push({dist_ep, ep});
    W.push({dist_ep, ep});
    try_add_topk(dist_ep, ep);

    std::unordered_set<int> prev_topk;
    int saturation_counter = 0;

    while (!C.empty()) {
        CandidateMin cur = C.top();
        C.pop();

        if (cur.dist > threshold()) break;

        for (int nb : layers_[layer][cur.id]) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_[nb]);
            if ((int)W.size() < ef) {
                C.push({d, nb});
                W.push({d, nb});
                try_add_topk(d, nb);
            } else if (d < threshold()) {
                C.push({d, nb});
                W.pop();
                W.push({d, nb});
                try_add_topk(d, nb);
            }
        }

        std::vector<int> curr = heap_topk_ids(Kheap, k);
        if ((int)curr.size() == k) {
            std::unordered_set<int> curr_set(curr.begin(), curr.end());
            if (!prev_topk.empty()) {
                int overlap = 0;
                for (int x : curr_set) {
                    if (prev_topk.find(x) != prev_topk.end()) ++overlap;
                }
                double phi = 100.0 * (double)overlap / (double)k;
                if (phi >= pip_gamma_) ++saturation_counter;
                else saturation_counter = 0;

                if (saturation_counter >= pip_delta_) break;
            }
            prev_topk = std::move(curr_set);
        }
    }

    std::vector<std::pair<float, int>> result;
    while (!W.empty()) {
        result.push_back({W.top().dist, W.top().id});
        W.pop();
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> ids;
    ids.reserve(result.size());
    for (const auto& p : result) ids.push_back(p.second);
    return ids;
}

std::vector<int> HNSWPiPIndex::select_neighbors_simple(const std::vector<float>& q,
                                                       const std::vector<int>& candidates,
                                                       int Mmax) const {
    std::vector<int> unique;
    std::unordered_set<int> seen;
    for (int c : candidates) {
        if (seen.insert(c).second) unique.push_back(c);
    }

    if ((int)unique.size() <= Mmax) return unique;

    std::vector<std::pair<float, int>> ds;
    ds.reserve(unique.size());
    for (int nb : unique) ds.push_back({dist(q, vectors_[nb]), nb});
    std::sort(ds.begin(), ds.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> out;
    for (int i = 0; i < Mmax; ++i) out.push_back(ds[i].second);
    return out;
}

std::vector<int> HNSWPiPIndex::select_neighbors_heuristic(const std::vector<float>& q,
                                                          const std::vector<int>& candidates,
                                                          int layer,
                                                          int Mmax,
                                                          bool extend_candidates,
                                                          bool keep_pruned_connections) const {
    if (Mmax <= 0) return {};

    std::unordered_set<int> Wset(candidates.begin(), candidates.end());
    if (extend_candidates) {
        std::vector<int> base(Wset.begin(), Wset.end());
        for (int e : base) {
            for (int adj : layers_[layer][e]) {
                Wset.insert(adj);
            }
        }
    }

    std::vector<std::pair<float, int>> cand;
    cand.reserve(Wset.size());
    for (int e : Wset) cand.push_back({dist(q, vectors_[e]), e});
    std::sort(cand.begin(), cand.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> R;
    std::vector<std::pair<float, int>> discarded;

    for (const auto& it : cand) {
        float d_qe = it.first;
        int e = it.second;
        bool good = true;
        for (int r : R) {
            if (dist(vectors_[e], vectors_[r]) < d_qe) {
                good = false;
                break;
            }
        }
        if (good) {
            R.push_back(e);
            if ((int)R.size() == Mmax) break;
        } else {
            discarded.push_back({d_qe, e});
        }
    }

    if (keep_pruned_connections && (int)R.size() < Mmax) {
        std::sort(discarded.begin(), discarded.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& it : discarded) {
            if (std::find(R.begin(), R.end(), it.second) == R.end()) {
                R.push_back(it.second);
                if ((int)R.size() == Mmax) break;
            }
        }
    }

    return R;
}

std::unordered_set<int> HNSWPiPIndex::prune_connections(int node_id, int layer, int Mmax) {
    const auto& neigh = layers_[layer][node_id];
    if ((int)neigh.size() <= Mmax) return neigh;

    std::vector<int> neighbors;
    for (int x : neigh) {
        if (x != node_id) neighbors.push_back(x);
    }

    std::vector<int> new_neigh_list =
        use_heuristic_
            ? select_neighbors_heuristic(vectors_[node_id], neighbors, layer, Mmax, true, false)
            : select_neighbors_simple(vectors_[node_id], neighbors, Mmax);

    std::unordered_set<int> new_set(new_neigh_list.begin(), new_neigh_list.end());

    for (int nb : neigh) {
        if (new_set.find(nb) == new_set.end()) {
            layers_[layer][nb].erase(node_id);
        }
    }

    layers_[layer][node_id] = new_set;
    return new_set;
}

std::vector<int> HNSWPiPIndex::query_pip(const std::vector<float>& q, int K, int efSearch) const {
    if (entry_id_ == -1) return {};
    int ep = entry_id_;
    for (int lc = maxlevel_; lc > 0; --lc) {
        ep = search_layer_greedy(q, ep, lc);
    }
    std::vector<int> W = search_layer_pip(q, ep, 0, efSearch, K);
    if ((int)W.size() > K) W.resize(K);
    return W;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>>
HNSWPiPIndex::search(const std::vector<std::vector<float>>& Xq, int k, int efSearch) const {
    std::vector<std::vector<float>> D(Xq.size(), std::vector<float>(k, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<int>> I(Xq.size(), std::vector<int>(k, -1));

    for (size_t i = 0; i < Xq.size(); ++i) {
        std::vector<int> ids = query_pip(Xq[i], k, efSearch);
        for (int j = 0; j < (int)ids.size() && j < k; ++j) {
            I[i][j] = ids[j];
            D[i][j] = dist(Xq[i], vectors_[ids[j]]);
        }
    }
    return {D, I};
}
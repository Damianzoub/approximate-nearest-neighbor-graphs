#include "hnsw_adaef.h"

HNSWAdaEFIndex::HNSWAdaEFIndex(int dim,
                               int M,
                               int efConstruction,
                               const std::string& metric,
                               uint32_t seed,
                               int adaef_bins,
                               double adaef_delta,
                               int adaef_sample_size)
    : dim_(dim),
      M_(M),
      M0_(2 * M),
      efConstruction_(efConstruction),
      metric_(metric),
      maxlevel_(-1),
      entry_id_(-1),
      mL_(1.0 / std::log((double)M)),
      use_heuristic_(true),
      adaef_bins_(adaef_bins),
      adaef_delta_(adaef_delta),
      adaef_sample_size_(adaef_sample_size),
      offline_ready_(false),
      rng_(seed),
      uni_(0.0, 1.0) {
    if (dim_ <= 0) throw std::invalid_argument("dim must be positive");
    if (M_ <= 1) throw std::invalid_argument("M must be > 1");
    if (metric_ != "cosine") {
        throw std::invalid_argument("paper-based Ada-ef here supports cosine only");
    }
}

float HNSWAdaEFIndex::dist(const std::vector<float>& a, const std::vector<float>& b) const {
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

int HNSWAdaEFIndex::sample_level() {
    double U = std::max(uni_(rng_), 1e-12);
    return (int)(-std::log(U) * mL_);
}

void HNSWAdaEFIndex::add(const std::vector<std::vector<float>>& X) {
    for (int i = 0; i < (int)X.size(); ++i) {
        add_point(X[i], (int)vectors_.size());
    }
}

int HNSWAdaEFIndex::add_point(const std::vector<float>& vec, int node_id) {
    if ((int)vec.size() != dim_) throw std::invalid_argument("vector dimension mismatch");
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
        if ((int)layer.size() <= node_id) layer.resize(node_id + 1);
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

    if (l > old_top) entry_id_ = node_id;
    return node_id;
}

int HNSWAdaEFIndex::search_layer_greedy(const std::vector<float>& q, int ep, int lc) const {
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

std::vector<int> HNSWAdaEFIndex::search_layer_standard(const std::vector<float>& q,
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

        float worst = W.top().dist;
        if (cur.dist > worst) break;

        for (int nb : layers_[layer][cur.id]) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_[nb]);
            if ((int)W.size() < ef) {
                C.push({d, nb});
                W.push({d, nb});
            } else if (d < W.top().dist) {
                C.push({d, nb});
                W.pop();
                W.push({d, nb});
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

std::vector<int> HNSWAdaEFIndex::select_neighbors_simple(const std::vector<float>& q,
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

std::vector<int> HNSWAdaEFIndex::select_neighbors_heuristic(const std::vector<float>& q,
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

std::unordered_set<int> HNSWAdaEFIndex::prune_connections(int node_id, int layer, int Mmax) {
    const auto& neigh = layers_[layer][node_id];
    if ((int)neigh.size() <= Mmax) return neigh;

    std::vector<int> neighbors;
    for (int x : neigh) if (x != node_id) neighbors.push_back(x);

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

void HNSWAdaEFIndex::compute_dataset_statistics() {
    int n = (int)vectors_.size();
    dataset_mean_.assign(dim_, 0.0);
    dataset_cov_.assign(dim_, std::vector<double>(dim_, 0.0));

    std::vector<std::vector<double>> Xn(n, std::vector<double>(dim_, 0.0));

    for (int i = 0; i < n; ++i) {
        double norm = 0.0;
        for (int j = 0; j < dim_; ++j) norm += (double)vectors_[i][j] * vectors_[i][j];
        norm = std::sqrt(norm);
        if (norm == 0.0) norm = 1.0;

        for (int j = 0; j < dim_; ++j) {
            Xn[i][j] = (double)vectors_[i][j] / norm;
            dataset_mean_[j] += Xn[i][j];
        }
    }

    for (int j = 0; j < dim_; ++j) dataset_mean_[j] /= (double)n;

    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < dim_; ++a) {
            double da = Xn[i][a] - dataset_mean_[a];
            for (int b = 0; b < dim_; ++b) {
                double db = Xn[i][b] - dataset_mean_[b];
                dataset_cov_[a][b] += da * db;
            }
        }
    }

    double denom = std::max(1, n - 1);
    for (int a = 0; a < dim_; ++a) {
        for (int b = 0; b < dim_; ++b) {
            dataset_cov_[a][b] /= denom;
        }
    }
}

std::pair<double, double> HNSWAdaEFIndex::estimate_fdl_params(const std::vector<float>& q) const {
    std::vector<double> qn(dim_, 0.0);
    double qnorm = 0.0;
    for (int i = 0; i < dim_; ++i) qnorm += (double)q[i] * q[i];
    qnorm = std::sqrt(qnorm);
    if (qnorm == 0.0) qnorm = 1.0;
    for (int i = 0; i < dim_; ++i) qn[i] = (double)q[i] / qnorm;

    double mu_cs = 0.0;
    for (int i = 0; i < dim_; ++i) mu_cs += qn[i] * dataset_mean_[i];

    double var_cs = 0.0;
    for (int i = 0; i < dim_; ++i) {
        double s = 0.0;
        for (int j = 0; j < dim_; ++j) s += dataset_cov_[i][j] * qn[j];
        var_cs += qn[i] * s;
    }
    var_cs = std::max(var_cs, 1e-12);

    double mu_cd = 1.0 - mu_cs;
    double sigma_cd = std::sqrt(var_cs);
    return {mu_cd, sigma_cd};
}

// Acklam-style rational approximation
double HNSWAdaEFIndex::inv_norm_cdf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::invalid_argument("p must be in (0,1)");
    }

    static const double a1 = -3.969683028665376e+01;
    static const double a2 =  2.209460984245205e+02;
    static const double a3 = -2.759285104469687e+02;
    static const double a4 =  1.383577518672690e+02;
    static const double a5 = -3.066479806614716e+01;
    static const double a6 =  2.506628277459239e+00;

    static const double b1 = -5.447609879822406e+01;
    static const double b2 =  1.615858368580409e+02;
    static const double b3 = -1.556989798598866e+02;
    static const double b4 =  6.680131188771972e+01;
    static const double b5 = -1.328068155288572e+01;

    static const double c1 = -7.784894002430293e-03;
    static const double c2 = -3.223964580411365e-01;
    static const double c3 = -2.400758277161838e+00;
    static const double c4 = -2.549732539343734e+00;
    static const double c5 =  4.374664141464968e+00;
    static const double c6 =  2.938163982698783e+00;

    static const double d1 =  7.784695709041462e-03;
    static const double d2 =  3.224671290700398e-01;
    static const double d3 =  2.445134137142996e+00;
    static const double d4 =  3.754408661907416e+00;

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;

    if (p < plow) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    } else if (p <= phigh) {
        double q = p - 0.5;
        double r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
}

std::vector<double> HNSWAdaEFIndex::compute_bins(double mu, double sigma) const {
    std::vector<double> thresholds;
    thresholds.reserve(adaef_bins_);
    for (int i = 1; i <= adaef_bins_; ++i) {
        double p = adaef_delta_ * i;
        p = std::min(std::max(p, 1e-12), 1.0 - 1e-12);
        thresholds.push_back(mu + sigma * inv_norm_cdf(p));
    }
    return thresholds;
}

double HNSWAdaEFIndex::compute_query_score(const std::vector<float>& q,
                                           const std::vector<float>& D) const {
    if (D.empty()) return 0.0;
    auto [mu, sigma] = estimate_fdl_params(q);
    std::vector<double> thresholds = compute_bins(mu, sigma);

    std::vector<int> counts(adaef_bins_, 0);
    for (float d : D) {
        for (int i = 0; i < adaef_bins_; ++i) {
            if ((double)d <= thresholds[i]) {
                counts[i]++;
                break;
            }
        }
    }

    double score = 0.0;
    double denom = (double)D.size();
    for (int i = 0; i < adaef_bins_; ++i) {
        double w = 100.0 * std::exp(-(double)i);
        score += w * ((double)counts[i] / denom);
    }
    return score;
}

std::vector<int> HNSWAdaEFIndex::exact_knn_excluding_self(const std::vector<float>& q, int qid, int k) const {
    std::vector<std::pair<float, int>> arr;
    for (int nid = 0; nid < (int)vectors_.size(); ++nid) {
        if (nid == qid) continue;
        arr.push_back({dist(q, vectors_[nid]), nid});
    }
    std::sort(arr.begin(), arr.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> out;
    for (int i = 0; i < std::min(k, (int)arr.size()); ++i) out.push_back(arr[i].second);
    return out;
}

double HNSWAdaEFIndex::recall_at_k(const std::vector<int>& gt, const std::vector<int>& pred, int k) const {
    std::unordered_set<int> g;
    for (int i = 0; i < std::min(k, (int)gt.size()); ++i) g.insert(gt[i]);
    int hits = 0;
    for (int i = 0; i < std::min(k, (int)pred.size()); ++i) {
        if (g.find(pred[i]) != g.end()) ++hits;
    }
    return (double)hits / (double)std::max(k, 1);
}

int HNSWAdaEFIndex::entry_after_upper_layers(const std::vector<float>& q) const {
    int ep = entry_id_;
    for (int lc = maxlevel_; lc > 0; --lc) {
        ep = search_layer_greedy(q, ep, lc);
    }
    return ep;
}

int HNSWAdaEFIndex::two_hop_size(int ep_id, int layer) const {
    if (ep_id < 0 || ep_id >= (int)layers_[layer].size()) return 1;

    std::unordered_set<int> reach;
    reach.insert(ep_id);

    std::vector<int> one_hop;
    for (int nb : layers_[layer][ep_id]) {
        reach.insert(nb);
        one_hop.push_back(nb);
    }

    for (int nb : one_hop) {
        for (int x : layers_[layer][nb]) {
            reach.insert(x);
        }
    }

    return std::max(1, (int)reach.size());
}

std::vector<float> HNSWAdaEFIndex::collect_distance_list(const std::vector<float>& q, int ep_id, int layer) const {
    int l = two_hop_size(ep_id, layer);

    std::unordered_set<int> visited;
    std::priority_queue<CandidateMin, std::vector<CandidateMin>, std::greater<CandidateMin>> C;
    std::vector<float> D;

    float dist_ep = dist(q, vectors_[ep_id]);
    visited.insert(ep_id);
    C.push({dist_ep, ep_id});
    D.push_back(dist_ep);

    while (!C.empty() && (int)D.size() < l) {
        CandidateMin cur = C.top();
        C.pop();

        for (int nb : layers_[layer][cur.id]) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_[nb]);
            D.push_back(d);
            C.push({d, nb});

            if ((int)D.size() >= l) break;
        }
    }
    return D;
}

void HNSWAdaEFIndex::build_adaef_offline(int k,
                                         double target_recall,
                                         const std::vector<int>& ef_values) {
    if (vectors_.empty()) throw std::runtime_error("index is empty");

    compute_dataset_statistics();

    std::vector<int> all_ids(vectors_.size());
    for (int i = 0; i < (int)vectors_.size(); ++i) all_ids[i] = i;
    std::shuffle(all_ids.begin(), all_ids.end(), rng_);
    if ((int)all_ids.size() > adaef_sample_size_) all_ids.resize(adaef_sample_size_);

    std::unordered_map<int, std::vector<int>> score_groups;
    std::unordered_map<int, std::vector<int>> gt_cache;

    for (int qid : all_ids) {
        const auto& q = vectors_[qid];
        gt_cache[qid] = exact_knn_excluding_self(q, qid, k);

        int ep = entry_after_upper_layers(q);
        std::vector<float> D = collect_distance_list(q, ep, 0);

        int score_int = (int)compute_query_score(q, D);
        score_groups[score_int].push_back(qid);
    }

    ef_estimation_table_.clear();
    int target_key = (int)std::llround(target_recall * 1000.0);

    double weighted_sum = 0.0;
    int total_queries = 0;

    for (const auto& kv : score_groups) {
        int score_int = kv.first;
        const auto& qids = kv.second;

        std::vector<std::pair<int, double>> pairs;
        for (int ef : ef_values) {
            std::vector<double> recalls;
            recalls.reserve(qids.size());

            for (int qid : qids) {
                std::vector<int> pred = search_layer_standard(vectors_[qid], entry_after_upper_layers(vectors_[qid]), 0, ef);
                pred.erase(std::remove(pred.begin(), pred.end(), qid), pred.end());
                if ((int)pred.size() > k) pred.resize(k);
                recalls.push_back(recall_at_k(gt_cache[qid], pred, k));
            }

            double avg = 0.0;
            for (double r : recalls) avg += r;
            avg /= std::max(1, (int)recalls.size());
            pairs.push_back({ef, avg});
        }

        ef_estimation_table_[score_int] = pairs;

        int chosen_ef = pairs.back().first;
        for (const auto& pr : pairs) {
            if (pr.second >= target_recall) {
                chosen_ef = pr.first;
                break;
            }
        }

        weighted_sum += (double)qids.size() * chosen_ef;
        total_queries += (int)qids.size();
    }

    wae_by_target_[target_key] = weighted_sum / std::max(1, total_queries);
    offline_ready_ = true;
}

int HNSWAdaEFIndex::estimate_ef(const std::vector<float>& q,
                                const std::vector<float>& D,
                                double target_recall) const {
    if (!offline_ready_) throw std::runtime_error("build_adaef_offline() must be called first");
    int score_int = (int)compute_query_score(q, D);

    if (ef_estimation_table_.empty()) throw std::runtime_error("ef_estimation_table is empty");

    auto it = ef_estimation_table_.find(score_int);
    if (it == ef_estimation_table_.end()) {
        int best_key = ef_estimation_table_.begin()->first;
        int best_diff = std::abs(best_key - score_int);
        for (const auto& kv : ef_estimation_table_) {
            int diff = std::abs(kv.first - score_int);
            if (diff < best_diff) {
                best_diff = diff;
                best_key = kv.first;
            }
        }
        it = ef_estimation_table_.find(best_key);
    }

    int target_key = (int)std::llround(target_recall * 1000.0);
    double wae = it->second.back().first;
    auto wit = wae_by_target_.find(target_key);
    if (wit != wae_by_target_.end()) wae = wit->second;

    int ef = it->second.back().first;
    for (const auto& pr : it->second) {
        if (pr.second >= target_recall) {
            ef = pr.first;
            break;
        }
    }

    ef = std::max(ef, (int)std::ceil(wae));
    return ef;
}

std::vector<int> HNSWAdaEFIndex::search_layer_adaef(const std::vector<float>& q,
                                                    int ep_id,
                                                    int layer,
                                                    double target_recall) const {
    if (layer < 0 || layer >= (int)layers_.size()) return {};
    if (ep_id < 0 || ep_id >= (int)vectors_.size()) return {};

    std::unordered_set<int> visited;
    std::priority_queue<CandidateMin, std::vector<CandidateMin>, std::greater<CandidateMin>> C;
    std::priority_queue<CandidateMax> W;

    float dist_ep = dist(q, vectors_[ep_id]);
    visited.insert(ep_id);
    C.push({dist_ep, ep_id});
    W.push({dist_ep, ep_id});

    std::vector<float> D = {dist_ep};
    int l = two_hop_size(ep_id, layer);
    bool ef_estimated = false;
    int ef = std::numeric_limits<int>::max() / 4;

    auto current_threshold = [&]() -> float {
        if (!ef_estimated) return std::numeric_limits<float>::infinity();
        return W.top().dist;
    };

    while (!C.empty()) {
        CandidateMin cur = C.top();
        C.pop();

        if (ef_estimated && cur.dist > current_threshold()) break;

        for (int nb : layers_[layer][cur.id]) {
            if (visited.find(nb) != visited.end()) continue;
            visited.insert(nb);

            float d = dist(q, vectors_[nb]);

            if (!ef_estimated) {
                D.push_back(d);
                C.push({d, nb});
                W.push({d, nb});

                if ((int)D.size() >= l) {
                    ef = estimate_ef(q, D, target_recall);
                    ef_estimated = true;

                    if ((int)W.size() > ef) {
                        std::vector<std::pair<float, int>> best;
                        while (!W.empty()) {
                            best.push_back({W.top().dist, W.top().id});
                            W.pop();
                        }
                        std::sort(best.begin(), best.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
                        if ((int)best.size() > ef) best.resize(ef);
                        for (const auto& p : best) W.push({p.first, p.second});
                    }
                }
            } else {
                if ((int)W.size() < ef) {
                    C.push({d, nb});
                    W.push({d, nb});
                } else if (d < W.top().dist) {
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

std::vector<int> HNSWAdaEFIndex::query_adaef(const std::vector<float>& q, int K, double target_recall) const {
    if (entry_id_ == -1) return {};
    if (!offline_ready_) throw std::runtime_error("build_adaef_offline() must be called first");

    int ep = entry_after_upper_layers(q);
    std::vector<int> W = search_layer_adaef(q, ep, 0, target_recall);
    if ((int)W.size() > K) W.resize(K);
    return W;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>>
HNSWAdaEFIndex::search(const std::vector<std::vector<float>>& Xq, int k, double target_recall) const {
    std::vector<std::vector<float>> D(Xq.size(), std::vector<float>(k, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<int>> I(Xq.size(), std::vector<int>(k, -1));

    for (size_t i = 0; i < Xq.size(); ++i) {
        std::vector<int> ids = query_adaef(Xq[i], k, target_recall);
        for (int j = 0; j < (int)ids.size() && j < k; ++j) {
            I[i][j] = ids[j];
            D[i][j] = dist(Xq[i], vectors_[ids[j]]);
        }
    }
    return {D, I};
}
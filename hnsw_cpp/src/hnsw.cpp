#include "hnsw_new.h"
#include <cmath>
#include <stdexcept>
#include <queue>
#include <algorithm>

void HNSW_NEW::check_dim(const std::vector<float>& v) const{
    if (static_cast<int>(v.size()) != dim_){
        throw std::runtime_error("Vector dimension mismatch: expected " + std::to_string(dim_) + ", got" + std::to_string(v.size()));
    }
}

HNSW_NEW::HNSW_NEW(int dim ,int M ,int efConstruction, Metric metric, uint64_t seed) :
    dim_(dim),
      M_(M),
      M0_(2 * M),
      efConstruction_(efConstruction),
      metric_(metric),
      maxlevel_(-1),
      entry_id_(-1),
      mL_(1.0 / std::log(static_cast<double>(M))),
      rng_(seed),
      unif_(0.0, 1.0),
      use_heuristic_(true){

        if (dim_ <= 0) throw std::runtime_error("dim must be > 0");
        if (M_ < 2) throw std::runtime_error("M must be >=2");
        if (efConstruction_ < 1) throw std::runtime_error("efConstruction must be >=1");
}
//distance
float HNSW_NEW::dist(const std::vector<float>& a, const std::vector<float>& b) const{
    if (metric_ == Metric::L2){
        double s=0.0;
        for (int i=0; i < dim_; ++i){
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            s +=d*d;
        }
        return static_cast<float>(s);

    }else{
        double dot = 0.0, na =0.0, nb=0.0;
        for (int i=0; i < dim_; ++i){
            double da = a[i],db=b[i];
            dot += da*db;
            na += da*da;
            nb += db*db;
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom <= 0.0) return 1.0f;
        double c = dot/denom;
        return static_cast<float>(1.0-c);
    }
}
//level sampling
int HNSW_NEW::sample_level() const{
    double U = unif_(rng_);
    if (U <1e-12) U = 1e-12;
    return static_cast<int>(-std::log(U) * mL_);
}
// updating levels
void HNSW_NEW::ensure_layers(int new_maxlevel){
    if (new_maxlevel <0) return;
    if (static_cast<int>(layers_.size()) <= new_maxlevel){
        layers_.resize(new_maxlevel+1);
    }
}

//greedy search 
int HNSW_NEW::search_layer_greedy(const std::vector<float>& q, int ep, int layer) const {
    int best = ep;
    float bestDist = dist(q,vectors_.at(best));

    while(true){
        bool improved = false;
        auto itNode = layers_[layer].find(best);
        if (itNode == layers_[layer].end()) break;

        const auto& neigh = itNode -> second;
        for (int nb : neigh){
            float d = dist(q,vectors_.at(nb));
            if (d < bestDist){
                bestDist = d;
                best = nb;
                improved =true;
            }
        }
        if (!improved) break;
    }
    return best;
}

std::vector<int> HNSW_NEW::search_layer_beam(const std::vector<float>& q, int ep, int layer, int ef) const{
    if (layer <0 || layer >= static_cast<int>(layers_.size())) return {};
    if (layers_[layer].empty()) return {};
    if (vectors_.find(ep) == vectors_.end()) return {};
    if (ef <= 0 )return {};

    using MinItem = std::pair<float,int>;
    struct MinCmp{
        bool operator()(const MinItem& a, const MinItem& b) const {
            return a.first > b.first;
        }
    };
    std::priority_queue<MinItem,std::vector<MinItem>,MinCmp> C;

    using MaxItem = std::pair<float,int>;
    struct  MaxCmp
    {
        bool operator()(const MaxItem& a , const MaxItem& b )const {
            return a.first < b.first;
        }
    };
    std::priority_queue<MaxItem,std::vector<MaxItem>,MaxCmp> W;

    std::unordered_set<int> visited;
    visited.reserve(static_cast<size_t>(ef*8));

    float dist_ep = dist(q,vectors_.at(ep));
    visited.insert(ep);
    C.push({dist_ep,ep});
    W.push({dist_ep,ep});

    while (!C.empty()){
        auto [dist_c,c_id] = C.top();
        C.pop();

        float worstDist = W.top().first;
        if (dist_c > worstDist) break;

        auto it = layers_[layer].find(c_id);
        if (it == layers_[layer].end()) continue;

        for (int nb : it-> second){
            if (visited.find(nb) != visited.end()) continue;
            

            float d = dist(q,vectors_.at(nb));
            if (static_cast<int>(W.size()) < ef ){
                visited.insert(nb);
                C.push({d,nb});
                W.push({d,nb});
            }else{
                worstDist = W.top().first;
                if (d < worstDist){
                    visited.insert(nb);
                    C.push({d,nb});
                    W.pop();
                    W.push({d,nb});
                }
            }
        }
    }
    std::vector<std::pair<float,int>> tmp;
    tmp.reserve(W.size());
    while (!W.empty()){
        tmp.push_back(W.top());
        W.pop();
    }
    std::sort(tmp.begin(),tmp.end(),[](const auto& a, const auto& b){ return a.first < b.first;});
    std::vector<int> ids;
    ids.reserve(tmp.size());
    for (auto& p : tmp) ids.push_back(p.second);
    return ids;
}

std::vector<int> HNSW_NEW::select_neighbors_simple(const std::vector<float>& q, const std::vector<int>& candidates, int layer, int Mmax) const{

    if (Mmax <= 0) return {};
    std::vector<int> uniq;
    uniq.reserve(candidates.size());{
        std::unordered_set<int> seen;
        seen.reserve(candidates.size() * 2 +1);
        for (int id : candidates) {
            if (seen.insert(id).second) uniq.push_back(id);
        }
    }

    if (static_cast<int>(uniq.size()) <= Mmax) return uniq;
    std::vector<std::pair<float,int>> dlist;
    dlist.reserve(uniq.size());
    for (int nb :uniq){
        dlist.push_back({dist(q,vectors_.at(nb)),nb});
    }
    std::sort(dlist.begin(),dlist.end(), [](const auto& a, const auto& b){return a.first < b.first;});

    std::vector<int> out;
    out.reserve(Mmax);
    for (int i =0; i<Mmax; ++i) out.push_back(dlist[i].second);
    return out;
}

std::vector<int> HNSW_NEW::select_neighbors_heuristic_paper(
    const std::vector<float>& q,
    const std::vector<int>& candidates,
    int layer, int Mmax,
    bool extendCandidates,
    bool keepPruned
) const {
    if (Mmax <= 0) return {};
    std::unordered_set<int> Wset;
    Wset.reserve(candidates.size()*3+1);
    for (int id : candidates) Wset.insert(id);
    
    if (extendCandidates){
        std::vector<int> base(Wset.begin(),Wset.end());
        for (int e : base){
            auto it = layers_[layer].find(e);
            if (it ==layers_[layer].end()) continue;
            for (int adj : it->second) Wset.insert(adj);
        }
    }

    std::vector<std::pair<float,int>> cand;
    cand.reserve(Wset.size());
    for (int e : Wset){
        cand.push_back({dist(q,vectors_.at(e)),e});
    }
    std::sort(cand.begin(),cand.end(), [](const auto& a, const auto& b){return a.first < b.first;});
    std::vector<int> R;
    R.reserve(Mmax);

    std::vector<std::pair<float,int>> discarded;
    discarded.reserve(cand.size());
    
    for (const auto& [d_qe,e] : cand){
        bool good = true;
        for (int r : R){
            float d_er = dist(vectors_.at(e), vectors_.at(r));
            if (d_er < d_qe){
                good = false;
                break;
            }
        }
        if (good){
            R.push_back(e);
            if(static_cast<int>(R.size()) == Mmax) break;
        }else{
            discarded.push_back({d_qe,e});
        }
    }
    if (keepPruned && static_cast<int>(R.size()) < Mmax){
        std::sort(discarded.begin(),discarded.end(), [](const auto& a, const auto& b){return a.first < b.first;});
        for (auto& p : discarded){
            int e = p.second;
            if (std::find(R.begin(),R.end(),e) == R.end()){
                R.push_back(e);
                if (static_cast<int>(R.size()) == Mmax) break;
            }
        }
    }
    return R;
}


std::unordered_set<int> HNSW_NEW::prune_connection(int node_id,int layer ,int Mmax){
    auto it = layers_[layer].find(node_id);
    if (it == layers_[layer].end()) return {};

    auto& neigh = it -> second;
    if (static_cast<int>(neigh.size()) <= Mmax) return neigh;

    std::vector<int> neighbors(neigh.begin(),neigh.end());
    const auto& q = vectors_.at(node_id);

    std::vector<int> newList;
    if (use_heuristic_){
        newList = select_neighbors_heuristic_paper(
            q,neighbors,layer,Mmax,true,false
        );
    }else{
        newList = select_neighbors_simple(q,neighbors,layer,Mmax);
    }

    std::unordered_set<int> newSet;
    newSet.reserve(newList.size()*2+1);
    for (int x : newList) newSet.insert(x);

    std::vector<int> removed;
    removed.reserve(neigh.size());
    for (int nb : neigh){
        if (newSet.find(nb) == newSet.end()) removed.push_back(nb);
    }
    for (int nb : removed){
        auto it2 = layers_[layer].find(nb);
        if (it2 != layers_[layer].end()){
            it2 -> second.erase(node_id);
        }
    }
    neigh = newSet;
    return neigh;
}

std::vector<int> HNSW_NEW::query(const std::vector<float>& q, int k,int efSearch) const {
    check_dim(q);
    if (entry_id_ <0) return {};
    if (k<= 0) return {};
    if (efSearch <=0) efSearch =1;
    int ep = entry_id_;

    for (int lc = maxlevel_; lc >0; --lc){
        ep = search_layer_greedy(q,ep,lc);

    }
    std::vector<int> W = search_layer_beam(q,ep,0,efSearch);
    if (static_cast<int>(W.size()) >k) W.resize(k);
    return W;

}


int HNSW_NEW::insert(const std::vector<float>& vec_in, int node_id)  {
    check_dim(vec_in);

    // assign id
    if (node_id < 0) {
        // not perfect (ids could be non-contiguous if user inserts custom ids),
        // but matches your current approach
        node_id = static_cast<int>(vectors_.size());
        while (vectors_.find(node_id) != vectors_.end()) ++node_id;
    }
    if (vectors_.find(node_id) != vectors_.end()) {
        throw std::runtime_error("Node id already exists: " + std::to_string(node_id));
    }

    vectors_[node_id] = vec_in;

    int l = sample_level();

    // first node
    if (entry_id_ < 0) {
        ensure_layers(l);
        // create empty adjacency set for node on each layer it belongs to
        for (int lc = 0; lc <= l; ++lc) {
            layers_[lc][node_id] = std::unordered_set<int>{};
        }
        entry_id_ = node_id;
        maxlevel_ = l;
        return node_id;
    }

    int old_top = maxlevel_;
    if (l > maxlevel_) {
        ensure_layers(l);
        maxlevel_ = l;
    }

    int ep = entry_id_;
    int L = old_top;

    // Phase 1: greedy from top down to level l+1
    for (int lc = L; lc > l; --lc) {
        ep = search_layer_greedy(vec_in, ep, lc);
    }

    // Phase 2: from min(L,l) down to 0: searchLayer + selectNeighbors + connect + prune
    for (int lc = std::min(L, l); lc >= 0; --lc) {
        // ensure node exists in this layer adjacency map
        layers_[lc].try_emplace(node_id, std::unordered_set<int>{});

        // W = searchLayer(q, ep, lc, efConstruction)
        std::vector<int> W = search_layer_beam(vec_in, ep, lc, efConstruction_);

        int Mmax = (lc == 0 ? M0_ : M_);

        std::vector<int> neighbors;
        if (use_heuristic_) {
            neighbors = select_neighbors_heuristic_paper(
                vec_in, W, lc, Mmax,
                /*extendCandidates=*/true,
                /*keepPruned=*/false
            );
        } else {
            neighbors = select_neighbors_simple(vec_in, W, lc, Mmax);
        }

        // connect bidirectionally
        for (int nb : neighbors) {
            layers_[lc].try_emplace(nb, std::unordered_set<int>{});
            layers_[lc][node_id].insert(nb);
            layers_[lc][nb].insert(node_id);

            // prune nb if it exceeds Mmax
            if (static_cast<int>(layers_[lc][nb].size()) > Mmax) {
                prune_connection(nb, lc, Mmax);
            }
        }

        // update ep to the best candidate for next layer down
        if (!W.empty()) ep = W.front();
    }

    // if new node has higher level, it becomes the new entry point
    if (l > old_top) {
        entry_id_ = node_id;
    }

    return node_id;
}

void HNSW_NEW::search(const std::vector<std::vector<float>>& Xq,int k ,int efSearch, std::vector<std::vector<float>>& D, std::vector<std::vector<int>>& I) const{
    D.assign(Xq.size(), std::vector<float>(k, std::numeric_limits<float>::infinity()));
    I.assign(Xq.size(), std::vector<int>(k, -1));

    for (size_t i = 0; i < Xq.size(); ++i) {
        const auto& q = Xq[i];
        auto ids = query(q, k, efSearch);

        for (size_t j = 0; j < ids.size(); ++j) {
            I[i][j] = ids[j];
            D[i][j] = dist(q, vectors_.at(ids[j]));
        }
    }
}

const std::vector<int>& HNSW_NEW::darth_neigh_cb(int node_id, void* ctx) {
    auto* self = static_cast<HNSW_NEW*>(ctx);

    // Cached?
    if (self->darth_neigh_owner_ == node_id) return self->darth_neigh_buf_;

    self->darth_neigh_owner_ = node_id;
    self->darth_neigh_buf_.clear();

    if (self->layers_.empty()) return self->darth_neigh_buf_;

    auto it = self->layers_[0].find(node_id);
    if (it == self->layers_[0].end()) return self->darth_neigh_buf_;

    const auto& s = it->second; // unordered_set<int>
    self->darth_neigh_buf_.reserve(s.size());
    for (int nb : s) self->darth_neigh_buf_.push_back(nb);

    return self->darth_neigh_buf_;
}

float HNSW_NEW::darth_dist_cb(const float* q, int node_id, void* ctx) {
    auto* self = static_cast<HNSW_NEW*>(ctx);

    auto it = self->vectors_.find(node_id);
    if (it == self->vectors_.end()) return std::numeric_limits<float>::infinity();
    const std::vector<float>& b = it->second;

    if (self->metric_ == Metric::L2) {
        double s = 0.0;
        for (int i = 0; i < self->dim_; ++i) {
            double d = (double)q[i] - (double)b[i];
            s += d * d;
        }
        return (float)s; // squared L2 (matches your dist())
    } else {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < self->dim_; ++i) {
            double da = (double)q[i];
            double db = (double)b[i];
            dot += da * db;
            na += da * da;
            nb += db * db;
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom <= 0.0) return 1.0f;
        double c = dot / denom;
        return (float)(1.0 - c);
    }
}

std::vector<int> HNSW_NEW::query_darth(const std::vector<float>& q,
                                      const DarthParams& params,
                                      const IRecallPredictor& predictor) const {
    check_dim(q);
    if (entry_id_ < 0) return {};
    if (params.k <= 0) return {};

    int ep = entry_id_;

    // upper layers greedy
    for (int lc = maxlevel_; lc > 0; --lc) {
        ep = search_layer_greedy(q, ep, lc);
    }

    // base layer: DARTH
    return DarthSearcher::search_layer0_darth(
        q.data(),
        ep,
        params,
        predictor,
        &HNSW_NEW::darth_neigh_cb,
        &HNSW_NEW::darth_dist_cb,
        (void*)this
    );
}

void HNSW_NEW::search_darth(const std::vector<std::vector<float>>& Xq,
                           const DarthParams& params,
                           const IRecallPredictor& predictor,
                           std::vector<std::vector<float>>& D,
                           std::vector<std::vector<int>>& I) const {
    int k = params.k;
    D.assign(Xq.size(), std::vector<float>(k, std::numeric_limits<float>::infinity()));
    I.assign(Xq.size(), std::vector<int>(k, -1));

    for (size_t i = 0; i < Xq.size(); ++i) {
        const auto& q = Xq[i];
        auto ids = query_darth(q, params, predictor);

        for (size_t j = 0; j < ids.size() && (int)j < k; ++j) {
            I[i][j] = ids[j];
            D[i][j] = dist(q, vectors_.at(ids[j]));
        }
    }
}

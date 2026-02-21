#include "hnsw_new.h"
#include <cmath>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <cstring>
void HNSW_NEW::check_dim(const std::vector<float>& v) const{
    if (static_cast<int>(v.size()) != dim_){
        throw std::runtime_error("Vector dimension mismatch: expected " + std::to_string(dim_) + ", got" + std::to_string(v.size()));
    }
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

//level sampling
int HNSW_NEW::sample_level() const{
    double U = unif_(rng_);
    if (U <1e-12) U = 1e-12;
    return static_cast<int>(-std::log(U) * mL_);
}
// updating levels
void HNSW_NEW::ensure_node_capacity(int slot,int upto_level){
    for (int lc = 0; lc <= upto_level; ++lc){
        auto &adj = layers_[lc];
        if ((int)adj.size() <= slot) adj.resize(slot+1);
    }
}

void HNSW_NEW::ensure_layers(int new_maxlevel){
    if (new_maxlevel < 0) return ;
    if ((int)layers_.size() <= new_maxlevel){
        layers_.resize(new_maxlevel+1);
    }
}

//greedy search 
int HNSW_NEW::search_layer_greedy(const std::vector<float>& q, int ep, int layer) const {
    int best = ep;
    const float* qptr = q.data();
    float q_inv_norm = 1.0f;
    if (metric_ == Metric::Cosine) q_inv_norm = inv_norm_ptr(qptr, dim_);

    float bestDist = dist_ptr(qptr, best, q_inv_norm);

    while (true) {
        bool improved = false;

        if (best < 0 || best >= (int)level_.size() || level_[best] < layer) break;
        const auto& neigh = layers_[layer][best];

        for (int nb : neigh) {
            float d = dist_ptr(qptr, nb, q_inv_norm);
            if (d < bestDist) {
                bestDist = d;
                best = nb;
                improved = true;
            }
        }
        if (!improved) break;
    }
    return best;
}


std::vector<int> HNSW_NEW::search_layer_beam(const std::vector<float>& q, int ep, int layer, int ef) const {
    using Item = std::pair<float,int>;
    if (layer < 0 || layer >= (int)layers_.size()) return {};
    if (ep < 0) return {};
    if (ef <= 0) return {};

    const float* qptr = q.data();
    float q_inv_norm = 1.0f;
    if (metric_ == Metric::Cosine) q_inv_norm = inv_norm_ptr(qptr, dim_);

    begin_search_tag();

    auto& C = scratch_.C;   // min-heap by distance
    auto& W = scratch_.W;   // max-heap by distance
    auto& tmp = scratch_.tmp;

    C.clear(); W.clear(); tmp.clear();
    C.reserve((size_t)ef * 2);
    W.reserve((size_t)ef + 1);
    tmp.reserve((size_t)ef + 1);

    auto min_cmp = [](const auto& a, const auto& b) { return a.first > b.first; }; // smallest on top
    auto max_cmp = [](const auto& a, const auto& b) { return a.first < b.first; }; // largest on top

    auto heap_push = [](auto& h, const auto& item, auto cmp) {
        h.push_back(item);
        std::push_heap(h.begin(), h.end(), cmp);
    };
    auto heap_pop = [](auto& h, auto cmp) {
        std::pop_heap(h.begin(), h.end(), cmp);
        auto item = h.back();
        h.pop_back();
        return item;
    };

    float dist_ep = dist_ptr(qptr, ep,q_inv_norm);
    mark_visited(ep);

    heap_push(C, Item(dist_ep, ep), min_cmp);
    heap_push(W, Item(dist_ep,ep), max_cmp);

    while (!C.empty()) {
        auto [dist_c, c_id] = heap_pop(C, min_cmp);

        float worstDist = W.front().first; // max-heap: worst is at front
        if (dist_c > worstDist) break;

        if (c_id < 0 || c_id >= (int)level_.size() || level_[c_id] < layer) continue;
        const auto& neigh = layers_[layer][c_id];

        for (int nb : neigh) {
            if (is_visited(nb)) continue;
            

            float d = dist_ptr(qptr, nb,q_inv_norm);

            if ((int)W.size() < ef) {
                mark_visited(nb);
                heap_push(C, Item(d, nb), min_cmp);
                heap_push(W, Item(d, nb), max_cmp);
            } else {
                float worstDist = W.front().first;
                if (d < worstDist) {
                    mark_visited(nb);
                    heap_push(C, Item(d, nb), min_cmp);

                    // replace worst in W
                    std::pop_heap(W.begin(), W.end(), max_cmp);
                    W.back() = {d, nb};
                    std::push_heap(W.begin(), W.end(), max_cmp);
                }
            }
        }
    }

    tmp = W;
    std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> out;
    out.reserve(tmp.size());
    for (auto& p : tmp) out.push_back(p.second);
    return out;
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
    const float* qptr = q.data();
    float q_inv_norm = 1.0f;
    if (metric_ == Metric::Cosine) q_inv_norm = inv_norm_ptr(qptr, dim_);

    for (int nb : uniq){
        dlist.push_back({dist_ptr(qptr, nb, q_inv_norm), nb});
    }
    std::sort(dlist.begin(),dlist.end(), [](const auto& a, const auto& b){return a.first < b.first;});

    std::vector<int> out;
    out.reserve(Mmax);
    for (int i =0; i<Mmax; ++i) out.push_back(dlist[i].second);
    return out;
}

std::vector<int> HNSW_NEW::select_neighbors_heuristic_paper_slot(
    int qslot,
    const std::vector<int>& candidates,
    int layer, int Mmax,
    bool extendCandidates,
    bool keepPruned
) const {
    if (Mmax <= 0) return {};

    std::unordered_set<int> Wset;
    Wset.reserve(candidates.size()*3 + 1);
    for (int id : candidates) { 
        if (id == qslot) continue; 
        Wset.insert(id);
    }

    if (extendCandidates) {
        std::vector<int> base(Wset.begin(), Wset.end());
        for (int e : base) {
            if (e < 0 || e >= (int)level_.size() || level_[e] < layer) continue;
            for (int adj : layers_[layer][e]) Wset.insert(adj);
        }
    }

    std::vector<std::pair<float,int>> cand;
    cand.reserve(Wset.size());
    for (int e : Wset) {
        cand.push_back({dist_slots(qslot, e), e});
    }
    std::sort(cand.begin(), cand.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    std::vector<int> R;
    R.reserve(Mmax);

    std::vector<std::pair<float,int>> discarded;
    discarded.reserve(cand.size());

    for (auto& [d_qe, e] : cand) {
        bool good = true;
        for (int r : R) {
            float d_er = dist_slots(e, r);
            if (d_er < d_qe) { good = false; break; }
        }
        if (good) {
            R.push_back(e);
            if ((int)R.size() == Mmax) break;
        } else {
            discarded.push_back({d_qe, e});
        }
    }

    if (keepPruned && (int)R.size() < Mmax) {
        std::sort(discarded.begin(), discarded.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
        for (auto& p : discarded) {
            int e = p.second;
            bool exists = false;
            for (int x : R) if (x == e) { exists = true; break; }
            if (!exists) {
                R.push_back(e);
                if ((int)R.size() == Mmax) break;
            }
        }
    }
    return R;
}


std::vector<int> HNSW_NEW::prune_connection(int node_id, int layer, int Mmax) {
    if (node_id < 0 || node_id >= (int)level_.size() || level_[node_id] < layer) return {};
    auto& neigh = layers_[layer][node_id];
    if ((int)neigh.size() <= Mmax) return neigh;

    std::vector<int> neighbors = neigh;

    std::vector<int> newList;
    if (use_heuristic_) {
        newList = select_neighbors_heuristic_paper_slot(node_id, neighbors, layer, Mmax, true, false);
    } else {
        // you can also create a slot-based simple selector similarly, or keep your old one
        // but old one needs q vector; better to write select_neighbors_simple_slot too.
        newList = select_neighbors_heuristic_paper_slot(node_id, neighbors, layer, Mmax, false, true);
    }

    std::unordered_set<int> keep;
    keep.reserve(newList.size()*2 + 1);
    for (int x : newList) keep.insert(x);

    for (int old_nb : neigh) {
        if (!keep.count(old_nb)) {
            if (old_nb >= 0 && old_nb < (int)level_.size() && level_[old_nb] >= layer) {
                erase_swap(layers_[layer][old_nb], node_id);
            }
        }
    }

    neigh = std::move(newList);
    return neigh;
}

std::vector<int> HNSW_NEW::query(const std::vector<float>& q, int k, int efSearch) const {
    auto slots = query_slots(q, k, efSearch);
    for (int& s : slots) s = slot_to_id_[s];
    return slots;
}

std::vector<int> HNSW_NEW::query_slots(const std::vector<float>& q, int k, int efSearch) const {
    check_dim(q);
    if (entry_id_ < 0) return {};
    if (k <= 0) return {};
    if (efSearch <= 0) efSearch = 1;

    int ep = entry_id_;
    for (int lc = maxlevel_; lc > 0; --lc) {
        ep = search_layer_greedy(q, ep, lc);
    }

    auto slots = search_layer_beam(q, ep, 0, efSearch);
    if ((int)slots.size() > k) slots.resize(k);
    return slots;
}


int HNSW_NEW::insert(const std::vector<float>& vec_in, int node_id) {
    check_dim(vec_in);

    // assign external id if not provided
    if (node_id < 0) {
        node_id = (int)slot_to_id_.size();
        while (id_to_slot_.find(node_id) != id_to_slot_.end()) ++node_id;
    }
    if (id_to_slot_.find(node_id) != id_to_slot_.end()) {
        throw std::runtime_error("Node id already exists: " + std::to_string(node_id));
    }

    // create slot and store vector contiguously
    int slot = (int)slot_to_id_.size();
    slot_to_id_.push_back(node_id);
    id_to_slot_[node_id] = slot;
    size_t off = data_.size();
    data_.resize(off + (size_t)dim_);
    std::memcpy(data_.data()+off,vec_in.data(),(size_t)dim_ * sizeof(float));
    visited_tag_.resize(slot_to_id_.size(),0);

    if (metric_ == Metric::Cosine){
        const float* vptr = data_.data()+off;
        inv_norms_.push_back(inv_norm_ptr(vptr,dim_));
    }else{
        inv_norms_.push_back(1.0f); // dummy value for L2
    }

    int l = sample_level();
    level_.push_back(l);
    ensure_layers(std::max(maxlevel_, l));
    ensure_node_capacity(slot,l);
    // first node
    if (entry_id_ < 0) {
        ensure_layers(l);
        ensure_node_capacity(slot,l);
        for (int lc = 0; lc <= l; ++lc) {
            int Mmax = (lc == 0 ? M0_ : M_);
            layers_[lc][slot].clear();
            layers_[lc][slot].reserve((size_t)Mmax+1);
        }
        entry_id_ = slot;
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
        ep = search_layer_greedy(vec_in, ep, lc); // vec_in used as query vector, OK
    }

    // Phase 2
    for (int lc = std::min(L, l); lc >= 0; --lc) {
    int Mmax = (lc == 0 ? M0_ : M_);

    layers_[lc][slot].reserve((size_t)Mmax + 1);
    
    std::vector<int> W = search_layer_beam(vec_in, ep, lc, efConstruction_);

    std::vector<int> neighbors;
    if (use_heuristic_) {
        neighbors = select_neighbors_heuristic_paper_slot(slot, W, lc, Mmax, true, false);
    } else {
        neighbors = select_neighbors_simple(vec_in, W, lc, Mmax);
    }
    auto& a = layers_[lc][slot];
    for (int nb : neighbors) {
        if (nb < 0 || nb >= (int)level_.size() || level_[nb] < lc) continue;
        
        auto& b = layers_[lc][nb];
        b.reserve((size_t)Mmax+1);
        a.push_back(nb);
        bool exists = false;
        for (int x : b) {
            if (x == slot) {
                exists = true;
                break;
            }
        }
        if (!exists) b.push_back(slot);

        if ((int)b.size() > Mmax) {
            prune_connection(nb, lc, Mmax);
        }
    }

    if (!W.empty()) ep = W.front();
    }


    if (l > old_top) {
        entry_id_ = slot;
    }

    return node_id;
}


void HNSW_NEW::search(const std::vector<std::vector<float>>& Xq,
                      int k, int efSearch,
                      std::vector<std::vector<float>>& D,
                      std::vector<std::vector<int>>& I) const {
    D.assign(Xq.size(), std::vector<float>(k, std::numeric_limits<float>::infinity()));
    I.assign(Xq.size(), std::vector<int>(k, -1));

    for (size_t i = 0; i < Xq.size(); ++i) {
        const auto& q = Xq[i];
        check_dim(q);

        const float* qptr = q.data();
        float q_inv_norm = 1.0f;
        if (metric_ == Metric::Cosine) {
            q_inv_norm = inv_norm_ptr(qptr, dim_);
        }

        auto slots = query_slots(q, k, efSearch);

        for (size_t j = 0; j < slots.size(); ++j) {
            int slot = slots[j];
            I[i][j] = slot_to_id_[slot];
            D[i][j] = dist_ptr(qptr, slot, q_inv_norm);  // <-- use cached q_inv_norm
        }
    }
}


#pragma once

#include <vector>
#include <unordered_set>
#include <queue>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <string>

class HNSWPiPIndex {
public:
    HNSWPiPIndex(int dim,
                 int M = 16,
                 int efConstruction = 200,
                 const std::string& metric = "l2",
                 uint32_t seed = 42,
                 double pip_gamma = 95.0,
                 int pip_delta = 20);

    void add(const std::vector<std::vector<float>>& X);
    int add_point(const std::vector<float>& vec, int node_id = -1);

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>>
    search(const std::vector<std::vector<float>>& Xq, int k, int efSearch) const;

    int entry_point() const { return entry_id_; }
    int max_level() const { return maxlevel_; }

private:
    struct CandidateMin {
        float dist;
        int id;
        bool operator>(const CandidateMin& other) const {
            return dist > other.dist;
        }
    };

    struct CandidateMax {
        float dist;
        int id;
        bool operator<(const CandidateMax& other) const {
            return dist < other.dist;
        }
    };

    int dim_;
    int M_;
    int M0_;
    int efConstruction_;
    std::string metric_;
    int maxlevel_;
    int entry_id_;
    double mL_;
    bool use_heuristic_;
    double pip_gamma_;
    int pip_delta_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> uni_;

    std::vector<std::unordered_set<int>> layer_dummy_sizes_;
    std::vector<std::vector<std::unordered_set<int>>> layers_;
    std::vector<std::vector<float>> vectors_;

private:
    float dist(const std::vector<float>& a, const std::vector<float>& b) const;
    int sample_level();
    int search_layer_greedy(const std::vector<float>& q, int ep, int lc) const;

    std::vector<int> search_layer_standard(const std::vector<float>& q,
                                           int ep,
                                           int layer,
                                           int ef) const;

    std::vector<int> search_layer_pip(const std::vector<float>& q,
                                      int ep,
                                      int layer,
                                      int ef,
                                      int k) const;

    std::vector<int> select_neighbors_simple(const std::vector<float>& q,
                                             const std::vector<int>& candidates,
                                             int Mmax) const;

    std::vector<int> select_neighbors_heuristic(const std::vector<float>& q,
                                                const std::vector<int>& candidates,
                                                int layer,
                                                int Mmax,
                                                bool extend_candidates = true,
                                                bool keep_pruned_connections = false) const;

    std::unordered_set<int> prune_connections(int node_id, int layer, int Mmax);

    std::vector<int> query_pip(const std::vector<float>& q, int K, int efSearch) const;

    static std::vector<int> heap_topk_ids(const std::priority_queue<CandidateMax>& heap, int k);
};
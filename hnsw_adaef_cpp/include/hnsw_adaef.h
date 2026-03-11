#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <string>

class HNSWAdaEFIndex {
public:
    HNSWAdaEFIndex(int dim,
                   int M = 16,
                   int efConstruction = 200,
                   const std::string& metric = "cosine",
                   uint32_t seed = 42,
                   int adaef_bins = 5,
                   double adaef_delta = 0.001,
                   int adaef_sample_size = 200);

    void add(const std::vector<std::vector<float>>& X);
    int add_point(const std::vector<float>& vec, int node_id = -1);

    void build_adaef_offline(int k,
                             double target_recall = 0.95,
                             const std::vector<int>& ef_values = {50, 75, 100, 150, 200, 300, 400, 500});

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>>
    search(const std::vector<std::vector<float>>& Xq, int k, double target_recall) const;

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

    int adaef_bins_;
    double adaef_delta_;
    int adaef_sample_size_;
    bool offline_ready_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> uni_;

    std::vector<std::vector<std::unordered_set<int>>> layers_;
    std::vector<std::vector<float>> vectors_;

    std::vector<double> dataset_mean_;
    std::vector<std::vector<double>> dataset_cov_;
    std::unordered_map<int, std::vector<std::pair<int, double>>> ef_estimation_table_;
    std::unordered_map<int, double> wae_by_target_;

private:
    float dist(const std::vector<float>& a, const std::vector<float>& b) const;
    int sample_level();
    int search_layer_greedy(const std::vector<float>& q, int ep, int lc) const;

    std::vector<int> search_layer_standard(const std::vector<float>& q,
                                           int ep,
                                           int layer,
                                           int ef) const;

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

    void compute_dataset_statistics();
    std::pair<double, double> estimate_fdl_params(const std::vector<float>& q) const;
    std::vector<double> compute_bins(double mu, double sigma) const;
    double compute_query_score(const std::vector<float>& q, const std::vector<float>& D) const;

    std::vector<int> exact_knn_excluding_self(const std::vector<float>& q, int qid, int k) const;
    double recall_at_k(const std::vector<int>& gt, const std::vector<int>& pred, int k) const;

    int entry_after_upper_layers(const std::vector<float>& q) const;
    int two_hop_size(int ep_id, int layer = 0) const;
    std::vector<float> collect_distance_list(const std::vector<float>& q, int ep_id, int layer = 0) const;

    int estimate_ef(const std::vector<float>& q, const std::vector<float>& D, double target_recall) const;
    std::vector<int> search_layer_adaef(const std::vector<float>& q,
                                        int ep_id,
                                        int layer,
                                        double target_recall) const;
    std::vector<int> query_adaef(const std::vector<float>& q, int K, double target_recall) const;

    static double inv_norm_cdf(double p);
};
#pragma once 

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cstdint>
#include <limits>
#include <string>

class HNSW_NEW {
    public: 
        enum class Metric {L2,Cosine};
        HNSW_NEW(int dim, int M, int efConstruction, Metric metric = Metric::L2, uint64_t seed=42);

        int insert(const std::vector<float>& vec, int node_id=-1);
        std::vector<int> query(const std::vector<float>& q, int k ,int efSearch) const;
        void search(const std::vector<std::vector<float>>& Xq, int k ,int efSearch, std::vector<std::vector<float>>& D, std::vector<std::vector<int>>& I) const;

        int max_level() const {return maxlevel_;}
        int entry_point() const {return entry_id_;}
        int size() const {return static_cast<int>(vectors_.size());}
        void set_use_heuristic(bool v) {use_heuristic_ = v;}
        bool use_heuristic() const {return use_heuristic;}
    
    private:
        int dim_;
        int M_;
        int M0_;
        int efConstruction_;
        Metric metric_;

        std::vector<std::unordered_map<int, std::unordered_set<int>>> layers_;
        std::unordered_map<int, std::vector<float>> vectors_;
        int maxlevel_;
        int entry_id_;
        double mL_;
        //RNG
        mutable std::mt19937_64 rng_;
        mutable std::uniform_real_distribution<double> unif_;
        bool use_heuristic_;

        void ensure_layers(int new_maxlevel);
        int sample_level() const;
        float dist(const std::vector<float>& a , const std::vector<float>& b) const;

        int search_layer_greedy(const std::vector<float>& q,int ep, int layer) const;

        std::vector<int> search_layer_beam(const std::vector<float>& q, int ep, int layer, int ef ) const;

        std::vector<int> select_neighbors_simple(const std::vector<float>& q, const std::vector<int>& candidates, int layer, int Mmax) const;

        std::vector<int> select_neighbors_heuristic_paper(const std::vector<float>& q, const std::vector<int>& candidates, int layer, int Mmax, bool extendCandidates, bool keepPruned) const;

        std::unordered_set<int> prune_connection(int node_id,int layer, int Mmax);
        void check_dim(const std::vector<float>& v) const;
};
#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <random>
#include <limits>
#include <cstdint>


class HNSW_DARTH{
    public:
        enum class Metric {L2,Cosine};

        struct DarthFeatures{
            int ndis=0;
            int nstep=0;
            int ninserts=0;
            float firstNN = std::numeric_limits<float>::infinity();
            float closestNN = std::numeric_limits<float>::infinity();
            float furthestNN = std::numeric_limits<float>::infinity();
            float meanNN = std::numeric_limits<float>::infinity();
            float varNN = 0.0f;
            float p25NN = std::numeric_limits<float>::infinity();
            float p50NN = std::numeric_limits<float>::infinity();
            float p75NN = std::numeric_limits<float>::infinity();
        };
        // predictor interface
        struct IPredictor{
            virtual ~IPredictor() = default;
            virtual float predict(const DarthFeatures& f) const = 0;
        };

        HNSW_DARTH(int dim,int M, int efConstruction , Metric metric = Metric::L2, uint64_t seed=42);

        //build
        int insert(const std::vector<float>& vec, int node_id=-1);

        // -----------------------------
        // Query APIs
        // -----------------------------
        // Classic HNSW query (no DARTH). Returns top-k internal ids.
        std::vector<int> query(const std::vector<float>& q, int k, int efSearch) const;

        // DARTH query (early termination on layer 0). Returns top-k internal ids.
        std::vector<int> query_darth(const std::vector<float>& q,int k ,int efSearch, float Rt, const IPredictor& predictor, int ipi =200,int mpi=20) const;

        // -----------------------------
        // Batch search (exposed for Python bindings)
        // -----------------------------
        void search(const std::vector<std::vector<float>>& Xq,
                    int k,
                    int efSearch,
                    std::vector<std::vector<float>>& D,
                    std::vector<std::vector<int>>& I) const;

        void search_darth(const std::vector<std::vector<float>>& Xq,
                          int k,
                          int efSearch,
                          float Rt,
                          const IPredictor& predictor,
                          int ipi,
                          int mpi,
                          std::vector<std::vector<float>>& D,
                          std::vector<std::vector<int>>& I) const;

        //basic info 
        int max_level() const {return maxlevel_;}
        int entry_point() const {return entry_id_;}
    private:
        //distance
        float dist(const std::vector<float>& a , const std::vector<float>& b) const;
        //sampling
        int sample_level() const;
        
        int search_layer_greedy(const std::vector<float>& q, int ep, int layer) const;
        std::vector<int> search_layer(const std::vector<float>& q, int ep_id, int lc, int ef) const;

        //paper heuristic
        std::vector<int> select_neighbors_heuristic_paper(const std::vector<float>& q,const std::vector<int>& candidates,int lc,int M,bool extend_candidates,bool keep_pruned_connections) const;

        //pruning
        void prune_connections(int node_id,int lc, int Mmax);
        
        //darth base layer search
        std::vector<int> search_layer_darth(const std::vector<float>& q,int ep_id,int lc,int efSearch,int k,float Rt, const IPredictor& predictor, int ipi,int mpi) const;

        static DarthFeatures darth_extract_features(const std::vector<std::pair<float,int>>& result_maxheap,int ndis,int nstep,float firstNN,int ninserts);

    private:
        int dim_;
        int M_;
        int M0_;
        int efConstruction_;
        Metric metric_;

        // graph storage: [level][node_id] -> neighbors
        std::vector<std::unordered_map<int,std::vector<int>>> layers_;

        // vectors_[id] is the stored point
        std::unordered_map<int, std::vector<float>> vectors_;

        int maxlevel_=-1;
        int entry_id_=-1;

        //level sampling
        double mL_;
        mutable std::mt19937_64 rng_;
        mutable std::uniform_real_distribution<double> unif_;

        bool use_heuristic_ =true;

    private:
        void check_dim(const std::vector<float>& v) const {
            if ((int)v.size() != dim_) throw std::runtime_error("dim mismatch");
        }
};
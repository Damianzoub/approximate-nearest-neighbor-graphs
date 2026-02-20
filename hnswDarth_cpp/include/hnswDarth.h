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

        //query
        std::vector<int> query_darth(const std::vector<float>& q,int k ,int efSearch, float Rt, const IPredictor& predictor, int ipi =200,int mpi=20) const;

        //basic info 
        int max_level() const {return maxlevel_;}
        int entry_point() const {return entry_id_;}
    private:
        using Adj = std::vector<int>;
        using Layer = std::unordered_map<int,Adj>;
        std::vector<Layer> layers_;

        //vector store
        std::unordered_map<int,std::vector<float>> vectors_;
        
        int dim_;
        int M_;
        int M0_;
        int efConstruction_;
        Metric metric_;
        int maxlevel_;
        int entry_id_;

        //level sampling
        double mL_;
        mutable std::mt19937_64 rng_;
        mutable std::uniform_real_distribution<double> unif_;

        bool use_heuristic_ =true;
    
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
};